import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from data.config import NOMINAL
from data.fixed_wing import DEFAULT_PRESET, PRESETS, STANDARD_GRAVITY_M_S2, UP, fly_steps
from filters.flags import DEFAULT_THRESHOLDS
from web import live_flight
from web.evade import (
    CLIMB_INTERVAL_S,
    JINK_INTERVAL_S,
    alternating_signs,
    beam_direction,
    climb_within_speed_limits,
    floor_binds,
    floor_recovery_direction,
    pull_up_drop_m,
    evade_rows,
    steer_row,
)

ENVELOPE = PRESETS[DEFAULT_PRESET]
DT_S = NOMINAL.dt
HEAD_ON_POSITION = np.array([0.0, 4000.0, 0.0])
HEAD_ON_VELOCITY = np.array([0.0, -110.0, 0.0])
NORTH_OF_SENSOR = np.array([0.0, 1000.0, 0.0])
EAST = np.array([100.0, 0.0, 0.0])
WEST = -EAST
ON_COURSE_EAST = EAST / 100.0
RIGHT_OF_EAST = np.array([0.0, -1.0, 0.0])
EVADE_WITHIN_S = 5.0
BEAM_TOLERANCE_DEG = 15.0
FLOAT_ROUNDING = 1e-12
JINK_OFF_BEAM_DEG = 5.0
CLIMB_ANGLE_DEG = 15.0
SEED = [8, 50]
ENGAGE_STEP = 10
LOWERED_AT_ROW = 100
LOWERED_MAX_LOAD_FACTOR = 3.0


class EnvelopeAtRow(dict):
    """fly_steps' envelope_changes, answered per row so a test can change the envelope mid-flight."""

    def __init__(self, envelope_at):
        super().__init__()
        self.envelope_at = envelope_at

    def get(self, step, default):
        return self.envelope_at(step)


def fly_evading(seed, steps, envelope_at=lambda row: ENVELOPE):
    latest = {"truth": (HEAD_ON_POSITION, HEAD_ON_VELOCITY, None)}
    recorded = []
    rows = evade_rows(
        lambda: envelope_at(len(recorded)), DT_S, np.random.default_rng(seed), lambda: latest["truth"]
    )

    def feedback_rows():
        for _ in range(steps):
            recorded.append(next(rows))
            yield recorded[-1]

    states = []
    flown = fly_steps(
        ENVELOPE, HEAD_ON_POSITION, HEAD_ON_VELOCITY, 0.0, feedback_rows(), DT_S, EnvelopeAtRow(envelope_at)
    )
    for state, lift in flown:
        latest["truth"] = (state[0:3], state[3:6], lift)
        states.append(state)
    return np.array(states), np.array(recorded)


def angle_to_line_of_sight_deg(position, velocity):
    return float(
        np.degrees(np.arccos(position @ velocity / (np.linalg.norm(position) * np.linalg.norm(velocity))))
    )


def test_head_on_target_flies_across_the_line_of_sight_within_5_s():
    states, _ = fly_evading(seed=SEED, steps=round(EVADE_WITHIN_S / DT_S) + 1)

    assert angle_to_line_of_sight_deg(states[0, 0:3], states[0, 3:6]) == 180.0
    assert abs(angle_to_line_of_sight_deg(states[-1, 0:3], states[-1, 3:6]) - 90.0) <= BEAM_TOLERANCE_DEG


@pytest.mark.parametrize(
    "jink, angle_deg", [(1.0, 90.0 - JINK_OFF_BEAM_DEG), (-1.0, 90.0 + JINK_OFF_BEAM_DEG)]
)
def test_beam_direction_is_a_unit_vector_off_the_beam_by_the_jink_angle(jink, angle_deg):
    direction = beam_direction(NORTH_OF_SENSOR, EAST, jink, 1.0)

    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=FLOAT_ROUNDING)
    assert angle_to_line_of_sight_deg(NORTH_OF_SENSOR, direction) == pytest.approx(
        angle_deg, abs=FLOAT_ROUNDING
    )


@pytest.mark.parametrize("velocity, east_sign", [(EAST, 1.0), (WEST, -1.0), (HEAD_ON_VELOCITY, -1.0)])
def test_beam_direction_crosses_to_the_side_the_target_leans_to_and_head_on_breaks_west_of_north(
    velocity, east_sign
):
    assert np.sign(beam_direction(NORTH_OF_SENSOR, velocity, 1.0, 1.0)[0]) == east_sign


@pytest.mark.parametrize("climb", [1.0, -1.0])
def test_beam_direction_climbs_or_dives_by_the_climb_angle_in_the_beam_plane(climb):
    direction = beam_direction(NORTH_OF_SENSOR, EAST, 1.0, climb)

    assert np.arctan2(direction[2], direction[0]) == pytest.approx(
        climb * np.deg2rad(CLIMB_ANGLE_DEG), abs=FLOAT_ROUNDING
    )


def test_beam_direction_straight_over_the_sensor_is_across_the_vertical_line_of_sight():
    overhead = np.array([0.0, 0.0, 1000.0])

    direction = beam_direction(overhead, EAST, 1.0, 1.0)

    assert angle_to_line_of_sight_deg(overhead, direction) == pytest.approx(
        90.0 - JINK_OFF_BEAM_DEG, abs=FLOAT_ROUNDING
    )
    assert direction[0] > 0.0


def test_on_course_wings_level_holds_one_g_at_full_throttle_without_rolling():
    assert steer_row(ENVELOPE, DT_S, EAST, UP, ON_COURSE_EAST) == (0.0, 1.0, 1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("turn, roll_rate_sign", [(RIGHT_OF_EAST, 1.0), (-RIGHT_OF_EAST, -1.0)])
def test_a_level_turn_off_the_nose_rolls_toward_it_at_the_envelope_roll_rate(turn, roll_rate_sign):
    roll_rate, load_factor, *_ = steer_row(ENVELOPE, DT_S, EAST, UP, turn)

    assert roll_rate == roll_rate_sign * ENVELOPE.max_roll_rate_rad_s
    assert load_factor == 1.0


def test_a_pull_straight_up_off_the_nose_is_capped_at_the_envelope_max_g():
    assert steer_row(ENVELOPE, DT_S, EAST, UP, UP)[1] == ENVELOPE.max_load_factor


def test_inverted_on_course_pushes_one_negative_g_and_rolls_upright():
    roll_rate, load_factor, *_ = steer_row(ENVELOPE, DT_S, EAST, -UP, ON_COURSE_EAST)

    assert (roll_rate, load_factor) == (ENVELOPE.max_roll_rate_rad_s, -1.0)


@pytest.mark.parametrize(
    "envelope, push_limit",
    [
        (ENVELOPE, ENVELOPE.max_negative_load_factor),
        (replace(ENVELOPE, max_negative_load_factor=None), -ENVELOPE.max_load_factor),
    ],
)
def test_inverted_pull_up_is_a_push_capped_at_the_envelope_negative_g(envelope, push_limit):
    assert steer_row(envelope, DT_S, EAST, -UP, UP)[1] == push_limit


def test_a_small_roll_is_taken_in_one_row():
    lift = np.array([0.0, -np.sin(0.01), np.cos(0.01)])

    roll_rate, *_ = steer_row(ENVELOPE, DT_S, EAST, lift, ON_COURSE_EAST)

    assert roll_rate * DT_S == pytest.approx(-0.01, abs=FLOAT_ROUNDING)


def run_lengths(signs, runs):
    lengths, current, count = [], next(signs), 1
    while len(lengths) < runs:
        sign = next(signs)
        if sign == current:
            count += 1
            continue
        assert sign == -current
        lengths.append(count)
        current, count = sign, 1
    return lengths


def test_jink_runs_last_every_whole_number_of_rows_from_one_to_three_seconds_inclusive():
    lengths = run_lengths(alternating_signs(np.random.default_rng(0), JINK_INTERVAL_S, DT_S), 2000)

    assert set(lengths) == set(range(50, 151))


def test_first_jink_side_is_drawn_both_ways_across_seeds():
    first = {
        next(alternating_signs(np.random.default_rng(seed), JINK_INTERVAL_S, DT_S)) for seed in range(20)
    }

    assert first == {-1.0, 1.0}


@pytest.mark.parametrize(
    "speed, climb, expected",
    [
        (ENVELOPE.max_speed_m_s, -1.0, 1.0),
        (np.nextafter(ENVELOPE.max_speed_m_s, 0.0), -1.0, -1.0),
        (1.5 * ENVELOPE.min_speed_m_s, 1.0, -1.0),
        (np.nextafter(1.5 * ENVELOPE.min_speed_m_s, np.inf), 1.0, 1.0),
    ],
)
def test_climbs_at_max_speed_and_dives_at_one_and_a_half_times_stall_speed(speed, climb, expected):
    assert climb_within_speed_limits(ENVELOPE, speed, climb) == expected


def test_first_row_before_any_flown_lift_steers_from_wings_level_on_a_climb():
    climbing = np.array([100.0, 0.0, 10.0])
    wings_level = np.array([-10.0, 0.0, 100.0]) / np.hypot(10.0, 100.0)

    def first_row(lift):
        rows = evade_rows(
            lambda: ENVELOPE, DT_S, np.random.default_rng(0), lambda: (NORTH_OF_SENSOR, climbing, lift)
        )
        return next(rows)

    assert first_row(None) == pytest.approx(first_row(wings_level), abs=FLOAT_ROUNDING)


def test_climb_runs_last_every_whole_number_of_rows_from_two_to_five_seconds_inclusive():
    lengths = run_lengths(alternating_signs(np.random.default_rng(0), CLIMB_INTERVAL_S, DT_S), 3000)

    assert set(lengths) == set(range(100, 251))


def test_a_small_pull_up_off_the_nose_adds_the_g_that_turns_it_at_sine_of_the_angle_per_half_second():
    off_nose_rad = 0.05
    direction = np.array([np.cos(off_nose_rad), 0.0, np.sin(off_nose_rad)])

    _, load_factor, *_ = steer_row(ENVELOPE, DT_S, EAST, UP, direction)

    turn_rate_rad_s = np.sin(off_nose_rad) / 0.5
    assert load_factor == pytest.approx(
        1.0 + 100.0 * turn_rate_rad_s / STANDARD_GRAVITY_M_S2, abs=FLOAT_ROUNDING
    )


def test_same_seed_and_start_give_identical_rows_and_another_seed_does_not():
    _, rows = fly_evading(seed=SEED, steps=1000)
    _, again = fly_evading(seed=SEED, steps=1000)
    _, other = fly_evading(seed=[SEED[0], SEED[1] + 1], steps=1000)

    assert rows.tobytes() == again.tobytes()
    assert not np.array_equal(rows, other)


def test_every_row_stays_inside_the_envelope_and_reaches_its_roll_g_and_negative_g_limits():
    _, rows = fly_evading(seed=SEED, steps=1500)

    assert np.abs(rows[:, 0]).max() == ENVELOPE.max_roll_rate_rad_s
    assert (rows[:, 1].min(), rows[:, 1].max()) == (
        ENVELOPE.max_negative_load_factor,
        ENVELOPE.max_load_factor,
    )
    assert {tuple(row) for row in rows[:, 2:]} == {(1.0, 0.0, 0.0, 0.0)}


def test_climbs_and_dives_alternate():
    states, _ = fly_evading(seed=SEED, steps=1500)

    assert np.count_nonzero(np.diff(np.sign(states[:, 5]))) >= 4


def test_max_g_lowered_mid_flight_caps_the_rows_from_that_row_on():
    lowered = replace(ENVELOPE, max_load_factor=LOWERED_MAX_LOAD_FACTOR)

    _, rows = fly_evading(
        seed=SEED, steps=1000, envelope_at=lambda row: ENVELOPE if row < LOWERED_AT_ROW else lowered
    )

    assert rows[:LOWERED_AT_ROW, 1].max() == ENVELOPE.max_load_factor
    assert rows[LOWERED_AT_ROW:, 1].max() == LOWERED_MAX_LOAD_FACTOR


PAGE_IMM_TUNING = Path(__file__).resolve().parents[1] / "web" / "imm_tuning.json"
FLIGHT_SEED = 8
PILOT_ROW = (0.0, 1.0, 0.6, 0.0, 0.0, 0.0)
ROW_LOAD_FACTOR = 1


def head_on_flight(tmp_path, steps=ENGAGE_STEP, imm_best=None, thresholds=DEFAULT_THRESHOLDS):
    setup = {
        "seed": FLIGHT_SEED,
        "envelope": asdict(ENVELOPE),
        "noise_regime": "nominal",
        "thresholds": thresholds,
        "start": {"position": HEAD_ON_POSITION.tolist(), "velocity": HEAD_ON_VELOCITY.tolist()},
    }
    imm_best = imm_best or json.loads(PAGE_IMM_TUNING.read_text())["best"]
    flight = live_flight.LiveFlight(setup, imm_best, tmp_path)
    for _ in range(steps):
        flight.advance(PILOT_ROW)
    return flight


def test_engaging_evade_drops_every_hold_and_the_running_manoeuvre(tmp_path):
    flight = head_on_flight(tmp_path)
    for name in ("heading", "altitude", "speed", "bank"):
        flight.toggle_hold(name)
    flight.start_manoeuvre("jink")

    flight.engage_evade()
    flight.advance(PILOT_ROW)

    frame = flight.frame()
    assert (frame["evade"], frame["manoeuvre"], frame["holds"]) == (True, None, {})


def test_engaging_a_hold_disengages_evade(tmp_path):
    flight = head_on_flight(tmp_path)
    flight.engage_evade()

    flight.toggle_hold("altitude")
    flight.advance(PILOT_ROW)

    assert (flight.frame()["evade"], list(flight.frame()["holds"])) == (False, ["altitude"])


def test_starting_a_manoeuvre_disengages_evade(tmp_path):
    flight = head_on_flight(tmp_path)
    flight.engage_evade()

    flight.start_manoeuvre("jink")
    flight.advance(PILOT_ROW)

    assert (flight.frame()["evade"], flight.frame()["manoeuvre"]) == (False, "jink")


def test_disengaged_evade_hands_the_very_next_row_back_to_the_pilot(tmp_path):
    flight = head_on_flight(tmp_path)
    flight.engage_evade()
    flight.advance(PILOT_ROW)

    flight.disengage_evade()
    flight.advance(PILOT_ROW)

    assert flight.controls[-2] != list(PILOT_ROW)
    assert (flight.controls[-1], flight.frame()["evade"]) == (list(PILOT_ROW), False)


def test_evade_is_seeded_by_the_scenario_seed_and_the_engage_step_apart_from_the_flight_draws(
    tmp_path, monkeypatch
):
    seeded = []
    monkeypatch.setattr(
        live_flight, "evade_rows", lambda read_envelope, dt_s, rng, read_truth, sensor_altitude_m: seeded.append(rng)
    )
    flight = head_on_flight(tmp_path)

    flight.engage_evade()

    expected = np.random.default_rng([FLIGHT_SEED, ENGAGE_STEP]).bit_generator.state
    assert [rng.bit_generator.state for rng in seeded] == [expected]


def test_live_evade_reads_the_truth_the_flight_flew_and_its_lift(tmp_path, monkeypatch):
    read = []
    monkeypatch.setattr(
        live_flight, "evade_rows", lambda read_envelope, dt_s, rng, read_truth, sensor_altitude_m: read.append(read_truth)
    )
    flight = head_on_flight(tmp_path)

    flight.engage_evade()

    position, velocity, lift = read[0]()
    frame = flight.frame()
    assert (position.tolist(), velocity.tolist(), lift.tolist()) == (
        frame["truth_m"],
        frame["truth_velocity_m_s"],
        frame["truth_lift_direction"],
    )


def test_max_g_lowered_while_evading_caps_the_next_row(tmp_path):
    flight = head_on_flight(tmp_path)
    flight.engage_evade()
    while not flight.controls or flight.controls[-1][ROW_LOAD_FACTOR] != ENVELOPE.max_load_factor:
        flight.advance(PILOT_ROW)

    flight.set_envelope({**asdict(ENVELOPE), "max_load_factor": LOWERED_MAX_LOAD_FACTOR})
    flight.advance(PILOT_ROW)

    assert flight.controls[-1][ROW_LOAD_FACTOR] == LOWERED_MAX_LOAD_FACTOR


def test_evade_rows_are_the_same_whatever_the_filters_estimate_or_flag(tmp_path):
    default = head_on_flight(tmp_path)
    retuned = head_on_flight(
        tmp_path,
        imm_best={
            "p_stay": 0.9,
            "omega_ct_deg": 5.0,
            "ca_sigma_a": 9.0,
            "ca_tau": 0.5,
            "ct_sigma_a": 9.0,
            "cv_sigma_a": 3.0,
        },
        thresholds={**DEFAULT_THRESHOLDS, "raw_error_m": 0.1},
    )
    for flight in (default, retuned):
        flight.engage_evade()
        for _ in range(200):
            flight.advance(PILOT_ROW)

    assert default.readouts()["IMM"]["estimate_m"] != retuned.readouts()["IMM"]["estimate_m"]
    assert default.controls == retuned.controls


@pytest.mark.parametrize("preset", PRESETS)
def test_every_row_stays_inside_each_preset_envelope(preset):
    envelope = PRESETS[preset]
    push_limit = envelope.max_negative_load_factor

    _, rows = fly_evading(seed=SEED, steps=1500, envelope_at=lambda row: envelope)

    assert np.abs(rows[:, 0]).max() <= envelope.max_roll_rate_rad_s
    assert push_limit <= rows[:, 1].min() and rows[:, 1].max() <= envelope.max_load_factor


def test_once_on_the_beam_the_target_jinks_toward_and_away_from_the_sensor():
    states, _ = fly_evading(seed=SEED, steps=1500)

    on_beam = states[round(EVADE_WITHIN_S / DT_S) :]
    radial_speed = np.einsum("ij,ij->i", on_beam[:, 0:3], on_beam[:, 3:6])
    assert np.count_nonzero(np.diff(np.sign(radial_speed))) >= 4


FLOOR_SENSOR_ALTITUDE_M = 100.0
FLOOR_M = 280.0
FLOOR_START = {"position": [0.0, 4000.0, 300.0], "velocity": [0.0, -110.0, 0.0]}
FLOOR_FLIGHT_STEPS = 1500
DIVING_SEED = 5


def lowest_world_altitude_m(tmp_path, envelope):
    setup = {
        "seed": DIVING_SEED,
        "envelope": asdict(envelope),
        "noise_regime": "nominal",
        "thresholds": DEFAULT_THRESHOLDS,
        "start": FLOOR_START,
        "sensor": {"position": [0.0, 0.0, FLOOR_SENSOR_ALTITUDE_M]},
    }
    imm_best = json.loads(PAGE_IMM_TUNING.read_text())["best"]
    flight = live_flight.LiveFlight(setup, imm_best, tmp_path)
    flight.engage_evade()
    lowest = np.inf
    for _ in range(FLOOR_FLIGHT_STEPS):
        flight.advance(PILOT_ROW)
        lowest = min(lowest, flight.frame()["truth_m"][2])
    return lowest


def test_evading_stays_above_the_envelope_min_altitude_over_a_raised_sensor(tmp_path):
    assert lowest_world_altitude_m(tmp_path, replace(ENVELOPE, min_altitude_m=FLOOR_M)) >= FLOOR_M


def test_the_same_flight_without_a_floor_dives_below_it(tmp_path):
    assert lowest_world_altitude_m(tmp_path, replace(ENVELOPE, min_altitude_m=None)) < FLOOR_M


FLOOR_WORLD_M = 100.0
STRAIGHT_DOWN_M_S = np.array([0.0, 0.0, -100.0])
CLIMB_RAD = np.deg2rad(CLIMB_ANGLE_DEG)


def test_a_floor_binds_only_within_the_margin_above_it_when_not_descending():
    floor = replace(ENVELOPE, min_altitude_m=FLOOR_WORLD_M)
    level = np.array([0.0, 100.0, 0.0])
    edge_m = 110.0

    assert floor_binds(floor, edge_m, level)
    assert not floor_binds(floor, np.nextafter(edge_m, np.inf), level)


def test_no_floor_never_binds():
    assert not floor_binds(replace(ENVELOPE, min_altitude_m=None), -1e9, STRAIGHT_DOWN_M_S)


def test_a_climbing_target_loses_no_altitude_pulling_up():
    assert pull_up_drop_m(ENVELOPE, np.array([0.0, 100.0, 5.0])) == 0.0


def test_a_vertical_dive_drops_the_roll_and_turn_lag_at_the_dive_rate_plus_a_quarter_circle_at_spare_g():
    roll_and_turn_lag_s = 0.5 + np.pi / ENVELOPE.max_roll_rate_rad_s
    spare_g = ENVELOPE.max_load_factor - 1.0

    drop = pull_up_drop_m(ENVELOPE, STRAIGHT_DOWN_M_S)

    assert drop == pytest.approx(
        100.0 * roll_and_turn_lag_s + 100.0**2 / (STANDARD_GRAVITY_M_S2 * spare_g), abs=FLOAT_ROUNDING
    )


def test_pulling_up_with_no_lift_to_spare_drops_without_bound():
    stall_dive = np.array([0.0, 0.0, -ENVELOPE.min_speed_m_s])

    assert pull_up_drop_m(ENVELOPE, stall_dive) == np.inf


@pytest.mark.parametrize(
    "speed, climb_rad",
    [(np.nextafter(1.5 * ENVELOPE.min_speed_m_s, np.inf), CLIMB_RAD), (1.5 * ENVELOPE.min_speed_m_s, 0.0)],
)
def test_floor_recovery_climbs_along_the_beam_heading_above_forced_dive_speed_and_holds_level_at_or_below(
    speed, climb_rad
):
    direction = floor_recovery_direction(ENVELOPE, np.array([1.0, 0.0, -0.5]), np.array([0.0, speed, 0.0]))

    heading = [1.0, 0.0] if climb_rad else [0.0, 1.0]
    assert direction == pytest.approx(
        [np.cos(climb_rad) * heading[0], np.cos(climb_rad) * heading[1], np.sin(climb_rad)], abs=FLOAT_ROUNDING
    )


@pytest.mark.parametrize(
    "direction, velocity, heading",
    [
        (UP, np.array([100.0, 0.0, 0.0]), [1.0, 0.0]),
        (UP, np.array([0.0, 0.0, 100.0]), [0.0, 1.0]),
    ],
)
def test_floor_recovery_from_a_vertical_beam_direction_keeps_the_velocity_heading_then_north(
    direction, velocity, heading
):
    recovery = floor_recovery_direction(ENVELOPE, direction, velocity)

    assert recovery == pytest.approx(
        [np.cos(CLIMB_RAD) * heading[0], np.cos(CLIMB_RAD) * heading[1], np.sin(CLIMB_RAD)], abs=FLOAT_ROUNDING
    )


def test_a_floor_set_while_evading_turns_the_very_next_row_from_a_dive_to_the_climb_up():
    live = {"envelope": replace(ENVELOPE, min_altitude_m=None)}
    diving = (np.array([0.0, 4000.0, 0.0]), np.array([0.0, -110.0, -30.0]), UP)
    rows = evade_rows(lambda: live["envelope"], DT_S, np.random.default_rng(0), lambda: diving)
    unfloored = next(rows)

    live["envelope"] = replace(ENVELOPE, min_altitude_m=0.0)

    assert next(rows)[1] > unfloored[1]


@pytest.mark.parametrize("seed", range(8))
def test_evade_without_a_sensor_altitude_reads_the_truth_z_as_world_altitude(seed):
    level = (np.array([0.0, 4000.0, 110.0]), np.array([0.0, -110.0, 0.0]), UP)

    def first_row(floor):
        envelope = replace(ENVELOPE, min_altitude_m=floor)
        return next(evade_rows(lambda: envelope, DT_S, np.random.default_rng(seed), lambda: level))

    assert first_row(100.0) != first_row(None)
