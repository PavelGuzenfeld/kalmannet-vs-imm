import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from data.config import NOMINAL
from data.fixed_wing import (
    AFTERBURNER,
    LOAD_FACTOR,
    PRESETS,
    ROLL_RATE,
    SPEED_BRAKE,
    STANDARD_GRAVITY_M_S2,
    THROTTLE,
    fly_steps,
)
from data.trajectory_gen import fly_scenario
from filters.flags import DEFAULT_THRESHOLDS
from web.holds import (
    ALTITUDE,
    BANK,
    HEADING,
    SPEED,
    Holds,
)
from web.live_flight import LiveFlight

PAGE_IMM_TUNING = Path(__file__).resolve().parents[1] / "web" / "imm_tuning.json"
START_ALTITUDE_M = 2000.0
CRUISE = [0.0, 1.0, 0.6, 0.0, 0.0, 0.0]
FULL_THROTTLE = [0.0, 1.0, 1.0]
ROLL_IN_S = 0.5
PULL_UP_S = 2.0
PULL_UP_G = 2.5
TURN_S = 5.0
CHECK_WINDOW_S = 2.0
HEADING_TOLERANCE_DEG = 0.1
ALTITUDE_TOLERANCE_M = 0.1
SPEED_TOLERANCE_M_S = 0.1
HEADING_TIME_CONSTANT_S = 2.0
SPEED_TIME_CONSTANT_S = 2.0
ALTITUDE_NATURAL_FREQUENCY_RAD_S = 0.5
MAX_HOLD_BANK_RAD = np.deg2rad(30.0)
LEVEL_POSITION_M = np.array([0.0, 0.0, START_ALTITUDE_M])
ALTITUDE_OFFSET_M = 4.0
FLOAT_ROUNDING_M_S2 = 1e-9
FLOAT_ROUNDING_RAD = 1e-12
FLY_STEPS_ACCELERATION = slice(6, 9)


def flight(preset, tmp_path):
    envelope = PRESETS[preset]
    speed = envelope.best_glide_speed_m_s
    setup = {
        "seed": 7,
        "envelope": asdict(envelope),
        "noise_regime": "nominal",
        "thresholds": DEFAULT_THRESHOLDS,
        "start": {"position": [0.0, 0.0, START_ALTITUDE_M], "velocity": [0.0, speed, 0.0]},
    }
    return LiveFlight(setup, json.loads(PAGE_IMM_TUNING.read_text())["best"], tmp_path)


def fly(live, control, steps, manual_axes=()):
    frames = []
    for _ in range(steps):
        live.advance(control, manual_axes)
        frames.append(live.frame())
    return frames


def heading_rad(frame):
    east, north, _ = frame["truth_velocity_m_s"]
    return np.arctan2(east, north)


def test_replay_of_a_held_flight_reproduces_it_bit_for_bit(tmp_path):
    live = flight("benchmark", tmp_path)
    frames = fly(live, [0.8, 2.5, 1.0], 50)
    for hold in (HEADING, ALTITUDE, SPEED):
        live.toggle_hold(hold)
    frames += fly(live, CRUISE, 300)
    live.toggle_hold(BANK)
    frames += fly(live, CRUISE, 100)

    states, _ = fly_scenario(json.loads(live.scenario_json()))

    live_states = np.array([frame["truth_m"] + frame["truth_velocity_m_s"] for frame in frames])
    assert np.array_equal(states[:, :6], live_states)
    assert live.controls[-1] != CRUISE


def test_manual_pull_disengages_altitude_hold_and_flies_the_pull(tmp_path):
    live = flight("benchmark", tmp_path)
    live.toggle_hold(ALTITUDE)
    live.toggle_hold(SPEED)
    fly(live, CRUISE, 10)

    fly(live, [0.0, 3.0, 0.6], 1, manual_axes=[LOAD_FACTOR])

    assert list(live.frame()["holds"]) == [SPEED]
    assert live.controls[-1][LOAD_FACTOR] == 3.0


@pytest.mark.parametrize("roll_hold", [HEADING, BANK])
def test_manual_roll_disengages_the_roll_hold_only(roll_hold, tmp_path):
    live = flight("benchmark", tmp_path)
    for hold in (roll_hold, ALTITUDE, SPEED):
        live.toggle_hold(hold)

    fly(live, [0.5, 1.0, 0.6], 1, manual_axes=[ROLL_RATE])

    assert sorted(live.frame()["holds"]) == sorted([ALTITUDE, SPEED])
    assert live.controls[-1][ROLL_RATE] == 0.5


@pytest.mark.parametrize(
    "axis, manual_row",
    [
        (THROTTLE, [0.0, 1.0, 0.25, 0.0, 0.0, 0.0]),
        (SPEED_BRAKE, [0.0, 1.0, 0.6, 0.0, 1.0, 0.0]),
        (AFTERBURNER, [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
    ],
)
def test_manual_throttle_brake_or_afterburner_disengages_speed_hold(axis, manual_row, tmp_path):
    live = flight("benchmark", tmp_path)
    live.toggle_hold(SPEED)

    fly(live, manual_row, 1, manual_axes=[axis])

    assert live.frame()["holds"] == {}
    assert live.controls[-1] == manual_row


def test_engaging_bank_hold_releases_heading_hold_on_the_shared_roll_axis(tmp_path):
    live = flight("benchmark", tmp_path)
    live.toggle_hold(HEADING)
    live.toggle_hold(ALTITUDE)

    live.toggle_hold(BANK)

    assert sorted(live.holds.targets) == sorted([ALTITUDE, BANK])


def test_toggling_an_engaged_hold_disengages_it(tmp_path):
    live = flight("benchmark", tmp_path)
    live.toggle_hold(ALTITUDE)
    live.toggle_hold(ALTITUDE)

    fly(live, [0.0, 2.0, 0.6], 1)

    assert live.controls[-1] == [0.0, 2.0, 0.6, 0.0, 0.0, 0.0]


def wrapped_deg(angle_rad):
    return np.degrees((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def worst_error_once_settled(live, control, settle_s, error_of_frame):
    fly(live, control, round(settle_s / NOMINAL.dt))
    return max(abs(error_of_frame(frame)) for frame in fly(live, control, round(CHECK_WINDOW_S / NOMINAL.dt)))


@pytest.mark.parametrize("preset", PRESETS)
def test_heading_hold_engaged_in_a_rolling_turn_rolls_out_and_returns_to_the_engaged_heading(
    preset, tmp_path
):
    envelope = PRESETS[preset]
    live = flight(preset, tmp_path)
    rolling = fly(live, [envelope.max_roll_rate_rad_s, 1.0, 1.0], round(ROLL_IN_S / NOMINAL.dt))
    engaged = heading_rad(rolling[-1])

    live.toggle_hold(HEADING)

    settle_s = 5.0 * HEADING_TIME_CONSTANT_S + 2.0 * MAX_HOLD_BANK_RAD / envelope.max_roll_rate_rad_s
    error = worst_error_once_settled(
        live, FULL_THROTTLE, settle_s, lambda f: wrapped_deg(heading_rad(f) - engaged)
    )
    assert error <= HEADING_TOLERANCE_DEG


@pytest.mark.parametrize("preset", PRESETS)
def test_altitude_hold_engaged_in_a_climb_levels_off_at_the_engaged_altitude(preset, tmp_path):
    envelope = PRESETS[preset]
    live = flight(preset, tmp_path)
    pull_up = [0.0, min(PULL_UP_G, envelope.max_load_factor), 1.0]
    engaged = fly(live, pull_up, round(PULL_UP_S / NOMINAL.dt))[-1]["truth_m"][2]

    live.toggle_hold(ALTITUDE)

    settle_s = 10.0 / ALTITUDE_NATURAL_FREQUENCY_RAD_S
    error = worst_error_once_settled(live, FULL_THROTTLE, settle_s, lambda f: f["truth_m"][2] - engaged)
    assert error <= ALTITUDE_TOLERANCE_M


@pytest.mark.parametrize("preset", PRESETS)
def test_speed_hold_through_a_two_g_turn_recovers_the_engaged_speed(preset, tmp_path):
    envelope = PRESETS[preset]
    roll_steps = round(np.deg2rad(60.0) / envelope.max_roll_rate_rad_s / NOMINAL.dt)
    live = flight(preset, tmp_path)
    rolled_in = fly(live, [envelope.max_roll_rate_rad_s, 1.0, 0.6], roll_steps)[-1]
    engaged = np.linalg.norm(rolled_in["truth_velocity_m_s"])

    live.toggle_hold(SPEED)
    fly(live, [0.0, 2.0, 0.6], round(TURN_S / NOMINAL.dt), manual_axes=[LOAD_FACTOR])
    fly(live, [-envelope.max_roll_rate_rad_s, 1.0, 0.6], roll_steps, manual_axes=[ROLL_RATE])

    settle_s = 5.0 * SPEED_TIME_CONSTANT_S
    error = worst_error_once_settled(
        live, CRUISE, settle_s, lambda f: np.linalg.norm(f["truth_velocity_m_s"]) - engaged
    )
    assert error <= SPEED_TOLERANCE_M_S


@pytest.mark.parametrize("preset", PRESETS)
def test_bank_hold_engaged_mid_roll_overrides_the_residual_roll_and_stops_on_the_next_step(
    preset, tmp_path
):
    envelope = PRESETS[preset]
    rolling = [envelope.max_roll_rate_rad_s, 1.0, 0.6]
    live = flight(preset, tmp_path)
    fly(live, rolling, round(ROLL_IN_S / NOMINAL.dt))

    live.toggle_hold(BANK)
    held = fly(live, rolling, round(CHECK_WINDOW_S / NOMINAL.dt))

    assert {row[ROLL_RATE] for row in live.controls[-len(held) :]} == {0.0}


def first_step_acceleration(envelope, velocity_m_s, bank_rad, row):
    state, _ = next(fly_steps(envelope, LEVEL_POSITION_M, np.asarray(velocity_m_s), bank_rad, [row], NOMINAL.dt))
    return state[FLY_STEPS_ACCELERATION]


def vertical_lift_m_s2(acceleration, velocity_m_s):
    unit_velocity = np.asarray(velocity_m_s) / np.linalg.norm(velocity_m_s)
    gravity = STANDARD_GRAVITY_M_S2 * np.array([0.0, 0.0, 1.0])
    perpendicular = acceleration + gravity
    lift = perpendicular - (perpendicular @ unit_velocity) * unit_velocity
    return lift[2]


@pytest.mark.parametrize("bank_deg", [0.0, 60.0, 80.0])
def test_altitude_hold_below_target_lifts_by_the_critically_damped_correction_at_any_bank(bank_deg):
    envelope = PRESETS["benchmark"]
    velocity = [0.0, 60.0, 0.0]
    holds = Holds(envelope, NOMINAL.dt, bank_rad=float(np.deg2rad(bank_deg)))
    holds.toggle(ALTITUDE, LEVEL_POSITION_M + [0.0, 0.0, ALTITUDE_OFFSET_M], velocity)

    row = holds.steer(CRUISE, LEVEL_POSITION_M, velocity)

    lift = vertical_lift_m_s2(first_step_acceleration(envelope, velocity, holds.bank_rad, row), velocity)
    omega = ALTITUDE_NATURAL_FREQUENCY_RAD_S
    assert lift == pytest.approx(STANDARD_GRAVITY_M_S2 + omega**2 * ALTITUDE_OFFSET_M, abs=FLOAT_ROUNDING_M_S2)


def test_altitude_hold_in_a_climb_lifts_by_the_climb_damping_term():
    envelope = PRESETS["benchmark"]
    velocity = [0.0, 60.0, 5.0]
    holds = Holds(envelope, NOMINAL.dt)
    holds.toggle(ALTITUDE, LEVEL_POSITION_M, velocity)

    row = holds.steer(CRUISE, LEVEL_POSITION_M, velocity)

    lift = vertical_lift_m_s2(first_step_acceleration(envelope, velocity, 0.0, row), velocity)
    damping = 2.0 * ALTITUDE_NATURAL_FREQUENCY_RAD_S * velocity[2]
    assert lift == pytest.approx(STANDARD_GRAVITY_M_S2 - damping, abs=FLOAT_ROUNDING_M_S2)


def test_speed_hold_one_metre_per_second_slow_accelerates_at_the_error_over_its_time_constant():
    envelope = PRESETS["benchmark"]
    holds = Holds(envelope, NOMINAL.dt)
    holds.toggle(SPEED, LEVEL_POSITION_M, [0.0, 60.0, 0.0])
    slow = [0.0, 59.0, 0.0]

    row = holds.steer(CRUISE, LEVEL_POSITION_M, slow)

    along_track = first_step_acceleration(envelope, slow, 0.0, row)[1]
    assert along_track == pytest.approx(1.0 / SPEED_TIME_CONSTANT_S, abs=FLOAT_ROUNDING_M_S2)


@pytest.mark.parametrize(
    "flying_m_s, throttle, speed_brake, afterburner",
    [(80.0, 0.0, 1.0, 0.0), (40.0, 1.0, 0.0, 1.0)],
    ids=["too_fast_for_idle_drag", "too_slow_for_full_throttle"],
)
def test_speed_hold_opens_the_brake_past_idle_and_lights_the_burner_past_full_throttle(
    flying_m_s, throttle, speed_brake, afterburner
):
    holds = Holds(PRESETS["benchmark"], NOMINAL.dt)
    holds.toggle(SPEED, LEVEL_POSITION_M, [0.0, 60.0, 0.0])

    row = holds.steer(CRUISE, LEVEL_POSITION_M, [0.0, flying_m_s, 0.0])

    assert (row[THROTTLE], row[SPEED_BRAKE], row[AFTERBURNER]) == (throttle, speed_brake, afterburner)


@pytest.mark.parametrize(
    "envelope, push_limit_g",
    [
        (PRESETS["benchmark"], PRESETS["benchmark"].max_negative_load_factor),
        (replace(PRESETS["benchmark"], max_negative_load_factor=None), -PRESETS["benchmark"].max_load_factor),
    ],
    ids=["max_negative_g", "recorded_before_negative_g"],
)
def test_altitude_hold_far_above_target_pushes_to_the_envelope_push_limit(envelope, push_limit_g):
    holds = Holds(envelope, NOMINAL.dt)
    holds.toggle(ALTITUDE, LEVEL_POSITION_M - [0.0, 0.0, 1000.0], [0.0, 60.0, 0.0])

    row = holds.steer(CRUISE, LEVEL_POSITION_M, [0.0, 60.0, 0.0])

    assert row[LOAD_FACTOR] == push_limit_g


@pytest.mark.parametrize(
    "engaged_deg, flying_deg, shortest_error_deg", [(1.0, 0.0, 1.0), (179.0, -179.0, -2.0), (-179.0, 179.0, 2.0)]
)
def test_heading_hold_banks_for_a_turn_rate_of_the_shortest_heading_error_over_its_time_constant(
    engaged_deg, flying_deg, shortest_error_deg
):
    speed = 25.0
    holds = Holds(PRESETS["benchmark"], NOMINAL.dt)
    holds.toggle(HEADING, LEVEL_POSITION_M, velocity_toward(engaged_deg, speed))

    holds.steer(CRUISE, LEVEL_POSITION_M, velocity_toward(flying_deg, speed))

    turn_rate = np.deg2rad(shortest_error_deg) / HEADING_TIME_CONSTANT_S
    expected = np.arctan(turn_rate * speed / STANDARD_GRAVITY_M_S2)
    assert holds.bank_rad == pytest.approx(expected, abs=FLOAT_ROUNDING_RAD)


def test_heading_hold_far_off_heading_banks_no_further_than_thirty_degrees():
    holds = Holds(PRESETS["benchmark"], NOMINAL.dt)
    holds.toggle(HEADING, LEVEL_POSITION_M, velocity_toward(90.0, 60.0))

    for _ in range(round(1.0 / NOMINAL.dt)):
        holds.steer(CRUISE, LEVEL_POSITION_M, velocity_toward(0.0, 60.0))

    assert holds.bank_rad == pytest.approx(np.deg2rad(30.0), abs=FLOAT_ROUNDING_RAD)


def velocity_toward(heading_deg, speed_m_s):
    heading = np.deg2rad(heading_deg)
    return np.array([np.sin(heading), np.cos(heading), 0.0]) * speed_m_s


RAISED_SENSOR_M = 100.0
FLOOR_ABOVE_SENSOR_M = 50.0
WORLD_FLOOR_M = RAISED_SENSOR_M + FLOOR_ABOVE_SENSOR_M
LEVEL_AT_SENSOR_ALTITUDE_M = np.zeros(3)


@pytest.mark.parametrize(
    "min_altitude_m, climb_m", [(WORLD_FLOOR_M, FLOOR_ABOVE_SENSOR_M), (None, 0.0), (RAISED_SENSOR_M - 10.0, 0.0)]
)
def test_altitude_hold_below_the_world_floor_climbs_to_it_from_a_raised_sensor(min_altitude_m, climb_m):
    envelope = replace(PRESETS["benchmark"], min_altitude_m=min_altitude_m)
    velocity = [0.0, 60.0, 0.0]
    holds = Holds(envelope, NOMINAL.dt, sensor_altitude_m=RAISED_SENSOR_M)
    holds.toggle(ALTITUDE, LEVEL_AT_SENSOR_ALTITUDE_M, velocity)

    row = holds.steer(CRUISE, LEVEL_AT_SENSOR_ALTITUDE_M, velocity)

    omega = ALTITUDE_NATURAL_FREQUENCY_RAD_S
    assert row[LOAD_FACTOR] == pytest.approx(
        1.0 + omega**2 * climb_m / STANDARD_GRAVITY_M_S2, abs=FLOAT_ROUNDING_M_S2
    )


def test_a_floor_raised_mid_hold_lifts_the_hold_target_without_re_engaging():
    holds = Holds(PRESETS["benchmark"], NOMINAL.dt, sensor_altitude_m=RAISED_SENSOR_M)
    velocity = [0.0, 60.0, 0.0]
    holds.toggle(ALTITUDE, LEVEL_AT_SENSOR_ALTITUDE_M, velocity)
    level = holds.steer(CRUISE, LEVEL_AT_SENSOR_ALTITUDE_M, velocity)[LOAD_FACTOR]

    holds.envelope = replace(holds.envelope, min_altitude_m=WORLD_FLOOR_M)

    assert holds.steer(CRUISE, LEVEL_AT_SENSOR_ALTITUDE_M, velocity)[LOAD_FACTOR] > level


def test_live_altitude_hold_below_the_world_floor_over_a_raised_sensor_pulls_up(tmp_path):
    envelope = replace(PRESETS["benchmark"], min_altitude_m=WORLD_FLOOR_M)
    setup = {
        "seed": 7,
        "envelope": asdict(envelope),
        "noise_regime": "nominal",
        "thresholds": DEFAULT_THRESHOLDS,
        "start": {"position": [0.0, 0.0, WORLD_FLOOR_M - 30.0], "velocity": [0.0, 60.0, 0.0]},
        "sensor": {"position": [0.0, 0.0, RAISED_SENSOR_M]},
    }
    live = LiveFlight(setup, json.loads(PAGE_IMM_TUNING.read_text())["best"], tmp_path)
    live.toggle_hold(ALTITUDE)

    live.advance(CRUISE)

    omega = ALTITUDE_NATURAL_FREQUENCY_RAD_S
    assert live.controls[-1][LOAD_FACTOR] == pytest.approx(
        1.0 + omega**2 * 30.0 / STANDARD_GRAVITY_M_S2, abs=FLOAT_ROUNDING_M_S2
    )
