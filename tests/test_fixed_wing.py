import os
import subprocess
import sys
from dataclasses import astuple

import numpy as np
import pytest

from data.config import NOMINAL
from data.fixed_wing import (
    AFTERBURNER,
    LOAD_FACTOR,
    DEFAULT_PRESET,
    PRESETS,
    ROLL_RATE,
    RUDDER,
    SPEED_BRAKE,
    THROTTLE,
    Envelope,
    fly,
)
from data.trajectory_gen import measure

G = 9.80665
DT = NOMINAL.dt
EAST = np.array([1.0, 0.0, 0.0])
ORIGIN = np.array([0.0, 0.0, 1000.0])
TEST_ENVELOPE = Envelope(min_speed_m_s=20.0, max_speed_m_s=200.0, max_load_factor=4.0, max_roll_rate_rad_s=2.0)


CONTROLS_ENVELOPE = Envelope(
    min_speed_m_s=20.0,
    max_speed_m_s=200.0,
    max_load_factor=4.0,
    max_roll_rate_rad_s=2.0,
    max_side_load_factor=1.0,
    max_negative_load_factor=-2.0,
    brake_drag_multiplier=3.0,
    afterburner_thrust_multiplier=1.5,
)


def held_controls(
    steps, roll_rate_rad_s=0.0, load_factor=1.0, throttle=0.0, rudder=0.0, speed_brake=0.0, afterburner=0.0
):
    controls = np.zeros((steps, 6))
    controls[:, ROLL_RATE] = roll_rate_rad_s
    controls[:, LOAD_FACTOR] = load_factor
    controls[:, THROTTLE] = throttle
    controls[:, RUDDER] = rudder
    controls[:, SPEED_BRAKE] = speed_brake
    controls[:, AFTERBURNER] = afterburner
    return controls


def fly_east(speed_m_s, controls, bank_rad=0.0, envelope=TEST_ENVELOPE):
    return fly(envelope, ORIGIN, speed_m_s * EAST, bank_rad, controls, DT)


def speeds(states):
    return np.linalg.norm(states[:, 3:6], axis=1)


def assert_step_acceleration(actual_m_s2, expected_m_s2):
    """rtol 1e-12: the column is a velocity difference over dt, so it carries float64 roundoff."""
    np.testing.assert_allclose(actual_m_s2, expected_m_s2, rtol=1e-12)


def test_scripted_flight_from_seeded_start_is_measurable_truth():
    rng = np.random.default_rng(5)
    azimuth = rng.uniform(-np.pi, np.pi)
    heading = rng.uniform(-np.pi, np.pi)
    start_range_m = rng.uniform(*NOMINAL.range0)
    start_speed_m_s = rng.uniform(*NOMINAL.speed0)
    position = start_range_m * np.array([np.cos(azimuth), np.sin(azimuth), 0.2])
    velocity = start_speed_m_s * np.array([np.cos(heading), np.sin(heading), 0.0])
    controls = np.concatenate(
        [
            held_controls(20, roll_rate_rad_s=1.0, load_factor=1.5, throttle=0.8),
            held_controls(40, load_factor=3.0, throttle=1.0),
            held_controls(20, roll_rate_rad_s=-1.0, load_factor=0.5, throttle=0.2),
        ]
    )

    states = fly(PRESETS[DEFAULT_PRESET], position, velocity, 0.0, controls, DT)
    z = measure(np.random.default_rng(6), states, NOMINAL)

    assert states.shape == (80, 9)
    assert states.dtype == np.float64
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(z)) and np.all(z[:, 2] > 0)


def test_acceleration_columns_are_the_step_acceleration_up_to_roundoff():
    controls = np.concatenate([held_controls(10, roll_rate_rad_s=1.5, load_factor=3.0, throttle=1.0)] * 2)
    states = fly_east(60.0, controls)
    position, velocity, acceleration = states[:, 0:3], states[:, 3:6], states[:, 6:9]

    np.testing.assert_allclose(velocity[1:], velocity[:-1] + DT * acceleration[:-1], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        position[1:],
        position[:-1] + DT * velocity[:-1] + 0.5 * DT**2 * acceleration[:-1],
        rtol=1e-12,
        atol=1e-9,
    )


def test_fixed_wing_import_pulls_in_no_filters_module():
    probe = "import sys, data.fixed_wing; print(sorted(m for m in sys.modules if m.split('.')[0] == 'filters'))"
    clean_env = {"PATH": os.environ["PATH"], "PYTHONPATH": os.getcwd()}
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=clean_env
    )
    assert result.stdout.strip() == "[]"


def test_load_factor_command_at_max_g_pulls_exactly_max_g():
    states = fly_east(100.0, held_controls(1, load_factor=4.0))
    assert_step_acceleration(states[0, 8], 4.0 * G - G)


def test_load_factor_command_above_max_g_is_clamped_to_max_g():
    at_max = fly_east(100.0, held_controls(20, load_factor=4.0))
    above_max = fly_east(100.0, held_controls(20, load_factor=4.5))
    assert np.array_equal(above_max, at_max)


def test_load_factor_command_below_max_g_is_not_clamped():
    states = fly_east(100.0, held_controls(1, load_factor=3.5))
    assert_step_acceleration(states[0, 8], 3.5 * G - G)


def test_negative_load_factor_command_beyond_max_g_is_clamped_to_minus_max_g():
    """An envelope without max_negative_load_factor, as recorded before #107, keeps the symmetric push."""
    states = fly_east(100.0, held_controls(1, load_factor=-4.5))
    assert_step_acceleration(states[0, 8], -4.0 * G - G)


def fly_east_with_controls(speed_m_s, controls, bank_rad=0.0, envelope=CONTROLS_ENVELOPE):
    return fly_east(speed_m_s, controls, bank_rad, envelope)


def test_push_at_max_negative_g_pushes_exactly_max_negative_g():
    states = fly_east_with_controls(100.0, held_controls(1, load_factor=-2.0))
    assert_step_acceleration(states[0, 8], -2.0 * G - G)


def test_push_beyond_max_negative_g_is_clipped_to_max_negative_g_not_minus_max_g():
    at_limit = fly_east_with_controls(100.0, held_controls(20, load_factor=-2.0))
    beyond = fly_east_with_controls(100.0, held_controls(20, load_factor=-2.5))
    assert np.array_equal(beyond, at_limit)


def test_push_short_of_max_negative_g_is_not_clipped():
    states = fly_east_with_controls(100.0, held_controls(1, load_factor=-1.5))
    assert_step_acceleration(states[0, 8], -1.5 * G - G)


def test_push_is_capped_by_lift_available_at_1_25_times_stall_speed():
    """(25 / 20)^2 = 1.5625 g, short of the 2 g push limit."""
    states = fly_east_with_controls(25.0, held_controls(1, load_factor=-2.0))
    assert_step_acceleration(states[0, 8], -1.5625 * G - G)


def test_right_rudder_at_max_side_g_slips_toward_the_right_wing_south_at_exactly_max_side_g():
    states = fly_east_with_controls(100.0, held_controls(1, rudder=1.0))
    assert_step_acceleration(states[0, 7], -1.0 * G)


def test_rudder_short_of_max_side_g_is_not_clipped():
    states = fly_east_with_controls(100.0, held_controls(1, rudder=0.5))
    assert_step_acceleration(states[0, 7], -0.5 * G)


def test_right_rudder_beyond_max_side_g_is_clipped_to_max_side_g():
    at_limit = fly_east_with_controls(100.0, held_controls(20, rudder=1.0))
    beyond = fly_east_with_controls(100.0, held_controls(20, rudder=1.5))
    assert np.array_equal(beyond, at_limit)


def test_left_rudder_beyond_max_side_g_slips_north_at_exactly_max_side_g():
    states = fly_east_with_controls(100.0, held_controls(1, rudder=-1.5))
    assert_step_acceleration(states[0, 7], 1.0 * G)


def test_rudder_is_capped_by_lift_available_at_1_25_times_stall_speed():
    """(25 / 20)^2 = 1.5625 g, short of a 2 g side limit."""
    envelope = Envelope(20.0, 200.0, 4.0, 2.0, max_side_load_factor=2.0)
    states = fly_east_with_controls(25.0, held_controls(1, rudder=2.0), envelope=envelope)
    assert_step_acceleration(states[0, 7], -1.5625 * G)


def test_right_rudder_banked_90_deg_right_pushes_along_the_banked_right_wing_down():
    """Zero load factor isolates the rudder from lift. atol 1e-12: cos(pi / 2) leaves about 1e-16 g along south."""
    controls = held_controls(1, load_factor=0.0, rudder=1.0)
    states = fly_east_with_controls(100.0, controls, bank_rad=np.pi / 2.0)
    np.testing.assert_allclose(states[0, 7:9], [0.0, -2.0 * G], rtol=1e-12, atol=1e-12)


def test_speed_brake_multiplies_drag_by_the_brake_factor():
    """Zero throttle, wings level: the whole x acceleration is drag."""
    states = fly_east_with_controls(100.0, held_controls(1, speed_brake=1.0))
    assert_step_acceleration(states[0, 6], -3.0 * drag_m_s2(100.0, 1.0))


def test_speed_brake_leaves_full_thrust_untouched():
    states = fly_east_with_controls(100.0, held_controls(1, throttle=1.0, speed_brake=1.0))
    assert_step_acceleration(states[0, 6], drag_m_s2(200.0, 1.0) - 3.0 * drag_m_s2(100.0, 1.0))


def test_afterburner_at_full_throttle_multiplies_max_thrust_by_the_afterburner_factor():
    states = fly_east_with_controls(100.0, held_controls(1, throttle=1.0, afterburner=1.0))
    assert_step_acceleration(states[0, 6], 1.5 * drag_m_s2(200.0, 1.0) - drag_m_s2(100.0, 1.0))


@pytest.mark.parametrize("throttle", [0.875, np.nextafter(1.0, 0.0)])
def test_afterburner_short_of_full_throttle_adds_no_thrust(throttle):
    with_afterburner = fly_east_with_controls(100.0, held_controls(1, throttle=throttle, afterburner=1.0))
    without = fly_east_with_controls(100.0, held_controls(1, throttle=throttle))
    assert np.array_equal(with_afterburner, without)


def test_afterburner_with_throttle_above_full_burns_as_at_full_throttle():
    above = fly_east_with_controls(100.0, held_controls(1, throttle=1.5, afterburner=1.0))
    full = fly_east_with_controls(100.0, held_controls(1, throttle=1.0, afterburner=1.0))
    assert np.array_equal(above, full)


def test_3_column_row_flies_bit_for_bit_as_the_full_row_with_every_new_input_neutral():
    """Under an envelope whose new limits all act, so a non-neutral pad would show."""
    controls = held_controls(20, roll_rate_rad_s=1.5, load_factor=-1.5, throttle=1.0)
    short = fly_east_with_controls(100.0, controls[:, :3])
    assert short.tobytes() == fly_east_with_controls(100.0, controls).tobytes()


def test_load_factor_is_capped_by_lift_available_at_twice_stall_speed():
    stall_limited = Envelope(min_speed_m_s=20.0, max_speed_m_s=200.0, max_load_factor=6.0, max_roll_rate_rad_s=2.0)
    states = fly_east(40.0, held_controls(1, load_factor=6.0), envelope=stall_limited)
    assert_step_acceleration(states[0, 8], 4.0 * G - G)


def test_roll_rate_command_above_max_is_clamped_to_max_roll_rate():
    at_max = fly_east(100.0, held_controls(20, roll_rate_rad_s=2.0, load_factor=2.0))
    above_max = fly_east(100.0, held_controls(20, roll_rate_rad_s=2.5, load_factor=2.0))
    assert np.array_equal(above_max, at_max)


def test_negative_roll_rate_command_beyond_max_is_clamped_to_minus_max_roll_rate():
    at_max = fly_east(100.0, held_controls(20, roll_rate_rad_s=-2.0, load_factor=2.0))
    beyond_max = fly_east(100.0, held_controls(20, roll_rate_rad_s=-2.5, load_factor=2.0))
    assert np.array_equal(beyond_max, at_max)


def test_roll_rate_command_below_max_rolls_slower_than_max():
    at_max = fly_east(100.0, held_controls(2, roll_rate_rad_s=2.0, load_factor=2.0))
    below_max = fly_east(100.0, held_controls(2, roll_rate_rad_s=1.5, load_factor=2.0))
    assert_step_acceleration(at_max[0, 7], -2.0 * G * np.sin(2.0 * DT))
    assert_step_acceleration(below_max[0, 7], -2.0 * G * np.sin(1.5 * DT))


def test_speed_is_floored_at_stall_speed_with_idle_throttle():
    """rtol 1e-15: rescaling onto the stall speed rounds within a few ulp."""
    states = fly_east(20.0, held_controls(30, throttle=0.0))
    np.testing.assert_allclose(speeds(states), 20.0, rtol=1e-15)


def test_pull_up_at_stall_speed_gets_only_one_g():
    pull_up = fly_east(20.0, held_controls(30, load_factor=3.0))
    level = fly_east(20.0, held_controls(30, load_factor=1.0))
    assert np.array_equal(pull_up, level)


def test_level_full_throttle_flight_at_max_speed_holds_speed_and_height():
    states = fly_east(200.0, held_controls(50, throttle=1.0))
    assert np.all(speeds(states) == 200.0)
    assert np.all(states[:, 2] == ORIGIN[2])


def test_level_flight_below_max_speed_at_full_throttle_accelerates():
    states = fly_east(100.0, held_controls(50, throttle=1.0))
    assert np.all(np.diff(speeds(states)) > 0.0)


def test_sustained_max_g_level_turn_at_full_throttle_loses_speed():
    level_turn_bank_rad = np.arccos(1.0 / 4.0)
    states = fly_east(200.0, held_controls(50, load_factor=4.0, throttle=1.0), bank_rad=level_turn_bank_rad)
    assert np.all(np.diff(speeds(states)) < 0.0)


def test_max_g_level_turn_decelerates_by_the_extra_induced_drag():
    """rtol 1e-12: one step of float64 roundoff on terms of order 10 m/s^2."""
    level_turn_bank_rad = np.arccos(1.0 / 4.0)
    states = fly_east(200.0, held_controls(1, load_factor=4.0, throttle=1.0), bank_rad=level_turn_bank_rad)
    best_glide_speed_m_s = np.sqrt(20.0 * 200.0)
    induced_drag_per_g_squared = G / (2.0 * 10.0) * (best_glide_speed_m_s / 200.0) ** 2
    np.testing.assert_allclose(states[0, 6], -(4.0**2 - 1.0) * induced_drag_per_g_squared, rtol=1e-12)


def test_climb_loses_speed_that_the_mirrored_descent_gains():
    """Same speed, load factor and drag; only gravity along the path differs: 2 g sin(30 deg) = g.
    rtol 1e-12: one step of float64 roundoff on terms of order 10 m/s^2."""
    path_rad = np.pi / 6.0
    controls = held_controls(1, load_factor=np.cos(path_rad), throttle=0.5)
    climb_direction = np.array([np.cos(path_rad), 0.0, np.sin(path_rad)])
    descent_direction = np.array([np.cos(path_rad), 0.0, -np.sin(path_rad)])
    climbing = fly(TEST_ENVELOPE, ORIGIN, 100.0 * climb_direction, 0.0, controls, DT)
    descending = fly(TEST_ENVELOPE, ORIGIN, 100.0 * descent_direction, 0.0, controls, DT)
    along_climb = climbing[0, 6:9] @ climb_direction
    along_descent = descending[0, 6:9] @ descent_direction
    np.testing.assert_allclose(along_climb - along_descent, -G, rtol=1e-12)


def test_wings_level_constant_pull_loops_through_vertical_up_and_down_back_to_east():
    """Pitch must rise monotonically through 90 and 270 deg; one step turns the path at most (n + 1) g dt / stall speed."""
    load_factor = 8.0
    benchmark = PRESETS[DEFAULT_PRESET]
    controls = held_controls(2500, load_factor=load_factor, throttle=1.0)
    states = fly(benchmark, ORIGIN, 200.0 * EAST, 0.0, controls, DT)
    pitch_rad = np.unwrap(np.arctan2(states[:, 5], states[:, 3]))
    looped = np.flatnonzero(pitch_rad >= 2.0 * np.pi)

    assert np.all(np.isfinite(states))
    assert np.all(states[:, 4] == 0.0)
    assert looped.size > 0
    assert np.all(np.diff(pitch_rad[: looped[0] + 1]) > 0.0)
    assert pitch_rad[looped[0]] - 2.0 * np.pi < (load_factor + 1.0) * G * DT / benchmark.min_speed_m_s


@pytest.mark.parametrize("vertical_speed_m_s", [100.0, -100.0])
def test_start_pointing_straight_up_or_down_flies_finite_states(vertical_speed_m_s):
    controls = held_controls(50, roll_rate_rad_s=1.0, load_factor=2.0, throttle=0.5)
    states = fly(TEST_ENVELOPE, ORIGIN, np.array([0.0, 0.0, vertical_speed_m_s]), 0.0, controls, DT)
    assert np.all(np.isfinite(states))


def test_pull_from_straight_up_pitches_toward_north():
    states = fly(TEST_ENVELOPE, ORIGIN, np.array([0.0, 0.0, 100.0]), 0.0, held_controls(1, load_factor=2.0), DT)
    assert states[0, 6] == 0.0
    assert_step_acceleration(states[0, 7], 2.0 * G)


def test_right_bank_from_straight_up_pulls_toward_the_right_wing_west():
    """atol 1e-12: cos(pi / 2) leaves about 1e-16 g of lift along north."""
    controls = held_controls(1, load_factor=2.0)
    states = fly(TEST_ENVELOPE, ORIGIN, np.array([0.0, 0.0, 100.0]), np.pi / 2.0, controls, DT)
    np.testing.assert_allclose(states[0, 6:8], [-2.0 * G, 0.0], rtol=1e-12, atol=1e-12)


def test_default_preset_spans_the_benchmark_speed_envelope():
    default = PRESETS[DEFAULT_PRESET]
    assert (default.min_speed_m_s, default.max_speed_m_s) == NOMINAL.speed0


def test_named_presets_pin_the_values_recorded_in_issue_2_decision_49():
    """Scenarios record the envelope by value, so this guards the preset table itself, not any replay."""
    assert {name: astuple(envelope) for name, envelope in PRESETS.items()} == {
        "benchmark": (20.0, 250.0, 9.0, np.deg2rad(180.0), 2.0, -4.5, 3.0, 1.5),
        "small aerial object": (12.0, 45.0, 4.0, np.deg2rad(120.0), 0.5, -2.0, 2.0, 1.0),
        "light aircraft": (25.0, 70.0, 3.8, np.deg2rad(60.0), 0.3, -1.5, 2.0, 1.0),
        "jet": (70.0, 300.0, 9.0, np.deg2rad(240.0), 1.0, -3.0, 3.0, 1.5),
    }


def first_step_acceleration(throttle):
    return fly_east(100.0, held_controls(1, throttle=throttle))[0, 6:9]


def test_throttle_exactly_one_is_stronger_than_half_throttle():
    assert first_step_acceleration(1.0)[0] > first_step_acceleration(0.5)[0]


def test_throttle_one_eighth_above_one_is_clipped_to_full_thrust():
    assert np.array_equal(first_step_acceleration(1.125), first_step_acceleration(1.0))


def test_throttle_well_above_one_is_clipped_to_full_thrust():
    assert np.array_equal(first_step_acceleration(1.5), first_step_acceleration(1.0))


def test_throttle_one_eighth_below_zero_is_clipped_to_idle():
    assert np.array_equal(first_step_acceleration(-0.125), first_step_acceleration(0.0))


def test_throttle_exactly_zero_is_weaker_than_half_throttle():
    assert first_step_acceleration(0.0)[0] < first_step_acceleration(0.5)[0]


def test_overthrottled_flight_holds_the_full_throttle_equilibrium_speed():
    benchmark = PRESETS["benchmark"]
    throttles = np.where(np.arange(100) % 2, 1.5, 1.0)
    states = fly_east(benchmark.max_speed_m_s, held_controls(100, throttle=throttles), envelope=benchmark)
    assert np.all(speeds(states) == benchmark.max_speed_m_s)

def drag_m_s2(speed_m_s, load_factor):
    return G / (2.0 * 10.0) * (speed_m_s**2 / 4000.0 + load_factor**2 * 4000.0 / speed_m_s**2)


def part_throttle_trim(speed_m_s):
    return drag_m_s2(speed_m_s, 1.0) / drag_m_s2(200.0, 1.0)


def test_level_flight_at_part_throttle_trim_holds_speed_and_height():
    """Best-glide speed squared is 20 * 200 = 4000. atol 1e-9: roundoff of a velocity difference over dt."""
    states = fly_east(100.0, held_controls(50, throttle=part_throttle_trim(100.0)))
    np.testing.assert_allclose(speeds(states), 100.0, atol=1e-9)
    np.testing.assert_allclose(states[:, 2], ORIGIN[2], atol=1e-9)


def test_wings_level_unaccelerated_flight_holds_heading():
    """Velocity (60, 80, 0) is the 3-4-5 heading at 100 m/s. atol 1e-9: velocity roundoff."""
    controls = held_controls(50, throttle=part_throttle_trim(100.0))
    states = fly(TEST_ENVELOPE, ORIGIN, np.array([60.0, 80.0, 0.0]), 0.0, controls, DT)
    np.testing.assert_allclose(states[:, 3], 60.0, atol=1e-9)
    np.testing.assert_allclose(states[:, 4], 80.0, atol=1e-9)
    np.testing.assert_allclose(states[:, 5], 0.0, atol=1e-9)


@pytest.mark.parametrize("load_factor", [2.0, 3.0])
def test_induced_drag_grows_with_the_square_of_load_factor(load_factor):
    """Zero throttle, wings level: lift is vertical so the whole x acceleration is drag. rtol 1e-12: one step."""
    states = fly_east(100.0, held_controls(1, load_factor=load_factor))
    assert_step_acceleration(states[0, 6], -drag_m_s2(100.0, load_factor))


def test_coasting_climb_trades_speed_for_height_at_g_sin_path_angle():
    """Thrust cancels drag at the first step, so along-path deceleration is g sin(30 deg) = g/2."""
    path_rad = np.pi / 6.0
    climb_direction = np.array([np.cos(path_rad), 0.0, np.sin(path_rad)])
    trim_throttle = drag_m_s2(100.0, np.cos(path_rad)) / drag_m_s2(200.0, 1.0)
    controls = held_controls(10, load_factor=np.cos(path_rad), throttle=trim_throttle)
    states = fly(TEST_ENVELOPE, ORIGIN, 100.0 * climb_direction, 0.0, controls, DT)
    assert_step_acceleration(states[0, 6:9] @ climb_direction, -G / 2.0)
    assert np.all(np.diff(speeds(states)) < 0.0)
    assert np.all(np.diff(states[:, 2]) > 0.0)
