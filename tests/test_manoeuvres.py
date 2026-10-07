import numpy as np
import pytest

from data.config import NOMINAL
from data.fixed_wing import DEFAULT_PRESET, PRESETS, STANDARD_GRAVITY_M_S2, THROTTLE, UP, fly_steps
from web.manoeuvres import MANOEUVRES, manoeuvre_rows

ENVELOPE = PRESETS[DEFAULT_PRESET]
ENTRY_SPEED_M_S = 150.0
ENTRY_VELOCITY = ENTRY_SPEED_M_S * np.array([1.0, 0.0, 0.0])
ENTRY_POSITION = np.array([0.0, 0.0, 3000.0])


def run(name):
    latest = {"velocity": ENTRY_VELOCITY}
    recorded = []

    def feedback_rows():
        for row in manoeuvre_rows(name, lambda: ENVELOPE, NOMINAL.dt, ENTRY_VELOCITY, lambda: latest["velocity"]):
            recorded.append(row)
            yield row

    states, lifts = [], []
    for state, lift in fly_steps(ENVELOPE, ENTRY_POSITION, ENTRY_VELOCITY, 0.0, feedback_rows(), NOMINAL.dt):
        latest["velocity"] = state[3:6]
        states.append(state)
        lifts.append(lift)
    return np.array(states), np.array(lifts), np.array(recorded)


def heading_deg(velocity):
    return float(np.degrees(np.arctan2(-velocity[1], velocity[0])))


def load_factors(states, lifts):
    return np.einsum("ij,ij->i", states[:, 6:9] + STANDARD_GRAVITY_M_S2 * UP, lifts) / STANDARD_GRAVITY_M_S2


def test_the_six_named_manoeuvres_are_keys_one_to_six_in_order():
    assert MANOEUVRES == ("break turn", "reversal", "split-S", "jink", "weave", "barrel roll")


def test_break_turn_turns_the_flight_path_ninety_degrees_and_rolls_out_level():
    states, lifts, _ = run("break turn")
    assert heading_deg(states[-1, 3:6]) == pytest.approx(90.0, abs=3.0)
    assert lifts[-1] @ UP > 0.99
    assert abs(states[-1, 5]) < 0.02 * np.linalg.norm(states[-1, 3:6])


def test_break_turn_pulls_the_envelope_load_factor():
    states, lifts, _ = run("break turn")
    assert np.isclose(load_factors(states, lifts), ENVELOPE.max_load_factor).sum() > 100


def test_split_s_loses_altitude_and_leaves_level_on_the_reversed_heading():
    states, lifts, _ = run("split-S")
    final_velocity = states[-1, 3:6]
    assert final_velocity[0] < 0.0
    assert abs(final_velocity[1]) < 0.1 * np.linalg.norm(final_velocity)
    assert abs(final_velocity[2]) < 0.1 * np.linalg.norm(final_velocity)
    assert ENTRY_POSITION[2] - states[-1, 2] > 300.0
    assert lifts[-1] @ UP > 0.99


def test_reversal_climbs_and_leaves_upright_on_the_reversed_heading():
    states, lifts, _ = run("reversal")
    final_velocity = states[-1, 3:6]
    assert final_velocity[0] < 0.0
    assert abs(final_velocity[1]) < 0.1 * np.linalg.norm(final_velocity)
    assert states[-1, 2] - ENTRY_POSITION[2] > 300.0
    assert lifts[-1] @ UP > 0.99


def test_jink_displaces_sideways_with_the_wings_level_and_the_heading_kept():
    states, lifts, _ = run("jink")
    assert states[-1, 1] == pytest.approx(-2 * 2.0 * STANDARD_GRAVITY_M_S2, rel=0.1)
    assert min(lifts @ UP) > 0.999
    assert abs(heading_deg(states[-1, 3:6])) < 5.0


def test_jink_alternates_the_rudder_starting_to_the_right():
    _, _, rows = run("jink")
    rudder = rows[:, 3]
    assert rudder[0] == ENVELOPE.max_side_load_factor
    assert rudder[-1] == -ENVELOPE.max_side_load_factor
    assert set(np.sign(rudder)) == {-1.0, 1.0}


def test_weave_swings_the_heading_both_ways_and_ends_wings_level():
    states, lifts, _ = run("weave")
    headings = [heading_deg(velocity) for velocity in states[:, 3:6]]
    assert max(headings) > 5.0
    assert min(headings) < -5.0
    assert lifts[-1] @ UP > 0.99


def test_weave_runs_two_half_swings_and_two_full_ones_of_three_seconds_plus_the_rolls():
    states, _, _ = run("weave")
    assert len(states) * NOMINAL.dt == pytest.approx(9.0 + 8.0 / 3.0, abs=0.1)


def test_weave_banks_to_both_sides_holding_the_level_turn_load_factor():
    states, lifts, rows = run("weave")
    assert np.isclose(load_factors(states, lifts), 2.0).sum() > 400
    assert rows[:, 1].max() == pytest.approx(2.0)
    assert set(np.sign(rows[:, 0])) == {-1.0, 0.0, 1.0}


def test_barrel_roll_takes_the_lift_through_inverted_and_returns_wings_level():
    states, lifts, _ = run("barrel roll")
    assert min(lifts @ UP) < -0.5
    assert lifts[-1] @ UP > 0.99
    assert states[-1, 0] > ENTRY_POSITION[0] + 100.0
    assert abs(heading_deg(states[-1, 3:6])) < 15.0


def test_barrel_roll_pulls_two_g_through_a_half_rate_roll_of_a_full_turn():
    states, lifts, rows = run("barrel roll")
    rates = rows[:, 0]
    assert np.isclose(load_factors(states, lifts), 2.0).sum() > 150
    assert rates.sum() * NOMINAL.dt == pytest.approx(2.0 * np.pi)
    assert rates.max() == pytest.approx(0.5 * ENVELOPE.max_roll_rate_rad_s)


@pytest.mark.parametrize("name", MANOEUVRES)
def test_every_manoeuvre_ends_flies_at_full_throttle_on_six_column_rows(name):
    states, _, rows = run(name)
    assert 50 < len(states) < 2000
    assert rows.shape[1] == 6
    assert set(rows[:, THROTTLE]) == {1.0}


@pytest.mark.parametrize("name", MANOEUVRES)
def test_no_manoeuvre_commands_past_the_envelope_load_or_side_limits(name):
    _, _, rows = run(name)
    assert rows[:, 1].max() <= ENVELOPE.max_load_factor
    assert np.abs(rows[:, 3]).max() <= ENVELOPE.max_side_load_factor


def test_jink_rudder_flips_sign_at_every_second_boundary_for_four_halves():
    _, _, rows = run("jink")
    signs = np.sign(rows[:, 3])
    assert np.count_nonzero(np.diff(signs)) == 3
