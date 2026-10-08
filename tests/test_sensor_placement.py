import json

import numpy as np
import pytest
import torch

from data.config import NOMINAL
from data.trajectory_gen import fly_scenario, scenario_start
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.torch_ops import export_npz
from scripts.page_parity import (
    IMM_TUNING,
    NO_WEIGHTS,
    SCENARIOS_DIR,
    SENSOR_MOVED_CASE,
    SENSOR_MOVED_M,
    SENSOR_MOVED_START,
    case_scenarios,
    native_flags,
)
from scripts.replay import RKN_LAYER_MULTS
from web.live_flight import FILTER_NAMES, LEARNED_WEIGHTS, LiveFlight

STAND_IN_SEED = 3
EVADE_ENGAGE_STEP = 50
EVADE_STEPS = 200
PILOT_ROW = [0.0, 1.0, 0.6, 0.0, 0.0, 0.0]
RAISED_SENSOR_M = [1000.0, -500.0, 50.0]
MOVED_SENSORS = [SENSOR_MOVED_M, RAISED_SENSOR_M]


@pytest.fixture(scope="module")
def weights_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("weights")
    torch.manual_seed(STAND_IN_SEED)
    networks = {
        "KalmanNet": KalmanGainGRU(),
        "RKN": RKNGainCovNetwork(**RKN_LAYER_MULTS),
        "BKN": KalmanGainGRU(),
    }
    for name, network in networks.items():
        (root / LEARNED_WEIGHTS[name]).parent.mkdir(parents=True)
        export_npz(network.double().state_dict(), root / LEARNED_WEIGHTS[name])
    return root


def break_turn():
    return json.loads((SCENARIOS_DIR / "break_turn_9g.json").read_text())


def placed_and_shifted(sensor, scenario=None):
    scenario = scenario or break_turn()
    shifted_start = {
        "position": [p - s for p, s in zip(SENSOR_MOVED_START["position"], sensor)],
        "velocity": SENSOR_MOVED_START["velocity"],
    }
    placed = {**scenario, "start": SENSOR_MOVED_START, "sensor": {"position": sensor}}
    return placed, {**scenario, "start": shifted_start}


def flown(setup, weights_root, evade_at=None, controls=None):
    flight = LiveFlight(setup, json.loads(IMM_TUNING.read_text())["best"], weights_root)
    frames = []
    for step, control in enumerate(controls or setup["controls"]):
        if step == evade_at:
            flight.engage_evade()
        flight.advance(control)
        frames.append(flight.frame())
    return flight, frames


def without_estimate(readout):
    return {**readout, "estimate_m": None}


def shifted(position, sensor):
    return None if position is None else [p + s for p, s in zip(position, sensor)]


def assert_frames_match_shifted(placed, origin, sensor):
    for at_placed, at_origin in zip(placed, origin, strict=True):
        assert at_placed["sensor_m"] == sensor
        assert at_placed["truth_m"] == shifted(at_origin["truth_m"], sensor)
        assert (at_placed["trips"], at_placed["flags"]) == (at_origin["trips"], at_origin["flags"])
        assert at_placed["filters"].keys() == at_origin["filters"].keys()
        for name, readout in at_placed["filters"].items():
            assert readout["estimate_m"] == shifted(at_origin["filters"][name]["estimate_m"], sensor)
            assert without_estimate(readout) == without_estimate(at_origin["filters"][name])


@pytest.mark.parametrize("sensor", MOVED_SENSORS)
def test_moved_sensor_scores_all_six_filters_as_the_origin_sensor_scores_the_start_shifted_by_minus_it(
    weights_root, sensor
):
    """MS-49: every filter runs sensor-centred, so moving sensor and start together changes no score."""
    placed_setup, origin_setup = placed_and_shifted(sensor)
    _, placed = flown(placed_setup, weights_root)
    _, origin = flown(origin_setup, weights_root)

    assert set(placed[0]["filters"]) == set(FILTER_NAMES)
    assert placed[-1]["truth_m"] != origin[-1]["truth_m"]
    assert_frames_match_shifted(placed, origin, sensor)


@pytest.mark.parametrize("sensor", MOVED_SENSORS)
def test_measurements_from_a_moved_sensor_are_those_of_the_origin_sensor_with_the_start_shifted(sensor):
    """MS-49: az/el/range are of the target relative to the sensor, its altitude included."""
    placed_setup, origin_setup = placed_and_shifted(sensor)
    placed_states, placed_z = fly_scenario(placed_setup)
    origin_states, origin_z = fly_scenario(origin_setup)

    assert (placed_states.tobytes(), placed_z.tobytes()) == (origin_states.tobytes(), origin_z.tobytes())
    assert not np.array_equal(
        placed_z, fly_scenario({**placed_setup, "sensor": {"position": [0.0, 0.0, 0.0]}})[1]
    )


@pytest.mark.parametrize("sensor", MOVED_SENSORS)
def test_evade_from_a_moved_sensor_flies_the_rows_it_flies_from_the_origin_with_the_start_shifted(sensor):
    """MS-49: evade aims across the line of sight from the sensor, wherever the sensor is."""
    placed_setup, origin_setup = placed_and_shifted(sensor)
    controls = [PILOT_ROW] * EVADE_STEPS
    placed_flight, placed = flown(placed_setup, NO_WEIGHTS, EVADE_ENGAGE_STEP, controls)
    origin_flight, origin = flown(origin_setup, NO_WEIGHTS, EVADE_ENGAGE_STEP, controls)

    assert placed_flight.controls[EVADE_ENGAGE_STEP:] == origin_flight.controls[EVADE_ENGAGE_STEP:]
    assert placed_flight.controls[EVADE_ENGAGE_STEP] != PILOT_ROW
    assert_frames_match_shifted(placed, origin, sensor)


def test_exported_moved_sensor_flight_replays_natively_to_the_flags_the_live_flight_showed():
    """MS-49: replay.py reads the sensor from the scenario and feeds the filters the same sensor-centred flight."""
    scenario = case_scenarios()[SENSOR_MOVED_CASE]
    flight, _ = flown(scenario, NO_WEIGHTS)
    exported = json.loads(flight.scenario_json())

    assert exported["sensor"] == {"position": SENSOR_MOVED_M}
    assert native_flags(exported) == {name: flight.frame()["flags"][name] for name in ("EKF", "UKF", "IMM")}
    assert native_flags(exported) != native_flags({**exported, "sensor": {"position": [0.0, 0.0, 0.0]}})


def test_a_sensor_at_the_origin_flies_bit_for_bit_as_a_scenario_without_one():
    """MS-49: scenarios from before the sensor could move replay unchanged."""
    scenario = break_turn()
    states, z = fly_scenario(scenario)
    with_origin = fly_scenario({**scenario, "sensor": {"position": [0.0, 0.0, 0.0]}})

    assert (states.tobytes(), z.tobytes()) == (with_origin[0].tobytes(), with_origin[1].tobytes())


def test_a_flight_without_a_sensor_shows_it_at_the_origin_and_exports_none():
    """MS-49: an absent sensor is the origin, and the export stays in the old form."""
    flight, frames = flown({**break_turn(), "controls": [PILOT_ROW]}, NO_WEIGHTS)

    assert frames[-1]["sensor_m"] == [0.0, 0.0, 0.0]
    assert "sensor" not in json.loads(flight.scenario_json())


def seeded_start(seed, sensor=None, spec=None):
    scenario = {**break_turn(), "seed": seed}
    if sensor is not None:
        scenario["sensor"] = {"position": sensor}
    if spec is not None:
        scenario["start_spec"] = spec
    return scenario_start(scenario, np.random.default_rng(seed), NOMINAL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_seeded_random_start_is_drawn_around_the_sensor(seed):
    """MS-49: a random flight around a moved sensor is the one drawn around the origin, carried with the sensor."""
    around_origin, around_sensor = seeded_start(seed), seeded_start(seed, RAISED_SENSOR_M)

    assert around_sensor["position"].tolist() == (around_origin["position"] + RAISED_SENSOR_M).tolist()
    assert around_sensor["velocity"].tolist() == around_origin["velocity"].tolist()


def test_set_range_and_bearing_are_from_the_sensor_and_set_altitude_is_above_z_0():
    """MS-49: start_spec range and bearing are from the sensor's horizontal position, altitude above z = 0."""
    start = seeded_start(5, RAISED_SENSOR_M, {"range_m": 2000.0, "bearing_deg": 0.0, "altitude_m": 300.0})

    assert start["position"].tolist() == [1000.0, 1500.0, 300.0]


def test_an_unset_range_keeps_the_drawn_ground_range_from_the_sensor():
    """MS-49: an unset start_spec field keeps the draw made around the sensor."""
    around_origin = seeded_start(6)["position"]
    start = seeded_start(6, RAISED_SENSOR_M, {"bearing_deg": 0.0})

    assert start["position"].tolist() == [
        1000.0,
        -500.0 + np.hypot(*around_origin[:2]),
        around_origin[2] + 50.0,
    ]


def test_moved_sensor_case_puts_the_sensor_at_the_ticket_offset_and_flies_west_from_a_world_start():
    """MS-49: the page_parity case is the ticket's sensor [1000, -500, 0] with a recorded world start."""
    scenario = case_scenarios()[SENSOR_MOVED_CASE]

    assert scenario["sensor"] == {"position": [1000.0, -500.0, 0.0]}
    assert scenario["start"] == {"position": [3123.456, 987.654, 300.0], "velocity": [-150.0, 0.0, 0.0]}
