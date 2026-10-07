import json
from dataclasses import asdict

import numpy as np
import pytest

from data.config import NOMINAL
from data.fixed_wing import DEFAULT_PRESET, PRESETS
from data.trajectory_gen import scenario_start
from filters.flags import DEFAULT_THRESHOLDS
from scripts.replay import break_turn_controls
from tests.test_live_flight import PAGE_IMM_TUNING, learned, preview_lines, replay_lines  # noqa: F401
from web.live_flight import LiveFlight

SEED = 8
ENVELOPE = PRESETS[DEFAULT_PRESET]
FULL_SPEC = {
    "range_m": 2000.0,
    "bearing_deg": 90.0,
    "altitude_m": 300.0,
    "heading_deg": 270.0,
    "speed_m_s": 150.0,
}
COSINE_OF_QUARTER_TURN_RESIDUE_M = 1e-9


def setup_for(seed=SEED, **extra):
    return {
        "seed": seed,
        "envelope": asdict(ENVELOPE),
        "noise_regime": "nominal",
        "thresholds": DEFAULT_THRESHOLDS,
        **extra,
    }


def flight_for(weights_root, **setup):
    return LiveFlight(setup_for(**setup), json.loads(PAGE_IMM_TUNING.read_text())["best"], weights_root)


def exported_start(tmp_path, **setup):
    flight = flight_for(tmp_path, **setup)
    flight.advance(break_turn_controls()[0])
    return json.loads(flight.scenario_json()).get("start")


def drawn_start(seed=SEED):
    return scenario_start(setup_for(seed), np.random.default_rng(seed), NOMINAL)


def bearing_deg(position):
    return np.rad2deg(np.arctan2(position[0], position[1]))


def test_east_bearing_west_heading_start_exports_east_north_up_position_and_westward_velocity(tmp_path):
    start = exported_start(tmp_path, start_spec=FULL_SPEC)

    np.testing.assert_allclose(
        start["position"], [2000.0, 0.0, 300.0], rtol=0.0, atol=COSINE_OF_QUARTER_TURN_RESIDUE_M
    )
    np.testing.assert_allclose(
        start["velocity"], [-150.0, 0.0, 0.0], rtol=0.0, atol=COSINE_OF_QUARTER_TURN_RESIDUE_M
    )


def test_start_spec_is_resolved_into_the_export_and_not_kept(tmp_path):
    flight = flight_for(tmp_path, start_spec=FULL_SPEC)
    flight.advance(break_turn_controls()[0])
    exported = json.loads(flight.scenario_json())

    assert "start_spec" not in exported
    assert set(exported["start"]) == {"position", "velocity"}


@pytest.mark.parametrize("spec", [{}, None])
def test_all_random_start_exports_the_same_bytes_as_a_setup_without_a_spec(tmp_path, spec):
    plain = flight_for(tmp_path)
    random = flight_for(tmp_path, **({} if spec is None else {"start_spec": spec}))
    for flight in (plain, random):
        for control in break_turn_controls()[:30]:
            flight.advance(control)

    assert random.scenario_json() == plain.scenario_json()
    assert "start" not in json.loads(plain.scenario_json())


def test_random_range_keeps_the_seeded_bearing_altitude_velocity(tmp_path):
    drawn = drawn_start()
    spec = {"range_m": 1000.0}

    start = exported_start(tmp_path, start_spec=spec)

    assert np.hypot(*start["position"][:2]) == pytest.approx(1000.0, abs=COSINE_OF_QUARTER_TURN_RESIDUE_M)
    assert bearing_deg(start["position"]) == pytest.approx(bearing_deg(drawn["position"]), abs=1e-9)
    assert start["position"][2] == drawn["position"][2]
    assert start["velocity"] == drawn["velocity"].tolist()


def test_random_altitude_keeps_the_seeded_ground_range(tmp_path):
    drawn = drawn_start()

    start = exported_start(tmp_path, start_spec={"altitude_m": 250.0})

    assert start["position"][2] == 250.0
    assert np.hypot(*start["position"][:2]) == pytest.approx(np.hypot(*drawn["position"][:2]), abs=1e-9)


def test_random_heading_keeps_the_seeded_heading_and_flies_level_at_the_set_speed(tmp_path):
    drawn = drawn_start()

    start = exported_start(tmp_path, start_spec={"speed_m_s": 100.0})

    assert start["velocity"][2] == 0.0
    assert np.hypot(*start["velocity"][:2]) == pytest.approx(100.0, abs=1e-9)
    assert bearing_deg(start["velocity"]) == pytest.approx(bearing_deg(drawn["velocity"]), abs=1e-9)
    assert start["position"] == drawn["position"].tolist()


def test_random_speed_keeps_the_seeded_speed_on_the_set_heading(tmp_path):
    drawn = drawn_start()

    start = exported_start(tmp_path, start_spec={"heading_deg": 0.0})

    assert start["velocity"][0] == 0.0
    assert start["velocity"][1] == pytest.approx(np.linalg.norm(drawn["velocity"]), abs=1e-9)


def test_partial_start_is_the_same_for_the_same_seed_and_differs_across_seeds(tmp_path):
    spec = {"range_m": 1500.0, "heading_deg": 45.0}

    first = exported_start(tmp_path, start_spec=spec)

    assert exported_start(tmp_path, start_spec=spec) == first
    assert exported_start(tmp_path, seed=SEED + 1, start_spec=spec) != first


@pytest.mark.parametrize("speed", [ENVELOPE.min_speed_m_s - 0.5, ENVELOPE.max_speed_m_s + 0.5])
def test_speed_outside_the_envelope_is_refused_with_the_limits(tmp_path, speed):
    with pytest.raises(
        ValueError, match=rf"speed {speed} m/s.*{ENVELOPE.min_speed_m_s}.*{ENVELOPE.max_speed_m_s}"
    ):
        flight_for(tmp_path, start_spec={"speed_m_s": speed})


@pytest.mark.parametrize("speed", [ENVELOPE.min_speed_m_s, ENVELOPE.max_speed_m_s])
def test_speed_at_an_envelope_limit_is_accepted(tmp_path, speed):
    start = exported_start(tmp_path, start_spec={"speed_m_s": speed})

    assert np.linalg.norm(start["velocity"]) == pytest.approx(speed, abs=1e-9)


def test_altitude_below_the_ground_plane_is_refused(tmp_path):
    with pytest.raises(ValueError, match="altitude -1.0 m"):
        flight_for(tmp_path, start_spec={"altitude_m": -1.0})


def test_altitude_on_the_ground_plane_is_accepted(tmp_path):
    assert exported_start(tmp_path, start_spec={"altitude_m": 0.0})["position"][2] == 0.0


def test_negative_range_is_refused(tmp_path):
    with pytest.raises(ValueError, match="range -1.0 m"):
        flight_for(tmp_path, start_spec={"range_m": -1.0})


def test_zero_range_is_accepted(tmp_path):
    assert exported_start(tmp_path, start_spec={"range_m": 0.0, "altitude_m": 100.0})["position"] == [
        0.0,
        0.0,
        100.0,
    ]


def test_unknown_start_spec_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="range"):
        flight_for(tmp_path, start_spec={"range": 1000.0})


def test_exported_set_start_replays_to_the_page_preview_flags(learned, tmp_path):  # noqa: F811
    weights_root, checkpoints = learned
    flight = flight_for(weights_root, start_spec=FULL_SPEC)
    for control in break_turn_controls():
        flight.advance(control)
    exported = tmp_path / "flight.json"
    exported.write_text(flight.scenario_json())

    assert replay_lines(exported, checkpoints) == preview_lines(flight)
    assert json.loads(exported.read_text())["start"]["position"][2] == 300.0
