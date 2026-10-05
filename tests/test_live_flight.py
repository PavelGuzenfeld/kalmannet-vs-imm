import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from data.config import NOMINAL
from data.fixed_wing import DEFAULT_PRESET, PRESETS
from data.trajectory_gen import fly_scenario
from filters.flags import DEFAULT_THRESHOLDS, FLAGS
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.numpy_forward import load_weights
from filters.motion_models import cv_model
from filters.torch_ops import export_npz
from scripts.export_weights import main as export_weights_main
from scripts.replay import SEED_SCENARIO_SEED, break_turn_controls, numpy_weights
from tests.test_numpy_forward import knet_stand_in, rkn_stand_in
from web.live_flight import FILTER_NAMES, LEARNED_WEIGHTS, LiveFlight, page_options

REPO = Path(__file__).resolve().parents[1]
SEED_SCENARIO = REPO / "data" / "scenarios" / "break_turn_9g.json"
PAGE_IMM_TUNING = REPO / "web" / "imm_tuning.json"
LOOSE_RAW_ERROR_M = 50.0
KNET_INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ESTIMATE_ROUND_TRIP_ATOL_M = 1e-9


@pytest.fixture
def learned(tmp_path):
    """Seeded stand-in checkpoints for replay, and their .npz where the page looks for them."""
    checkpoints = {}
    for name, net in (("KalmanNet", knet_stand_in()), ("RKN", rkn_stand_in())):
        checkpoints[name] = tmp_path / f"{name}.pt"
        torch.save(net.state_dict(), checkpoints[name])
        npz = tmp_path / "site" / LEARNED_WEIGHTS[name]
        npz.parent.mkdir(parents=True, exist_ok=True)
        export_npz(net.state_dict(), npz)
    return tmp_path / "site", checkpoints


def page_flight(weights_root, noise_regime="nominal", thresholds=DEFAULT_THRESHOLDS, preset=DEFAULT_PRESET):
    setup = {
        "seed": SEED_SCENARIO_SEED,
        "envelope": asdict(PRESETS[preset]),
        "noise_regime": noise_regime,
        "thresholds": thresholds,
    }
    flight = LiveFlight(setup, json.loads(PAGE_IMM_TUNING.read_text())["best"], weights_root)
    for control in break_turn_controls():
        flight.advance(control)
    return flight


def replay_lines(scenario_path, checkpoints):
    result = subprocess.run(
        [sys.executable, "scripts/replay.py", str(scenario_path), "--imm-tuning", str(PAGE_IMM_TUNING),
         "--knet-checkpoint", str(checkpoints["KalmanNet"]), "--rkn-checkpoint", str(checkpoints["RKN"])],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if ":" in line]


def preview_lines(flight):
    flags = flight.frame()["flags"]
    return [f"{name}: " + " ".join(f"{flag}={flags[name][flag]}" for flag in FLAGS) for name in FILTER_NAMES]


def test_exported_break_turn_replays_to_the_page_preview_flags(learned, tmp_path):
    """MS-10, MS-11, MS-12: page and replay fly the same seed and controls under nominal R, the benchmarked IMM,
    and the learned filters' numpy forward on the weights replay runs through torch."""
    weights_root, checkpoints = learned
    flight = page_flight(weights_root)
    exported = tmp_path / "flight.json"
    exported.write_text(flight.scenario_json())

    lines = replay_lines(exported, checkpoints)

    assert lines == preview_lines(flight)
    committed = json.loads(SEED_SCENARIO.read_text())
    assert json.loads(exported.read_text()) == {**committed, "git_tag": "unknown", "tripped": flight.trips()}
    assert flight.trips()["EKF"] == committed["tripped"]["EKF"]


@pytest.mark.parametrize("noise_regime, preset", zip(page_options()["noise_regimes"], reversed(PRESETS)))
def test_exported_regime_envelope_and_thresholds_replay_to_the_page_preview_flags(
    noise_regime, preset, learned, tmp_path
):
    """MS-11: the scenario's regime draws the noise, the filters still assume nominal R; replay reads the recorded
    envelope and thresholds."""
    weights_root, checkpoints = learned
    thresholds = {**DEFAULT_THRESHOLDS, "raw_error_m": LOOSE_RAW_ERROR_M}
    flight = page_flight(weights_root, noise_regime, thresholds, preset)
    exported = tmp_path / "flight.json"
    exported.write_text(flight.scenario_json())

    lines = replay_lines(exported, checkpoints)

    assert lines == preview_lines(flight)
    recorded = json.loads(exported.read_text())
    assert (recorded["noise_regime"], recorded["envelope"], recorded["thresholds"]) == (
        noise_regime, asdict(PRESETS[preset]), thresholds
    )


def test_picker_offers_the_four_noise_regimes_and_not_the_manoeuvre_regime():
    """MS-21: the page varies measurement noise only; high_maneuver shapes the random-trajectory truth."""
    assert page_options()["noise_regimes"] == ["nominal", "high_noise", "low_noise", "heavy_tailed"]


def test_learned_filter_without_exported_weights_reads_unavailable_and_is_left_out_of_the_export(tmp_path):
    """MS-10: a learned filter runs only on its trained weights; RKN has no innovation covariance to gate."""
    flight = page_flight(tmp_path)

    flags = flight.frame()["flags"]

    assert flags["KalmanNet"] == dict.fromkeys(FLAGS, "not available (no weights at results/knet_vanilla/best.npz)")
    assert flags["RKN"] == {
        "track_loss": "n/a (RKN defines no innovation covariance)",
        "overconfidence": "not available (no weights at results/rkn/best.npz)",
        "raw_error": "not available (no weights at results/rkn/best.npz)",
    }
    assert set(json.loads(flight.scenario_json())["tripped"]) == {"EKF", "UKF", "IMM"}


def test_export_weights_writes_the_npz_the_page_loads_beside_each_checkpoint(tmp_path, monkeypatch):
    """MS-10: the page's numpy forward reads the checkpoint's own state_dict arrays, by name."""
    checkpoint = tmp_path / "rkn" / "best.pt"
    checkpoint.parent.mkdir()
    torch.save(rkn_stand_in().state_dict(), checkpoint)
    monkeypatch.setattr(sys, "argv", ["export_weights.py", str(checkpoint)])

    export_weights_main()

    written = load_weights(tmp_path / "rkn" / "best.npz")
    expected = numpy_weights(checkpoint)
    assert written.keys() == expected.keys()
    assert all(np.array_equal(written[name], expected[name]) for name in expected)


def test_plotted_zero_gain_kalmannet_estimate_is_open_loop_cv_from_the_start_offset(tmp_path):
    """MS-10: KalmanNet keeps the CV f and learns only the gain, so zero gain dead-reckons. The page plots truth plus
    error; at km positions that round trip loses ~1e-13 m, so atol 1e-9 m."""
    net = KalmanGainGRU().double()
    for parameter in net.parameters():
        torch.nn.init.zeros_(parameter)
    npz = tmp_path / LEARNED_WEIGHTS["KalmanNet"]
    npz.parent.mkdir(parents=True)
    export_npz(net.state_dict(), npz)
    states, _ = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    F, _ = cv_model(NOMINAL.dt)
    dead_reckoned = states[0] + KNET_INIT_OFFSET
    for _ in states:
        dead_reckoned = F @ dead_reckoned

    plotted = page_flight(tmp_path).frame()["filters"]["KalmanNet"]["estimate_m"]

    np.testing.assert_allclose(plotted, dead_reckoned[:3], rtol=0.0, atol=ESTIMATE_ROUND_TRIP_ATOL_M)
