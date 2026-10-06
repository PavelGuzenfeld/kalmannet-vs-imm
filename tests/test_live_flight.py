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
from filters.flags import DEFAULT_THRESHOLDS, ELLIPSOID_95_DOF3, FLAGS, NEES_LOWER_025_DOF9
from filters.ekf import EKF
from filters.imm import IMM
from filters.plot_data import imm_kwargs_from_tuning
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.numpy_forward import load_weights
from filters.motion_models import cv_model
from filters.torch_ops import export_npz
from scripts.export_weights import main as export_weights_main
from scripts.replay import SEED_SCENARIO_SEED, break_turn_controls, numpy_weights
from tests.test_numpy_forward import knet_stand_in, rkn_stand_in
from scripts.eval import ekf_like_steps, imm_steps
from web.live_flight import FILTER_NAMES, LEARNED_WEIGHTS, LiveFlight, _ellipsoid, page_options

REPO = Path(__file__).resolve().parents[1]
SEED_SCENARIO = REPO / "data" / "scenarios" / "break_turn_9g.json"
OVERHEAD_SCENARIO = REPO / "data" / "scenarios" / "overhead_pass.json"
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


@pytest.fixture(scope="module")
def seed_frame(tmp_path_factory):
    return page_flight(tmp_path_factory.mktemp("no_weights")).frame()


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
        [
            sys.executable,
            "scripts/replay.py",
            str(scenario_path),
            "--imm-tuning",
            str(PAGE_IMM_TUNING),
            "--knet-checkpoint",
            str(checkpoints["KalmanNet"]),
            "--rkn-checkpoint",
            str(checkpoints["RKN"]),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if ":" in line]


def preview_lines(flight):
    flags = flight.frame()["flags"]
    return [f"{name}: " + " ".join(f"{flag}={flags[name][flag]}" for flag in FLAGS) for name in FILTER_NAMES]


def test_exported_break_turn_replays_to_the_page_preview_flags(learned, tmp_path):
    """MS-10, MS-11, MS-12: page and replay fly one seed and controls under nominal R, the benchmarked IMM,
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
    """MS-11: the scenario's regime draws the noise, filters still assume nominal R; replay reads the recorded
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
        noise_regime,
        asdict(PRESETS[preset]),
        thresholds,
    )


def test_picker_offers_the_four_noise_regimes_and_not_the_manoeuvre_regime():
    """MS-21: the page varies measurement noise only; high_maneuver shapes the random-trajectory truth."""
    assert page_options()["noise_regimes"] == ["nominal", "high_noise", "low_noise", "heavy_tailed"]


def test_learned_filter_without_exported_weights_reads_unavailable_and_is_left_out_of_the_export(tmp_path):
    """MS-10: a learned filter runs only on its trained weights; RKN has no innovation covariance to gate."""
    flight = page_flight(tmp_path)

    flags = flight.frame()["flags"]

    assert flags["KalmanNet"] == dict.fromkeys(
        FLAGS, "not available (no weights at results/knet_vanilla/best.npz)"
    )
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
    """MS-10: KalmanNet keeps CV f and learns only the gain, so zero gain dead-reckons. The page plots truth +
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


def test_scenario_with_a_recorded_start_flies_from_it_and_replays_to_the_page_preview_flags(
    learned, tmp_path
):
    """MS-35: the overhead pass starts at its recorded state, not a seeded draw; the page keeps that start."""
    weights_root, checkpoints = learned
    scenario = json.loads(OVERHEAD_SCENARIO.read_text())
    flight = LiveFlight(scenario, json.loads(PAGE_IMM_TUNING.read_text())["best"], weights_root)
    for control in scenario["controls"]:
        flight.advance(control)
    exported = tmp_path / "flight.json"
    exported.write_text(flight.scenario_json())

    lines = replay_lines(exported, checkpoints)

    assert lines == preview_lines(flight)
    assert json.loads(exported.read_text())["start"] == scenario["start"]
    assert flight.frame()["truth_m"] == fly_scenario(scenario)[0][-1, :3].tolist()


def test_ellipsoid_of_an_axis_aligned_position_covariance_has_the_95_percent_semi_axes_and_identity_axes():
    """Scaled by the pinned chi-square(3) 95% quantile; sqrt of exact binary variances, so atol only absorbs it."""
    P = np.diag([4.0, 16.0, 64.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    ellipsoid = _ellipsoid(P)

    expected_semi_axes = np.sqrt(np.array([4.0, 16.0, 64.0]) * ELLIPSOID_95_DOF3)
    np.testing.assert_allclose(ellipsoid["semi_axes_m"], expected_semi_axes, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(np.abs(ellipsoid["axes"]), np.eye(3))


def test_ellipsoid_axis_rows_point_along_the_long_direction_of_a_rotated_covariance():
    P = np.eye(9)
    P[0, 0], P[1, 1], P[0, 1], P[1, 0] = 5.0, 5.0, 4.0, 4.0

    ellipsoid = _ellipsoid(P)

    long_axis = np.abs(ellipsoid["axes"][-1])
    np.testing.assert_allclose(long_axis, [np.sqrt(0.5), np.sqrt(0.5), 0.0], rtol=0.0, atol=1e-12)
    assert ellipsoid["semi_axes_m"][-1] == pytest.approx(np.sqrt(9.0 * ELLIPSOID_95_DOF3), abs=1e-12)


def test_ellipsoid_of_a_covariance_with_a_negative_variance_has_a_zero_semi_axis():
    assert _ellipsoid(np.diag([-1.0, 1.0, 1.0, 1, 1, 1, 1, 1, 1.0]))["semi_axes_m"][0] == 0.0


def test_ellipsoid_of_a_non_finite_covariance_is_absent():
    P = np.eye(9)
    P[2, 2] = np.nan
    assert _ellipsoid(P) is None


def test_frame_carries_ellipsoids_for_covariance_filters_and_mode_probabilities_for_the_imm_only(seed_frame):
    """MS-12: the IMM's four modes are CV, CT+, CT-, CA; KalmanNet has no covariance to draw."""
    filters = seed_frame["filters"]
    assert all(filters[name]["ellipsoid"] is not None for name in ("EKF", "UKF", "IMM"))
    assert len(filters["IMM"]["mode_probabilities"]) == 4
    assert sum(filters["IMM"]["mode_probabilities"]) == pytest.approx(1.0, abs=1e-12)
    assert filters["EKF"]["mode_probabilities"] is None


def test_frame_ellipsoid_comes_from_the_covariance_that_scored_the_nees(learned):
    """MS-10: RKN's plotted covariance is the P its NEES is computed from."""
    weights_root, _ = learned
    frame = page_flight(weights_root).frame()["filters"]

    assert frame["RKN"]["ellipsoid"] is not None
    assert frame["KalmanNet"]["ellipsoid"] is None


def test_frame_trips_are_the_numeric_steps_the_flag_cells_show(seed_frame):
    committed = json.loads(SEED_SCENARIO.read_text())["tripped"]

    assert {name: seed_frame["trips"][name] for name in committed} == committed
    for name in ("EKF", "UKF", "IMM"):
        for flag, step in seed_frame["trips"][name].items():
            assert seed_frame["flags"][name][flag] == ("-" if step is None else str(step))
    assert seed_frame["trips"]["IMM"]["raw_error"] is None


def test_frame_truth_velocity_is_the_flown_state_velocity(seed_frame):
    states, _ = fly_scenario(json.loads(SEED_SCENARIO.read_text()))

    assert seed_frame["truth_velocity_m_s"] == states[-1, 3:6].tolist()


def test_ekf_snapshot_holds_the_covariance_the_last_nees_was_scored_with():
    states, z = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    snapshot = {}
    steps = ekf_like_steps(EKF(NOMINAL.dt, NOMINAL.R()), states, z, snapshot)

    err, nees, _ = list(steps)[-1]

    assert nees == err @ np.linalg.solve(snapshot["P"], err)


def test_imm_snapshot_holds_the_mode_probabilities_after_the_last_step(seed_frame):
    """MS-8: the IMM combines over [cv, ct+, ct-, ca]; the frame carries them in that order."""
    states, z = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    best = json.loads(PAGE_IMM_TUNING.read_text())["best"]
    snapshot = {}
    imm = IMM(NOMINAL.dt, NOMINAL.R(), **imm_kwargs_from_tuning(best))

    list(imm_steps(imm, states, z, snapshot))

    assert imm.mode_names == ["cv", "ct_pos", "ct_neg", "ca"]
    assert seed_frame["filters"]["IMM"]["mode_probabilities"] == snapshot["mu"].tolist()
    assert _ellipsoid(snapshot["P"]) == seed_frame["filters"]["IMM"]["ellipsoid"]
    assert snapshot["mu"].sum() == pytest.approx(1.0, abs=1e-12)


def test_page_options_offer_the_nees_lower_bound():
    assert page_options()["nees_lower"] == NEES_LOWER_025_DOF9
