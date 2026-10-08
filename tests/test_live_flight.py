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
from tests.test_replay import recorded_before_wider_row
from tests.test_unscented_kalmannet import stand_in_network
from filters.state import POS
from scripts.eval import ekf_like_steps, imm_steps, run_ekf_like
from web.live_flight import (
    FILTER_NAMES,
    LEARNED_WEIGHTS,
    LiveFlight,
    _ellipsoid,
    page_options,
)

REPO = Path(__file__).resolve().parents[1]
SEED_SCENARIO = REPO / "data" / "scenarios" / "break_turn_9g.json"
OVERHEAD_SCENARIO = REPO / "data" / "scenarios" / "overhead_pass.json"
PAGE_IMM_TUNING = REPO / "web" / "imm_tuning.json"
RAW_ERROR_LIMIT_OFFSET_M = {"just_below_peak": 0.0, "just_above_peak": 1.0}
KNET_INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ESTIMATE_ROUND_TRIP_ATOL_M = 1e-9
LEVEL_RUN_IN_STEPS = 50


@pytest.fixture
def learned(tmp_path):
    """Seeded stand-in checkpoints for replay, and their .npz where the page looks for them."""
    checkpoints = {}
    for name, net in (("KalmanNet", knet_stand_in()), ("RKN", rkn_stand_in()), ("BKN", knet_stand_in()), ("UKN", stand_in_network())):
        checkpoints[name] = tmp_path / f"{name}.pt"
        torch.save(net.state_dict(), checkpoints[name])
        npz = tmp_path / "site" / LEARNED_WEIGHTS[name]
        npz.parent.mkdir(parents=True, exist_ok=True)
        export_npz(net.state_dict(), npz)
    return tmp_path / "site", checkpoints


@pytest.fixture(scope="module")
def seed_frame(tmp_path_factory):
    return page_flight(tmp_path_factory.mktemp("no_weights")).frame()


def page_flight(weights_root, noise_regime="nominal", thresholds=DEFAULT_THRESHOLDS, preset=DEFAULT_PRESET, steps=None):
    setup = {
        "seed": SEED_SCENARIO_SEED,
        "envelope": asdict(PRESETS[preset]),
        "noise_regime": noise_regime,
        "thresholds": thresholds,
    }
    flight = LiveFlight(setup, json.loads(PAGE_IMM_TUNING.read_text())["best"], weights_root)
    for control in break_turn_controls()[:steps]:
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
            "--bkn-checkpoint",
            str(checkpoints["BKN"]),
            "--ukn-checkpoint",
            str(checkpoints["UKN"]),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if ":" in line]


def ekf_position_errors_m(flight):
    states, z = fly_scenario(json.loads(flight.scenario_json()))
    err, _, _ = run_ekf_like(EKF(NOMINAL.dt, NOMINAL.R()), states, z)
    return np.linalg.norm(err[:, POS], axis=1)


def preview_lines(flight):
    frame = flight.frame()
    steps = {name: readout["diverged_step"] for name, readout in frame["filters"].items()}
    return [
        f"{name}: "
        + " ".join(f"{flag}={frame['flags'][name][flag]}" for flag in FLAGS)
        + ("" if steps.get(name) is None else f" diverged={steps[name]}")
        for name in FILTER_NAMES
    ]


def test_exported_break_turn_replays_to_the_page_preview_flags(learned, tmp_path):
    """MS-10, MS-11, MS-12, MS-39: page and replay fly one seed and controls under nominal R, the benchmarked IMM,
    the learned filters' numpy forward on the weights replay runs through torch, and BKN's dropout seeded alike."""
    weights_root, checkpoints = learned
    flight = page_flight(weights_root)
    exported = tmp_path / "flight.json"
    exported.write_text(flight.scenario_json())

    lines = replay_lines(exported, checkpoints)

    assert lines == preview_lines(flight)
    assert [line.split(":")[0] for line in lines if " diverged=" in line] == ["KalmanNet", "RKN", "BKN"]
    committed = json.loads(SEED_SCENARIO.read_text())
    assert json.loads(exported.read_text()) == {**committed, "git_tag": "unknown", "tripped": flight.trips()}
    assert {name: flight.trips()[name] for name in committed["tripped"]} == committed["tripped"]


@pytest.mark.parametrize("raw_error_side", RAW_ERROR_LIMIT_OFFSET_M)
@pytest.mark.parametrize("noise_regime, preset", zip(page_options()["noise_regimes"], reversed(PRESETS)))
def test_exported_regime_envelope_and_thresholds_replay_to_the_page_preview_flags(
    noise_regime, preset, raw_error_side, learned, tmp_path
):
    """MS-11: the scenario's regime draws the noise, filters still assume nominal R; replay reads the recorded
    envelope and thresholds. The raw-error limit is the whole metre at or above the EKF's peak error, so the
    flag trips at the first step past it, or never."""
    weights_root, checkpoints = learned
    peak_m = ekf_position_errors_m(page_flight(tmp_path / "probe", noise_regime, preset=preset)).max()
    assert peak_m != np.floor(peak_m)
    limit_m = float(np.floor(peak_m)) + RAW_ERROR_LIMIT_OFFSET_M[raw_error_side]
    thresholds = {**DEFAULT_THRESHOLDS, "raw_error_m": limit_m}
    flight = page_flight(weights_root, noise_regime, thresholds, preset)
    errors_m = ekf_position_errors_m(flight)
    expected_step = int(np.argmax(errors_m > limit_m)) if raw_error_side == "just_below_peak" else None
    exported = tmp_path / "flight.json"
    exported.write_text(flight.scenario_json())

    lines = replay_lines(exported, checkpoints)

    assert lines == preview_lines(flight)
    assert lines[0].endswith(f"raw_error={'-' if expected_step is None else expected_step}")
    assert flight.trips()["EKF"]["raw_error"] == expected_step
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
    assert flags["BKN"] == dict.fromkeys(FLAGS, "not available (no weights at results/bkn/best.npz)")
    assert flags["UKN"] == dict.fromkeys(FLAGS, "not available (no weights at results/ukn/best.npz)")
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
    error; at km positions that round trip loses ~1e-13 m, so atol 1e-9 m. Over the level run-in only: the turn
    takes dead reckoning past the 200 m divergence limit, where the page stops plotting it."""
    net = KalmanGainGRU().double()
    for parameter in net.parameters():
        torch.nn.init.zeros_(parameter)
    npz = tmp_path / LEARNED_WEIGHTS["KalmanNet"]
    npz.parent.mkdir(parents=True)
    export_npz(net.state_dict(), npz)
    states, _ = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    F, _ = cv_model(NOMINAL.dt)
    dead_reckoned = states[0] + KNET_INIT_OFFSET
    for _ in states[:LEVEL_RUN_IN_STEPS]:
        dead_reckoned = F @ dead_reckoned

    plotted = page_flight(tmp_path, steps=LEVEL_RUN_IN_STEPS).frame()["filters"]["KalmanNet"]["estimate_m"]

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


def test_page_3_column_rows_export_as_full_rows_with_every_new_input_neutral(tmp_path):
    """MS-27, #106 decision 4: a recording writes the full control row whatever width the page sent."""
    committed = json.loads(OVERHEAD_SCENARIO.read_text())
    scenario = recorded_before_wider_row(committed)
    flight = LiveFlight(scenario, json.loads(PAGE_IMM_TUNING.read_text())["best"], tmp_path)
    for control in scenario["controls"]:
        flight.advance(control)

    assert json.loads(flight.scenario_json())["controls"] == committed["controls"]


def test_page_6_column_rows_fly_the_new_inputs_and_export_them_unchanged(tmp_path):
    """MS-27: the live flight is the replayed flight, rudder, brake and afterburner included."""
    scenario = {**json.loads(OVERHEAD_SCENARIO.read_text()), "envelope": asdict(PRESETS[DEFAULT_PRESET])}
    scenario["controls"] = [[0.0, 1.0, 1.0, 2.0, 1.0, 1.0]] * 3 + [[0.0, 1.0, 1.0, -1.0, 0.0, 1.0]] * 3
    flight = LiveFlight(scenario, json.loads(PAGE_IMM_TUNING.read_text())["best"], tmp_path)
    for control in scenario["controls"]:
        flight.advance(control)

    exported = json.loads(flight.scenario_json())
    assert exported["controls"] == scenario["controls"]
    assert flight.frame()["truth_m"] == fly_scenario(exported)[0][-1, :3].tolist()
    neutral = {**scenario, "controls": [row[:3] for row in scenario["controls"]]}
    assert flight.frame()["truth_m"] != fly_scenario(neutral)[0][-1, :3].tolist()


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
    """MS-10: RKN's plotted covariance is the P its NEES is computed from; read before the stand-in diverges."""
    weights_root, _ = learned
    frame = page_flight(weights_root, steps=LEVEL_RUN_IN_STEPS).frame()["filters"]

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


def lift_frame(roll_rate_rad_s, steps):
    setup = {
        "seed": SEED_SCENARIO_SEED,
        "envelope": asdict(PRESETS[DEFAULT_PRESET]),
        "noise_regime": "nominal",
        "thresholds": DEFAULT_THRESHOLDS,
    }
    flight = LiveFlight(setup, json.loads(PAGE_IMM_TUNING.read_text())["best"], REPO / "no_weights")
    for _ in range(steps):
        flight.advance([roll_rate_rad_s, 0.0, 1.0])
    return flight.frame()


def test_frame_lift_direction_is_a_unit_vector_perpendicular_to_the_velocity_and_level_flight_has_horizontal_wings():
    """MS-38: unit lift, perpendicular to the velocity; wings along velocity x lift are horizontal at zero bank."""
    frame = lift_frame(0.0, 20)
    lift, velocity = np.array(frame["truth_lift_direction"]), np.array(frame["truth_velocity_m_s"])

    wing_axis = np.cross(velocity, lift)

    assert np.linalg.norm(lift) == pytest.approx(1.0, abs=1e-12)
    assert lift @ velocity == pytest.approx(0.0, abs=1e-9)
    assert wing_axis[2] == pytest.approx(0.0, abs=1e-9)


def test_frame_lift_direction_is_rolled_from_the_level_lift_by_the_bank_angle():
    """MS-38: positive roll banks the lift toward the right wing, velocity x level lift, by the integrated angle."""
    steps, roll_rate = 20, 0.5
    bank_rad = roll_rate * NOMINAL.dt * steps
    level_frame, banked_frame = lift_frame(0.0, steps), lift_frame(roll_rate, steps)
    level = np.array(level_frame["truth_lift_direction"])
    velocity = np.array(level_frame["truth_velocity_m_s"])
    right_wing = np.cross(velocity / np.linalg.norm(velocity), level)

    assert banked_frame["truth_lift_direction"] == pytest.approx(
        (np.cos(bank_rad) * level + np.sin(bank_rad) * right_wing).tolist(), abs=1e-9
    )


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


def test_a_raw_error_threshold_far_under_the_peak_errors_diverges_no_filter(tmp_path):
    """Break-turn peaks: EKF and UKF about 171 m; a 1 m raw-error threshold trips them, yet none passes 200 m."""
    thresholds = {**DEFAULT_THRESHOLDS, "raw_error_m": 1.0}

    readouts = page_flight(tmp_path, thresholds=thresholds).frame()["filters"]

    assert {name: readout["diverged"] for name, readout in readouts.items()} == {
        "EKF": False,
        "UKF": False,
        "IMM": False,
    }


MANOEUVRE_CRUISE_CONTROL = [0.0, 1.0, 0.6, 0.0, 0.0, 0.0]
MANOEUVRE_START = {"position": [0.0, 0.0, 3000.0], "velocity": [150.0, 0.0, 0.0]}


def manoeuvre_flight(tmp_path):
    setup = {
        "seed": SEED_SCENARIO_SEED,
        "envelope": asdict(PRESETS[DEFAULT_PRESET]),
        "noise_regime": "nominal",
        "thresholds": DEFAULT_THRESHOLDS,
        "start": MANOEUVRE_START,
    }
    return LiveFlight(setup, json.loads(PAGE_IMM_TUNING.read_text())["best"], tmp_path)


def test_manoeuvre_runs_to_its_end_ignoring_the_pilot_rows_then_the_pilot_resumes(tmp_path):
    flight = manoeuvre_flight(tmp_path)
    flight.start_manoeuvre("split-S")
    assert flight.manoeuvre == "split-S"
    while flight.manoeuvre:
        flight.advance(MANOEUVRE_CRUISE_CONTROL)
    assert len(flight.controls) > 200
    assert flight.controls[-1] == MANOEUVRE_CRUISE_CONTROL
    assert MANOEUVRE_CRUISE_CONTROL not in flight.controls[:-1]
    assert flight.frame()["manoeuvre"] is None
    assert flight.frame()["truth_m"][2] < MANOEUVRE_START["position"][2] - 300.0


def test_cancelled_manoeuvre_hands_the_very_next_step_back_to_the_pilot_row(tmp_path):
    flight = manoeuvre_flight(tmp_path)
    flight.start_manoeuvre("jink")
    for _ in range(10):
        flight.advance(MANOEUVRE_CRUISE_CONTROL)
    assert flight.frame()["manoeuvre"] == "jink"
    flight.cancel_manoeuvre()
    flight.advance(MANOEUVRE_CRUISE_CONTROL)
    assert flight.controls[-1] == MANOEUVRE_CRUISE_CONTROL
    assert flight.frame()["manoeuvre"] is None
    assert flight.controls[0][3] == PRESETS[DEFAULT_PRESET].max_side_load_factor


SCRIPTED_DIVERGES_AT_STEP = 3
SCRIPTED_PAGE_STEPS = 10


def scripted_filter_steps(pulled):
    def steps(*args, **kwargs):
        count = [0]
        pulled.append(count)
        for step in range(SCRIPTED_PAGE_STEPS * 2):
            count[0] += 1
            error_m = 1.0 if step < SCRIPTED_DIVERGES_AT_STEP else 300.0
            err = np.zeros(9)
            err[0] = error_m
            yield err, error_m**2, 0.0 if step <= SCRIPTED_DIVERGES_AT_STEP else 1e3

    return steps


def scripted_page_flight(monkeypatch, tmp_path):
    from web import live_flight

    pulled = []
    monkeypatch.setattr(live_flight, "ekf_like_steps", scripted_filter_steps(pulled))
    monkeypatch.setattr(live_flight, "imm_steps", scripted_filter_steps(pulled))
    thresholds = {**DEFAULT_THRESHOLDS, "raw_error_m": 1e9}
    return page_flight(tmp_path, thresholds=thresholds, steps=SCRIPTED_PAGE_STEPS), pulled


def test_page_pulls_each_diverged_filter_through_its_divergence_step_and_no_further(monkeypatch, tmp_path):
    flight, pulled = scripted_page_flight(monkeypatch, tmp_path)

    assert [count[0] for count in pulled] == [SCRIPTED_DIVERGES_AT_STEP + 1] * 3
    assert {r["diverged_step"] for r in flight.readouts().values()} == {SCRIPTED_DIVERGES_AT_STEP}


def test_page_does_not_trip_a_flag_that_would_only_trip_after_the_divergence_step(monkeypatch, tmp_path):
    """Past the divergence step the innovation reads 1e3, which would trip track_loss at the next step."""
    flight, _ = scripted_page_flight(monkeypatch, tmp_path)

    assert {name: found["track_loss"] for name, found in flight.trips().items()} == {"EKF": None, "UKF": None, "IMM": None}


def test_starting_a_manoeuvre_drops_every_engaged_hold(tmp_path):
    flight = manoeuvre_flight(tmp_path)
    for name in ("heading", "altitude", "speed", "bank"):
        flight.toggle_hold(name)
    flight.start_manoeuvre("jink")
    assert flight.holds.targets == {}


def test_engaging_a_hold_cancels_the_running_manoeuvre(tmp_path):
    flight = manoeuvre_flight(tmp_path)
    flight.start_manoeuvre("jink")
    flight.toggle_hold("heading")
    assert flight.manoeuvre is None
    assert "heading" in flight.holds.targets


BENCHMARK = PRESETS[DEFAULT_PRESET]
LOWERED_MAX_LOAD_FACTOR = 3.0
ROW_LOAD_FACTOR = 1


def lowered(**fields):
    return {**asdict(BENCHMARK), "max_load_factor": LOWERED_MAX_LOAD_FACTOR, **fields}


def exported_changes(flight):
    return json.loads(flight.scenario_json()).get("envelope_changes")


def test_envelope_set_after_k_steps_is_exported_at_step_k_beside_the_starting_envelope(tmp_path):
    """MS-41: a change governs row k onward and the export keeps the starting envelope."""
    flight = page_flight(tmp_path, steps=LEVEL_RUN_IN_STEPS)

    flight.set_envelope(lowered())
    flight.advance(MANOEUVRE_CRUISE_CONTROL)

    assert exported_changes(flight) == [[LEVEL_RUN_IN_STEPS, lowered()]]
    assert json.loads(flight.scenario_json())["envelope"] == asdict(BENCHMARK)


def test_envelopes_set_twice_before_one_step_record_only_the_last(tmp_path):
    """MS-41: at most one change per step, the last one set."""
    flight = page_flight(tmp_path, steps=LEVEL_RUN_IN_STEPS)

    flight.set_envelope(lowered(max_load_factor=5.0))
    flight.set_envelope(lowered())
    flight.advance(MANOEUVRE_CRUISE_CONTROL)

    assert exported_changes(flight) == [[LEVEL_RUN_IN_STEPS, lowered()]]


def test_envelope_set_back_before_its_step_or_resent_unchanged_records_no_change(tmp_path):
    """MS-41: a change never repeats the envelope already in effect."""
    flight = page_flight(tmp_path, steps=LEVEL_RUN_IN_STEPS)

    flight.set_envelope(lowered())
    flight.set_envelope(asdict(BENCHMARK))
    flight.advance(MANOEUVRE_CRUISE_CONTROL)
    flight.set_envelope(asdict(BENCHMARK))
    flight.advance(MANOEUVRE_CRUISE_CONTROL)

    assert "envelope_changes" not in json.loads(flight.scenario_json())


def test_returning_to_the_starting_envelope_a_step_later_is_a_second_change(tmp_path):
    """MS-41: in effect means the previous row's envelope, not the starting one."""
    flight = page_flight(tmp_path, steps=LEVEL_RUN_IN_STEPS)

    flight.set_envelope(lowered())
    flight.advance(MANOEUVRE_CRUISE_CONTROL)
    flight.set_envelope(asdict(BENCHMARK))
    flight.advance(MANOEUVRE_CRUISE_CONTROL)

    assert exported_changes(flight) == [[LEVEL_RUN_IN_STEPS, lowered()], [LEVEL_RUN_IN_STEPS + 1, asdict(BENCHMARK)]]


def flown_truth(flight, controls, change_at_step=None, envelope=None):
    truths = []
    for step, control in enumerate(controls):
        if step == change_at_step:
            flight.set_envelope(envelope)
        flight.advance(control)
        truths.append(flight.frame()["truth_m"] + flight.frame()["truth_velocity_m_s"])
    return np.array(truths)


def test_live_envelope_change_mid_pull_flies_the_truth_its_export_replays_bit_for_bit(tmp_path):
    """MS-27, MS-41: page and replay fly one piecewise-constant envelope over steps."""
    controls = break_turn_controls()
    flight = page_flight(tmp_path, steps=0)

    truth = flown_truth(flight, controls, change_at_step=LEVEL_RUN_IN_STEPS + 50, envelope=lowered())

    states, _ = fly_scenario(json.loads(flight.scenario_json()))
    assert exported_changes(flight) == [[LEVEL_RUN_IN_STEPS + 50, lowered()]]
    assert truth.tobytes() == states[:, :6].tobytes()


@pytest.mark.parametrize("committed", [SEED_SCENARIO, OVERHEAD_SCENARIO], ids=lambda path: path.stem)
def test_committed_scenario_flown_live_with_its_envelope_resent_exports_no_change_and_the_same_truth(
    committed, tmp_path
):
    """MS-41: a scenario without envelope_changes flies one envelope throughout, as before."""
    scenario = json.loads(committed.read_text())
    flight = LiveFlight(scenario, json.loads(PAGE_IMM_TUNING.read_text())["best"], tmp_path)
    for control in scenario["controls"]:
        flight.set_envelope(scenario["envelope"])
        flight.advance(control)

    exported = json.loads(flight.scenario_json())

    assert "envelope_changes" not in exported
    assert fly_scenario(exported)[0].tobytes() == fly_scenario(scenario)[0].tobytes()


def test_altitude_hold_pushes_to_the_new_push_limit_from_the_step_the_envelope_changed(tmp_path):
    """MS-41: holds steer row k in the envelope that governs it."""
    flight = manoeuvre_flight(tmp_path)
    flight.toggle_hold("altitude")
    flight.holds.targets["altitude"] -= 1000.0

    flight.advance(MANOEUVRE_CRUISE_CONTROL)
    flight.set_envelope(lowered(max_negative_load_factor=-1.0))
    flight.advance(MANOEUVRE_CRUISE_CONTROL)

    assert [row[ROW_LOAD_FACTOR] for row in flight.controls] == [BENCHMARK.max_negative_load_factor, -1.0]


def test_break_turn_in_progress_pulls_the_lowered_max_g_from_the_step_it_was_lowered(tmp_path):
    """MS-41: a running manoeuvre generates row k from the envelope that governs it."""
    flight = manoeuvre_flight(tmp_path)
    flight.start_manoeuvre("break turn")
    while not flight.controls or flight.controls[-1][ROW_LOAD_FACTOR] != BENCHMARK.max_load_factor:
        flight.advance(MANOEUVRE_CRUISE_CONTROL)

    flight.set_envelope(lowered())
    flight.advance(MANOEUVRE_CRUISE_CONTROL)

    assert flight.manoeuvre == "break turn"
    assert flight.controls[-1][ROW_LOAD_FACTOR] == LOWERED_MAX_LOAD_FACTOR


def test_break_turn_started_after_max_g_is_lowered_pulls_the_lowered_max_g(tmp_path):
    """MS-41: a manoeuvre started after a change is planned in the new envelope."""
    flight = manoeuvre_flight(tmp_path)
    flight.set_envelope(lowered())
    flight.start_manoeuvre("break turn")
    while flight.manoeuvre or not flight.controls:
        flight.advance(MANOEUVRE_CRUISE_CONTROL)

    assert max(row[ROW_LOAD_FACTOR] for row in flight.controls) == LOWERED_MAX_LOAD_FACTOR
