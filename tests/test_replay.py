import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import chi2

from data.config import NOMINAL, REGIMES
from data.fixed_wing import Envelope
from data.trajectory_gen import fly_scenario, scenario_start
from filters.ekf import EKF
from filters.flags import DEFAULT_THRESHOLDS
from filters.imm import IMM
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.measurement_model import h
from filters.motion_models import cv_model
from filters.plot_data import imm_kwargs_from_tuning
from filters.state import STATE_DIM
from filters.ukf import UKF
from scripts import eval as bench_eval
from scripts.replay import main as replay_main
from tests import fixed_wing_before_controls

REPO = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO / "data" / "scenarios"
SEED_SCENARIO = SCENARIO_DIR / "break_turn_9g.json"
OVERHEAD_SCENARIO = SCENARIO_DIR / "overhead_pass.json"
PAGE_IMM_TUNING = REPO / "web" / "imm_tuning.json"
OVERHEAD_CROSSING_ERROR_BOUND_M = 3.0
OVERHEAD_ERROR_WINDOW_STEPS = 25
OVERCONFIDENT_STEPS = 25
IMM_CONSTRUCTOR_DEFAULTS = {
    "omega_ct_deg": 15.0, "p_stay": 0.97, "ca_sigma_a": 3.0, "ca_tau": 4.0, "cv_sigma_a": 0.05, "ct_sigma_a": 0.3,
}
FLAGS = ("track_loss", "overconfidence", "raw_error")
KNET_INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
BURN_IN = 20
BREAK_TURN_START_STEP, BREAK_TURN_END_STEP = 50, 50 + 23 + 150
RECORDED_TRUTH_POS_VEL = {
    0: [-1919.9007538401656, 153.80840024481452, 420.2908837135784,
        -137.6496699659416, 146.7731455011416, -7.653813872034979],
    74: [-2123.4321561971665, 371.63728900450354, 408.7458885693238,
         -135.01626655351177, 150.6628470648977, -9.16082734020019],
    148: [-2239.9650071107158, 639.6929546211076, 397.60718338081045,
          -17.316570397382403, 198.78973536723495, -5.376613470370975],
    222: [-2172.294855496557, 919.4780366442897, 393.88355298776327,
          105.09710146925633, 165.81085028027567, 0.5082357318531592],
    295: [-2013.5083892179464, 1159.473662862239, 393.09277288587424,
          108.89319763088594, 164.39841185829255, -0.6585178415275652],
}
CROSS_PLATFORM_ATOL = 1e-6
UNINFORMATIVE_R = np.diag([1e6, 1e6, 1e12])
INIT_OFFSET_EAST_SOUTH_M = [2.0, -1.0, 0.0]
START_ATOL_M = 1e-6
BENCHMARK_CONTROL_LIMITS_ISSUE_106 = {
    "max_side_load_factor": 2.0,
    "max_negative_load_factor": -4.5,
    "brake_drag_multiplier": 3.0,
    "afterburner_thrust_multiplier": 1.5,
}
G = 9.80665


def recorded_before_wider_row(committed):
    """A committed scenario as recorded before #107: 3-column rows, no #106 envelope fields."""
    return {
        **committed,
        "envelope": {k: v for k, v in committed["envelope"].items() if k not in BENCHMARK_CONTROL_LIMITS_ISSUE_106},
        "controls": [row[:3] for row in committed["controls"]],
    }


@pytest.fixture
def imm_tuning(tmp_path):
    path = tmp_path / "imm_tuning.json"
    path.write_text(json.dumps({"best": IMM_CONSTRUCTOR_DEFAULTS}))
    return path


def replay(*args):
    result = subprocess.run(
        [sys.executable, "scripts/replay.py", *map(str, args)], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return {line.split(":")[0]: line for line in result.stdout.splitlines() if ":" in line}


def shown(step):
    return "-" if step is None else str(step)


def zero_gain_knet(path):
    net = KalmanGainGRU().double()
    for parameter in net.parameters():
        torch.nn.init.zeros_(parameter)
    torch.save(net.state_dict(), path)


def test_record_rewrites_the_committed_seed_scenario_and_the_flags_it_tripped(tmp_path, monkeypatch):
    """MS-19 start drawn from the recorded seed; controls, envelope, thresholds and flags recorded by value."""
    recorded, missing = tmp_path / "scenarios" / "seed.json", tmp_path / "missing"
    argv = ["replay.py", str(recorded), "--record", "--imm-tuning", str(PAGE_IMM_TUNING)]
    monkeypatch.setattr(sys, "argv", [*argv, "--knet-checkpoint", str(missing), "--rkn-checkpoint", str(missing), "--bkn-checkpoint", str(missing)])
    monkeypatch.setenv("GIT_COMMIT", "recorded-sha")

    replay_main()

    assert recorded.read_text().endswith("}\n")
    committed = json.loads(SEED_SCENARIO.read_text())
    assert set(committed) == {"controls", "seed", "envelope", "noise_regime", "thresholds", "git_tag", "tripped"}
    assert json.loads(recorded.read_text()) == {**committed, "git_tag": "recorded-sha"}


def test_record_overhead_rewrites_the_committed_overhead_scenario(tmp_path, monkeypatch):
    """MS-35: --record --overhead regenerates the committed pass; only the git tag is the recording run's."""
    recorded, missing = tmp_path / "overhead.json", tmp_path / "missing"
    argv = ["replay.py", str(recorded), "--record", "--overhead", "--imm-tuning", str(PAGE_IMM_TUNING)]
    monkeypatch.setattr(sys, "argv", [*argv, "--knet-checkpoint", str(missing), "--rkn-checkpoint", str(missing), "--bkn-checkpoint", str(missing)])
    monkeypatch.setenv("GIT_COMMIT", "recorded-sha")

    replay_main()

    committed = json.loads(OVERHEAD_SCENARIO.read_text())
    assert json.loads(recorded.read_text()) == {**committed, "git_tag": "recorded-sha"}


@pytest.mark.parametrize("committed", [SEED_SCENARIO, OVERHEAD_SCENARIO], ids=lambda path: path.stem)
def test_3_column_form_of_committed_scenario_flies_bit_for_bit_the_truth_it_flew_before_the_wider_row(committed):
    """MS-27, #107: old rows replay with every new input neutral and, lacking the field, today's push limit."""
    scenario = recorded_before_wider_row(json.loads(committed.read_text()))
    regime = REGIMES[scenario["noise_regime"]]
    start = scenario_start(scenario, np.random.default_rng(scenario["seed"]), regime)
    controls = np.asarray(scenario["controls"], dtype=np.float64)
    envelope = Envelope(**scenario["envelope"])

    before = fixed_wing_before_controls.fly(envelope, start["position"], start["velocity"], 0.0, controls, regime.dt)
    states, _ = fly_scenario(scenario)

    assert states.tobytes() == before.tobytes()


def test_scenario_rudder_row_side_slips_the_overhead_pass_toward_the_right_wing_at_max_side_g():
    """MS-27: replay reads the 6-column row. The pass flies east wings level, so the right wing is south.
    rtol 1e-12: the acceleration column is a velocity difference over dt."""
    scenario = json.loads(OVERHEAD_SCENARIO.read_text())
    scenario["envelope"] = {**scenario["envelope"], **BENCHMARK_CONTROL_LIMITS_ISSUE_106}
    scenario["controls"] = [[0.0, 1.0, 1.0, 2.5, 0.0, 0.0]]

    states, _ = fly_scenario(scenario)

    np.testing.assert_allclose(states[0, 7], -2.0 * G, rtol=1e-12)


def test_model_change_silently_shifts_a_recorded_scenario():
    """MS-27: the fixed-wing flight is the Layer 5 truth; a model change moves it by metres. atol 1e-6 m
    (m/s) absorbs CPU-dependent libm/BLAS rounding over 296 steps, which a byte hash did not. Re-record, re-pin."""
    states, _ = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    for step, pos_vel in RECORDED_TRUTH_POS_VEL.items():
        np.testing.assert_allclose(states[step, :6], pos_vel, rtol=0.0, atol=CROSS_PLATFORM_ATOL)


def test_replay_trips_on_the_scenario_recorded_thresholds_not_the_defaults(tmp_path):
    """MS-11: same flight, same nominal R; only the recorded raw-error limit differs."""
    scenario = json.loads(SEED_SCENARIO.read_text())
    scenario["thresholds"]["raw_error_m"] = 1e9
    loose, missing = tmp_path / "loose.json", tmp_path / "missing"
    loose.write_text(json.dumps(scenario))

    lines = replay(loose, "--imm-tuning", missing, "--knet-checkpoint", missing, "--rkn-checkpoint", missing)

    assert lines["EKF"].endswith(" raw_error=-")


def test_replay_prints_the_recorded_first_trip_step_of_every_flag_per_filter(tmp_path):
    """MS-11: every filter replays under the nominal R, whatever the scenario's noise regime."""
    scenario = json.loads(SEED_SCENARIO.read_text())
    missing = tmp_path / "missing.pt"

    lines = replay(
        SEED_SCENARIO,
        "--imm-tuning",
        PAGE_IMM_TUNING,
        "--knet-checkpoint",
        missing,
        "--rkn-checkpoint",
        missing,
        "--bkn-checkpoint",
        missing,
    )

    assert set(lines) == {"EKF", "UKF", "IMM", "KalmanNet", "RKN", "BKN"}
    assert set(scenario["tripped"]) == {"EKF", "UKF", "IMM"}
    assert BREAK_TURN_START_STEP < scenario["tripped"]["EKF"]["track_loss"] < BREAK_TURN_END_STEP
    for name, trips in scenario["tripped"].items():
        assert lines[name] == f"{name}: " + " ".join(f"{flag}={shown(trips[flag])}" for flag in FLAGS)
    assert lines["KalmanNet"] == f"KalmanNet: skipped, no checkpoint at {missing}"
    assert lines["RKN"] == f"RKN: skipped, no checkpoint at {missing}"
    assert lines["BKN"] == f"BKN: skipped, no checkpoint at {missing}"


def test_replay_skips_imm_without_a_tuning_file_instead_of_inventing_parameters(tmp_path):
    """MS-8: the IMM runs only with its benchmarked tuning."""
    missing = tmp_path / "missing.json"

    lines = replay(SEED_SCENARIO, "--imm-tuning", missing, "--knet-checkpoint", missing, "--rkn-checkpoint", missing)

    assert lines["IMM"] == f"IMM: skipped, no tuning at {missing}"


def test_zero_gain_kalmannet_trips_raw_error_where_cv_dead_reckoning_leaves_20_m(imm_tuning, tmp_path):
    """MS-10: KalmanNet keeps the CV f and learns only the gain, so zero gain is open-loop CV."""
    knet, missing = tmp_path / "knet.pt", tmp_path / "missing"
    zero_gain_knet(knet)
    states, _ = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    F, _ = cv_model(NOMINAL.dt)
    estimate, error_m = states[0] + KNET_INIT_OFFSET, []
    for truth in states:
        estimate = F @ estimate
        error_m.append(np.linalg.norm(estimate[:3] - truth[:3]))
    expected_step = int(np.flatnonzero(np.array(error_m) > 20.0)[0])
    diverged_step = int(np.flatnonzero(np.array(error_m) > 200.0)[0])

    lines = replay(SEED_SCENARIO, "--imm-tuning", imm_tuning, "--knet-checkpoint", knet, "--rkn-checkpoint", missing)

    assert lines["KalmanNet"] == (
        f"KalmanNet: track_loss=n/a overconfidence=n/a raw_error={expected_step} diverged={diverged_step}"
    )


def test_rkn_track_loss_reads_n_a_with_the_reason_it_has_no_innovation_gate(tmp_path):
    """MS-10: RKN learns its gain and covariance, never an innovation covariance, so it gets no gate (#2 decision 51)."""
    rkn, missing = tmp_path / "rkn.pt", tmp_path / "missing"
    torch.save(RKNGainCovNetwork(fc1_mult=3, hidden_mult=3, fc2_mult=4).double().state_dict(), rkn)

    lines = replay(SEED_SCENARIO, "--imm-tuning", missing, "--knet-checkpoint", missing, "--rkn-checkpoint", rkn)

    rkn_line = re.fullmatch(r"RKN: track_loss=(.*) overconfidence=(-|\d+) raw_error=(-|\d+)( diverged=\d+)?", lines["RKN"])
    assert rkn_line is not None, lines["RKN"]
    assert rkn_line[1] == "n/a (RKN defines no innovation covariance)"


def test_eval_reports_the_committed_scenarios_as_their_own_regime(imm_tuning, tmp_path, monkeypatch):
    """MS-11: the scenarios row runs the classical filters under the nominal R, each flight past its burn-in."""
    data_dir, out = tmp_path / "datasets", tmp_path / "results"
    toy_sizes = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1"]
    toy_mc = ["--mc-test", "2", "--n-highmc", "1", "--mc-highmc", "2"]
    subprocess.run(
        [sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy_sizes, *toy_mc], cwd=REPO, check=True
    )
    runs = [
        bench_eval.run_ekf_like(EKF(NOMINAL.dt, NOMINAL.R()), *fly_scenario(json.loads(path.read_text())))
        for path in sorted(SCENARIO_DIR.glob("*.json"))
    ]
    err, nees, nis = (np.concatenate([run[i][BURN_IN:] for run in runs]) for i in range(3))
    ekf_pos_rmse = np.sqrt(np.mean(np.sum(err[:, :3] ** 2, axis=-1)))
    single_flight_nis_low, single_flight_nis_high = chi2.ppf(0.025, df=3), chi2.ppf(0.975, df=3)
    ekf_nis_inside = np.mean((nis >= single_flight_nis_low) & (nis <= single_flight_nis_high))
    eval_argv = ["eval.py", "--data-dir", str(data_dir), "--imm-tuning", str(imm_tuning), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*eval_argv, "--bkn-weights", str(tmp_path / "missing.npz")])
    monkeypatch.chdir(REPO)

    bench_eval.main()

    with open(out / "results_table.csv") as f:
        rows = {row["filter"]: row for row in csv.DictReader(f) if row["regime"] == "scenarios"}
    assert OVERHEAD_SCENARIO in SCENARIO_DIR.glob("*.json")
    assert list(rows) == ["EKF", "UKF", "IMM"]
    assert float(rows["EKF"]["pos_rmse"]) == ekf_pos_rmse
    assert float(rows["EKF"]["anees_mean"]) == np.mean(nees)
    assert float(rows["EKF"]["anis_fraction_inside_95"]) == ekf_nis_inside
    assert (rows["EKF"]["diverged_fraction"], rows["EKF"]["median_divergence_step"]) == ("0.0", "")


def overhead_crossing_step(scenario):
    start_x, speed = scenario["start"]["position"][0], scenario["start"]["velocity"][0]
    return round(-start_x / (speed * NOMINAL.dt))


def test_overhead_truth_crosses_the_sensor_exactly_at_the_geometric_crossing_step():
    """MS-35: straight pass at y = 0, so the truth is exactly overhead (rho = 0) at the step the geometry gives."""
    scenario = json.loads(OVERHEAD_SCENARIO.read_text())
    states, _ = fly_scenario(scenario)
    assert np.hypot(states[:, 0], states[:, 1]).argmin() == overhead_crossing_step(scenario)
    assert np.hypot(states[overhead_crossing_step(scenario), 0], states[overhead_crossing_step(scenario), 1]) == 0.0


def test_ekf_and_imm_trip_overconfidence_when_nees_leaves_its_band_at_the_overhead_crossing(imm_tuning, tmp_path):
    """MS-35: P collapses at rho = 0, so NEES is outside from the crossing step and the flag trips 25 steps in."""
    scenario = json.loads(OVERHEAD_SCENARIO.read_text())
    crossing, missing = overhead_crossing_step(scenario), tmp_path / "missing"
    expected = crossing + OVERCONFIDENT_STEPS - 1

    lines = replay(
        OVERHEAD_SCENARIO, "--imm-tuning", imm_tuning, "--knet-checkpoint", missing, "--rkn-checkpoint", missing
    )

    for name in ("EKF", "IMM"):
        assert f"overconfidence={expected} " in lines[name], lines[name]


def test_ekf_and_imm_position_error_in_the_25_steps_from_the_overhead_crossing_exceeds_3_m():
    """MS-35: the recorded approach error peaks at 1.25 m (EKF) and 1.16 m; 3 m is >2x that and below the IMM's 3.74 m."""
    scenario = json.loads(OVERHEAD_SCENARIO.read_text())
    crossing = overhead_crossing_step(scenario)
    states, z = fly_scenario(scenario)
    imm = IMM(NOMINAL.dt, NOMINAL.R(), **imm_kwargs_from_tuning(IMM_CONSTRUCTOR_DEFAULTS))
    runs = (
        bench_eval.run_ekf_like(EKF(NOMINAL.dt, NOMINAL.R()), states, z),
        bench_eval.run_imm(imm, states, z),
    )
    for err, _, _ in runs:
        position_error_m = np.linalg.norm(err[:, :3], axis=1)
        assert position_error_m[crossing : crossing + OVERHEAD_ERROR_WINDOW_STEPS].max() > OVERHEAD_CROSSING_ERROR_BOUND_M


@pytest.mark.parametrize(
    "run, make_filter",
    [
        (bench_eval.run_ekf_like, lambda R: EKF(NOMINAL.dt, R)),
        (bench_eval.run_ekf_like, lambda R: UKF(NOMINAL.dt, R)),
        (bench_eval.run_imm, lambda R: IMM(NOMINAL.dt, R)),
    ],
)
def test_classical_filters_start_2_m_east_and_1_m_south_of_the_truth(run, make_filter):
    """MS-8, MS-9: the same start offset as the learned filters. R this loose moves the first update by under
    1e-9 m (gain ~ 50 m^2 * 1e-3 rad/m / 1e6 rad^2), so atol 1e-6 m reads the start itself."""
    truth = np.zeros((1, STATE_DIM))
    truth[0, :3] = [1000.0, 0.0, 100.0]

    err, _, _ = run(make_filter(UNINFORMATIVE_R), truth, h(truth[0])[None])

    np.testing.assert_allclose(err[0, :3], INIT_OFFSET_EAST_SOUTH_M, rtol=0.0, atol=START_ATOL_M)


DIVERGES_AT_STEP = 3
SCRIPTED_STEPS = 40


class ScriptedFilter:
    """Estimates 1 m of position error, then 300 m from DIVERGES_AT_STEP on; its NIS would jump only after it."""

    def __init__(self):
        self.steps = 0

    def update(self, x, P, z):
        return self.step(x, P, z)

    def step(self, x, P, z):
        estimate, innovation = np.zeros(STATE_DIM), np.zeros(3)
        estimate[0] = 1.0 if self.steps < DIVERGES_AT_STEP else 300.0
        innovation[0] = 0.0 if self.steps <= DIVERGES_AT_STEP else 1e3
        self.steps += 1
        return estimate, np.eye(STATE_DIM), innovation, np.eye(3)

    def init_state(self, x, P):
        return {"mu": np.full(4, 0.25)}


class ScriptedIMM(ScriptedFilter):
    def __init__(self, dt, R, **tuned):
        super().__init__()

    def step(self, state, z):
        estimate, P, innovation, S = super().step(None, None, z)
        return estimate, P, state, innovation, S


def scripted_replay(monkeypatch, imm_tuning):
    from scripts import replay as replay_script

    made = []

    def scripted(*args, **kwargs):
        made.append(ScriptedFilter())
        return made[-1]

    def scripted_imm(dt, R, **tuned):
        made.append(ScriptedIMM(dt, R, **tuned))
        return made[-1]

    flight = (np.zeros((SCRIPTED_STEPS, STATE_DIM)), np.zeros((SCRIPTED_STEPS, 3)))
    monkeypatch.setattr(replay_script, "fly_scenario", lambda scenario: flight)
    monkeypatch.setattr(replay_script, "EKF", scripted)
    monkeypatch.setattr(replay_script, "UKF", scripted)
    monkeypatch.setattr(replay_script, "IMM", scripted_imm)
    scenario = {"thresholds": {**DEFAULT_THRESHOLDS, "raw_error_m": 1e9}}
    missing = imm_tuning.parent / "missing"
    return replay_script.replay(scenario, imm_tuning, missing, missing), made


def test_replay_pulls_each_diverged_filter_through_its_divergence_step_and_no_further(monkeypatch, imm_tuning):
    _, made = scripted_replay(monkeypatch, imm_tuning)

    assert [flt.steps for flt in made] == [DIVERGES_AT_STEP + 1] * 3


def test_replay_does_not_trip_a_flag_that_would_only_trip_after_the_divergence_step(monkeypatch, imm_tuning):
    """Past the divergence step the innovation reads 1e3 (track_loss) and NEES 9e4 for 25 steps (overconfidence)."""
    tripped, _ = scripted_replay(monkeypatch, imm_tuning)

    cut_flags = {"track_loss": None, "overconfidence": None, "raw_error": None, "diverged": DIVERGES_AT_STEP}
    assert [tripped[name] for name in ("EKF", "UKF", "IMM")] == [cut_flags] * 3
