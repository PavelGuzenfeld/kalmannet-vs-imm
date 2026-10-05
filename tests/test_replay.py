import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import chi2

from data.config import NOMINAL
from data.trajectory_gen import fly_scenario
from filters.ekf import EKF
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.motion_models import cv_model
from scripts import eval as bench_eval
from scripts.replay import main as replay_main

REPO = Path(__file__).resolve().parents[1]
SEED_SCENARIO = REPO / "data" / "scenarios" / "break_turn_9g.json"
IMM_CONSTRUCTOR_DEFAULTS = {"omega_ct_deg": 15.0, "p_stay": 0.97, "ca_sigma_a": 3.0, "ca_tau": 4.0}
FLAGS = ("track_loss", "overconfidence", "raw_error")
KNET_INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
BURN_IN = 20
BREAK_TURN_START_STEP, BREAK_TURN_END_STEP = 50, 50 + 23 + 150
RECORDED_TRUTH_SHA256 = "d0adbff6c23bf7f12f544d3de769e6a2c5595ee4f3f455b3a37ec469a2ad01ab"


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
    argv = ["replay.py", str(recorded), "--record", "--imm-tuning", str(missing)]
    monkeypatch.setattr(sys, "argv", [*argv, "--knet-checkpoint", str(missing), "--rkn-checkpoint", str(missing)])
    monkeypatch.setenv("GIT_COMMIT", "recorded-sha")

    replay_main()

    assert recorded.read_text().endswith("}\n")
    committed = json.loads(SEED_SCENARIO.read_text())
    assert set(committed) == {"controls", "seed", "envelope", "noise_regime", "thresholds", "git_tag", "tripped"}
    assert json.loads(recorded.read_text()) == {**committed, "git_tag": "recorded-sha"}


def test_model_change_silently_shifts_a_recorded_scenario():
    """MS-27: a scenario's fixed-wing flight is the Layer 5 truth; flags pin only trip steps. Re-record, re-pin."""
    states, _ = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    assert hashlib.sha256(states.tobytes()).hexdigest() == RECORDED_TRUTH_SHA256


def test_replay_trips_on_the_scenario_recorded_thresholds_not_the_defaults(tmp_path):
    """MS-11: same flight, same nominal R; only the recorded raw-error limit differs."""
    scenario = json.loads(SEED_SCENARIO.read_text())
    scenario["thresholds"]["raw_error_m"] = 1e9
    loose, missing = tmp_path / "loose.json", tmp_path / "missing"
    loose.write_text(json.dumps(scenario))

    lines = replay(loose, "--imm-tuning", missing, "--knet-checkpoint", missing, "--rkn-checkpoint", missing)

    assert lines["EKF"].endswith(" raw_error=-")


def test_replay_prints_the_recorded_first_trip_step_of_every_flag_per_filter(imm_tuning, tmp_path):
    """MS-11: every filter replays under the nominal R, whatever the scenario's noise regime."""
    scenario = json.loads(SEED_SCENARIO.read_text())
    missing = tmp_path / "missing.pt"

    lines = replay(
        SEED_SCENARIO, "--imm-tuning", imm_tuning, "--knet-checkpoint", missing, "--rkn-checkpoint", missing
    )

    assert set(lines) == {"EKF", "UKF", "IMM", "KalmanNet", "RKN"}
    assert BREAK_TURN_START_STEP < scenario["tripped"]["EKF"]["track_loss"] < BREAK_TURN_END_STEP
    for name, trips in scenario["tripped"].items():
        assert lines[name] == f"{name}: " + " ".join(f"{flag}={shown(trips[flag])}" for flag in FLAGS)
    assert all(f"{flag}=" in lines["IMM"] for flag in FLAGS)
    assert lines["KalmanNet"] == f"KalmanNet: skipped, no checkpoint at {missing}"
    assert lines["RKN"] == f"RKN: skipped, no checkpoint at {missing}"


def test_replay_skips_imm_without_a_tuning_file_instead_of_inventing_parameters(tmp_path):
    """MS-8: the IMM runs only with its benchmarked tuning."""
    missing = tmp_path / "missing.json"

    lines = replay(SEED_SCENARIO, "--imm-tuning", missing, "--knet-checkpoint", missing, "--rkn-checkpoint", missing)

    assert lines["IMM"] == f"IMM: skipped, no tuning at {missing}"


def test_zero_gain_kalmannet_trips_raw_error_where_cv_dead_reckoning_leaves_20_m(imm_tuning, tmp_path):
    """MS-10: KalmanNet keeps the CV f and learns only the gain, so zero gain is open-loop CV."""
    knet, rkn = tmp_path / "knet.pt", tmp_path / "rkn.pt"
    zero_gain_knet(knet)
    torch.save(RKNGainCovNetwork(fc1_mult=3, hidden_mult=3, fc2_mult=4).double().state_dict(), rkn)
    states, _ = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    F, _ = cv_model(NOMINAL.dt)
    estimate, error_m = states[0] + KNET_INIT_OFFSET, []
    for truth in states:
        estimate = F @ estimate
        error_m.append(np.linalg.norm(estimate[:3] - truth[:3]))
    expected_step = int(np.flatnonzero(np.array(error_m) > 20.0)[0])

    lines = replay(SEED_SCENARIO, "--imm-tuning", imm_tuning, "--knet-checkpoint", knet, "--rkn-checkpoint", rkn)

    assert lines["KalmanNet"] == f"KalmanNet: track_loss=n/a overconfidence=n/a raw_error={expected_step}"
    assert lines["RKN"].startswith("RKN: track_loss=n/a overconfidence=")


def test_eval_reports_the_committed_scenarios_as_their_own_regime(imm_tuning, tmp_path, monkeypatch):
    """MS-11: the scenarios row runs the classical filters under the nominal R, each flight past its burn-in."""
    data_dir, out = tmp_path / "datasets", tmp_path / "results"
    toy_sizes = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1"]
    toy_mc = ["--mc-test", "2", "--n-highmc", "1", "--mc-highmc", "2"]
    subprocess.run(
        [sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy_sizes, *toy_mc], cwd=REPO, check=True
    )
    states, z = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    err, nees, nis = bench_eval.run_ekf_like(EKF(NOMINAL.dt, NOMINAL.R()), states, z)
    ekf_pos_rmse = np.sqrt(np.mean(np.sum(err[BURN_IN:, :3] ** 2, axis=-1)))
    single_flight_nis_low, single_flight_nis_high = chi2.ppf(0.025, df=3), chi2.ppf(0.975, df=3)
    ekf_nis_inside = np.mean((nis[BURN_IN:] >= single_flight_nis_low) & (nis[BURN_IN:] <= single_flight_nis_high))
    eval_argv = ["eval.py", "--data-dir", str(data_dir), "--imm-tuning", str(imm_tuning), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", eval_argv)
    monkeypatch.chdir(REPO)

    bench_eval.main()

    with open(out / "results_table.csv") as f:
        rows = {row["filter"]: row for row in csv.DictReader(f) if row["regime"] == "scenarios"}
    assert list(rows) == ["EKF", "UKF", "IMM"]
    assert float(rows["EKF"]["pos_rmse"]) == ekf_pos_rmse
    assert float(rows["EKF"]["anees_mean"]) == np.mean(nees[BURN_IN:])
    assert float(rows["EKF"]["anis_fraction_inside_95"]) == ekf_nis_inside
