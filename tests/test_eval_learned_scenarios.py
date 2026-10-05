import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from data.config import NOMINAL
from data.trajectory_gen import fly_scenario
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.motion_models import cv_model
from scripts import eval_knet
from scripts.eval_knet import main as eval_knet_main
from scripts.eval_rkn import main as eval_rkn_main

REPO = Path(__file__).resolve().parents[1]
SEED_SCENARIO = REPO / "data" / "scenarios" / "break_turn_9g.json"
INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
BURN_IN = 20
INITIAL_P_DIAGONAL = 100.0
SUMMATION_ORDER_REL = 1e-9
CONSTANT_GAIN = np.zeros((9, 3))
CONSTANT_GAIN[0, 2] = -0.3
CONSTANT_GAIN[3, 2] = -0.05
RKN_MULTIPLIERS = {"fc1_mult": 3, "hidden_mult": 3, "fc2_mult": 4}


@pytest.fixture
def datasets(tmp_path):
    out = tmp_path / "datasets"
    toy = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1"]
    toy_mc = ["--mc-test", "2", "--n-highmc", "1", "--mc-highmc", "2"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", out, *toy, *toy_mc], cwd=REPO, check=True)
    return out


def zeroed(net, path):
    for parameter in net.double().parameters():
        torch.nn.init.zeros_(parameter)
    torch.save(net.state_dict(), path)
    return path


def open_loop_cv_errors():
    states, _ = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    F, _ = cv_model(NOMINAL.dt)
    estimate, errors = states[0] + INIT_OFFSET, []
    for truth in states:
        estimate = F @ estimate
        errors.append(estimate - truth)
    return np.array(errors)[BURN_IN:]


def fixed_gain_errors(gain):
    states, measurements = fly_scenario(json.loads(SEED_SCENARIO.read_text()))
    F, _ = cv_model(NOMINAL.dt)
    estimate, errors = states[0] + INIT_OFFSET, []
    for truth, z in zip(states, measurements):
        prior = F @ estimate
        rho = np.hypot(prior[0], prior[1])
        predicted = np.array([np.arctan2(prior[1], prior[0]), np.arctan2(prior[2], rho), np.hypot(rho, prior[2])])
        innovation = z - predicted
        innovation[:2] = (innovation[:2] + np.pi) % (2 * np.pi) - np.pi
        estimate = prior + gain @ innovation
        errors.append(estimate - truth)
    return np.array(errors)[BURN_IN:]


def constant_gain_checkpoint(net, final_layer, path):
    zeroed(net, path)
    final_layer.bias.data = torch.tensor(CONSTANT_GAIN.reshape(-1))
    torch.save(net.state_dict(), path)
    return path


def open_loop_cv_nees(errors):
    F, _ = cv_model(NOMINAL.dt)
    P, covariances = np.eye(9) * INITIAL_P_DIAGONAL, []
    for _ in range(len(errors) + BURN_IN):
        P = F @ P @ F.T
        covariances.append(P)
    return np.array([e @ np.linalg.solve(P, e) for e, P in zip(errors, covariances[BURN_IN:])])


def scenarios_rows(csv_path):
    with open(csv_path) as f:
        return [row for row in csv.DictReader(f) if row["regime"] == "scenarios"]


def rms_norm(errors, columns):
    return np.sqrt(np.mean(np.sum(errors[:, columns] ** 2, axis=-1)))


def assert_rmse(row, errors):
    assert float(row["pos_rmse"]) == pytest.approx(rms_norm(errors, slice(0, 3)), rel=SUMMATION_ORDER_REL)
    assert float(row["vel_rmse"]) == pytest.approx(rms_norm(errors, slice(3, 6)), rel=SUMMATION_ORDER_REL)


def test_eval_knet_reports_the_committed_scenarios_as_a_regime_row(datasets, tmp_path, monkeypatch):
    """MS-11: zero gain is open-loop CV, so the row is the CV dead-reckoning error past each flight's burn-in."""
    checkpoint, out = zeroed(KalmanGainGRU(), tmp_path / "knet.pt"), tmp_path / "knet.csv"
    argv = ["eval_knet.py", "--data-dir", str(datasets), "--checkpoint", str(checkpoint), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(REPO)

    eval_knet_main()

    rows = scenarios_rows(out)
    assert [row["filter"] for row in rows] == ["KalmanNet-vanilla"]
    assert_rmse(rows[0], open_loop_cv_errors())


def test_eval_rkn_reports_the_committed_scenarios_as_a_regime_row_with_its_nees(datasets, tmp_path, monkeypatch):
    """MS-11: zero gain and zero B leave RKN open-loop CV with P propagated as F P F^T from the 100 I start."""
    checkpoint = zeroed(RKNGainCovNetwork(**RKN_MULTIPLIERS), tmp_path / "rkn.pt")
    out, out_csv = tmp_path / "rkn.json", tmp_path / "rkn.csv"
    argv = ["eval_rkn.py", "--data-dir", str(datasets), "--checkpoint", str(checkpoint), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--out-csv", str(out_csv)])
    monkeypatch.chdir(REPO)

    eval_rkn_main()

    rows = scenarios_rows(out_csv)
    errors = open_loop_cv_errors()
    assert [row["filter"] for row in rows] == ["RKN"]
    assert_rmse(rows[0], errors)
    assert json.loads(out.read_text())["scenarios"]["anees_mean"] == pytest.approx(
        np.mean(open_loop_cv_nees(errors)), rel=SUMMATION_ORDER_REL
    )


def test_eval_knet_scenarios_row_applies_the_checkpoint_gain_as_a_fixed_gain_observer(datasets, tmp_path, monkeypatch):
    """MS-10: only the gain is learned, so a constant-output net is CV prediction plus a fixed K times the innovation."""
    net = KalmanGainGRU()
    checkpoint = constant_gain_checkpoint(net, net.FC2[-1], tmp_path / "knet.pt")
    out = tmp_path / "knet.csv"
    argv = ["eval_knet.py", "--data-dir", str(datasets), "--checkpoint", str(checkpoint), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(REPO)

    eval_knet_main()

    errors = fixed_gain_errors(CONSTANT_GAIN)
    assert rms_norm(errors, slice(0, 3)) != pytest.approx(rms_norm(open_loop_cv_errors(), slice(0, 3)), rel=0.01)
    assert_rmse(scenarios_rows(out)[0], errors)


def test_eval_rkn_scenarios_row_applies_the_checkpoint_gain_as_a_fixed_gain_observer(datasets, tmp_path, monkeypatch):
    """MS-10: RKN learns only the gain, so a constant rnn_K output is the same fixed-gain observer as KalmanNet's."""
    net = RKNGainCovNetwork(**RKN_MULTIPLIERS)
    checkpoint = constant_gain_checkpoint(net, net.rnn_K.fc2[-1], tmp_path / "rkn.pt")
    out, out_csv = tmp_path / "rkn.json", tmp_path / "rkn.csv"
    argv = ["eval_rkn.py", "--data-dir", str(datasets), "--checkpoint", str(checkpoint), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--out-csv", str(out_csv)])
    monkeypatch.chdir(REPO)

    eval_rkn_main()

    assert_rmse(scenarios_rows(out_csv)[0], fixed_gain_errors(CONSTANT_GAIN))


class NonFiniteFilter:
    def __init__(self, net, dt):
        pass

    def init_sequence(self, *priors):
        pass

    def step(self, z):
        return torch.tensor([[-np.inf, np.inf, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)


def test_diverged_knet_estimate_is_clipped_to_1e5_per_component_so_the_rmse_stays_finite(monkeypatch):
    """-inf, +inf and NaN position errors become -1e5, 1e5 and 1e5."""
    monkeypatch.setattr(eval_knet, "KalmanNetFilter", NonFiniteFilter)
    states = np.zeros((BURN_IN + 1, 9))

    errors = eval_knet.flight_errors(None, NOMINAL.dt, NOMINAL.R(), states, np.zeros((BURN_IN + 1, 3)))

    assert errors[0, :3].tolist() == [-1e5, 1e5, 1e5]
