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
SCENARIO_FILES = sorted((REPO / "data" / "scenarios").glob("*.json"))
INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
BURN_IN = 20
INITIAL_P_DIAGONAL = 100.0
SUMMATION_ORDER_REL = 1e-9
DIVERGED_POSITION_ERROR_M = 200.0
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


def flights():
    return [fly_scenario(json.loads(path.read_text())) for path in SCENARIO_FILES]


def first_past_limit(errors):
    over = np.flatnonzero(np.linalg.norm(errors[:, :3], axis=1) > DIVERGED_POSITION_ERROR_M)
    return int(over[0]) if over.size else None


def counted(errors):
    """A run counts past its burn-in and up to, not including, the step its position error first passes 200 m."""
    step = first_past_limit(errors)
    return errors[: len(errors) if step is None else step][BURN_IN:]


def open_loop_cv_run(states):
    F, _ = cv_model(NOMINAL.dt)
    estimate, errors = states[0] + INIT_OFFSET, []
    for truth in states:
        estimate = F @ estimate
        errors.append(estimate - truth)
    return np.array(errors)


def open_loop_cv_flight_errors(states):
    return counted(open_loop_cv_run(states))


def assert_open_loop_cv_divergence(row):
    """Dead reckoning leaves the break turn past 200 m and stays within it on the unaccelerated overhead pass."""
    break_turn, overhead = (first_past_limit(open_loop_cv_run(states)) for states, _ in flights())
    assert overhead is None
    assert (float(row["diverged_fraction"]), float(row["median_divergence_step"])) == (0.5, float(break_turn))


def open_loop_cv_errors():
    return np.concatenate([open_loop_cv_flight_errors(states) for states, _ in flights()])


def fixed_gain_flight_errors(gain_at, states, measurements):
    F, _ = cv_model(NOMINAL.dt)
    estimate, errors = states[0] + INIT_OFFSET, []
    for truth, z in zip(states, measurements):
        prior = F @ estimate
        rho = np.hypot(prior[0], prior[1])
        predicted = np.array([np.arctan2(prior[1], prior[0]), np.arctan2(prior[2], rho), np.hypot(rho, prior[2])])
        innovation = z - predicted
        innovation[:2] = (innovation[:2] + np.pi) % (2 * np.pi) - np.pi
        estimate = prior + gain_at(prior) @ innovation
        errors.append(estimate - truth)
    return counted(np.array(errors))


def fixed_gain_errors(gain_at):
    return np.concatenate([fixed_gain_flight_errors(gain_at, *flight) for flight in flights()])


def los_gain_at(los_gain):
    def gain(prior):
        px, py, pz = prior[:3]
        rho = np.hypot(px, py)
        r = np.hypot(rho, pz)
        u_az = np.array([-py, px, 0.0]) / rho
        u_el = np.array([-pz * px / rho, -pz * py / rho, rho]) / r
        rotation = np.array([u_az, u_el, prior[:3] / r])
        return np.kron(np.eye(3), rotation).T @ los_gain * np.array([rho, r, 1.0])

    return gain


def constant_gain_checkpoint(net, final_layer, path):
    zeroed(net, path)
    final_layer.bias.data = torch.tensor(CONSTANT_GAIN.reshape(-1))
    torch.save(net.state_dict(), path)
    return path


def open_loop_cv_flight_nees(errors):
    F, _ = cv_model(NOMINAL.dt)
    P, covariances = np.eye(9) * INITIAL_P_DIAGONAL, []
    for _ in range(len(errors) + BURN_IN):
        P = F @ P @ F.T
        covariances.append(P)
    return np.array([e @ np.linalg.solve(P, e) for e, P in zip(errors, covariances[BURN_IN:])])


def open_loop_cv_nees():
    return np.concatenate([open_loop_cv_flight_nees(open_loop_cv_flight_errors(states)) for states, _ in flights()])


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
    assert_open_loop_cv_divergence(rows[0])


def test_eval_rkn_reports_the_committed_scenarios_as_a_regime_row_with_its_nees(datasets, tmp_path, monkeypatch):
    """MS-11: zero gain and zero B leave RKN open-loop CV with P propagated as F P F^T from the 100 I start."""
    checkpoint = zeroed(RKNGainCovNetwork(**RKN_MULTIPLIERS), tmp_path / "rkn.pt")
    out, out_csv = tmp_path / "rkn.json", tmp_path / "rkn.csv"
    argv = ["eval_rkn.py", "--data-dir", str(datasets), "--checkpoint", str(checkpoint), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--out-csv", str(out_csv)])
    monkeypatch.chdir(REPO)

    eval_rkn_main()

    rows = scenarios_rows(out_csv)
    assert [row["filter"] for row in rows] == ["RKN"]
    assert_rmse(rows[0], open_loop_cv_errors())
    assert_open_loop_cv_divergence(rows[0])
    assert json.loads(out.read_text())["scenarios"]["anees_mean"] == pytest.approx(
        np.mean(open_loop_cv_nees()), rel=SUMMATION_ORDER_REL
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

    errors = fixed_gain_errors(lambda prior: CONSTANT_GAIN)
    assert rms_norm(errors, slice(0, 3)) != pytest.approx(rms_norm(open_loop_cv_errors(), slice(0, 3)), rel=0.01)
    assert_rmse(scenarios_rows(out)[0], errors)


def test_eval_rkn_scenarios_row_applies_the_checkpoint_gain_as_a_fixed_gain_observer(datasets, tmp_path, monkeypatch):
    """MS-10, MS-37: RKN learns only the gain, so a constant rnn_K output is a fixed gain in the line-of-sight frame."""
    net = RKNGainCovNetwork(**RKN_MULTIPLIERS)
    checkpoint = constant_gain_checkpoint(net, net.rnn_K.fc2[-1], tmp_path / "rkn.pt")
    out, out_csv = tmp_path / "rkn.json", tmp_path / "rkn.csv"
    argv = ["eval_rkn.py", "--data-dir", str(datasets), "--checkpoint", str(checkpoint), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*argv, "--out-csv", str(out_csv)])
    monkeypatch.chdir(REPO)

    eval_rkn_main()

    assert_rmse(scenarios_rows(out_csv)[0], fixed_gain_errors(los_gain_at(CONSTANT_GAIN)))


class NonFiniteFilter:
    steps = 0

    def __init__(self, net, dt):
        NonFiniteFilter.steps = 0

    def init_sequence(self, *priors):
        pass

    def step(self, z):
        NonFiniteFilter.steps += 1
        return torch.tensor([[-np.inf, np.inf, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)


def test_knet_run_whose_first_estimate_is_non_finite_is_stepped_no_further(monkeypatch):
    monkeypatch.setattr(eval_knet, "KalmanNetFilter", NonFiniteFilter)
    states = np.zeros((BURN_IN + 1, 9))

    errors = eval_knet.flight_errors(None, NOMINAL.dt, NOMINAL.R(), states, np.zeros((BURN_IN + 1, 3)))

    assert (errors.shape, NonFiniteFilter.steps) == ((1, 9), 1)
