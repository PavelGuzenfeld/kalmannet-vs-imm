import argparse
import subprocess
import sys
import json

import numpy as np
import pytest
import torch

from data.config import NOMINAL
from data.trajectory_gen import fly_scenario
from filters.state import STATE_DIM
from filters.torch_ops import export_npz
from scripts.train_ckn import RELEASED_BETA, ckn_loss, run_training_epoch
from scripts.tune_bkn import RMSE_SLACK
from scripts.tune_ckn import selected
from tests.test_cholesky_kalmannet import BREAK_TURN, stand_in_network
from tests.test_numpy_forward import REPO

LOSS_RTOL = 4 * np.finfo(np.float64).eps
ONE_SIGMA_ERROR = torch.tensor([[30.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)


def zero_covariance():
    return torch.zeros(1, STATE_DIM, STATE_DIM, dtype=torch.float64)


def test_released_beta_is_the_reference_repos_0_05():
    """MS-53: trainer.py calls empirical_averaging_all with beta=0.05."""
    assert RELEASED_BETA == 0.05


@pytest.mark.parametrize("beta, expected", [(0.0, 2.0 / 9.0), (1.0, 4.0), (0.25, 0.75 * 2.0 / 9.0 + 0.25 * 4.0)])
def test_loss_mixes_state_mean_squared_error_and_the_l1_over_every_covariance_entry(beta, expected):
    """MS-53: e is one σ_init on px and on vx, P = 0. MSE averages w e² over the 9 states: 2/9. The L1 sums
    sqrt(w_i w_j)|e_i e_j| over all 81 entries: (1 + 1)² = 4, off-diagonals included."""
    loss = ckn_loss(ONE_SIGMA_ERROR, zero_covariance(), beta)
    assert loss.item() == pytest.approx(expected, rel=LOSS_RTOL)


def test_loss_covariance_term_counts_the_off_diagonal_entries_a_diagonal_p_misses():
    """MS-53: P = diag(e²) matches the diagonal, leaving |e_px e_vx| sqrt(w_px w_vx) = 1 twice: 2."""
    P = torch.diag_embed(ONE_SIGMA_ERROR**2)
    assert ckn_loss(ONE_SIGMA_ERROR, P, 1.0).item() == pytest.approx(2.0, rel=LOSS_RTOL)


def test_loss_covariance_term_is_zero_when_p_is_the_outer_product_of_the_error():
    """MS-53: P = e eᵀ leaves only the MSE term."""
    P = ONE_SIGMA_ERROR.unsqueeze(-1) * ONE_SIGMA_ERROR.unsqueeze(-2)
    assert ckn_loss(ONE_SIGMA_ERROR, P, 0.5).item() == pytest.approx(0.5 * 2.0 / 9.0, rel=LOSS_RTOL)


def test_loss_averages_over_the_batch():
    """MS-53: a second trajectory at zero error with P = 0 halves both terms."""
    error = torch.cat([ONE_SIGMA_ERROR, torch.zeros_like(ONE_SIGMA_ERROR)])
    loss = ckn_loss(error, zero_covariance().expand(2, -1, -1), 0.25)
    assert loss.item() == pytest.approx(0.5 * (0.75 * 2.0 / 9.0 + 0.25 * 4.0), rel=LOSS_RTOL)


def parameters_of(branch):
    return [parameter.detach().clone() for parameter in branch.parameters()]


def moved(before, branch):
    return any(not torch.equal(old, new) for old, new in zip(before, branch.parameters(), strict=True))


def test_training_steps_one_branch_at_a_time_alternating_from_the_prior_branch():
    """MS-53: as the reference's train_batch_alternative at alter_period 1, step k updates the G¹ branch on even k
    and the G² branch on odd k; the other branch's weights do not move."""
    states, z = fly_scenario(json.loads(BREAK_TURN.read_text()))
    states, measurements = states[None, :2], z[None, None, :2]
    net = stand_in_network().train()
    optimizers = [torch.optim.Adam(branch.parameters(), lr=1e-3) for branch in (net.prior, net.innovation)]
    args = argparse.Namespace(batch_size=1, trunc_len=2, beta=RELEASED_BETA)
    rng = np.random.default_rng(0)
    for step_count, (active, idle) in enumerate([(net.prior, net.innovation), (net.innovation, net.prior)]):
        active_before, idle_before = parameters_of(active), parameters_of(idle)
        _, steps_taken = run_training_epoch(net, NOMINAL.dt, NOMINAL.R(), states, measurements, args, rng, optimizers, step_count)
        assert steps_taken == step_count + 1
        assert moved(active_before, active)
        assert not moved(idle_before, idle)


def row(beta, diverged, rmse, anees):
    return {"beta": beta, "diverged_fraction": diverged, "pos_rmse": rmse, "anees_mean": anees}


def test_tuning_takes_the_anees_closest_to_nine_among_runs_near_the_best_rmse_and_fewest_diverged():
    """MS-53: 0.25 is outside the 10% RMSE band, 0.05 is in it and closer to 9 in log ratio than 0.01."""
    rows = [row(0.01, 0.0, 10.0, 40.0), row(0.05, 0.0, 10.9, 12.0), row(0.25, 0.0, 11.2, 9.0)]
    assert selected(rows)["beta"] == 0.05


def test_tuning_drops_runs_diverging_more_than_five_points_above_the_fewest():
    """MS-53: 0.25 calibrates best but diverges on 0.06 of runs, 6 points above 0.05's zero."""
    rows = [row(0.05, 0.0, 10.0, 40.0), row(0.25, 0.06, 10.0, 9.0)]
    assert selected(rows)["beta"] == 0.05


def test_tuning_keeps_the_smaller_beta_on_an_anees_tie():
    """MS-53: equal log ratios (18 and 4.5 are a factor of 2 either side of 9) go to the smaller β."""
    rows = [row(0.25, 0.0, 10.0, 18.0), row(0.05, 0.0, 10.0, 4.5)]
    assert selected(rows)["beta"] == 0.05


def test_tuning_script_scores_each_checkpoint_on_val_under_the_nominal_r_with_beta_from_its_training_log(tmp_path, monkeypatch):
    """MS-53: rows come from val.npz through the page's numpy forward under nominal R, β read off training_log.json."""
    from scripts import eval as bench_eval
    from scripts import tune_ckn

    data_dir, weights = tmp_path / "datasets", tmp_path / "ckn_b0.25" / "best.npz"
    toy = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1", "--mc-test", "1", "--n-highmc", "1"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy, "--mc-highmc", "1"], cwd=REPO, check=True)
    weights.parent.mkdir()
    export_npz(stand_in_network().state_dict(), weights)
    (weights.parent / "training_log.json").write_text(json.dumps({"hyperparams": {"beta": 0.25}}))
    out = tmp_path / "tuning.json"
    monkeypatch.setattr(sys, "argv", ["tune_ckn.py", str(weights), "--data-dir", str(data_dir), "--out", str(out)])
    monkeypatch.chdir(REPO)

    tune_ckn.main()

    tuning = json.loads(out.read_text())
    states, measurements = bench_eval._load(data_dir / "val.npz")
    expected = tune_ckn.validation_row(weights, NOMINAL.dt, NOMINAL.R(), states, measurements)
    assert expected["beta"] == 0.25
    assert tuning["rows"] == json.loads(json.dumps([expected]))
    assert tuning["best"] == json.loads(json.dumps(expected))


def test_tuning_keeps_a_run_exactly_five_points_above_the_fewest_diverged():
    """MS-53: "within 5 points" includes the edge: 0.05 diverged against 0.0 stays a candidate."""
    rows = [row(0.05, 0.0, 10.0, 40.0), row(0.25, 0.05, 10.0, 9.0)]
    assert selected(rows)["beta"] == 0.25


def test_tuning_keeps_a_run_exactly_at_ten_percent_above_the_best_rmse():
    """MS-53: "within 10%" includes the edge: RMSE_SLACK times the best RMSE stays a candidate."""
    rows = [row(0.05, 0.0, 10.0, 40.0), row(0.25, 0.0, RMSE_SLACK * 10.0, 9.0)]
    assert selected(rows)["beta"] == 0.25


def test_tuning_measures_anees_against_nine_as_a_ratio():
    """MS-53: ANEES 9 is a log ratio of 0 and beats ANEES 1 (ln 9 away), whatever the β order."""
    rows = [row(0.05, 0.0, 10.0, 1.0), row(0.25, 0.0, 10.0, 9.0)]
    assert selected(rows)["beta"] == 0.25
