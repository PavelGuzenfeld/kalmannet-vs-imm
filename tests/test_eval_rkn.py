import numpy as np
import torch
from scipy.stats import chi2

from filters.state import STATE_DIM
from scripts import eval_rkn

TINY_DIAGONAL = 2.0**-30


class _IllConditionedFilter:
    def __init__(self, net, dt):
        factor = np.eye(STATE_DIM)
        factor[1, 0] = 1.0
        factor[1, 1] = TINY_DIAGONAL
        self.S = torch.tensor(factor).unsqueeze(0)

    def init_sequence(self, x0, P0):
        pass

    def step(self, z):
        x_upd = torch.zeros(1, STATE_DIM, dtype=torch.float64)
        x_upd[0, 0] = 1.0
        return x_upd, self.S @ self.S.transpose(1, 2)


def test_nees_is_whitened_through_s_when_p_rounds_to_singular(monkeypatch):
    """MS-36: P = S S^T rounds to singular here; e = [1, 0] is outside range(P) and whitens through S to [1, -2^30], so NEES is 2^60."""
    monkeypatch.setattr(eval_rkn, "RKNFilter", _IllConditionedFilter)
    states = np.zeros((1, 1, STATE_DIM))
    measurements = np.zeros((1, 1, 1, 3))

    result = eval_rkn.evaluate_dataset(None, 0.02, states, measurements)

    assert result["nees"][0, 0, 0] == 2.0**60


def test_rkn_anees_cells_average_over_the_monte_carlo_runs_with_bounds_for_the_run_count():
    """Averaged over the two runs each step reads 0, 0 and 30: outside the two-run band. Averaged over the three
    steps instead it would read 10, inside the three-step band."""
    alpha = 1 - 0.95
    result = {"err": np.zeros((1, 2, 3, STATE_DIM)), "nees": np.array([[[0.0, 0.0, 30.0]] * 2])}

    summary = eval_rkn.summarize(result, burn_in=0)

    assert summary["anees_fraction_inside_95"] == 0.0
    assert summary["anees_bounds"] == (chi2.ppf(alpha / 2, df=18) / 2, chi2.ppf(1 - alpha / 2, df=18) / 2)


class _RunawayFilter:
    """Position estimate 100 m further off each step: exactly 200 m at step 2, past it at step 3."""

    made = []

    def __init__(self, net, dt):
        self.steps = 0
        self.S = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0)
        _RunawayFilter.made.append(self)

    def init_sequence(self, x0, P0):
        pass

    def step(self, z):
        x_upd = torch.zeros(1, STATE_DIM, dtype=torch.float64)
        x_upd[0, 0] = 100.0 * self.steps
        self.steps += 1
        return x_upd, self.S


def test_rkn_run_is_stepped_through_its_divergence_step_and_no_further(monkeypatch):
    _RunawayFilter.made = []
    monkeypatch.setattr(eval_rkn, "RKNFilter", _RunawayFilter)

    result = eval_rkn.evaluate_dataset(None, 0.02, np.zeros((1, 5, STATE_DIM)), np.zeros((1, 1, 5, 3)))

    assert [flt.steps for flt in _RunawayFilter.made] == [4]
    assert np.isnan(result["err"][0, 0, 4]).all() and result["err"][0, 0, 3, 0] == 300.0
