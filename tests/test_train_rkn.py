import math

import numpy as np
import pytest
import torch

from scripts.train_rkn import _gaussian_nll, run_epoch

_DIM = 3
_STATE_DIM = 9
_MEAS_DIM = 3


class _NonFiniteFactorNet(torch.nn.Module):
    """Zero gain and L = scale * NaN, so the covariance factor and the NLL are non-finite from the first step."""

    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64))

    def init_hidden(self, batch, device):
        return (torch.zeros(1, dtype=torch.float64),)

    def forward(self, features, hidden):
        batch = features.shape[0]
        K = torch.zeros(batch, _STATE_DIM, _MEAS_DIM, dtype=torch.float64)
        L = self.scale * torch.full((batch, _STATE_DIM, _STATE_DIM), math.nan, dtype=torch.float64)
        return K, L, hidden


def test_training_epoch_on_a_non_finite_covariance_factor_completes_without_updating_weights():
    torch.manual_seed(0)
    steps, range_m = 2, 1000.0
    states = np.zeros((1, steps, _STATE_DIM))
    states[..., 0] = range_m
    measurements = np.zeros((1, 1, steps, _MEAS_DIM))
    measurements[..., 2] = range_m
    net = _NonFiniteFactorNet(1000.0)
    optimizer = torch.optim.SGD(net.parameters(), lr=1.0)

    run_epoch(net, 0.02, states, measurements, 1, steps, np.random.default_rng(0), optimizer)

    assert net.scale.item() == 1000.0


def _nll_with_grad(e, S):
    S = S.clone().requires_grad_(True)
    loss = _gaussian_nll(e, S)
    loss.backward()
    return loss.item()


def test_nll_of_a_singular_covariance_factor_is_infinite_instead_of_a_solver_crash():
    S = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)).unsqueeze(0)
    e = torch.ones(1, _DIM, dtype=torch.float64)
    assert _nll_with_grad(e, S) == math.inf


def test_nll_of_a_factor_with_a_negative_diagonal_is_infinite_since_it_is_no_cholesky_factor():
    S = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)).unsqueeze(0)
    e = torch.ones(1, _DIM, dtype=torch.float64)
    assert _nll_with_grad(e, S) == math.inf


def test_one_singular_factor_makes_the_batch_nll_infinite():
    S = torch.eye(_DIM, dtype=torch.float64).repeat(2, 1, 1)
    S[1, -1, -1] = 0.0
    e = torch.ones(2, _DIM, dtype=torch.float64)
    assert _nll_with_grad(e, S) == math.inf


def test_nll_of_a_cholesky_factor_is_the_batch_mean_of_quadratic_plus_logdet():
    """Element 0: S = [[2,0],[1,2]], P = S S^T = [[4,2],[2,5]], e = [2,1] -> quad 1, logdet ln 16.
    Element 1: S = I, e = 0 -> 0. rel 1e-15 covers last-ulp rounding of the log() calls."""
    S = torch.tensor([[[2.0, 0.0], [1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    e = torch.tensor([[2.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    assert _nll_with_grad(e, S) == pytest.approx((1.0 + math.log(16.0)) / 2.0, rel=1e-15)
