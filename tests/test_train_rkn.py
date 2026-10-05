import math

import numpy as np
import pytest
import torch

from scripts.train_rkn import _gaussian_nll, run_epoch

_DIM = 3
_STATE_DIM = 9
_MEAS_DIM = 3


class _NegativeCovarianceNet(torch.nn.Module):
    """Zero gain and B = -scale * I, so P = A + B is negative definite from the first step."""

    def __init__(self, scale):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float64))

    def init_hidden(self, batch, device):
        return (torch.zeros(1, dtype=torch.float64),)

    def forward(self, features, hidden):
        batch = features.shape[0]
        K = torch.zeros(batch, _STATE_DIM, _MEAS_DIM, dtype=torch.float64)
        B = -self.scale * torch.eye(_STATE_DIM, dtype=torch.float64).repeat(batch, 1, 1)
        return K, B, hidden


def test_training_epoch_on_a_non_positive_definite_covariance_completes_without_updating_weights():
    torch.manual_seed(0)
    steps, range_m = 2, 1000.0
    states = np.zeros((1, steps, _STATE_DIM))
    states[..., 0] = range_m
    measurements = np.zeros((1, 1, steps, _MEAS_DIM))
    measurements[..., 2] = range_m
    net = _NegativeCovarianceNet(1000.0)
    optimizer = torch.optim.SGD(net.parameters(), lr=1.0)

    run_epoch(net, 0.02, states, measurements, 1, steps, np.random.default_rng(0), optimizer)

    assert net.scale.item() == 1000.0


def _nll_with_grad(e, P):
    P = P.clone().requires_grad_(True)
    loss = _gaussian_nll(e, P)
    loss.backward()
    return loss.item()


def test_nll_of_a_singular_covariance_is_infinite_instead_of_a_solver_crash():
    P = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)).unsqueeze(0)
    e = torch.ones(1, _DIM, dtype=torch.float64)
    assert _nll_with_grad(e, P) == math.inf


def test_nll_of_an_indefinite_covariance_with_positive_determinant_is_infinite():
    P = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64)).unsqueeze(0)
    e = torch.ones(1, _DIM, dtype=torch.float64)
    assert _nll_with_grad(e, P) == math.inf


def test_one_non_positive_definite_element_makes_the_batch_nll_infinite():
    P = torch.eye(_DIM, dtype=torch.float64).repeat(2, 1, 1)
    P[1, -1, -1] = 0.0
    e = torch.ones(2, _DIM, dtype=torch.float64)
    assert _nll_with_grad(e, P) == math.inf


def test_nll_of_a_positive_definite_covariance_is_the_batch_mean_of_quadratic_plus_logdet():
    """Element 0: P = [[4,2],[2,5]] = L L^T, L = [[2,0],[1,2]], e = [2,1] -> quad 1, logdet ln 16.
    Element 1: P = I, e = 0 -> 0. rel 1e-15 covers last-ulp rounding of the log() calls."""
    P = torch.tensor([[[4.0, 2.0], [2.0, 5.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    e = torch.tensor([[2.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    assert _nll_with_grad(e, P) == pytest.approx((1.0 + math.log(16.0)) / 2.0, rel=1e-15)
