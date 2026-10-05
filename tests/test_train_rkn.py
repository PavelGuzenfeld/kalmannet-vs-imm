import math

import pytest
import torch

from scripts.train_rkn import _gaussian_nll

_DIM = 3


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
