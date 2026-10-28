"""Validate van_loan_discretize against the textbook closed-form discrete
white-noise-acceleration (DWNA) model (Bar-Shalom, Li & Kirubarajan,
"Estimation with Applications to Tracking and Navigation," sec. 6.2) before
trusting it for the Singer/CT motion models built on top of it."""

import numpy as np
import pytest
from scipy.linalg import expm

from filters.discretize import van_loan_discretize
from filters.motion_models import cv_model


def test_matches_dwna_closed_form():
    dt, q = 0.25, 3.0
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    G = np.array([[0.0], [1.0]])
    Qc = np.array([[q]])

    F, Q = van_loan_discretize(A, G, Qc, dt)

    F_expected = np.array([[1.0, dt], [0.0, 1.0]])
    Q_expected = q * np.array([
        [dt**3 / 3, dt**2 / 2],
        [dt**2 / 2, dt],
    ])

    assert np.allclose(F, F_expected)
    assert np.allclose(Q, Q_expected)


def test_discrete_Q_is_symmetric_positive_semidefinite():
    dt = 0.02
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -0.2]])
    G = np.array([[0.0], [0.0], [1.0]])
    Qc = np.array([[1.5]])

    _, Q = van_loan_discretize(A, G, Qc, dt)

    assert np.allclose(Q, Q.T)
    eigvals = np.linalg.eigvalsh(Q)
    assert np.all(eigvals > -1e-12)


@pytest.mark.parametrize("dt", [1.0, 2.0, 5.0, 10.0])
def test_cv_model_Q_stays_symmetric_psd_at_large_dt(dt):
    """MS-4, MS-7: the CV Singer pole (tau = 0.05 s) must not make Q indefinite at dt = 1 to 10 s."""
    _, Q = cv_model(dt)

    assert np.allclose(Q, Q.T, rtol=1e-12, atol=0.0)
    assert np.linalg.eigvalsh((Q + Q.T) / 2).min() >= -1e-12 * np.linalg.norm(Q, 2)


@pytest.mark.parametrize("dt", [1.0, 3.7, 10.0])
def test_fast_pole_Q_matches_ornstein_uhlenbeck_variance_and_semigroup_at_large_dt(dt):
    """MS-4, MS-7: Q_aa = sigma^2 (1 - exp(-2 dt / tau)) and Q(2 dt) = F Q(dt) F^T + Q(dt), both exact for LTI."""
    sigma, tau = 0.05, 0.05
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0 / tau]])
    G = np.array([[0.0], [0.0], [1.0]])
    Qc = np.array([[2.0 * sigma**2 / tau]])

    F, Q = van_loan_discretize(A, G, Qc, dt)
    _, Q_double = van_loan_discretize(A, G, Qc, 2.0 * dt)

    assert Q[2, 2] == pytest.approx(sigma**2 * (1.0 - np.exp(-2.0 * dt / tau)), rel=1e-12)
    assert np.allclose(Q_double, F @ Q @ F.T + Q, rtol=1e-9, atol=0.0)


def test_Q_matches_cancelling_van_loan_form_at_50_hz():
    """MS-3, MS-7: at dt = 0.02 s the old F @ Phi12 form is accurate to its cancellation error, ~1e-13."""
    dt, alpha, q_c = 0.02, 20.0, 2.0 * 0.05**2 * 20.0
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -alpha]])
    G = np.array([[0.0], [0.0], [1.0]])
    n = 3
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = G @ G.T * q_c
    M[n:, n:] = A.T
    phi = expm(M * dt)
    Q_old = phi[n:, n:].T @ phi[:n, n:]

    F, Q = van_loan_discretize(A, G, np.array([[q_c]]), dt)

    assert np.allclose(F, expm(A * dt), rtol=1e-12, atol=0.0)
    assert np.allclose(Q, Q_old, rtol=1e-9, atol=1e-9 * np.abs(Q_old).max())
