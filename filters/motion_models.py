"""Discrete-time process models sharing the 9-dim [pos, vel, acc] state
(filters/state.py). Three motion-model families, matching both the ground-
truth trajectory generator and the IMM's per-mode filters:

  - CV  (constant velocity): acceleration is a fast-decaying nuisance state
                              with small driving noise, so it stays near zero
                              and doesn't perturb velocity.
  - CA / Singer:              acceleration follows a mean-reverting
                              (Ornstein-Uhlenbeck) process with time constant
                              tau. CV above is the tau->0 special case of the
                              same continuous-time model; this is the
                              multi-second-correlation, large-driving-noise
                              case.
  - CT  (coordinated turn):   horizontal velocity rotates at a fixed known
                              turn rate omega (Li & Jilkov, "Survey of
                              Maneuvering Target Tracking," Part I, 2003);
                              every axis carries the same Singer acceleration
                              as CV/CA, feeding velocity alongside the turn.

CV, CA and CT are all discretized via van_loan_discretize (verified against
the closed-form DWNA case in test_discretize.py) rather than hand-typed
closed-form formulas.
"""

import numpy as np

from filters.discretize import van_loan_discretize
from filters.state import STATE_DIM, PX, PY, PZ, VX, VY, VZ, A_X, A_Y, A_Z


def singer_axis(dt, tau, sigma_a):
    """One position/velocity/acceleration axis: a_dot = -a/tau + w, with w's
    intensity set so the stationary acceleration variance is sigma_a^2."""
    alpha = 1.0 / tau
    A = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -alpha],
    ])
    G = np.array([[0.0], [0.0], [1.0]])
    q_c = 2.0 * sigma_a**2 * alpha
    return van_loan_discretize(A, G, np.array([[q_c]]), dt)


def _singer_9d(dt, tau, sigma_a):
    F = np.eye(STATE_DIM)
    Q = np.zeros((STATE_DIM, STATE_DIM))
    for p, v, a in ((PX, VX, A_X), (PY, VY, A_Y), (PZ, VZ, A_Z)):
        F3, Q3 = singer_axis(dt, tau, sigma_a)
        idx = [p, v, a]
        F[np.ix_(idx, idx)] = F3
        Q[np.ix_(idx, idx)] = Q3
    return F, Q


def cv_model(dt, sigma_a=0.05, tau=0.05):
    return _singer_9d(dt, tau=tau, sigma_a=sigma_a)


def ca_model(dt, sigma_a=2.0, tau=5.0):
    return _singer_9d(dt, tau=tau, sigma_a=sigma_a)


def ct_model(dt, omega, sigma_a=0.3, tau=5.0):
    """omega in rad/s, positive = counter-clockwise. Velocity rotates at omega and is driven by
    the Singer acceleration of every axis, so omega = 0 is the ca_model Singer model."""
    alpha = 1.0 / tau
    A = np.zeros((STATE_DIM, STATE_DIM))
    for p, v, a in ((PX, VX, A_X), (PY, VY, A_Y), (PZ, VZ, A_Z)):
        A[p, v] = 1.0
        A[v, a] = 1.0
        A[a, a] = -alpha
    A[VX, VY] = -omega
    A[VY, VX] = omega
    G = np.zeros((STATE_DIM, 3))
    G[[A_X, A_Y, A_Z], [0, 1, 2]] = 1.0
    q_c = 2.0 * sigma_a**2 * alpha
    return van_loan_discretize(A, G, q_c * np.eye(3), dt)


def step(x, F):
    return F @ x


MOTION_LABELS = {"cv": 0, "ct": 1, "ca": 2}
LABEL_NAMES = {v: k for k, v in MOTION_LABELS.items()}
