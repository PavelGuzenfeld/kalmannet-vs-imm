"""Exact continuous-to-discrete LTI conversion, F = expm(A dt). Q is the Lyapunov form, not
Van Loan's F @ Phi12: its -A block grows as exp(dt/tau) and the product cancels away for fast
poles, so Q lost PSD from about dt = 2 s at tau = 0.05 s."""

import numpy as np
from scipy.linalg import expm


def van_loan_discretize(A, G, Qc, dt):
    n = A.shape[0]
    eye = np.eye(n)
    M = np.zeros((n * n + 1, n * n + 1))
    M[: n * n, : n * n] = np.kron(A, eye) + np.kron(eye, A)
    M[: n * n, n * n] = (G @ Qc @ G.T).ravel()
    Q = expm(M * dt)[: n * n, n * n].reshape(n, n)
    return expm(A * dt), Q
