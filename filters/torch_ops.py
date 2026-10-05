"""Differentiable torch counterparts of filters/measurement_model.py and the
CV motion model, used inside KalmanNet's BPTT loop. Must match the numpy
versions exactly (same wrap convention, same MS-23 singularity handling) -- verified
in tests/test_torch_ops.py -- or KalmanNet would train against a subtly
different model than the one the classical filters are evaluated with,
breaking the fairness the brief demands.
"""

import numpy as np
import torch

from filters.measurement_model import AZIMUTH_ROW_FLOOR_M
from filters.motion_models import cv_model
from filters.state import STATE_DIM

_EPS = 1e-6


def wrap_angle_torch(a):
    return torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi


def h_torch(x):
    """x: [..., 9] -> [..., 3] (azimuth, elevation, range)."""
    px, py, pz = x[..., 0], x[..., 1], x[..., 2]
    rho2 = px * px + py * py
    overhead = rho2 == 0.0
    rho = torch.where(
        overhead, torch.zeros_like(rho2), torch.sqrt(torch.where(overhead, torch.ones_like(rho2), rho2))
    )
    r = torch.sqrt(rho2 + pz * pz)
    az = torch.where(
        overhead, torch.zeros_like(px), torch.atan2(py, torch.where(overhead, torch.ones_like(px), px))
    )
    el = torch.atan2(pz, rho)
    return torch.stack([az, el, r], dim=-1)


def wrap_measurement_diff(diff):
    """diff: [..., 3] (az, el, range) -- wrap the angular components only."""
    az = wrap_angle_torch(diff[..., 0])
    el = wrap_angle_torch(diff[..., 1])
    return torch.stack([az, el, diff[..., 2]], dim=-1)


def innovation_torch(z_meas, x_pred):
    return wrap_measurement_diff(z_meas - h_torch(x_pred))


def H_jacobian_torch(x):
    """Batched torch counterpart of filters.measurement_model.H_jacobian --
    same analytic formula and singularity floor, needed by Recursive
    KalmanNet's closed-form A_t term and its Jacobian input feature.
    x: [batch, 9] -> [batch, 3, 9]."""
    batch = x.shape[0]
    px, py, pz = x[:, 0], x[:, 1], x[:, 2]
    rho2 = torch.clamp(px * px + py * py, min=_EPS)
    rho = torch.sqrt(rho2)
    r2 = torch.clamp(rho2 + pz * pz, min=_EPS)
    r = torch.sqrt(r2)

    H = torch.zeros(batch, 3, STATE_DIM, dtype=x.dtype, device=x.device)
    horizontal_range_sq = px * px + py * py
    overhead = horizontal_range_sq == 0.0
    horizontal_range = torch.sqrt(torch.where(overhead, torch.ones_like(px), horizontal_range_sq))
    azimuth_gain = 1.0 / torch.clamp(horizontal_range, min=AZIMUTH_ROW_FLOOR_M)
    H[:, 0, 0] = -py / horizontal_range * azimuth_gain
    H[:, 0, 1] = px / horizontal_range * azimuth_gain
    H[:, 1, 0] = -pz * px / (r2 * rho)
    H[:, 1, 1] = -pz * py / (r2 * rho)
    H[:, 1, 2] = rho / r2
    H[:, 2, 0] = px / r
    H[:, 2, 1] = py / r
    H[:, 2, 2] = pz / r
    return H


def export_npz(state_dict, destination):
    """The torch-free weight format filters/knet/numpy_forward.py reads, keyed by the state_dict names."""
    np.savez(destination, **{name: tensor.detach().cpu().numpy() for name, tensor in state_dict.items()})


class CVMotionModel:
    """Constant-velocity f(x) = F_cv @ x as a fixed (non-trainable) linear
    torch op -- the same F every classical filter uses, built once from
    filters.motion_models.cv_model so there is one source of truth for the
    CV transition matrix."""

    def __init__(self, dt, sigma_a=0.05, tau=0.05, device="cpu"):
        F, _ = cv_model(dt, sigma_a=sigma_a, tau=tau)
        self.F = torch.tensor(F, dtype=torch.float64, device=device)

    def __call__(self, x):
        return x @ self.F.T
