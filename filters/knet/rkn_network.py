"""Recursive KalmanNet (Mortada, Falcon, Kahil, Clavaud, Michel, EUSIPCO 2025,
arXiv:2506.11639) -- the consistency-corrected KalmanNet variant, adapted
from the official reference (`github.com/ixblue/RecursiveKalmanNet`,
`Algo/RecursiveKalmanNet.py`) onto our az/el/range measurement model. Two
independent GRU networks, matching the reference's `GRUNetwork` structure
(FC pre-layer -> GRU -> FC post-layer -> linear output):

  - rnn_K:   outputs the Kalman gain in the line-of-sight frame, K in R^{m x n},
             rotated to Cartesian (decision 58 in #2, #85).
  - rnn_cov: outputs the p = m(m+1)/2 free lower-triangular entries of a
             Cholesky factor L; B_t = L_t L_t^T is the learned half of the
             Joseph-form covariance (issue #14 sec. 2). The other half, A_t
             (the propagated (I-K_tH_t)F_t P_{t-1} F_t^T (I-K_tH_t)^T term),
             is propagated in square-root form with L (decision 54 in
             #2) -- this is what distinguishes RKN
             from vanilla KalmanNet, Split-KalmanNet, and Cholesky-KalmanNet.

Both networks read the same 4 input features (innovation, observation
difference, Jacobian H_t, state correction), each group L2-normalised. rnn_cov
squares them elementwise first -- per the reference and issue #14's extraction
("covariance is a second-order/quadratic quantity") -- so they stay non-negative
and bounded (decision 59 in #2, #86); rnn_K reads them signed (decision 55, #70).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from filters.measurement_model import HORIZONTAL_RANGE_FLOOR_M
from filters.state import ACC, AZIMUTH, MEAS_DIM, POS, STATE_DIM, VEL

_M, _N = STATE_DIM, MEAS_DIM
_FEATURE_GROUPS = (_N, _N, _M * _N, _M)
_FEATURE_DIM = sum(_FEATURE_GROUPS)
_FEATURE_AXIS = 1
_KINEMATIC_BLOCKS = len((POS, VEL, ACC))


def normalise_groups(features):
    """MS-10, #70: each group L2-normalised as KalmanNet's features are, so signs survive (squared, a mirror
    through the z axis gave the same K) and a runaway state cannot saturate the GRU."""
    groups = features.split(_FEATURE_GROUPS, dim=_FEATURE_AXIS)
    normalised = [F.normalize(group, dim=_FEATURE_AXIS) for group in groups]
    return torch.cat(normalised, dim=_FEATURE_AXIS)


def los_frame(H):
    """MS-37: H's position rows are u_az/ρ, u_el/r and u_r. Returns blockdiag(R, R, R), R's rows those directions,
    and each measurement's metres per unit. Where H floors ρ (ρ < 1 mm, read off the azimuth row: 0 overhead, else
    1/max(ρ, 2⁻¹⁰ m)) u_el collapses onto u_r and both angles are undefined: R keeps only u_r, angle scales 0 (#85)."""
    position_rows = H[:, :, POS]
    norms = position_rows.norm(dim=-1)
    azimuth_norm = norms[:, [AZIMUTH]]
    angles_defined = (azimuth_norm > 0.0) & (azimuth_norm <= 1.0 / HORIZONTAL_RANGE_FLOOR_M)
    measured = torch.cat([angles_defined, angles_defined, torch.ones_like(angles_defined)], dim=-1)
    metres_per_unit = torch.where(measured, 1.0 / torch.where(measured, norms, torch.ones_like(norms)), 0.0)
    rotation = F.normalize(position_rows, dim=-1) * measured.unsqueeze(-1)
    identity = torch.eye(_KINEMATIC_BLOCKS, dtype=H.dtype, device=H.device).unsqueeze(0)
    return torch.kron(identity, rotation), metres_per_unit


class GRUNetwork(nn.Module):
    """FC1 -> GRU -> FC2 -> linear output, matching the reference
    `Algo.RecursiveKalmanNet.GRUNetwork` (one FC layer each side, default
    multipliers 10/10/20 -- kept faithful to the reference's defaults)."""

    def __init__(self, output_size, fc1_mult=10, hidden_mult=10, fc2_mult=20):
        super().__init__()
        self.output_size = output_size
        hidden_size = output_size * hidden_mult

        self.fc1 = nn.Sequential(nn.Linear(_FEATURE_DIM, _FEATURE_DIM * fc1_mult), nn.ReLU())
        self.gru = nn.GRU(_FEATURE_DIM * fc1_mult, hidden_size)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * fc2_mult),
            nn.ReLU(),
            nn.Linear(hidden_size * fc2_mult, output_size),
        )
        self.hidden_size = hidden_size

        # Same rationale as vanilla KalmanNet's gain network (see
        # gain_network.py): a fresh recurrent gain/covariance output has no
        # stability guarantee, so start near zero (K~0 => trust the CV
        # prediction; L~0 => P starts as just the closed-form A_t term).
        final_linear = self.fc2[-1]
        nn.init.uniform_(final_linear.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_linear.bias)

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float64, device=device)

    def forward(self, features, hidden):
        """features: [batch, feature_dim]. Returns (output [batch,
        output_size], new_hidden)."""
        x = features.unsqueeze(0)  # [1, batch, feature_dim], seq-first for GRU
        x = self.fc1(x)
        x, hidden = self.gru(x, hidden)
        x = self.fc2(x)
        return x.squeeze(0), hidden


class RKNGainCovNetwork(nn.Module):
    def __init__(self, m=STATE_DIM, n=MEAS_DIM, fc1_mult=10, hidden_mult=10, fc2_mult=20):
        """Multiplier defaults (10, 10, 20) match the reference exactly. At
        our m=9 (the reference's own examples use smaller m), that default
        sizing produces a ~7.8M-parameter network whose forward pass alone
        measures ~3.3 ms/step single-threaded -- already over the brief's
        2 ms budget before any accelerator is involved (see issue #15 sec.
        3). Pass smaller multipliers for a latency-competitive trained
        variant; the reference-default size is reported separately as a
        deployment-audit data point, not silently discarded."""
        super().__init__()
        self.m, self.n = m, n
        self.rnn_K = GRUNetwork(output_size=m * n, fc1_mult=fc1_mult, hidden_mult=hidden_mult, fc2_mult=fc2_mult)
        self.rnn_cov = GRUNetwork(output_size=m * (m + 1) // 2, fc1_mult=fc1_mult, hidden_mult=hidden_mult, fc2_mult=fc2_mult)
        self._tril_rows, self._tril_cols = torch.tril_indices(m, m)

    def init_hidden(self, batch_size, device):
        return self.rnn_K.init_hidden(batch_size, device), self.rnn_cov.init_hidden(batch_size, device)

    def forward(self, features, hidden):
        """MS-37, #85: rnn_K reads H and the correction in the line-of-sight frame and emits N there; K =
        blockdiag(Rᵀ) N diag(ρ, r, 1), so on position K H = Rᵀ N R: the loop is N's alone, whatever the range
        and bearing, and a mirror through the z axis gives diag(mirror) K exactly."""
        h_K, h_cov = hidden
        batch = features.shape[0]
        innovation, z_diff, H_flat, correction = features.split(_FEATURE_GROUPS, dim=_FEATURE_AXIS)
        H = H_flat.view(batch, self.n, self.m)
        blocks, metres_per_unit = los_frame(H)
        los_H = (H @ blocks.transpose(1, 2)).reshape(batch, -1)
        los_correction = (blocks @ correction.unsqueeze(-1)).squeeze(-1)
        los_features = torch.cat([innovation, z_diff, los_H, los_correction], dim=_FEATURE_AXIS)
        los_gain, h_K = self.rnn_K(normalise_groups(los_features), h_K)
        C, h_cov = self.rnn_cov(normalise_groups(features**2), h_cov)

        K = blocks.transpose(1, 2) @ los_gain.view(batch, self.m, self.n) * metres_per_unit.unsqueeze(1)

        L = torch.zeros(batch, self.m, self.m, dtype=features.dtype, device=features.device)
        L[:, self._tril_rows, self._tril_cols] = C

        return K, L, (h_K, h_cov)
