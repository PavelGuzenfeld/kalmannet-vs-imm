"""Cholesky-KalmanNet's two Split-KalmanNet branches (Ko & Shafieezadeh, IEEE SPL 32 (2025) 326-330; `DNN_CKN` in
github.com/RAMSIS-Lab/ckn-spl-public), sized by the reference's dnn.py formulas at m = 9, n = 3. The PDEL emits the
factor in units of D_x = σ_init (MS-10's loss scale) and D_z = sqrt(diag R_nominal), so its entries are O(1) (MS-51)."""

import torch
import torch.nn as nn

from filters.knet.numpy_forward import (
    CKN_BRANCHES,
    CKN_GRU_LAYERS,
    CKN_INNOVATION_FEATURE_DIM,
    CKN_PRIOR_FEATURE_DIM,
    PDEL_EPSILON,
)
from filters.state import MEAS_DIM, STATE_DIM

ENCODER_DIM = (STATE_DIM + MEAS_DIM) * 10 * 8
DECODER_DIM = STATE_DIM * MEAS_DIM * 4
GRU_HIDDEN_DIM = 2 * (STATE_DIM * STATE_DIM + MEAS_DIM * MEAS_DIM)
OUTPUT_WEIGHT_RANGE = 1e-3
PRIOR_DIAGONAL_INIT = 2.0**-6
INNOVATION_DIAGONAL_INIT = 1.0
INIT_PERTURB_STD = (30.0, 30.0, 30.0, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5)


def lower_triangular_factor(entries, size):
    """Eq 4's PDEL, batched: [b, q(q+1)/2] fill the lower triangle row by row, the diagonal through |a| + ε."""
    rows, columns = torch.tril_indices(size, size)
    factor = entries.new_zeros(entries.shape[0], size, size)
    factor[:, rows, columns] = torch.where(rows == columns, torch.abs(entries) + PDEL_EPSILON, entries)
    return factor


class CovarianceBranch(nn.Module):
    """One Split-KalmanNet branch: Linear-ReLU, a 2-layer GRU, Linear-ReLU-Linear to the PDEL entries. The output
    layer starts near zero with diagonal bias b, so the untrained factor is b·I and the filter stays bounded (MS-51)."""

    def __init__(self, feature_dim, size, diagonal_init):
        super().__init__()
        self.size = size
        self.encoder = nn.Sequential(nn.Linear(feature_dim, ENCODER_DIM), nn.ReLU())
        self.gru = nn.GRU(ENCODER_DIM, GRU_HIDDEN_DIM, CKN_GRU_LAYERS)
        self.decoder = nn.Sequential(
            nn.Linear(GRU_HIDDEN_DIM, DECODER_DIM), nn.ReLU(), nn.Linear(DECODER_DIM, size * (size + 1) // 2)
        )
        output = self.decoder[-1]
        nn.init.uniform_(output.weight, -OUTPUT_WEIGHT_RANGE, OUTPUT_WEIGHT_RANGE)
        rows, columns = torch.tril_indices(size, size)
        with torch.no_grad():
            output.bias.copy_((rows == columns) * diagonal_init)

    def forward(self, features, hidden):
        out, hidden = self.gru(self.encoder(features).unsqueeze(0), hidden)
        return lower_triangular_factor(self.decoder(out[0]), self.size), hidden


class CholeskyKalmanNetwork(nn.Module):
    def __init__(self, R_nominal):
        super().__init__()
        self.prior = CovarianceBranch(CKN_PRIOR_FEATURE_DIM, CKN_BRANCHES["prior"], PRIOR_DIAGONAL_INIT)
        self.innovation = CovarianceBranch(
            CKN_INNOVATION_FEATURE_DIM, CKN_BRANCHES["innovation"], INNOVATION_DIAGONAL_INIT
        )
        R_nominal = torch.as_tensor(R_nominal, dtype=torch.float64)
        self.register_buffer("state_scale", torch.tensor(INIT_PERTURB_STD, dtype=torch.float64))
        self.register_buffer("measurement_scale", torch.sqrt(torch.diagonal(R_nominal)))

    def init_hidden(self, batch, dtype=torch.float64):
        return {
            branch: torch.zeros(CKN_GRU_LAYERS, batch, GRU_HIDDEN_DIM, dtype=dtype) for branch in CKN_BRANCHES
        }
