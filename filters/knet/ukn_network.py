"""Unscented KalmanNet's NoiseNet and GainNet (Ko & Shafieezadeh, arXiv 2608.04201v2, Fig. 2), sized as its
manoeuvring-target example (d_e 32, H_G 128, H_N 64). Output layers start at zero and the stage adapters at the
identity, so the untrained filter is the UKF (MS-43)."""

import torch
import torch.nn as nn

from filters.knet.numpy_forward import (
    GAIN_FEATURE_DIM,
    LOG_VARIANCE_SATURATION,
    NOISE_FEATURE_DIM,
    NOISE_STAGES,
    OFF_DIAGONAL_SATURATION,
    VARIANCE_MULTIPLIER_MAX,
    VARIANCE_MULTIPLIER_MIN,
)
from filters.state import MEAS_DIM, STATE_DIM

EMBEDDING_DIM = 32
GAIN_HIDDEN_DIM = 128
NOISE_HIDDEN_DIM = 64


def encoder(feature_dim):
    return nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, EMBEDDING_DIM),
        nn.ReLU(),
        nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
        nn.ReLU(),
    )


def zeroed(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def identity_adapter(feature_dim):
    adapter = nn.Linear(feature_dim, feature_dim)
    nn.init.eye_(adapter.weight)
    nn.init.zeros_(adapter.bias)
    return adapter


def cholesky_multiplier(diagonal, lower):
    """Eq 12-13, batched: [b, d] diagonal and [b, d(d-1)/2] strictly lower outputs to a lower-triangular [b, d, d]."""
    size = diagonal.shape[1]
    rows, columns = torch.tril_indices(size, size, offset=-1)
    multiplier = torch.zeros(*diagonal.shape, size, dtype=diagonal.dtype, device=diagonal.device)
    multiplier[:, rows, columns] = OFF_DIAGONAL_SATURATION * torch.tanh(lower)
    variance_scale = torch.exp(LOG_VARIANCE_SATURATION * torch.tanh(diagonal))
    return multiplier + torch.diag_embed(
        torch.sqrt(torch.clamp(variance_scale, VARIANCE_MULTIPLIER_MIN, VARIANCE_MULTIPLIER_MAX))
    )


class NoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder(NOISE_FEATURE_DIM)
        self.gru = nn.GRU(EMBEDDING_DIM, NOISE_HIDDEN_DIM)
        self.post = nn.Sequential(nn.Linear(NOISE_HIDDEN_DIM, NOISE_HIDDEN_DIM), nn.ReLU())
        for stage, size in NOISE_STAGES.items():
            setattr(self, f"{stage}_adapter", identity_adapter(NOISE_FEATURE_DIM))
            setattr(self, f"{stage}_diagonal", zeroed(nn.Linear(NOISE_HIDDEN_DIM, size)))
            setattr(self, f"{stage}_lower", zeroed(nn.Linear(NOISE_HIDDEN_DIM, size * (size - 1) // 2)))

    def forward(self, stage, features, hidden):
        embedded = self.encoder(getattr(self, f"{stage}_adapter")(features)).unsqueeze(0)
        out, hidden = self.gru(embedded, hidden)
        shared = self.post(out.squeeze(0))
        diagonal = getattr(self, f"{stage}_diagonal")(shared)
        return cholesky_multiplier(diagonal, getattr(self, f"{stage}_lower")(shared)), hidden


class GainNet(nn.Module):
    def __init__(self, gain_scale):
        super().__init__()
        self.encoder = encoder(GAIN_FEATURE_DIM)
        self.gru = nn.GRU(EMBEDDING_DIM, GAIN_HIDDEN_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(GAIN_HIDDEN_DIM, GAIN_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(GAIN_HIDDEN_DIM, GAIN_HIDDEN_DIM),
            nn.ReLU(),
            zeroed(nn.Linear(GAIN_HIDDEN_DIM, STATE_DIM * MEAS_DIM)),
        )
        self.register_buffer("gain_scale", torch.tensor(gain_scale, dtype=torch.float64))

    def forward(self, features, gain, hidden):
        """Eq 20 per measurement column (MS-46): column j of ΔK is within c_K ‖K_UKF[:, j]‖, its own units."""
        out, hidden = self.gru(self.encoder(features).unsqueeze(0), hidden)
        bound = self.gain_scale * torch.linalg.vector_norm(gain, dim=-2, keepdim=True)
        return bound * torch.tanh(self.decoder(out[0])).reshape(-1, STATE_DIM, MEAS_DIM), hidden


class UnscentedKalmanNetwork(nn.Module):
    def __init__(self, gain_scale):
        super().__init__()
        self.noise_net = NoiseNet()
        self.gain_net = GainNet(gain_scale)

    def init_hidden(self, batch, dtype=torch.float64):
        noise_hidden = torch.zeros(1, batch, NOISE_HIDDEN_DIM, dtype=dtype)
        return {
            **dict.fromkeys(NOISE_STAGES, noise_hidden),
            "gain": torch.zeros(1, batch, GAIN_HIDDEN_DIM, dtype=dtype),
        }
