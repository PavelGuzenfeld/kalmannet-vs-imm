"""Batched, differentiable Cholesky-KalmanNet for training: numpy_forward.CholeskyKalmanNetFilter's step in torch,
with P as the Joseph sum of two Gram matrices rather than its QR factor (MS-51, MS-52)."""

import torch

from filters.state import AZIMUTH, ELEVATION, STATE_DIM
from filters.torch_ops import CVMotionModel, H_jacobian_torch, h_torch, innovation_torch, wrap_measurement_diff


def linearisation_feature(x_prior):
    z_prior = h_torch(x_prior)
    azimuth = z_prior[:, AZIMUTH]
    return torch.stack([torch.sin(azimuth), torch.cos(azimuth), z_prior[:, ELEVATION]], dim=1)


class CholeskyKalmanNetFilter:
    def __init__(self, network, dt, R):
        self.net = network
        self.f = CVMotionModel(dt)
        self.R_factor = torch.linalg.cholesky(torch.as_tensor(R, dtype=torch.float64))

    def init_sequence(self, x0):
        self.hidden = self.net.init_hidden(x0.shape[0])
        self.x = self.x_previous = x0
        self.x_prior_previous = self.f(x0)
        self.z_previous = None

    def step(self, z):
        """Returns (x, P, NIS) for the batch."""
        net, hidden = self.net, self.hidden
        x_prior = self.f(self.x)
        H = H_jacobian_torch(x_prior)
        y = innovation_torch(z, x_prior)
        z_diff = wrap_measurement_diff(z - (z if self.z_previous is None else self.z_previous))
        scaled_H = (H * net.state_scale / net.measurement_scale[:, None]).flatten(1)
        linearisation = linearisation_feature(x_prior)
        correction = (self.x - self.x_prior_previous) / net.state_scale
        increment = (self.x - self.x_previous) / net.state_scale
        prior_features = torch.cat([correction, increment, linearisation, scaled_H], dim=1)
        innovation_features = torch.cat(
            [y / net.measurement_scale, z_diff / net.measurement_scale, linearisation, scaled_H], dim=1
        )
        L_prior, hidden["prior"] = net.prior(prior_features, hidden["prior"])
        L_innovation, hidden["innovation"] = net.innovation(innovation_features, hidden["innovation"])
        prior_factor = net.state_scale[:, None] * L_prior
        inverse_innovation_factor = L_innovation / net.measurement_scale[:, None]
        K = prior_factor @ prior_factor.mT @ H.mT @ inverse_innovation_factor @ inverse_innovation_factor.mT
        self.x_prior_previous, self.x_previous, self.z_previous = x_prior, self.x, z
        self.x = x_prior + (K @ y.unsqueeze(-1)).squeeze(-1)
        IKH = torch.eye(STATE_DIM, dtype=K.dtype) - K @ H
        propagated, measured = IKH @ prior_factor, K @ self.R_factor
        P = propagated @ propagated.mT + measured @ measured.mT
        whitened = (inverse_innovation_factor.mT @ y.unsqueeze(-1)).squeeze(-1)
        return self.x, P, (whitened * whitened).sum(1)

    def detach_state(self):
        self.x, self.x_previous, self.x_prior_previous = self.x.detach(), self.x_previous.detach(), self.x_prior_previous.detach()
        self.z_previous = self.z_previous.detach()
        self.hidden = {branch: hidden.detach() for branch, hidden in self.hidden.items()}
