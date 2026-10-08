"""Batched, differentiable Unscented KalmanNet for training: filters/ukf.py's sigma-point recursion in torch, with
the same weights, jitter, circular azimuth mean and wrapped residuals, so numpy_forward replays it (MS-43 to MS-48)."""

import torch

from filters.state import AZIMUTH, ELEVATION, RANGE, STATE_DIM
from filters.torch_ops import h_torch, wrap_measurement_diff
from filters.ukf import CHOLESKY_JITTER_RELATIVE, UKF


def float64(array):
    return torch.tensor(array, dtype=torch.float64)


def weighted_outer(Wc, left, right):
    return torch.einsum("s,bsi,bsj->bij", Wc, left, right)


def scaled_covariance(base, base_factor, multiplier):
    identity = torch.eye(len(base), dtype=base.dtype)
    return base + base_factor @ (multiplier @ multiplier.mT - identity) @ base_factor.T


def log_diagonal(P):
    return torch.log(torch.diagonal(P, dim1=-2, dim2=-1))


def noise_features(z, y, x, P):
    measurement = [torch.sin(z[:, AZIMUTH]), torch.cos(z[:, AZIMUTH]), z[:, ELEVATION], z[:, RANGE]]
    return torch.cat([torch.stack(measurement, dim=1), y, x, log_diagonal(P)], dim=1)


def gain_features(y, S, x_increment, P_pred, K):
    S_factor = torch.linalg.cholesky(S)
    whitened = torch.linalg.solve_triangular(S_factor, y.unsqueeze(-1), upper=False).squeeze(-1)
    nis = (whitened * whitened).sum(1, keepdim=True)
    log_det_S = 2.0 * log_diagonal(S_factor).sum(1, keepdim=True)
    correction = (K @ y.unsqueeze(-1)).squeeze(-1)
    return torch.cat([whitened, nis, log_det_S, x_increment, log_diagonal(P_pred), correction], dim=1)


class UnscentedKalmanNetFilter:
    def __init__(self, network, dt, R):
        ukf = UKF(dt, R)
        self.net = network
        self.F, self.Q, self.R = float64(ukf.F), float64(ukf.Q), float64(R)
        self.Q_factor, self.R_factor = torch.linalg.cholesky(self.Q), torch.linalg.cholesky(self.R)
        self.Wm, self.Wc = float64(ukf.Wm), float64(ukf.Wc)
        self.spread = STATE_DIM + ukf.lam

    def sigma_points(self, x, P):
        jitter = CHOLESKY_JITTER_RELATIVE * torch.diagonal(P, dim1=-2, dim2=-1).sum(1) / STATE_DIM
        identity = torch.eye(STATE_DIM, dtype=P.dtype)
        columns = torch.linalg.cholesky(self.spread * (P + jitter[:, None, None] * identity)).mT
        centre = x.unsqueeze(1)
        return torch.cat([centre, centre + columns, centre - columns], dim=1)

    def predict(self, x, P, Q):
        pts_pred = self.sigma_points(x, P) @ self.F.T
        x_pred = torch.einsum("s,bsn->bn", self.Wm, pts_pred)
        diff = pts_pred - x_pred.unsqueeze(1)
        P_pred = weighted_outer(self.Wc, diff, diff) + Q
        return x_pred, P_pred, self.sigma_points(x_pred, P_pred)

    def innovation(self, x_pred, pts_pred, z):
        Z = h_torch(pts_pred)
        sin_mean = torch.einsum("s,bs->b", self.Wm, torch.sin(Z[..., AZIMUTH]))
        cos_mean = torch.einsum("s,bs->b", self.Wm, torch.cos(Z[..., AZIMUTH]))
        other_means = torch.einsum("s,bsm->bm", self.Wm, Z[..., ELEVATION:])
        z_mean = torch.cat([torch.atan2(sin_mean, cos_mean).unsqueeze(1), other_means], dim=1)
        dz = wrap_measurement_diff(Z - z_mean.unsqueeze(1))
        return pts_pred - x_pred.unsqueeze(1), dz, wrap_measurement_diff(z - z_mean)

    def analytic_update(self, P_pred, dx, dz, R):
        S = weighted_outer(self.Wc, dz, dz) + R
        Pxz = weighted_outer(self.Wc, dx, dz)
        K = Pxz @ torch.linalg.inv(S)
        cross_term = Pxz @ K.mT
        return K, P_pred - cross_term - cross_term.mT + K @ S @ K.mT, S

    def init_sequence(self, x0, P0):
        self.hidden = self.net.init_hidden(x0.shape[0])
        self.x, self.P = x0, P0
        self.z_previous = self.y_previous = None

    def step(self, z):
        """Returns (x, P, y, S, ΔK, K_UKF) for the batch; step 0 is update-only, as eval.py starts the UKF."""
        hidden = self.hidden
        if self.z_previous is None:
            x_pred, P_pred, pts_pred = self.x, self.P, self.sigma_points(self.x, self.P)
        else:
            features = noise_features(self.z_previous, self.y_previous, self.x, self.P)
            A_Q, hidden["process"] = self.net.noise_net("process", features, hidden["process"])
            x_pred, P_pred, pts_pred = self.predict(
                self.x, self.P, scaled_covariance(self.Q, self.Q_factor, A_Q)
            )
        dx, dz, y = self.innovation(x_pred, pts_pred, z)
        features = noise_features(z, y, x_pred, P_pred)
        A_R, hidden["measurement"] = self.net.noise_net("measurement", features, hidden["measurement"])
        K, P_ukf, S = self.analytic_update(P_pred, dx, dz, scaled_covariance(self.R, self.R_factor, A_R))
        features = gain_features(y, S, x_pred - self.x, P_pred, K)
        delta_K, hidden["gain"] = self.net.gain_net(features, K, hidden["gain"])
        self.x = x_pred + ((K + delta_K) @ y.unsqueeze(-1)).squeeze(-1)
        self.P = P_ukf + delta_K @ S @ delta_K.mT
        self.z_previous, self.y_previous = z, y
        return self.x, self.P, y, S, delta_K, K

    def detach_state(self):
        self.x, self.P = self.x.detach(), self.P.detach()
        self.z_previous, self.y_previous = self.z_previous.detach(), self.y_previous.detach()
        self.hidden = {stage: hidden.detach() for stage, hidden in self.hidden.items()}
