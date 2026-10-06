"""Torch-free forward pass of KalmanNet and RKN for Pyodide, which has numpy but no torch (#2 decision 5).
Mirrors gain_network/knet_filter/rkn_network/rkn_filter on one unbatched flight; weights are the torch
state_dict as arrays keyed by its own names, read by load_weights from torch_ops.export_npz's .npz."""

import numpy as np

from filters.measurement_model import HORIZONTAL_RANGE_FLOOR_M, H_jacobian, h, innovation, wrap_angle
from filters.motion_models import cv_model
from filters.state import ACC, AZIMUTH, MEAS_DIM, POS, STATE_DIM, VEL

GRU_GATES = 3
RKN_FEATURE_SPLITS = np.cumsum([MEAS_DIM, MEAS_DIM, STATE_DIM * MEAS_DIM])
KINEMATIC_BLOCKS = len((POS, VEL, ACC))


def load_weights(npz):
    """Read every array once: an NpzFile re-reads its archive on each lookup."""
    with np.load(npz) as archive:
        return dict(archive)


def linear(weights, prefix, x):
    return weights[f"{prefix}weight"] @ x + weights[f"{prefix}bias"]


def relu(x):
    return np.maximum(x, 0.0)


def sigmoid(x):
    decay = np.exp(-np.abs(x))
    return np.where(x >= 0.0, 1.0, decay) / (1.0 + decay)


def gru_cell(weights, prefix, x, hidden):
    """One step of a single-layer torch.nn.GRU; gate rows stacked (reset, update, new) as torch stores them."""
    input_gates = weights[f"{prefix}weight_ih_l0"] @ x + weights[f"{prefix}bias_ih_l0"]
    hidden_gates = weights[f"{prefix}weight_hh_l0"] @ hidden + weights[f"{prefix}bias_hh_l0"]
    i_r, i_z, i_n = np.split(input_gates, GRU_GATES)
    h_r, h_z, h_n = np.split(hidden_gates, GRU_GATES)
    r = sigmoid(i_r + h_r)
    z = sigmoid(i_z + h_z)
    n = np.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * hidden


def normalize(v):
    torch_normalize_eps = 1e-12
    return v / max(np.linalg.norm(v), torch_normalize_eps)


def normalise_rkn_groups(features):
    return np.concatenate([normalize(group) for group in np.split(features, RKN_FEATURE_SPLITS)])


def los_frame(H):
    position_rows = H[:, POS]
    norms = np.linalg.norm(position_rows, axis=1)
    angles_defined = 0.0 < norms[AZIMUTH] <= 1.0 / HORIZONTAL_RANGE_FLOOR_M
    measured = np.array([angles_defined, angles_defined, True])
    metres_per_unit = np.divide(1.0, norms, out=np.zeros_like(norms), where=measured)
    rotation = np.array([normalize(row) for row in position_rows])
    return np.kron(np.eye(KINEMATIC_BLOCKS), rotation), metres_per_unit


def wrap_measurement_diff(diff):
    wrapped = diff.copy()
    wrapped[AZIMUTH] = wrap_angle(diff[AZIMUTH])
    return wrapped


class KalmanGainGRU:
    def __init__(self, weights, m=STATE_DIM, n=MEAS_DIM):
        self.weights, self.m, self.n = weights, m, n

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, hidden):
        w = self.weights
        h_Q, h_Sigma, h_S = hidden
        out_Q = h_Q = gru_cell(w, "GRU_Q.", relu(linear(w, "FC5.0.", fw_update_diff)), h_Q)
        out_FC6 = relu(linear(w, "FC6.0.", fw_evol_diff))
        out_Sigma = gru_cell(w, "GRU_Sigma.", np.concatenate([out_Q, out_FC6]), h_Sigma)
        out_FC1 = relu(linear(w, "FC1.0.", out_Sigma))
        out_FC7 = relu(linear(w, "FC7.0.", np.concatenate([obs_diff, obs_innov_diff])))
        out_S = h_S = gru_cell(w, "GRU_S.", np.concatenate([out_FC1, out_FC7]), h_S)
        out_FC2 = linear(w, "FC2.2.", relu(linear(w, "FC2.0.", np.concatenate([out_Sigma, out_S]))))
        out_FC3 = relu(linear(w, "FC3.0.", np.concatenate([out_S, out_FC2])))
        h_Sigma = relu(linear(w, "FC4.0.", np.concatenate([out_Sigma, out_FC3])))
        return out_FC2.reshape(self.m, self.n), (h_Q, h_Sigma, h_S)


class KalmanNetFilter:
    def __init__(self, gain_net, dt):
        self.net = gain_net
        self.F, _ = cv_model(dt)

    def init_sequence(self, x0, prior_Q, prior_Sigma, prior_S):
        self.hidden = (prior_Q.flatten(), prior_Sigma.flatten(), prior_S.flatten())
        self.x_posterior = self.x_posterior_prev = self.x_prior_prev = x0
        self.y_prev = h(x0)

    def step(self, z):
        x_prior = self.F @ self.x_posterior
        obs_diff = wrap_measurement_diff(z - self.y_prev)
        innov = innovation(z, x_prior)
        fw_evol_diff = self.x_posterior - self.x_posterior_prev
        fw_update_diff = self.x_posterior - self.x_prior_prev

        gain, self.hidden = self.net.forward(
            normalize(obs_diff), normalize(innov), normalize(fw_evol_diff), normalize(fw_update_diff), self.hidden
        )
        x_upd = x_prior + gain @ innov

        self.x_posterior_prev = self.x_posterior
        self.x_prior_prev = x_prior
        self.x_posterior = x_upd
        self.y_prev = z
        return x_upd


class GRUNetwork:
    def __init__(self, weights, prefix):
        self.weights, self.prefix = weights, prefix
        self.hidden_size = weights[f"{prefix}gru.weight_hh_l0"].shape[1]

    def forward(self, features, hidden):
        w, p = self.weights, self.prefix
        hidden = gru_cell(w, f"{p}gru.", relu(linear(w, f"{p}fc1.0.", features)), hidden)
        return linear(w, f"{p}fc2.2.", relu(linear(w, f"{p}fc2.0.", hidden))), hidden


class RKNGainCovNetwork:
    def __init__(self, weights, m=STATE_DIM, n=MEAS_DIM):
        self.m, self.n = m, n
        self.rnn_K = GRUNetwork(weights, "rnn_K.")
        self.rnn_cov = GRUNetwork(weights, "rnn_cov.")
        self._tril = np.tril_indices(m)

    def init_hidden(self):
        return np.zeros(self.rnn_K.hidden_size), np.zeros(self.rnn_cov.hidden_size)

    def forward(self, features, hidden):
        h_K, h_cov = hidden
        y, z_diff, H_flat, correction = np.split(features, RKN_FEATURE_SPLITS)
        H = H_flat.reshape(self.n, self.m)
        blocks, metres_per_unit = los_frame(H)
        los_features = np.concatenate([y, z_diff, (H @ blocks.T).flatten(), blocks @ correction])
        los_gain, h_K = self.rnn_K.forward(normalise_rkn_groups(los_features), h_K)
        C, h_cov = self.rnn_cov.forward(normalise_rkn_groups(features**2), h_cov)
        L = np.zeros((self.m, self.m))
        L[self._tril] = C
        K = blocks.T @ los_gain.reshape(self.m, self.n) * metres_per_unit
        return K, L, (h_K, h_cov)


class RKNFilter:
    def __init__(self, network, dt):
        self.net = network
        self.F, _ = cv_model(dt)

    def init_sequence(self, x0, P0):
        self.hidden = self.net.init_hidden()
        self.x = x0
        self.S = np.linalg.cholesky(P0)
        self.x_prior_previous = self.F @ x0
        self.z_previous = h(x0)

    def step(self, z):
        x_prior = self.F @ self.x
        H_t = H_jacobian(x_prior)
        y = innovation(z, x_prior)

        correction = self.x - self.x_prior_previous
        z_diff = wrap_measurement_diff(z - self.z_previous)

        self.z_previous = z
        self.x_prior_previous = x_prior

        features = np.concatenate([y, z_diff, H_t.flatten(), correction])
        K, L, self.hidden = self.net.forward(features, self.hidden)

        x_upd = x_prior + K @ y
        IKH = np.eye(STATE_DIM) - K @ H_t
        self.S = np.linalg.qr(np.vstack([(IKH @ self.F @ self.S).T, L.T]), mode="r").T
        self.x = x_upd
        return x_upd, self.S @ self.S.T
