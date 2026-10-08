"""Torch-free forward pass of KalmanNet, Bayesian KalmanNet, RKN and Unscented KalmanNet for Pyodide, which has numpy but no torch (#2 decision 5).
Mirrors gain_network/knet_filter/rkn_network/rkn_filter on one unbatched flight; weights are the torch
state_dict as arrays keyed by its own names, read by load_weights from torch_ops.export_npz's .npz."""

import itertools

import numpy as np

from filters.measurement_model import HORIZONTAL_RANGE_FLOOR_M, H_jacobian, h, innovation, wrap_angle
from filters.motion_models import cv_model
from filters.state import ACC, AZIMUTH, ELEVATION, MEAS_DIM, POS, RANGE, STATE_DIM, VEL
from filters.ukf import UKF

GRU_GATES = 3
RKN_FEATURE_SPLITS = np.cumsum([MEAS_DIM, MEAS_DIM, STATE_DIM * MEAS_DIM])
KINEMATIC_BLOCKS = len((POS, VEL, ACC))
DROPOUT_LAYERS = ("FC5", "FC6", "FC1", "FC7", "FC2", "FC3", "FC4")
NOISE_STAGES = {"process": STATE_DIM, "measurement": MEAS_DIM}
MEASUREMENT_FEATURE_DIM = MEAS_DIM + 1
NOISE_FEATURE_DIM = MEASUREMENT_FEATURE_DIM + MEAS_DIM + 2 * STATE_DIM
GAIN_FEATURE_DIM = MEAS_DIM + 2 + 3 * STATE_DIM
VARIANCE_MULTIPLIER_MIN, VARIANCE_MULTIPLIER_MAX = 0.1, 50.0
LOG_VARIANCE_SATURATION = 4.0
OFF_DIAGONAL_SATURATION = 1.0
TORCH_LAYER_NORM_EPS = 1e-5


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
    rotation = np.array([normalize(row) for row in position_rows]) * measured[:, np.newaxis]
    return np.kron(np.eye(KINEMATIC_BLOCKS), rotation), metres_per_unit


def wrap_measurement_diff(diff):
    wrapped = diff.copy()
    wrapped[AZIMUTH] = wrap_angle(diff[AZIMUTH])
    return wrapped


def unchanged(x):
    return x


def bernoulli_dropout(rng, dropout_p):
    """torch.nn.Dropout's convention: zero with probability p, scale the kept units by 1/(1-p).
    dropout_p is one p, or one per DROPOUT_LAYERS entry: the order KalmanGainGRU.forward applies them in."""
    keep_ps = itertools.cycle(1.0 - np.atleast_1d(dropout_p))

    def drop(x):
        keep_p = next(keep_ps)
        return x * (rng.random(x.shape) < keep_p) / keep_p

    return drop


def learned_dropout_p(weights):
    """The per-layer p a concrete-dropout checkpoint learned (#149), in DROPOUT_LAYERS order; None at a fixed p."""
    if "dropout.FC5.p_logit" not in weights:
        return None
    logits = np.concatenate([weights[f"dropout.{layer}.p_logit"] for layer in DROPOUT_LAYERS])
    return sigmoid(logits)


def dropout_rng(trajectory_seed):
    """Spawned off the seed, so never the stream default_rng(seed) draws measurement noise from."""
    return np.random.default_rng(np.random.SeedSequence(trajectory_seed).spawn(1)[0])


def ensemble_mean_and_covariance(estimates):
    """Eq 22a/22b over [J, m], normalised by J. Anchored at the first estimate, so J equal estimates average exactly."""
    mean = estimates[0] + np.mean(estimates - estimates[0], axis=0)
    spread = estimates - mean
    return mean, spread.T @ spread / len(estimates)


class KalmanGainGRU:
    def __init__(self, weights, m=STATE_DIM, n=MEAS_DIM, dropout=unchanged):
        self.weights, self.m, self.n, self.dropout = weights, m, n, dropout

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, hidden):
        w, drop = self.weights, self.dropout
        h_Q, h_Sigma, h_S = hidden
        out_Q = h_Q = gru_cell(w, "GRU_Q.", drop(relu(linear(w, "FC5.0.", fw_update_diff))), h_Q)
        out_FC6 = drop(relu(linear(w, "FC6.0.", fw_evol_diff)))
        out_Sigma = gru_cell(w, "GRU_Sigma.", np.concatenate([out_Q, out_FC6]), h_Sigma)
        out_FC1 = drop(relu(linear(w, "FC1.0.", out_Sigma)))
        out_FC7 = drop(relu(linear(w, "FC7.0.", np.concatenate([obs_diff, obs_innov_diff]))))
        out_S = h_S = gru_cell(w, "GRU_S.", np.concatenate([out_FC1, out_FC7]), h_S)
        out_FC2 = linear(w, "FC2.2.", drop(relu(linear(w, "FC2.0.", np.concatenate([out_Sigma, out_S])))))
        out_FC3 = drop(relu(linear(w, "FC3.0.", np.concatenate([out_S, out_FC2]))))
        h_Sigma = drop(relu(linear(w, "FC4.0.", np.concatenate([out_Sigma, out_FC3]))))
        return out_FC2.reshape(self.m, self.n), (h_Q, h_Sigma, h_S)


def rowwise_matvec(matrix, rows):
    """matrix @ row for every row of [J, in]. Not rows @ matrix.T: that is a gemm, whose summation order differs
    from the gemv a single `matrix @ row` runs, so the last bits would move."""
    return (matrix @ rows[:, :, np.newaxis])[:, :, 0]


def stacked_linear(weights, prefix, x):
    return rowwise_matvec(weights[f"{prefix}weight"], x) + weights[f"{prefix}bias"]


def stacked_gru_cell(weights, prefix, x, hidden):
    """gru_cell over [J, ·] rows."""
    input_gates = rowwise_matvec(weights[f"{prefix}weight_ih_l0"], x) + weights[f"{prefix}bias_ih_l0"]
    hidden_gates = rowwise_matvec(weights[f"{prefix}weight_hh_l0"], hidden) + weights[f"{prefix}bias_hh_l0"]
    i_r, i_z, i_n = np.split(input_gates, GRU_GATES, axis=-1)
    h_r, h_z, h_n = np.split(hidden_gates, GRU_GATES, axis=-1)
    r = sigmoid(i_r + h_r)
    z = sigmoid(i_z + h_z)
    n = np.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * hidden


class BatchedKalmanGainGRU:
    """J dropout realisations of KalmanGainGRU in one pass. The inputs are shared, the hidden states are [J, ·].
    Masks come from one Generator, drawn realisation by realisation and layer by layer in DROPOUT_LAYERS order:
    the order J serial passes draw them in. dropout_p is one p, or one per layer (#149)."""

    def __init__(self, weights, dropout_p, rng, realisations, m=STATE_DIM, n=MEAS_DIM):
        self.weights, self.rng, self.realisations, self.m, self.n = weights, rng, realisations, m, n
        self.keep_p = 1.0 - np.broadcast_to(dropout_p, len(DROPOUT_LAYERS))
        widths = [weights[f"{layer}.0.weight"].shape[0] for layer in DROPOUT_LAYERS]
        self.keep_thresholds = np.repeat(self.keep_p, widths)
        self.mask_splits = np.cumsum(widths)[:-1]

    def draw_masks(self):
        draws = self.rng.random((self.realisations, len(self.keep_thresholds)))
        return iter(zip(np.split(draws < self.keep_thresholds, self.mask_splits, axis=-1), self.keep_p))

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, hidden):
        w, masks = self.weights, self.draw_masks()

        def drop(x):
            mask, keep_p = next(masks)
            return x * mask / keep_p

        h_Q, h_Sigma, h_S = hidden
        out_Q = h_Q = stacked_gru_cell(w, "GRU_Q.", drop(relu(linear(w, "FC5.0.", fw_update_diff))), h_Q)
        out_FC6 = drop(relu(linear(w, "FC6.0.", fw_evol_diff)))
        out_Sigma = stacked_gru_cell(w, "GRU_Sigma.", np.concatenate([out_Q, out_FC6], axis=-1), h_Sigma)
        out_FC1 = drop(relu(stacked_linear(w, "FC1.0.", out_Sigma)))
        out_FC7 = drop(relu(linear(w, "FC7.0.", np.concatenate([obs_diff, obs_innov_diff]))))
        out_S = h_S = stacked_gru_cell(w, "GRU_S.", np.concatenate([out_FC1, out_FC7], axis=-1), h_S)
        hidden_FC2 = drop(relu(stacked_linear(w, "FC2.0.", np.concatenate([out_Sigma, out_S], axis=-1))))
        out_FC2 = stacked_linear(w, "FC2.2.", hidden_FC2)
        out_FC3 = drop(relu(stacked_linear(w, "FC3.0.", np.concatenate([out_S, out_FC2], axis=-1))))
        h_Sigma = drop(relu(stacked_linear(w, "FC4.0.", np.concatenate([out_Sigma, out_FC3], axis=-1))))
        return out_FC2.reshape(self.realisations, self.m, self.n), (h_Q, h_Sigma, h_S)


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


class BayesianKalmanNetFilter:
    """Dahan et al. 2025 Algorithm 1: every realisation predicts from the ensemble mean, keeping its own GRU state.
    S = H F P Fᵀ Hᵀ + R is this benchmark's, for NIS; the paper defines none (MS-40)."""

    def __init__(self, weights, dt, R, dropout_p, rng, realisations):
        self.net = BatchedKalmanGainGRU(weights, dropout_p, rng, realisations)
        self.realisations = realisations
        self.F, _ = cv_model(dt)
        self.R = R

    def init_sequence(self, x0, prior_Q, prior_Sigma, prior_S):
        priors = (prior_Q, prior_Sigma, prior_S)
        self.hidden = tuple(np.tile(prior.flatten(), (self.realisations, 1)) for prior in priors)
        self.x = self.x_prev = self.x_prior_prev = x0
        self.y_prev = h(x0)
        self.P = np.zeros((STATE_DIM, STATE_DIM))

    def step(self, z):
        x_prior = self.F @ self.x
        H = H_jacobian(x_prior)
        S = H @ self.F @ self.P @ self.F.T @ H.T + self.R
        innov = innovation(z, x_prior)
        gains, self.hidden = self.net.forward(
            normalize(wrap_measurement_diff(z - self.y_prev)),
            normalize(innov),
            normalize(self.x - self.x_prev),
            normalize(self.x - self.x_prior_prev),
            self.hidden,
        )
        estimates = x_prior + gains @ innov
        self.x_prev, self.x_prior_prev, self.y_prev = self.x, x_prior, z
        self.x, self.P = ensemble_mean_and_covariance(estimates)
        return self.x, self.P, innov, S


def layer_norm(weights, prefix, x):
    centred = x - np.mean(x)
    return centred / np.sqrt(np.mean(centred**2) + TORCH_LAYER_NORM_EPS) * weights[f"{prefix}weight"] + weights[
        f"{prefix}bias"
    ]


def encoded(weights, prefix, x):
    hidden = relu(linear(weights, f"{prefix}1.", layer_norm(weights, f"{prefix}0.", x)))
    return relu(linear(weights, f"{prefix}3.", hidden))


def cholesky_multiplier(diagonal, lower):
    """Eq 12-13: each variance direction scaled into [0.1, 50]x its baseline, strictly lower entries within ±c_off."""
    size = len(diagonal)
    multiplier = np.zeros((size, size))
    multiplier[np.tril_indices(size, -1)] = OFF_DIAGONAL_SATURATION * np.tanh(lower)
    variance_scale = np.exp(LOG_VARIANCE_SATURATION * np.tanh(diagonal))
    bounded_scale = np.clip(variance_scale, VARIANCE_MULTIPLIER_MIN, VARIANCE_MULTIPLIER_MAX)
    multiplier[np.diag_indices(size)] = np.sqrt(bounded_scale)
    return multiplier


def scaled_covariance(base, base_factor, multiplier):
    """Eq 9 as base + L (A Aᵀ - I) Lᵀ: (L A)(L A)ᵀ in exact arithmetic, and base bit for bit at A = I (MS-44)."""
    return base + base_factor @ (multiplier @ multiplier.T - np.eye(len(base))) @ base_factor.T


def noise_features(z, y, x, P):
    """Eq 10-11 without the control term; azimuth enters as its sine and cosine, so the ±π cut never shows."""
    measurement = [np.sin(z[AZIMUTH]), np.cos(z[AZIMUTH]), z[ELEVATION], z[RANGE]]
    return np.concatenate([measurement, y, x, np.log(np.diag(P))])


def gain_features(y, S, x_increment, P_pred, K):
    """Eq 19: whitened innovation, NIS, log det S, predicted increment, log diag P, K_UKF y; y is wrapped."""
    S_factor = np.linalg.cholesky(S)
    whitened = np.linalg.solve(S_factor, y)
    surprise = [whitened @ whitened, 2.0 * np.sum(np.log(np.diag(S_factor)))]
    return np.concatenate([whitened, surprise, x_increment, np.log(np.diag(P_pred)), K @ y])


class UnscentedKalmanNetwork:
    def __init__(self, weights):
        self.weights = weights

    def init_hidden(self):
        w = self.weights
        noise_hidden = np.zeros(w["noise_net.gru.weight_hh_l0"].shape[1])
        return {**dict.fromkeys(NOISE_STAGES, noise_hidden), "gain": np.zeros(w["gain_net.gru.weight_hh_l0"].shape[1])}

    def noise(self, stage, features, hidden):
        w = self.weights
        adapted = linear(w, f"noise_net.{stage}_adapter.", features)
        hidden = gru_cell(w, "noise_net.gru.", encoded(w, "noise_net.encoder.", adapted), hidden)
        shared = relu(linear(w, "noise_net.post.0.", hidden))
        diagonal = linear(w, f"noise_net.{stage}_diagonal.", shared)
        return cholesky_multiplier(diagonal, linear(w, f"noise_net.{stage}_lower.", shared)), hidden

    def gain_residual(self, features, gain, hidden):
        w = self.weights
        hidden = gru_cell(w, "gain_net.gru.", encoded(w, "gain_net.encoder.", features), hidden)
        decoded = relu(linear(w, "gain_net.decoder.2.", relu(linear(w, "gain_net.decoder.0.", hidden))))
        residual = np.tanh(linear(w, "gain_net.decoder.4.", decoded)).reshape(STATE_DIM, MEAS_DIM)
        return w["gain_net.gain_scale"] * np.linalg.norm(gain, axis=0) * residual, hidden


class UnscentedKalmanNetFilter:
    """Ko & Shafieezadeh on filters/ukf.py's sigma points (MS-43 to MS-48): NoiseNet scales Q and R, GainNet adds
    ΔK to K_UKF, and P = P_UKF + ΔK S ΔKᵀ. Step 0 is update-only, as eval.py starts the UKF."""

    def __init__(self, network, dt, R):
        self.net = network
        self.ukf = UKF(dt, R)
        self.Q_factor = np.linalg.cholesky(self.ukf.Q)
        self.R_factor = np.linalg.cholesky(R)

    def init_sequence(self, x0, P0):
        self.hidden = self.net.init_hidden()
        self.x, self.P = x0, P0
        self.z_previous = self.y_previous = None

    def step(self, z):
        ukf, hidden = self.ukf, self.hidden
        if self.z_previous is None:
            x_pred, P_pred, pts_pred = self.x, self.P, ukf._sigma_points(self.x, self.P)
        else:
            features = noise_features(self.z_previous, self.y_previous, self.x, self.P)
            A_Q, hidden["process"] = self.net.noise("process", features, hidden["process"])
            x_pred, P_pred, pts_pred = ukf.predict(self.x, self.P, scaled_covariance(ukf.Q, self.Q_factor, A_Q))
        dx, dz, y = ukf.innovation(x_pred, pts_pred, z)
        features = noise_features(z, y, x_pred, P_pred)
        A_R, hidden["measurement"] = self.net.noise("measurement", features, hidden["measurement"])
        K, P_ukf, S = ukf.analytic_update(P_pred, dx, dz, scaled_covariance(ukf.R, self.R_factor, A_R))
        features = gain_features(y, S, x_pred - self.x, P_pred, K)
        delta_K, hidden["gain"] = self.net.gain_residual(features, K, hidden["gain"])
        self.x = x_pred + (K + delta_K) @ y
        self.P = P_ukf + delta_K @ S @ delta_K.T
        self.z_previous, self.y_previous = z, y
        return self.x, self.P, y, S


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
