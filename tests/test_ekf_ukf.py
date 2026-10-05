import numpy as np

from data.config import Regime
from data.trajectory_gen import measure
from filters.ekf import EKF
from filters.measurement_model import h
from filters.motion_models import cv_model, step
from filters.ukf import UKF
from filters.state import STATE_DIM


def _pure_cv_trajectory(rng, regime, T):
    """Ground truth generated directly from the CV model, bypassing the
    random segment-kind draw in data.trajectory_gen -- these tests want the
    filter's own model assumption to be exactly correct, not whatever motion
    kind the segment sampler happens to pick for a given seed."""
    F, Q = cv_model(regime.dt, sigma_a=regime.cv_sigma_a)
    L = np.linalg.cholesky(Q + 1e-12 * np.eye(STATE_DIM))
    x = np.zeros(STATE_DIM)
    x[:3] = [3000.0, 500.0, 200.0]
    x[3:6] = [50.0, -20.0, 5.0]
    states = np.zeros((T, STATE_DIM))
    for t in range(T):
        states[t] = x
        x = step(x, F) + L @ rng.standard_normal(STATE_DIM)
    return states


def _run_filter(flt, states, measurements):
    T = states.shape[0]
    x = states[0].copy()
    x[3:] += np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])  # perturbed init
    P = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])

    xs, Ps, innovs, Ss = [], [], [], []
    for t in range(T):
        if t == 0:
            x_pred, P_pred = x, P
            if isinstance(flt, UKF):
                pts_pred = flt._sigma_points(x_pred, P_pred)
                x_upd, P_upd, y, S = flt.update(x_pred, P_pred, pts_pred, measurements[t])
            else:
                x_upd, P_upd, y, S = flt.update(x_pred, P_pred, measurements[t])
        else:
            x_upd, P_upd, y, S = flt.step(xs[-1], Ps[-1], measurements[t])
        xs.append(x_upd)
        Ps.append(P_upd)
        innovs.append(y)
        Ss.append(S)
    return np.array(xs), np.array(Ps), np.array(innovs), np.array(Ss)


def test_ekf_tracks_cv_trajectory_with_bounded_error_and_valid_covariance():
    regime = Regime(name="cv_only", T=200)
    states = _pure_cv_trajectory(np.random.default_rng(3), regime, regime.T)
    z = measure(np.random.default_rng(4), states, regime)

    ekf = EKF(regime.dt, regime.R())
    xs, Ps, _, _ = _run_filter(ekf, states, z)

    pos_err = np.linalg.norm(xs[-20:, :3] - states[-20:, :3], axis=-1)
    assert np.mean(pos_err) < 20.0  # converges to sub-20m tracking error

    for P in Ps:
        assert np.allclose(P, P.T, atol=1e-6)
        assert np.all(np.linalg.eigvalsh(P) > -1e-8)


def test_ukf_handles_target_crossing_the_azimuth_wrap_boundary():
    """Construct sigma points that straddle +-pi and confirm the UKF's
    circular-mean measurement update doesn't diverge."""
    regime = Regime(name="wrap", T=1)
    dt = regime.dt
    ukf = UKF(dt, regime.R())

    x = np.zeros(STATE_DIM)
    r = 1000.0
    x[0] = r * np.cos(np.pi - 0.005)
    x[1] = r * np.sin(np.pi - 0.005)  # azimuth just under +pi
    x[3] = 50.0  # moving so sigma points spread across the seam
    P = np.diag([200.0, 200.0, 50.0, 20.0, 20.0, 5.0, 1.0, 1.0, 1.0])

    x_pred, P_pred, pts_pred = ukf.predict(x, P)
    z = h(x_pred)
    x_upd, P_upd, y, S = ukf.update(x_pred, P_pred, pts_pred, z)

    assert np.all(np.isfinite(x_upd))
    assert abs(y[0]) < 0.1  # innovation on a noiseless self-consistent measurement stays tiny
    assert np.all(np.linalg.eigvalsh(P_upd) > -1e-6)


def test_ukf_range_innovation_variance_includes_process_noise_in_measurement_sigma_points():
    """MS-9/MS-32. At (1000, 0, 0) range is px to first order, so S[range] must equal
    (F P F^T + Q)[px, px] + R[range]. rtol 1e-6 covers the O(sigma^2 / r) curvature of
    range at the sigma points. dt = 5 tau keeps van Loan clear of the e^(dt/tau) cancellation that
    made dt = 20 tau CPU-dependent on CI."""
    R = Regime(name="range_axis", T=1).R()
    dt_long, sigma_a_large = 0.25, 20.0
    ukf = UKF(dt_long, R, sigma_a=sigma_a_large)
    x = np.zeros(STATE_DIM)
    x[0] = 1000.0
    P = np.diag([1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])
    P_pred_expected = ukf.F @ P @ ukf.F.T + ukf.Q
    process_noise_share = ukf.Q[0, 0] / P_pred_expected[0, 0]
    assert process_noise_share > 0.9

    x_pred, P_pred, pts_pred = ukf.predict(x, P)
    _, _, _, S = ukf.update(x_pred, P_pred, pts_pred, h(x_pred))

    assert np.isclose(S[2, 2], P_pred_expected[0, 0] + R[2, 2], rtol=1e-6, atol=0.0)


LINE_OF_SIGHT_RANGE_M = 100.0
CROSS_RANGE_SIGMA_M = 10.0
CROSS_RANGE_AXES = 2
WEIGHT_PER_SIGMA_POINT = 1.0 / (2 * STATE_DIM)
FLOAT_TOLERANCE_M = 1e-9


def _predicted_range_offset(update):
    x = np.zeros(STATE_DIM)
    x[0] = LINE_OF_SIGHT_RANGE_M
    P = np.zeros((STATE_DIM, STATE_DIM))
    P[1, 1] = P[2, 2] = CROSS_RANGE_SIGMA_M**2
    y = update(x, P, h(x))[2]
    return -y[2]


def test_ukf_predicted_range_sits_at_alpha_one_sigma_points_ms9_ms31():
    """MS-9/MS-31. alpha = 1, kappa = 0 gives lambda = 0: the 4 cross-range points sit at
    +-3 sigma with weight 1/18 each, so the range bias is 4/18 (sqrt(r^2 + 9 s^2) - r) = 0.978 m,
    not the 1.0 m second-order r + s^2/r. Tolerance is the 1e-12 jitter on P plus float rounding."""
    ukf = UKF(0.1, np.eye(3))
    spread_m = STATE_DIM**0.5 * CROSS_RANGE_SIGMA_M
    offset_at_spread = np.hypot(LINE_OF_SIGHT_RANGE_M, spread_m) - LINE_OF_SIGHT_RANGE_M
    expected = 2 * CROSS_RANGE_AXES * WEIGHT_PER_SIGMA_POINT * offset_at_spread
    bias = _predicted_range_offset(lambda x, P, z: ukf.update(x, P, ukf._sigma_points(x, P), z))
    assert abs(bias - expected) < FLOAT_TOLERANCE_M


def test_ekf_predicted_range_omits_second_order_bias_ms9():
    """MS-9. The EKF mean is h(x), so the second-order range bias is exactly zero."""
    assert _predicted_range_offset(EKF(0.1, np.eye(3)).update) == 0.0


def test_ukf_default_process_noise_matches_ekf_defaults_ms9():
    """MS-9. EKF and UKF run the same CV model, so their default Q must be identical."""
    dt = 0.1
    assert np.array_equal(UKF(dt, np.eye(3)).Q, EKF(dt, np.eye(3)).Q)


UKF_ALPHA = 1.0
UKF_BETA = 2.0
UKF_LAMBDA = 0.0
WC0_FROM_MS9 = UKF_LAMBDA / (STATE_DIM + UKF_LAMBDA) + 1 - UKF_ALPHA**2 + UKF_BETA
NEAR_LINEAR_RANGE_M = 3000.0
NEAR_LINEAR_SIGMA_M = 1.0


def test_ukf_zeroth_covariance_weight_carries_beta_ms9():
    """MS-9. alpha = 1, kappa = 0 gives lambda = 0, so Wc[0] = 0 + 1 - 1 + beta = 2 exactly."""
    assert WC0_FROM_MS9 == 2.0
    assert UKF(0.1, np.eye(3)).Wc[0] == WC0_FROM_MS9


def _ukf_and_ekf_posteriors(range_m, cross_range_sigma_m):
    x = np.zeros(STATE_DIM)
    x[0] = range_m
    P = np.diag([cross_range_sigma_m**2] * 3 + [1e-6] * 3 + [1e-8] * 3)
    R = Regime(name="posterior_pair", T=1).R()
    ukf = UKF(0.1, R)
    ukf_post = ukf.update(x, P, ukf._sigma_points(x, P), h(x))
    ekf_post = EKF(0.1, R).update(x, P, h(x))
    return ukf_post, ekf_post


def _range_bias_m(range_m, cross_range_sigma_m):
    spread_m = STATE_DIM**0.5 * cross_range_sigma_m
    return 2 * CROSS_RANGE_AXES * WEIGHT_PER_SIGMA_POINT * (np.hypot(range_m, spread_m) - range_m)


def test_ukf_agrees_with_ekf_within_range_bias_when_measurement_is_near_linear_ms9():
    """MS-9. At 3 km with 1 m position sigma the only UKF-EKF gap is the second-order range
    bias (3.3e-4 m); the gain is below 1, so the posterior position gap is bounded by it."""
    (x_ukf, P_ukf, _, _), (x_ekf, _, _, _) = _ukf_and_ekf_posteriors(NEAR_LINEAR_RANGE_M, NEAR_LINEAR_SIGMA_M)
    bound_m = _range_bias_m(NEAR_LINEAR_RANGE_M, NEAR_LINEAR_SIGMA_M)
    assert np.max(np.abs(x_ukf[:3] - x_ekf[:3])) < bound_m
    assert np.allclose(P_ukf, P_ukf.T, atol=1e-9)


def test_ukf_posterior_differs_from_ekf_by_gain_times_range_bias_when_measurement_is_curved_ms9():
    """MS-9. At 100 m with 10 m sigma the UKF innovation is -bias on range only, so the
    px posterior gap is K[px, range] * bias with Pxz[px, range] = sigma^2. Tolerance as FLOAT_TOLERANCE_M."""
    (x_ukf, _, _, S), (x_ekf, _, _, _) = _ukf_and_ekf_posteriors(LINE_OF_SIGHT_RANGE_M, CROSS_RANGE_SIGMA_M)
    gain_px_range = CROSS_RANGE_SIGMA_M**2 * np.linalg.inv(S)[2, 2]
    expected_gap_m = -gain_px_range * _range_bias_m(LINE_OF_SIGHT_RANGE_M, CROSS_RANGE_SIGMA_M)
    assert abs((x_ukf[0] - x_ekf[0]) - expected_gap_m) < FLOAT_TOLERANCE_M
