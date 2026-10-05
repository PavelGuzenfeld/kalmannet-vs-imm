import numpy as np

from filters.motion_models import cv_model, ca_model, ct_model, step
from filters.state import STATE_DIM, PX, PY, VX, VY, A_X, A_Y


def test_cv_position_update_matches_constant_velocity():
    dt = 0.02
    F, Q = cv_model(dt)
    x = np.array([0.0, 0.0, 100.0, 10.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    x1 = step(x, F)
    assert np.isclose(x1[PX], x[PX] + x[VX] * dt, atol=1e-3)
    assert np.isclose(x1[PY], x[PY] + x[VY] * dt, atol=1e-3)
    assert np.all(np.linalg.eigvalsh(Q) > -1e-12)


def test_ct_zero_omega_matches_cv_horizontal_block():
    dt = 0.02
    F_cv, _ = cv_model(dt)
    F_ct, _ = ct_model(dt, omega=0.0)
    h_idx = [PX, VX, PY, VY]
    assert np.allclose(F_ct[np.ix_(h_idx, h_idx)], F_cv[np.ix_(h_idx, h_idx)])


def test_ct_rotation_preserves_horizontal_speed_noiseless():
    dt = 0.02
    omega = np.deg2rad(15.0)
    F, _ = ct_model(dt, omega=omega)
    x = np.zeros(STATE_DIM)
    x[VX], x[VY] = 20.0, 0.0
    speed0 = np.hypot(x[VX], x[VY])
    for _ in range(100):
        x = step(x, F)
    speed1 = np.hypot(x[VX], x[VY])
    assert np.isclose(speed0, speed1, rtol=1e-6)


def test_ca_has_larger_stationary_accel_variance_than_cv():
    dt = 0.02
    _, Q_cv = cv_model(dt)
    _, Q_ca = ca_model(dt)
    from filters.state import A_X
    assert Q_ca[A_X, A_X] > Q_cv[A_X, A_X]


def test_all_process_noise_covariances_are_psd():
    dt = 0.02
    for F, Q in (cv_model(dt), ca_model(dt), ct_model(dt, omega=np.deg2rad(10))):
        assert np.allclose(Q, Q.T)
        assert np.all(np.linalg.eigvalsh(Q) > -1e-10)


def test_ct_horizontal_acceleration_decays_with_singer_time_constant():
    """MS-6/MS-30: inside CT, ax and ay follow the Singer axis, a_k+1 = exp(-dt/tau) a_k.
    rtol 1e-12: expm Pade rounding against np.exp, a few ulp."""
    dt, tau = 0.02, 2.0
    F, _ = ct_model(dt, omega=np.deg2rad(25.0), sigma_a=0.3, tau=tau)
    x = np.zeros(STATE_DIM)
    x[A_X], x[A_Y] = 1.0, -2.0
    x1 = step(x, F)
    decay = np.exp(-dt / tau)
    assert np.isclose(x1[A_X], decay, rtol=1e-12, atol=0.0)
    assert np.isclose(x1[A_Y], -2.0 * decay, rtol=1e-12, atol=0.0)


def test_ct_horizontal_acceleration_feeds_the_turning_velocity():
    """MS-6/MS-30: with v = vx + i vy, a = ax + i ay, v' = i omega v + a, a' = -a/tau, so a unit ax
    reaches v as (exp(i omega dt) - exp(-dt/tau)) / (i omega + 1/tau). rtol 1e-12: expm rounding."""
    dt, tau, omega = 0.02, 2.0, np.deg2rad(25.0)
    F, _ = ct_model(dt, omega=omega, sigma_a=0.3, tau=tau)
    v = (np.exp(1j * omega * dt) - np.exp(-dt / tau)) / (1j * omega + 1.0 / tau)
    assert np.isclose(F[VX, A_X], v.real, rtol=1e-12, atol=0.0)
    assert np.isclose(F[VY, A_X], v.imag, rtol=1e-12, atol=0.0)


def test_ct_horizontal_acceleration_has_singer_process_noise_while_turning():
    """MS-6/MS-30: the ax, ay rows are an Ornstein-Uhlenbeck process with stationary variance
    sigma_a^2, so one step adds sigma_a^2 (1 - exp(-2 dt/tau)). rtol 1e-12: expm rounding."""
    dt, tau, sigma_a = 0.02, 2.0, 0.3
    _, Q = ct_model(dt, omega=np.deg2rad(25.0), sigma_a=sigma_a, tau=tau)
    q_a = sigma_a**2 * (1.0 - np.exp(-2.0 * dt / tau))
    assert np.isclose(Q[A_X, A_X], q_a, rtol=1e-12, atol=0.0)
    assert np.isclose(Q[A_Y, A_Y], q_a, rtol=1e-12, atol=0.0)


def test_ct_positive_turn_rate_rotates_velocity_counter_clockwise():
    """MS-6: the exact CT velocity rotation, vy_k+1 = sin(omega dt) vx_k for positive omega.
    rtol 1e-12: expm Pade rounding against np.sin, a few ulp."""
    dt, omega = 0.02, np.deg2rad(25.0)
    F, _ = ct_model(dt, omega=omega)
    assert np.isclose(F[VY, VX], np.sin(omega * dt), rtol=1e-12, atol=0.0)
    assert np.isclose(F[VX, VY], -np.sin(omega * dt), rtol=1e-12, atol=0.0)


def test_ct_at_zero_turn_rate_is_the_three_axis_singer_model():
    """MS-6: omega -> 0 reduces the CT continuous model to the Singer axis of MS-4/MS-5 on x, y, z.
    atol 1e-15: both sides are expm of the same blocks, differing only in rounding."""
    dt, tau, sigma_a = 0.02, 5.0, 0.3
    F_ct, Q_ct = ct_model(dt, omega=0.0, sigma_a=sigma_a, tau=tau)
    F_singer, Q_singer = ca_model(dt, sigma_a=sigma_a, tau=tau)
    assert np.allclose(F_ct, F_singer, rtol=0.0, atol=1e-15)
    assert np.allclose(Q_ct, Q_singer, rtol=0.0, atol=1e-15)
