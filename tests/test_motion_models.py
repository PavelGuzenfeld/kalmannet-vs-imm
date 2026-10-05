import numpy as np

from filters.motion_models import cv_model, ca_model, ct_model, step
from filters.state import STATE_DIM, PX, PY, VX, VY


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


CT_H_IDX = [PX, VX, PY, VY]
EXACT_DT = 2.0**-5
EXACT_SIGMA_A = 0.5
BENCHMARK_DT = 0.02


def test_ct_white_accel_psd_does_not_scale_with_dt():
    """MS-7: Q[vx,vx]/dt is the continuous PSD under isotropic rotation; rtol 1e-12 covers expm roundoff."""
    _, Q_full = ct_model(EXACT_DT, omega=np.deg2rad(25.0), sigma_a=EXACT_SIGMA_A)
    _, Q_half = ct_model(EXACT_DT / 2, omega=np.deg2rad(25.0), sigma_a=EXACT_SIGMA_A)
    assert np.isclose(Q_full[VX, VX] / EXACT_DT, Q_half[VX, VX] / (EXACT_DT / 2), rtol=1e-12, atol=0.0)


def test_ct_zero_omega_q_is_dwna_closed_form_per_horizontal_axis():
    """MS-7. rtol 1e-12 covers expm roundoff; the PSD is sigma_a^2 * 0.02 s."""
    _, Q = ct_model(EXACT_DT, omega=0.0, sigma_a=EXACT_SIGMA_A)
    q_c = EXACT_SIGMA_A**2 * BENCHMARK_DT
    dwna = q_c * np.array([[EXACT_DT**3 / 3, EXACT_DT**2 / 2], [EXACT_DT**2 / 2, EXACT_DT]])
    expected = np.zeros((4, 4))
    expected[:2, :2] = dwna
    expected[2:, 2:] = dwna
    assert np.allclose(Q[np.ix_(CT_H_IDX, CT_H_IDX)], expected, rtol=1e-12, atol=0.0)


def test_ct_velocity_variance_at_benchmark_dt_keeps_former_hand_typed_value():
    """MS-7: the PSD is pinned so 50 Hz matches the old (sigma_a*dt)^2; rtol 1e-12 covers expm roundoff."""
    _, Q = ct_model(BENCHMARK_DT, omega=np.deg2rad(25.0), sigma_a=0.3)
    assert np.isclose(Q[VX, VX], (0.3 * BENCHMARK_DT) ** 2, rtol=1e-12, atol=0.0)
    assert np.isclose(Q[VY, VY], (0.3 * BENCHMARK_DT) ** 2, rtol=1e-12, atol=0.0)


def test_ct_horizontal_q_is_symmetric_positive_definite_with_position_terms():
    """MS-7: white acceleration drives position too, so the 4x4 block is full rank.
    Symmetry atol is 1e-12 of the largest entry: expm roundoff, not a modelled asymmetry."""
    _, Q = ct_model(BENCHMARK_DT, omega=np.deg2rad(25.0))
    Q_h = Q[np.ix_(CT_H_IDX, CT_H_IDX)]
    assert np.allclose(Q_h, Q_h.T, rtol=0.0, atol=1e-12 * np.abs(Q_h).max())
    assert np.all(np.linalg.eigvalsh(Q_h) > 0.0)
