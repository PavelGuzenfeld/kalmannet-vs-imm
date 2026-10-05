"""Angle-wrap is the #1 bug in angle-measurement filters (per the brief) --
this is the first executable test in the whole benchmark, written and passing
before the data generator, before any filter, before anything else."""

import numpy as np

from data.config import NOMINAL
from data.fixed_wing import DEFAULT_PRESET, PRESETS, fly
from data.trajectory_gen import measure
from filters.ekf import EKF
from filters.imm import IMM
from filters.measurement_model import AZIMUTH_ROW_FLOOR_M, H_jacobian, h, innovation, wrap_angle
from filters.state import PX, PY, PZ, STATE_DIM
from filters.ukf import UKF
from scripts.eval import run_ekf_like, run_imm


def _state_at(az, el, r):
    px = r * np.cos(el) * np.cos(az)
    py = r * np.cos(el) * np.sin(az)
    pz = r * np.sin(el)
    x = np.zeros(STATE_DIM)
    x[0], x[1], x[2] = px, py, pz
    return x


def test_innovation_wraps_across_pi_boundary():
    r, el = 100.0, 0.0
    x_true = _state_at(az=np.pi - 0.01, el=el, r=r)
    x_pred = _state_at(az=-np.pi + 0.01, el=el, r=r)

    z_meas = h(x_true)
    naive_diff = z_meas[0] - h(x_pred)[0]
    wrapped = innovation(z_meas, x_pred)

    # True azimuth separation is only 0.02 rad; a naive subtraction across
    # the +-pi seam produces a spurious ~2*pi difference instead.
    assert abs(naive_diff) > 6.0
    assert abs(wrapped[0]) < 0.05


def test_wrap_angle_is_idempotent_and_bounded():
    angles = np.linspace(-10 * np.pi, 10 * np.pi, 4001)
    wrapped = wrap_angle(angles)
    assert np.all(wrapped >= -np.pi) and np.all(wrapped < np.pi)
    # wrapping twice must be a no-op
    assert np.allclose(wrap_angle(wrapped), wrapped)


def test_measurement_jacobian_matches_finite_difference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = np.zeros(STATE_DIM)
        x[0:3] = rng.uniform(-500, 500, size=3)
        x[2] = rng.uniform(50, 500)  # keep pz away from the rho=0 singularity path
        H = H_jacobian(x)

        eps = 1e-4
        H_fd = np.zeros((3, STATE_DIM))
        for j in range(3):  # h only depends on position columns
            dx = np.zeros(STATE_DIM)
            dx[j] = eps
            H_fd[:, j] = (h(x + dx) - h(x - dx)) / (2 * eps)

        assert np.allclose(H[:, :3], H_fd[:, :3], atol=1e-3)
        assert np.allclose(H[:, 3:], 0.0)


OVERHEAD_ALTITUDE_M = 300.0
OVERHEAD_PASS_STEPS = 200
OVERHEAD_PASS_SEEDS = range(5)


def _position(px, py, pz=OVERHEAD_ALTITUDE_M):
    x = np.zeros(STATE_DIM)
    x[PX], x[PY], x[PZ] = px, py, pz
    return x


def test_h_directly_overhead_is_finite_with_zero_azimuth():
    """MS-2/MS-23: atan2(0, 0) = 0, so h stays finite at rho = 0 and needs no guard."""
    assert np.array_equal(h(_position(0.0, 0.0)), [0.0, np.pi / 2, OVERHEAD_ALTITUDE_M])


def test_azimuth_row_directly_overhead_is_zero():
    """MS-23: azimuth has no gradient at rho = 0, so the row is zero and an EKF takes no azimuth update."""
    assert np.array_equal(H_jacobian(_position(0.0, 0.0))[0], np.zeros(STATE_DIM))


def test_azimuth_row_at_the_floor_is_inverse_floor():
    """MS-23: |d az / d(px, py)| = 1/rho exactly at rho = AZIMUTH_ROW_FLOOR_M."""
    assert H_jacobian(_position(AZIMUTH_ROW_FLOOR_M, 0.0))[0, PY] == 1.0 / AZIMUTH_ROW_FLOOR_M


def test_azimuth_row_above_the_floor_is_exact_inverse_horizontal_range():
    """MS-23: above the floor the azimuth row is the analytic 1/rho, uncapped."""
    rho = 2 * AZIMUTH_ROW_FLOOR_M
    assert H_jacobian(_position(rho, 0.0))[0, PY] == 1.0 / rho


def test_azimuth_row_below_the_floor_is_capped_not_shrunk():
    """MS-23: a row shrinking to 0 drives an unbounded EKF correction y/|H|; capped, it is <= pi * floor."""
    rho = AZIMUTH_ROW_FLOOR_M / 2
    assert H_jacobian(_position(0.0, rho))[0, PX] == -1.0 / AZIMUTH_ROW_FLOOR_M


def test_azimuth_row_half_a_millimetre_off_overhead_is_capped_at_the_spec_floor():
    """MS-23: the floor is 2^-10 m, so at rho = 2^-11 m the row holds 2^10 rad/m and pi * floor stays ~3 mm."""
    assert H_jacobian(_position(2.0**-11, 0.0))[0, PY] == 2.0**10


def _overhead_pass():
    envelope = PRESETS[DEFAULT_PRESET]
    start = np.array([-500.0, 0.0, OVERHEAD_ALTITUDE_M])
    velocity = np.array([envelope.max_speed_m_s, 0.0, 0.0])
    trim_at_max_speed = np.array([[0.0, 1.0, 1.0]] * OVERHEAD_PASS_STEPS)
    return fly(envelope, start, velocity, 0.0, trim_at_max_speed, NOMINAL.dt)


def test_fixed_wing_overhead_pass_keeps_every_filter_state_finite():
    """MS-2/MS-23: full-throttle trim at max speed is unaccelerated, so the truth crosses rho = 0 exactly."""
    states = _overhead_pass()
    assert np.any(np.hypot(states[:, PX], states[:, PY]) == 0.0)
    for seed in OVERHEAD_PASS_SEEDS:
        z = measure(np.random.default_rng(seed), states, NOMINAL)
        for err, nees, nis in (
            run_ekf_like(EKF(NOMINAL.dt, NOMINAL.R()), states, z),
            run_ekf_like(UKF(NOMINAL.dt, NOMINAL.R()), states, z),
            run_imm(IMM(NOMINAL.dt, NOMINAL.R()), states, z),
        ):
            assert np.isfinite(err).all() and np.isfinite(nees).all() and np.isfinite(nis).all()


def test_elevation_innovation_at_the_pi_boundary_is_not_wrapped():
    """MS-34/MS-2: elevation lies in [-pi/2, pi/2], so +pi/2 against -pi/2 is an innovation of +pi, not -pi."""
    z_meas = np.array([0.0, np.pi / 2, 100.0])
    x_pred = _state_at(az=0.0, el=-np.pi / 2, r=100.0)
    assert innovation(z_meas, x_pred)[1] == np.pi


def test_azimuth_innovation_at_the_pi_boundary_wraps_to_minus_pi():
    """MS-34: angles wrap to [-pi, pi), so an azimuth innovation of exactly +pi comes back as -pi."""
    z_meas = np.array([np.pi / 2, 0.0, 100.0])
    x_pred = _state_at(az=-np.pi / 2, el=0.0, r=100.0)
    assert innovation(z_meas, x_pred)[0] == -np.pi
