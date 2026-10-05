"""KalmanNet trains through torch h()/f(); the classical filters evaluate
through the numpy versions. If the two diverge even slightly, KalmanNet
trains against a different model than it's judged against -- verify
agreement directly, and verify the wrap fix survives in the torch path (the
brief's #1 bug, again, on a second code path)."""

import numpy as np
import torch

from filters.measurement_model import AZIMUTH_ROW_FLOOR_M, H_jacobian, h, innovation, wrap_angle
from filters.state import STATE_DIM
from filters.torch_ops import CVMotionModel, H_jacobian_torch, h_torch, innovation_torch, wrap_angle_torch
from filters.motion_models import cv_model, step


def test_wrap_angle_torch_matches_numpy():
    angles = np.linspace(-10 * np.pi, 10 * np.pi, 2001)
    np_wrapped = wrap_angle(angles)
    torch_wrapped = wrap_angle_torch(torch.tensor(angles)).numpy()
    assert np.allclose(np_wrapped, torch_wrapped, atol=1e-10)


def test_h_torch_matches_numpy_on_random_states():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = np.zeros(STATE_DIM)
        x[:3] = rng.uniform(-2000, 2000, size=3)
        z_np = h(x)
        z_torch = h_torch(torch.tensor(x)).numpy()
        assert np.allclose(z_np, z_torch, atol=1e-8)


def _h_both_at(px, py, pz):
    x = np.zeros(STATE_DIM)
    x[:3] = px, py, pz
    return h(x), h_torch(torch.tensor(x)).numpy()


def test_h_torch_equals_numpy_directly_overhead():
    """MS-23: overhead, azimuth reads 0, elevation pi/2 and range |pz|, with no rho floor in either path."""
    z_np, z_torch = _h_both_at(0.0, 0.0, 300.0)
    assert np.array_equal(z_torch, z_np)
    assert np.array_equal(z_torch, [0.0, np.pi / 2, 300.0])


def test_h_torch_range_equals_numpy_inside_one_millimetre_of_the_z_axis():
    """MS-23: a 1e-6 floor on rho squared would read 1 mm of range for a target 0.49 mm from the sensor."""
    z_np, z_torch = _h_both_at(2.0**-11, 0.0, 0.0)
    assert np.array_equal(z_torch, z_np)
    assert z_torch[2] == 2.0**-11


def test_h_torch_elevation_matches_numpy_inside_one_millimetre_overhead():
    """MS-23: a rho floor shifts elevation by ~1.7e-6 rad here. Torch and libm atan2 differ by ulps only,
    so the tolerance is 4 ulp relative."""
    z_np, z_torch = _h_both_at(2.0**-11, 2.0**-12, 300.0)
    np.testing.assert_allclose(z_torch, z_np, rtol=4 * np.finfo(np.float64).eps, atol=0.0)


def test_h_torch_equals_numpy_at_and_above_the_old_floor():
    """MS-23: rho = 1 mm and rho = 2 mm read their exact horizontal range in both paths."""
    for px in (1e-3, 2.0**-9):
        z_np, z_torch = _h_both_at(px, 0.0, 0.0)
        assert np.array_equal(z_torch, z_np)
        assert z_torch[2] == px


def test_h_torch_gradient_is_finite_directly_overhead():
    """MS-23: KalmanNet backpropagates through h at rho = 0, so azimuth, elevation and range stay finite."""
    x = torch.zeros(STATE_DIM, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        x[2] = 300.0
    h_torch(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_innovation_torch_matches_numpy_including_wrap_boundary():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x_true = np.zeros(STATE_DIM)
        x_true[:3] = rng.uniform(-2000, 2000, size=3)
        x_pred = np.zeros(STATE_DIM)
        x_pred[:3] = rng.uniform(-2000, 2000, size=3)
        z = h(x_true)

        innov_np = innovation(z.copy(), x_pred)
        innov_torch = innovation_torch(torch.tensor(z), torch.tensor(x_pred)).numpy()
        assert np.allclose(innov_np, innov_torch, atol=1e-8)

    # explicit wrap-boundary case, mirroring test_measurement_model.py
    r, el = 100.0, 0.0
    az_true, az_pred = np.pi - 0.01, -np.pi + 0.01
    x_true = np.array([
        r * np.cos(el) * np.cos(az_true), r * np.cos(el) * np.sin(az_true), r * np.sin(el),
        0, 0, 0, 0, 0, 0,
    ])
    x_pred = np.array([
        r * np.cos(el) * np.cos(az_pred), r * np.cos(el) * np.sin(az_pred), r * np.sin(el),
        0, 0, 0, 0, 0, 0,
    ])
    z = h(x_true)
    innov_torch = innovation_torch(torch.tensor(z), torch.tensor(x_pred)).numpy()
    assert abs(innov_torch[0]) < 0.05  # not a spurious ~2*pi jump


def test_H_jacobian_torch_matches_numpy_batched():
    rng = np.random.default_rng(2)
    xs = np.zeros((10, STATE_DIM))
    xs[:, :3] = rng.uniform(-2000, 2000, size=(10, 3))
    xs[:, 2] = rng.uniform(50, 500, size=10)  # keep pz away from the rho=0 singularity

    H_np = np.stack([H_jacobian(xs[i]) for i in range(10)])
    H_t = H_jacobian_torch(torch.tensor(xs)).numpy()
    assert np.allclose(H_np, H_t, atol=1e-8)


def test_H_jacobian_torch_matches_numpy_at_and_below_the_azimuth_floor():
    """MS-23: RKN's torch Jacobian carries the same azimuth cap and overhead zero row as the numpy one."""
    xs = np.zeros((4, STATE_DIM))
    xs[:, 2] = 300.0
    xs[:, 0] = [0.0, AZIMUTH_ROW_FLOOR_M / 2, AZIMUTH_ROW_FLOOR_M, 2 * AZIMUTH_ROW_FLOOR_M]
    H_np = np.stack([H_jacobian(x) for x in xs])
    assert np.array_equal(H_jacobian_torch(torch.tensor(xs)).numpy()[:, 0], H_np[:, 0])


def test_H_jacobian_torch_gradient_is_finite_directly_overhead():
    """MS-23: RKN backpropagates through the Jacobian, so the rho = 0 branch must not leak a 0/0 gradient."""
    x = torch.zeros(1, STATE_DIM, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        x[0, 2] = 300.0
    H_jacobian_torch(x)[:, 0].sum().backward()
    assert torch.isfinite(x.grad).all()


def test_cv_motion_model_matches_numpy_step():
    dt = 0.02
    F_np, _ = cv_model(dt)
    x = np.array([100.0, 200.0, 50.0, 10.0, -5.0, 1.0, 0.0, 0.0, 0.0])

    x1_np = step(x, F_np)
    torch_model = CVMotionModel(dt)
    x1_torch = torch_model(torch.tensor(x, dtype=torch.float64)).numpy()
    assert np.allclose(x1_np, x1_torch, atol=1e-10)
