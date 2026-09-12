import numpy as np
import torch

from filters.knet.rkn_network import RKNGainCovNetwork
from filters.knet.rkn_filter import RKNFilter
from filters.torch_ops import CVMotionModel, H_jacobian_torch, h_torch
from filters.measurement_model import h
from filters.state import STATE_DIM


def _p0(batch):
    return torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0).repeat(batch, 1, 1) * 100.0


def test_rkn_forward_pass_is_finite_and_p_stays_valid():
    torch.manual_seed(0)
    net = RKNGainCovNetwork().double()
    flt = RKNFilter(net, dt=0.02)

    batch = 3
    x0 = torch.zeros(batch, STATE_DIM, dtype=torch.float64)
    x0[:, 0] = 1000.0
    x0[:, 1] = 200.0
    x0[:, 2] = 50.0
    flt.init_sequence(x0, _p0(batch))

    for t in range(5):
        z = torch.tensor(
            np.stack([h(x0[b].numpy() + t) for b in range(batch)]), dtype=torch.float64
        )
        x_upd, P_upd = flt.step(z)
        assert x_upd.shape == (batch, STATE_DIM)
        assert P_upd.shape == (batch, STATE_DIM, STATE_DIM)
        assert torch.all(torch.isfinite(x_upd))
        assert torch.all(torch.isfinite(P_upd))
        for b in range(batch):
            P_b = P_upd[b].detach().numpy()
            assert np.allclose(P_b, P_b.T, atol=1e-6)
            assert np.all(np.linalg.eigvalsh(P_b) > -1e-6)


def test_gradient_flows_through_both_networks():
    torch.manual_seed(1)
    net = RKNGainCovNetwork().double()
    flt = RKNFilter(net, dt=0.02)

    batch = 2
    x0 = torch.zeros(batch, STATE_DIM, dtype=torch.float64)
    x0[:, 0] = 1500.0
    x0[:, 2] = 100.0
    flt.init_sequence(x0, _p0(batch))

    loss = torch.tensor(0.0, dtype=torch.float64)
    x_true = x0.clone()
    for _ in range(4):
        x_true = x_true.clone()
        x_true[:, 0] += 20.0 * 0.02
        z = torch.stack([
            torch.tensor(h(x_true[b].detach().numpy()), dtype=torch.float64) for b in range(batch)
        ])
        x_upd, P_upd = flt.step(z)
        e = (x_upd - x_true).unsqueeze(-1)
        nll = torch.mean(torch.bmm(torch.bmm(e.transpose(1, 2), torch.linalg.inv(P_upd)), e))
        loss = loss + nll

    loss.backward()
    grad_norms_K = [p.grad.norm().item() for p in net.rnn_K.parameters() if p.grad is not None]
    grad_norms_cov = [p.grad.norm().item() for p in net.rnn_cov.parameters() if p.grad is not None]
    assert len(grad_norms_K) > 0 and len(grad_norms_cov) > 0
    assert all(np.isfinite(g) for g in grad_norms_K + grad_norms_cov)
    assert any(g > 0 for g in grad_norms_K)
    assert any(g > 0 for g in grad_norms_cov)


def test_rkn_survives_azimuth_wrap_boundary():
    """Mirrors the equivalent vanilla-KalmanNet test -- the reference this
    was ported from never wraps angles, so this is where a regression could
    silently reappear on RKN's own feature computation (z_diff)."""
    torch.manual_seed(2)
    net = RKNGainCovNetwork().double()
    flt = RKNFilter(net, dt=0.02)

    r, el = 100.0, 0.0

    def state_at(az):
        x = np.zeros(STATE_DIM)
        x[0] = r * np.cos(el) * np.cos(az)
        x[1] = r * np.cos(el) * np.sin(az)
        x[2] = r * np.sin(el)
        return x

    x0 = torch.tensor(state_at(np.pi - 0.01), dtype=torch.float64).unsqueeze(0)
    flt.init_sequence(x0, _p0(1))

    z0 = torch.tensor(h(state_at(np.pi - 0.01)), dtype=torch.float64).unsqueeze(0)
    z1 = torch.tensor(h(state_at(-np.pi + 0.01)), dtype=torch.float64).unsqueeze(0)

    x1, _ = flt.step(z0)
    x2, P2 = flt.step(z1)

    assert torch.all(torch.isfinite(x2))
    assert torch.all(torch.isfinite(P2))
    assert torch.norm(x2[:, :2] - x1[:, :2]).item() < 50.0


class _FixedGain:
    """Network stub returning a prescribed K and B, so the covariance recursion
    can be checked against a closed-form oracle instead of an untrained net."""

    def __init__(self, K, B):
        self.K, self.B = K, B

    def init_hidden(self, batch, device):
        return (torch.zeros(1, dtype=torch.float64),)

    def __call__(self, features, hidden):
        return self.K, self.B, hidden


def test_optimal_gain_joseph_update_shrinks_covariance_to_simple_form():
    """Joseph form with the *optimal* Kalman gain must collapse to the simple
    form (I - KH)P-, the identity that makes a measurement shrink uncertainty.
    Flip the sign in the (I - KH) factor and the update inflates it instead.
    """

    dt = 0.02
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor([1200.0, -300.0, 80.0], dtype=torch.float64)
    x0[0, 3:6] = torch.tensor([40.0, 10.0, -2.0], dtype=torch.float64)
    P0 = _p0(1)

    f = CVMotionModel(dt)
    F = f.F.unsqueeze(0)
    P_minus = F @ P0 @ F.transpose(1, 2)
    H = H_jacobian_torch(f(x0))
    R = torch.diag(torch.tensor([4e-6, 4e-6, 25.0], dtype=torch.float64)).unsqueeze(0)

    S = H @ P_minus @ H.transpose(1, 2) + R
    K = P_minus @ H.transpose(1, 2) @ torch.linalg.inv(S)
    B = K @ R @ K.transpose(1, 2)

    flt = RKNFilter(_FixedGain(K, B), dt=dt)
    flt.init_sequence(x0, P0)
    _, P_upd = flt.step(torch.tensor(h(x0[0].numpy()), dtype=torch.float64).unsqueeze(0))

    eye = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0)
    expected = (eye - K @ H) @ P_minus

    # Exact algebra in float64: measured residual is 7e-15 on entries of order
    # 1e2. 1e-12 is two decades of headroom for a different BLAS, no more.
    assert torch.allclose(P_upd, expected, rtol=0, atol=1e-12)
    assert torch.trace(P_upd[0]) < torch.trace(P_minus[0])


def test_joseph_form_holds_for_a_suboptimal_gain():
    """The gain is learned, so it is not the optimal one -- and under the optimal
    gain Joseph and the simple form (I-KH)P- coincide by identity, which makes
    the requirement vacuous there. Pin it where the two actually differ.
    """
    dt = 0.02
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor([640.0, 410.0, -95.0], dtype=torch.float64)
    x0[0, 3:6] = torch.tensor([12.0, -7.0, 3.0], dtype=torch.float64)
    P0 = _p0(1)

    f = CVMotionModel(dt)
    F = f.F.unsqueeze(0)
    P_minus = F @ P0 @ F.transpose(1, 2)
    H = H_jacobian_torch(f(x0))

    torch.manual_seed(7)
    K = torch.randn(1, STATE_DIM, 3, dtype=torch.float64) * 5.0
    B = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0) * 0.75

    flt = RKNFilter(_FixedGain(K, B), dt=dt)
    flt.init_sequence(x0, P0)
    _, P_upd = flt.step(torch.tensor(h(x0[0].numpy()), dtype=torch.float64).unsqueeze(0))

    eye = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0)
    IKH = eye - K @ H
    joseph = IKH @ P_minus @ IKH.transpose(1, 2) + B
    simple = IKH @ P_minus

    assert torch.allclose(P_upd, joseph, rtol=0, atol=1e-9)
    # Without this the test would pass on the simple form too, i.e. assert nothing.
    assert not torch.allclose(P_upd, simple, rtol=0, atol=1.0)
    # Joseph's reason for existing: symmetric and PSD for any gain, optimal or not.
    assert torch.allclose(P_upd, P_upd.transpose(1, 2), rtol=0, atol=1e-9)
    assert torch.linalg.eigvalsh(P_upd[0]).min() > 0.0


def test_state_update_adds_the_gain_weighted_innovation():
    """x+ = x- + K(z - h(x-)). Both signs are load-bearing: flipping the
    innovation or the correction still yields a finite, plausible state, so
    only an exact oracle separates them. z is chosen off the wrap boundary so
    the expected innovation is a plain subtraction.
    """
    dt = 0.02
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor([900.0, 250.0, -40.0], dtype=torch.float64)
    x0[0, 3:6] = torch.tensor([-15.0, 30.0, 5.0], dtype=torch.float64)

    f = CVMotionModel(dt)
    x_prior = f(x0)
    # Distinct rows, so a permuted or row-mixed gain cannot pass: every state
    # component must receive its own row of K, not a shared correction.
    torch.manual_seed(11)
    K = torch.randn(1, STATE_DIM, 3, dtype=torch.float64) * 4.0
    B = torch.zeros(1, STATE_DIM, STATE_DIM, dtype=torch.float64)

    z = h_torch(x_prior) + torch.tensor([0.01, -0.02, 12.0], dtype=torch.float64)
    y = z - h_torch(x_prior)
    expected = x_prior + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)

    flt = RKNFilter(_FixedGain(K, B), dt=dt)
    flt.init_sequence(x0, _p0(1))
    x_upd, _ = flt.step(z)

    assert torch.allclose(x_upd, expected, rtol=0, atol=1e-12)
    assert not torch.allclose(x_upd, x_prior, rtol=0, atol=1e-6)


def test_feature_vector_is_the_specified_squared_quadruple():
    """RKN's network input is [innovation, z_diff, vec(H), correction], squared
    (reference Algo/RecursiveKalmanNet.py get_features). Nothing downstream of
    a learned net can observe a wrong feature, so assert the vector itself.
    Squaring hides a sign only when the two operands are equal, so the states
    below are chosen to make every difference non-degenerate.
    """
    dt = 0.02
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor([700.0, -120.0, 60.0], dtype=torch.float64)
    x0[0, 3:6] = torch.tensor([25.0, -40.0, 8.0], dtype=torch.float64)

    f = CVMotionModel(dt)
    x_prior = f(x0)
    z = h_torch(x_prior) + torch.tensor([0.003, 0.004, 9.0], dtype=torch.float64)

    expected = torch.cat([
        z - h_torch(x_prior),
        z - h_torch(x0),
        H_jacobian_torch(x_prior).reshape(1, -1),
        x0 - f(x0),
    ], dim=-1) ** 2

    seen = {}

    class _Capture(_FixedGain):
        def __call__(self, features, hidden):
            seen["features"] = features
            return super().__call__(features, hidden)

    net = _Capture(
        torch.zeros(1, STATE_DIM, 3, dtype=torch.float64),
        torch.zeros(1, STATE_DIM, STATE_DIM, dtype=torch.float64),
    )
    flt = RKNFilter(net, dt=dt)
    flt.init_sequence(x0, _p0(1))
    flt.step(z)

    assert torch.allclose(seen["features"], expected, rtol=0, atol=1e-12)
