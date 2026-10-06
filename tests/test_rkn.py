import math

import numpy as np
import torch

from filters.knet.rkn_network import RKNGainCovNetwork, normalise_groups
from filters.knet.rkn_filter import RKNFilter
from filters.torch_ops import CVMotionModel, H_jacobian_torch, h_torch
from filters.measurement_model import HORIZONTAL_RANGE_FLOOR_M, h, wrap_angle
from filters.state import STATE_DIM
from scripts.train_rkn import _gaussian_nll

DIVERGED_POSITION_VARIANCE = 1e32


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
            # P is IKH.P-.IKH^T + L.L^T -- symmetric PSD by construction, not
            # merely approximately so. Measured residual is 6e-15 on entries
            # of order 1e2; 1e-12 is two decades of headroom, no more.
            assert np.allclose(P_b, P_b.T, rtol=0, atol=1e-12)
            assert np.linalg.eigvalsh(P_b).min() > 0.0


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
    """Smoke test: a real (untrained) network stays numerically sane across the
    boundary. The wrap requirement itself is pinned exactly by
    test_innovation_wraps_across_the_azimuth_boundary and its z_diff twin --
    a magnitude bound under a random gain cannot separate wrapped from not."""
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
    """Network stub returning a prescribed K and factor L of B = L L^T, so the covariance
    recursion can be checked against a closed-form oracle instead of an untrained net."""

    def __init__(self, K, L):
        self.K, self.L = K, L

    def init_hidden(self, batch, device):
        return (torch.zeros(1, dtype=torch.float64),)

    def __call__(self, features, hidden):
        return self.K, self.L, hidden


class _CaptureFeatures(_FixedGain):
    """Records one feature row per step. Zero gain by default, so the state
    stays pinned at the prior across steps; pass K to also exercise a
    genuinely moving state (needed once "recursive" must be told apart from
    "state never changes")."""

    def __init__(self, K=None):
        if K is None:
            K = torch.zeros(1, STATE_DIM, 3, dtype=torch.float64)
        super().__init__(K, torch.zeros(1, STATE_DIM, STATE_DIM, dtype=torch.float64))
        self.seen = []

    def __call__(self, features, hidden):
        self.seen.append(features)
        return super().__call__(features, hidden)


def _state_at_azimuth(az, r=800.0):
    x = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x[0, 0] = r * torch.cos(torch.tensor(az, dtype=torch.float64))
    x[0, 1] = r * torch.sin(torch.tensor(az, dtype=torch.float64))
    return x


def test_optimal_gain_joseph_update_shrinks_covariance_to_simple_form():
    """MS-10: Joseph form with the *optimal* Kalman gain must collapse to the simple
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
    L = K @ torch.sqrt(R)

    flt = RKNFilter(_FixedGain(K, L), dt=dt)
    flt.init_sequence(x0, P0)
    _, P_upd = flt.step(torch.tensor(h(x0[0].numpy()), dtype=torch.float64).unsqueeze(0))

    eye = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0)
    expected = (eye - K @ H) @ P_minus

    # Exact algebra in float64: measured residual is 7e-14 on entries of order
    # 1e2 through the QR factor. 1e-12 is ~14x headroom for a different BLAS.
    assert torch.allclose(P_upd, expected, rtol=0, atol=1e-12)
    assert torch.trace(P_upd[0]) < torch.trace(P_minus[0])


def test_joseph_form_holds_for_a_suboptimal_gain():
    """MS-10: the gain is learned, so it is not the optimal one -- and under the optimal
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
    L = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0) * 0.75 ** 0.5
    B = L @ L.transpose(1, 2)

    flt = RKNFilter(_FixedGain(K, L), dt=dt)
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


def test_learned_factor_adds_L_L_transpose_not_L_transpose_L():
    """MS-10: P+ = A + L L^T, the network's B (#14 sec. 2); a non-normal L separates it from L^T L,
    579 apart. The QR factor reproduces it to 2.3e-13 at |P| <= 7e2; atol 1e-11 is ~40x headroom."""
    dt = 0.02
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor([300.0, -600.0, 40.0], dtype=torch.float64)
    x0[0, 3:6] = torch.tensor([5.0, 15.0, -3.0], dtype=torch.float64)
    P0 = _p0(1)

    f = CVMotionModel(dt)
    F = f.F.unsqueeze(0)
    A_expected = F @ P0 @ F.transpose(1, 2)

    K = torch.zeros(1, STATE_DIM, 3, dtype=torch.float64)
    idx = torch.arange(STATE_DIM, dtype=torch.float64)
    L = torch.tril(idx.unsqueeze(1) * STATE_DIM + idx.unsqueeze(0) + 1.0).unsqueeze(0) / STATE_DIM

    flt = RKNFilter(_FixedGain(K, L), dt=dt)
    flt.init_sequence(x0, P0)
    _, P_upd = flt.step(torch.tensor(h(x0[0].numpy()), dtype=torch.float64).unsqueeze(0))

    assert torch.allclose(P_upd, A_expected + L @ L.transpose(1, 2), rtol=0, atol=1e-11)
    assert not torch.allclose(P_upd, A_expected + L.transpose(1, 2) @ L, rtol=0, atol=1.0)


def test_covariance_factor_stays_positive_definite_where_the_closed_form_goes_indefinite():
    """MS-10: a diverged run's 1e32 m^2 position variance next to unit ones (#62). B = I bounds lambda_min(P)
    by 1, yet float64 rounds the closed-form A + B indefinite. Triangular S has S_jj >= sigma_min(S) =
    sqrt(lambda_min(P)) >= 1 in exact arithmetic; measured min 1.21."""
    dt = 0.02
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor([640.0, 410.0, -95.0], dtype=torch.float64)
    P0 = torch.diag(torch.tensor([DIVERGED_POSITION_VARIANCE] * 3 + [1.0] * 6, dtype=torch.float64)).unsqueeze(0)
    torch.manual_seed(7)
    K = torch.randn(1, STATE_DIM, 3, dtype=torch.float64) * 5.0
    L = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0)
    f = CVMotionModel(dt)
    F = f.F.unsqueeze(0)
    IKH = torch.eye(STATE_DIM, dtype=torch.float64) - K @ H_jacobian_torch(f(x0))
    closed_form = IKH @ F @ P0 @ F.transpose(1, 2) @ IKH.transpose(1, 2) + L @ L.transpose(1, 2)

    flt = RKNFilter(_FixedGain(K, L), dt=dt)
    flt.init_sequence(x0, P0)
    flt.step(h_torch(f(x0)))

    assert torch.linalg.cholesky_ex(closed_form).info.item() != 0
    S_diagonal = torch.diagonal(flt.S, dim1=-2, dim2=-1)
    assert torch.all(torch.isfinite(S_diagonal)) and torch.all(S_diagonal >= 1.0)
    assert torch.isfinite(_gaussian_nll(torch.ones(1, STATE_DIM, dtype=torch.float64), flt.S))


def test_covariance_factor_is_lower_triangular_with_a_positive_diagonal():
    """MS-10: the NLL reads S as a Cholesky factor, triangular solve and log of its diagonal. Householder
    QR leaves 8 of R's 9 diagonal entries negative here, all of magnitude below 1 (P0 = 1e-4 I)."""
    dt = 0.02
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor([640.0, 410.0, -95.0], dtype=torch.float64)
    torch.manual_seed(7)
    K = torch.randn(1, STATE_DIM, 3, dtype=torch.float64) * 5.0
    L = torch.eye(STATE_DIM, dtype=torch.float64).unsqueeze(0) * 1e-3

    flt = RKNFilter(_FixedGain(K, L), dt=dt)
    flt.init_sequence(x0, _p0(1) * 1e-6)
    flt.step(h_torch(CVMotionModel(dt)(x0)))

    assert torch.equal(flt.S, torch.tril(flt.S))
    assert torch.all(torch.diagonal(flt.S, dim1=-2, dim2=-1) > 0.0)


def test_state_update_adds_the_gain_weighted_innovation():
    """MS-10: x+ = x- + K(z - h(x-)). Both signs are load-bearing: flipping the
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
    L = torch.zeros(1, STATE_DIM, STATE_DIM, dtype=torch.float64)

    z = h_torch(x_prior) + torch.tensor([0.01, -0.02, 12.0], dtype=torch.float64)
    y = z - h_torch(x_prior)
    expected = x_prior + torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)

    flt = RKNFilter(_FixedGain(K, L), dt=dt)
    flt.init_sequence(x0, _p0(1))
    x_upd, _ = flt.step(z)

    assert torch.allclose(x_upd, expected, rtol=0, atol=1e-12)
    assert not torch.allclose(x_upd, x_prior, rtol=0, atol=1e-6)


def test_feature_vector_is_the_specified_signed_quadruple():
    """MS-10: the network input is [innovation, z_diff, vec(H), correction] (reference get_features), signed:
    each branch encodes it its own way (#70). Nothing downstream of a learned net can observe a wrong
    feature, so assert the vector itself."""
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
    ], dim=-1)

    net = _CaptureFeatures()
    flt = RKNFilter(net, dt=dt)
    flt.init_sequence(x0, _p0(1))
    flt.step(z)
    seen = {"features": net.seen[0]}

    assert torch.allclose(seen["features"], expected, rtol=0, atol=1e-12)


def test_feature_correction_is_recursive_at_the_second_step():
    """MS-10: correction = self.x - self.x_prior_previous must carry the prior saved
    from the *previous* step, not the prior just recomputed this step. At
    t=1 those coincide (x_prior_previous is seeded from x0 at init), so an
    implementation that stores this step's own state instead of its prior
    only diverges from step 2 onward.
    """
    dt = 0.02
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor([700.0, -120.0, 60.0], dtype=torch.float64)
    x0[0, 3:6] = torch.tensor([25.0, -40.0, 8.0], dtype=torch.float64)

    f = CVMotionModel(dt)
    torch.manual_seed(23)
    K = torch.randn(1, STATE_DIM, 3, dtype=torch.float64) * 3.0

    x_prior1 = f(x0)
    z1 = h_torch(x_prior1) + torch.tensor([0.004, -0.003, 15.0], dtype=torch.float64)
    y1 = z1 - h_torch(x_prior1)
    x_upd1 = x_prior1 + torch.bmm(K, y1.unsqueeze(-1)).squeeze(-1)

    z2 = z1 + torch.tensor([0.002, -0.001, 5.0], dtype=torch.float64)
    x_prior2 = f(x_upd1)
    y2 = z2 - h_torch(x_prior2)
    z_diff2 = z2 - z1
    correction2 = x_upd1 - x_prior1
    expected2 = torch.cat(
        [y2, z_diff2, H_jacobian_torch(x_prior2).reshape(1, -1), correction2], dim=-1
    )

    net = _CaptureFeatures(K=K)
    flt = RKNFilter(net, dt=dt)
    flt.init_sequence(x0, _p0(1))
    flt.step(z1)
    flt.step(z2)

    assert torch.allclose(net.seen[1], expected2, rtol=0, atol=1e-9)
    # Without this, correcting against this step's own prior (x_prior2)
    # instead of the one saved from step 1 would also pass.
    wrong_correction2 = x_upd1 - x_prior2
    wrong_expected2 = torch.cat(
        [y2, z_diff2, H_jacobian_torch(x_prior2).reshape(1, -1), wrong_correction2], dim=-1
    )
    assert not torch.allclose(net.seen[1], wrong_expected2, rtol=0, atol=1.0)


def test_innovation_wraps_across_the_azimuth_boundary():
    """MS-34: a target at az = pi - e measured at az = -pi + e is 2e away, not 2pi - 2e.
    The reference port never wraps, so without the wrap the innovation feature
    is 2e - 2pi instead of 2e, yet still finite and symmetric, which is all the
    older tests check.
    """
    eps = 0.01
    dt = 0.02
    x0 = _state_at_azimuth(torch.pi - eps)
    z = h_torch(_state_at_azimuth(-torch.pi + eps))

    net = _CaptureFeatures()
    flt = RKNFilter(net, dt=dt)
    flt.init_sequence(x0, _p0(1))
    flt.step(z)

    innovation_az = net.seen[0][0, 0]
    assert torch.allclose(innovation_az, torch.tensor(2 * eps, dtype=torch.float64),
                          rtol=0, atol=1e-9)


def test_z_diff_feature_wraps_independently_of_the_innovation():
    """MS-34: z_diff = z_t - z_{t-1} carries azimuth too and needs its own wrap. Here
    step 2's innovation is exactly zero -- the measurement matches the prior --
    so only z_diff crosses the boundary, and an implementation that wraps the
    innovation but not z_diff is separated from a correct one.
    """
    eps = 0.01
    dt = 0.02
    x0 = _state_at_azimuth(torch.pi - eps)
    z_far = h_torch(_state_at_azimuth(-torch.pi + eps))
    z_here = h_torch(x0)

    net = _CaptureFeatures()
    flt = RKNFilter(net, dt=dt)
    flt.init_sequence(x0, _p0(1))
    flt.step(z_far)
    flt.step(z_here)

    innovation_az, z_diff_az = net.seen[1][0, 0], net.seen[1][0, 3]
    assert torch.allclose(innovation_az, torch.zeros((), dtype=torch.float64),
                          rtol=0, atol=1e-18)
    assert torch.allclose(z_diff_az, torch.tensor(-2 * eps, dtype=torch.float64),
                          rtol=0, atol=1e-9)


def test_hidden_state_is_threaded_between_steps():
    """MS-10: "Recursive" is the requirement: step t must receive the hidden state step
    t-1 returned, not a freshly initialized one. Re-initializing every step
    leaves every other assertion in this file satisfied.
    """
    class _Recurrent(_FixedGain):
        def __init__(self):
            super().__init__(
                torch.zeros(1, STATE_DIM, 3, dtype=torch.float64),
                torch.zeros(1, STATE_DIM, STATE_DIM, dtype=torch.float64),
            )
            self.received, self.returned, self.inits = [], [], 0

        def init_hidden(self, batch, device):
            self.inits += 1
            return (torch.zeros(1, dtype=torch.float64),)

        def __call__(self, features, hidden):
            self.received.append(hidden)
            nxt = (hidden[0] + 1.0,)
            self.returned.append(nxt)
            return self.K, self.L, nxt

    dt = 0.02
    x0 = _state_at_azimuth(0.4)
    net = _Recurrent()
    flt = RKNFilter(net, dt=dt)
    flt.init_sequence(x0, _p0(1))
    for _ in range(3):
        flt.step(h_torch(x0))

    assert net.inits == 1
    assert net.received[0][0].item() == 0.0
    for t in range(1, 3):
        assert net.received[t][0] is net.returned[t - 1][0]


MIRROR_X_Y = torch.tensor([-1.0, -1.0, 1.0] * 3, dtype=torch.float64)
FEATURE_DIM = 42
FEATURE_SLICES = {"H": slice(6, 33), "correction": slice(33, 42)}
MIRROR_FEATURE_ATOL = 1e-13
MIRROR_GAIN_RTOL = 2e-15
MIRROR_WEIGHT_SCALE = 0.1
MIRROR_STEPS = 4
MIRROR_NOISE = torch.tensor([0.003, 0.004, 9.0], dtype=torch.float64)
RUNAWAY_FEATURE_SCALE = 1e17
LOS_TEST_RANGE_M = 1024.0
FAR_RANGE_M = 2.0**20
LEVEL_POSITION_M = (700.0, -120.0, 60.0)
ELEVATED_POSITION_M = (300.0, -400.0, 1200.0)
LOOP_GAIN_ATOL = 4e-14
UNDER_FLOOR_RHO_M = 5e-4
ABOVE_FLOOR_RHO_M = 2.0**-9
ABOVE_FLOOR_RTOL = 1e-14
UNDER_FLOOR_RANGE_RTOL = 1e-12


def _exact_raw_features():
    raw = torch.zeros(1, FEATURE_DIM, dtype=torch.float64)
    raw[0, 0:2] = torch.tensor([3.0, -4.0], dtype=torch.float64)
    raw[0, 6] = -600.0
    raw[0, 20] = 800.0
    raw[0, 35] = -0.5
    return raw


def _exact_gain_features():
    expected = torch.zeros(1, FEATURE_DIM, dtype=torch.float64)
    expected[0, 0:2] = torch.tensor([0.6, -0.8], dtype=torch.float64)
    expected[0, 6] = -0.6
    expected[0, 20] = 0.8
    expected[0, 35] = -1.0
    return expected


def _record_input(seen, key):
    def hook(_, args):
        seen[key] = args[0]

    return hook


def _branch_inputs(raw):
    net = RKNGainCovNetwork(fc1_mult=1, hidden_mult=1, fc2_mult=1).double()
    seen = {}
    net.rnn_K.register_forward_pre_hook(_record_input(seen, "K"))
    net.rnn_cov.register_forward_pre_hook(_record_input(seen, "cov"))
    with torch.no_grad():
        net(raw, net.init_hidden(1, "cpu"))
    return seen


def test_gain_features_unit_normalise_each_group_keeping_signs_and_leave_a_zero_group_zero():
    """MS-10, #70: per-group L2 like KalmanNet's features, so groups 1e3 apart in scale both reach unit norm.
    Inputs are exact in float64 (3-4-5 triangles), so equality is exact; z_diff = 0 pins the eps guard."""
    assert torch.equal(normalise_groups(_exact_raw_features()), _exact_gain_features())


def _feature_groups(features):
    return features[0].split((3, 3, 27, 9))


def test_covariance_network_reads_squares_normalised_per_group():
    """MS-36, #86: square first, then normalise, so 3 : -4 reaches rnn_cov as 9 : 16 on the unit sphere.
    atol 1e-16 is one rounding of an entry below 1."""
    raw = _exact_raw_features()
    squares = _feature_groups(raw ** 2)
    expected = torch.cat([group / max(group.norm().item(), 1e-12) for group in squares]).unsqueeze(0)
    assert torch.allclose(_branch_inputs(raw)["cov"], expected, rtol=0, atol=1e-16)


def test_covariance_features_stay_on_the_unit_sphere_for_a_runaway_state_at_1e17():
    """MS-36, #86: a runaway state fed rnn_cov features of 1e6-1e17 (#70). Each nonzero group must reach it with norm 1
    and no negative entry; atol 1e-15 is a few roundings of a unit-norm sum of 27 squares."""
    raw = _exact_raw_features() * RUNAWAY_FEATURE_SCALE
    groups = _feature_groups(_branch_inputs(raw)["cov"])
    assert all(torch.all(group >= 0.0) for group in groups)
    norms = torch.stack([group.norm() for group in groups])
    assert torch.allclose(norms, torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64), rtol=0, atol=1e-15)


def test_runaway_state_reaches_rnn_cov_with_the_ratios_of_the_same_state_at_unit_scale():
    """MS-36, #86: only the direction of each group may survive, so 1e17 x the features reads as 1 x them.
    1e17 is not a power of two, so the squares round differently: atol 1e-16 is one rounding of an entry below 1."""
    raw = _exact_raw_features()
    runaway = _branch_inputs(raw * RUNAWAY_FEATURE_SCALE)["cov"]
    assert torch.allclose(runaway, _branch_inputs(raw)["cov"], rtol=0, atol=1e-16)


def _mirror_run(mirror, position=LEVEL_POSITION_M):
    """MIRROR_STEPS steps on a geometry, or its mirror through the z axis: x,y negated, azimuth shifted by pi."""
    x0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x0[0, :3] = torch.tensor(position, dtype=torch.float64)
    x0[0, 3:6] = torch.tensor([25.0, -40.0, 8.0], dtype=torch.float64)
    f = CVMotionModel(0.02)
    truth, measurements = x0, []
    for step in range(MIRROR_STEPS):
        truth = f(truth)
        measurements.append(h_torch(truth) + MIRROR_NOISE * (-1.0) ** step)
    if mirror:
        x0 = x0 * MIRROR_X_Y
        for z in measurements:
            z[0, 0] = wrap_angle(z[0, 0] + torch.pi)
    torch.manual_seed(5)
    net = RKNGainCovNetwork(fc1_mult=1, hidden_mult=1, fc2_mult=1).double()
    for parameter in net.parameters():
        torch.nn.init.uniform_(parameter, -MIRROR_WEIGHT_SCALE, MIRROR_WEIGHT_SCALE)
    seen = {"features": [], "K": []}
    net.rnn_K.register_forward_pre_hook(lambda _, args: seen["features"].append(args[0]))
    net.register_forward_hook(lambda _, args, out: seen["K"].append(out[0]))
    flt = RKNFilter(net, dt=0.02)
    flt.init_sequence(x0, _p0(1))
    with torch.no_grad():
        for z in measurements:
            flt.step(z)
    return seen


def test_mirrored_geometry_feeds_rnn_K_the_same_input():
    """MS-37, #85: in the line-of-sight frame H and the correction no longer flip sign under the mirror, so
    the gain network cannot tell the two apart. az + pi rounds h(), and the correction x - x_prior cancels the
    700 m state: measured 1.2e-14 by step 4, atol 1e-13 is about one ulp of that state."""
    original, mirrored = _mirror_run(False)["features"], _mirror_run(True)["features"]
    for step in range(MIRROR_STEPS):
        assert torch.allclose(mirrored[step], original[step], rtol=0, atol=MIRROR_FEATURE_ATOL)


def test_mirrored_geometry_gets_the_mirrored_gain_at_every_step():
    """MS-37, #85: K' = diag(mirror) K by construction, untrained weights included (#84 trained: |K' - MK|/|K|
    median 1.0). Only az + pi rounding separates the runs: measured 2.7e-16 of max|K| by step 4, rtol 2e-15 ~7x."""
    original, mirrored = _mirror_run(False)["K"], _mirror_run(True)["K"]
    for step in range(MIRROR_STEPS):
        expected = MIRROR_X_Y.view(1, STATE_DIM, 1) * original[step]
        assert original[step].abs().max() > 0.0
        assert (mirrored[step] - expected).abs().max() <= MIRROR_GAIN_RTOL * original[step].abs().max()


def test_mirrored_elevated_geometry_gets_the_mirrored_gain_at_every_step():
    """MS-37, #85: 67 deg up, horizontal and slant range differ (500 m vs 1300 m) and u_el leans on x,y; the
    mirror still gives K' = diag(mirror) K. Measured 1.4e-16 of max|K| by step 4, rtol 2e-15 as on the level run."""
    original = _mirror_run(False, ELEVATED_POSITION_M)["K"]
    mirrored = _mirror_run(True, ELEVATED_POSITION_M)["K"]
    for step in range(MIRROR_STEPS):
        expected = MIRROR_X_Y.view(1, STATE_DIM, 1) * original[step]
        assert original[step].abs().max() > 0.0
        assert (mirrored[step] - expected).abs().max() <= MIRROR_GAIN_RTOL * original[step].abs().max()


def _features_at(position, correction=None):
    x = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    x[0, :3] = torch.tensor(position, dtype=torch.float64)
    features = torch.zeros(1, FEATURE_DIM, dtype=torch.float64)
    features[0, FEATURE_SLICES["H"]] = H_jacobian_torch(x).reshape(-1)
    if correction is not None:
        features[0, FEATURE_SLICES["correction"]] = torch.tensor(correction, dtype=torch.float64)
    return features


def _los_rotation(position):
    """Rows u_az, u_el, u_r from the position itself, not from H."""
    px, py, pz = position
    rho, r = math.hypot(px, py), math.hypot(px, py, pz)
    return torch.tensor(
        [[-py / rho, px / rho, 0.0], [-pz * px / (rho * r), -pz * py / (rho * r), rho / r], [px / r, py / r, pz / r]],
        dtype=torch.float64,
    )


def _gain_for_los_output(los_gain, features):
    """rnn_K's last layer zeroed but for its bias, so it emits los_gain exactly whatever the GRU does."""
    net = RKNGainCovNetwork(fc1_mult=1, hidden_mult=1, fc2_mult=1).double()
    with torch.no_grad():
        net.rnn_K.fc2[-1].weight.zero_()
        net.rnn_K.fc2[-1].bias.copy_(los_gain.reshape(-1))
        return net(features, net.init_hidden(1, "cpu"))[0][0]


def test_gain_network_reads_the_correction_in_the_line_of_sight_frame():
    """MS-37, #85: on the +x axis the azimuth, elevation and range axes are +y, +z and +x, so a position
    correction (3, -4, 0) reads (-4, 0, 3), normalised to (-0.8, 0, 0.6). Exact in float64."""
    correction = [3.0, -4.0, 0.0] + [0.0] * 6
    seen = _branch_inputs(_features_at([LOS_TEST_RANGE_M, 0.0, 0.0], correction))["K"]
    expected = torch.tensor([-0.8, 0.0, 0.6] + [0.0] * 6, dtype=torch.float64)
    assert torch.equal(seen[0, FEATURE_SLICES["correction"]], expected)


def test_los_gain_rows_land_on_the_cartesian_axes_of_the_line_of_sight():
    """MS-37, #85: on the +x axis LOS rows (az, el, range) are Cartesian (y, z, x) in every 3-block, and the
    angle columns scale by horizontal and slant range (both 1024 m here), the range column by 1. Exact."""
    los_gain = torch.arange(1.0, STATE_DIM * 3 + 1.0, dtype=torch.float64).view(STATE_DIM, 3)
    K = _gain_for_los_output(los_gain, _features_at([LOS_TEST_RANGE_M, 0.0, 0.0]))
    angle_scale = torch.tensor([LOS_TEST_RANGE_M, LOS_TEST_RANGE_M, 1.0], dtype=torch.float64)
    assert torch.equal(K, los_gain[[2, 0, 1, 5, 3, 4, 8, 6, 7]] * angle_scale)


def test_identity_los_gain_fully_corrects_position_whatever_the_range_and_bearing():
    """MS-37, #85: K H = blockdiag(Rᵀ) N P blockdiag(R), so the error loop depends on N alone. N = I on
    position must give K H = I on position at 1024 m on +x and at 2^20 m on -y. Exact in float64."""
    los_gain = torch.zeros(STATE_DIM, 3, dtype=torch.float64)
    los_gain[:3] = torch.eye(3, dtype=torch.float64)
    expected = torch.zeros(STATE_DIM, STATE_DIM, dtype=torch.float64)
    expected[:3, :3] = torch.eye(3, dtype=torch.float64)
    for position in ([LOS_TEST_RANGE_M, 0.0, 0.0], [0.0, -FAR_RANGE_M, 0.0]):
        features = _features_at(position)
        H = features[0, FEATURE_SLICES["H"]].view(3, STATE_DIM)
        assert torch.equal(_gain_for_los_output(los_gain, features) @ H, expected)


def test_target_within_the_horizontal_range_floor_takes_no_angle_correction_and_stays_finite():
    """MS-37, #85: below rho = 1 mm H floors rho, u_el collapses onto u_r and r²/1e-3 would scale the elevation
    column (5e10 here). Both angles are undefined there, so both columns are exactly 0, overhead and at 0.5 mm."""
    los_gain = torch.arange(1.0, STATE_DIM * 3 + 1.0, dtype=torch.float64).view(STATE_DIM, 3)
    for position in ([0.0, 0.0, LOS_TEST_RANGE_M], [UNDER_FLOOR_RHO_M, 0.0, LOS_TEST_RANGE_M]):
        K = _gain_for_los_output(los_gain, _features_at(position))
        assert torch.all(torch.isfinite(K))
        assert torch.equal(K[:, :2], torch.zeros(STATE_DIM, 2, dtype=torch.float64))


def test_target_within_the_horizontal_range_floor_gets_a_range_column_of_n_range_rows_along_u_r():
    """MS-37, #85: below the floor R keeps only u_r, so each 3-block of K's range column is N[range, range] u_r with
    no N[el, range] term (that would add 8..26). H's floor adds 1e-6 m² to r², so its range row is 1 - 4.8e-13 long
    and K's range scale its inverse: measured 4.8e-13 relative, rtol 1e-12."""
    los_gain = torch.arange(1.0, STATE_DIM * 3 + 1.0, dtype=torch.float64).view(STATE_DIM, 3)
    for rho in (0.0, UNDER_FLOOR_RHO_M):
        position = torch.tensor([rho, 0.0, LOS_TEST_RANGE_M], dtype=torch.float64)
        K = _gain_for_los_output(los_gain, _features_at(position.tolist()))
        expected = los_gain[2::3, 2].unsqueeze(1) * position / position.norm()
        assert torch.allclose(K[:, 2].view(3, 3), expected, rtol=UNDER_FLOOR_RANGE_RTOL, atol=0)


def test_gain_network_reads_no_angle_components_within_the_horizontal_range_floor():
    """MS-37, #85: below the floor B keeps only u_r, so vec(H Bᵀ)'s and B·correction's azimuth and elevation
    entries are exactly 0 and every rnn_K input is finite, overhead and at 0.5 mm."""
    correction = [3.0, -4.0, 5.0, 1.0, 2.0, -6.0, 0.5, -0.25, 7.0]
    for position in ([0.0, 0.0, LOS_TEST_RANGE_M], [UNDER_FLOOR_RHO_M, 0.0, LOS_TEST_RANGE_M]):
        seen = _branch_inputs(_features_at(position, correction))["K"][0]
        assert torch.all(torch.isfinite(seen))
        los_H = seen[FEATURE_SLICES["H"]].view(3, 3, 3)
        los_correction = seen[FEATURE_SLICES["correction"]].view(3, 3)
        assert torch.equal(los_H[:, :, :2], torch.zeros(3, 3, 2, dtype=torch.float64))
        assert torch.equal(los_correction[:, :2], torch.zeros(3, 2, dtype=torch.float64))
        assert torch.all(los_correction[:, 2] != 0.0)


def test_angle_columns_at_and_above_the_horizontal_range_floor_scale_by_horizontal_and_slant_range():
    """MS-37, #85: at rho = 1 mm H's floor is not yet active (azimuth row norm exactly 1000), and at 2 mm R is a
    rotation: K's angle columns are blockdiag(Rᵀ) N's times rho and r. Measured 2.7e-16 of max, rtol 1e-14."""
    los_gain = torch.arange(1.0, STATE_DIM * 3 + 1.0, dtype=torch.float64).view(STATE_DIM, 3)
    for rho in (HORIZONTAL_RANGE_FLOOR_M, ABOVE_FLOOR_RHO_M):
        position = (rho, 0.0, LOS_TEST_RANGE_M)
        K = _gain_for_los_output(los_gain, _features_at(list(position)))
        scale = torch.tensor([rho, math.hypot(rho, LOS_TEST_RANGE_M)], dtype=torch.float64)
        expected = torch.kron(torch.eye(3, dtype=torch.float64), _los_rotation(position)).T @ los_gain[:, :2] * scale
        assert torch.allclose(K[:, :2], expected, rtol=ABOVE_FLOOR_RTOL, atol=0)


def test_gain_gradients_stay_finite_for_a_target_directly_overhead():
    """MS-37, #85: the zero azimuth row must not reach 1/norm on the backward pass either."""
    net = RKNGainCovNetwork(fc1_mult=1, hidden_mult=1, fc2_mult=1).double()
    features = _features_at([0.0, 0.0, LOS_TEST_RANGE_M]).requires_grad_()
    net(features, net.init_hidden(1, "cpu"))[0].sum().backward()
    assert torch.all(torch.isfinite(features.grad))
    assert all(torch.all(torch.isfinite(p.grad)) for p in net.rnn_K.parameters())


def test_non_identity_los_gain_closes_the_position_loop_as_rotated_n_at_an_elevated_off_axis_target():
    """MS-37, #85: K H = blockdiag(Rᵀ) N R on position and 0 elsewhere, R's rows u_az, u_el, u_r built here
    from the position, not from H. Entries reach 44; measured 3.6e-15, atol 4e-14 is ~10x."""
    rotation = _los_rotation(ELEVATED_POSITION_M)
    los_gain = torch.arange(1.0, STATE_DIM * 3 + 1.0, dtype=torch.float64).view(STATE_DIM, 3)
    features = _features_at(list(ELEVATED_POSITION_M))
    H = features[0, FEATURE_SLICES["H"]].view(3, STATE_DIM)
    expected = torch.zeros(STATE_DIM, STATE_DIM, dtype=torch.float64)
    expected[:, :3] = torch.kron(torch.eye(3, dtype=torch.float64), rotation).T @ los_gain @ rotation
    assert torch.allclose(_gain_for_los_output(los_gain, features) @ H, expected, rtol=0, atol=LOOP_GAIN_ATOL)
