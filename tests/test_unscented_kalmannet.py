import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from data.config import NOMINAL
from data.trajectory_gen import fly_scenario
from filters.knet import numpy_forward, ukn_filter, ukn_network
from filters.knet.numpy_forward import load_weights
from filters.knet.ukn_filter import UnscentedKalmanNetFilter
from filters.knet.ukn_network import UnscentedKalmanNetwork
from filters.measurement_model import h
from filters.state import AZIMUTH, MEAS_DIM, STATE_DIM
from filters.torch_ops import export_npz
from filters.ukf import UKF
from scripts import eval as bench_eval
from scripts.eval import _INIT_OFFSET, _P0, ekf_like_steps, ukn_steps
from tests.test_numpy_forward import REPO
from tests.test_replay import imm_tuning  # noqa: F401

OVERHEAD_PASS = REPO / "data" / "scenarios" / "overhead_pass.json"
BREAK_TURN = REPO / "data" / "scenarios" / "break_turn_9g.json"
HEADS = ("process_diagonal", "process_lower", "measurement_diagonal", "measurement_lower")
PARITY_ATOL_STATE = 1e-11
PARITY_ATOL_P = 1e-12
SEAM_ATOL = 1e-9
BATCH_ATOL_STATE = 2e-10
BATCH_ATOL_P = 2e-12


def zero_init_weights(tmp_path):
    destination = tmp_path / "ukn.npz"
    export_npz(UnscentedKalmanNetwork(gain_scale=1.0).double().state_dict(), destination)
    return load_weights(destination)


def test_untrained_unscented_kalmannet_tracks_the_overhead_pass_exactly_like_the_ukf(tmp_path):
    """MS-43: zero-initialised heads give A^Q = A^R = I and ΔK = 0, so UKN is filters/ukf.py bit for bit."""
    states, z = fly_scenario(json.loads(OVERHEAD_PASS.read_text()))
    R = NOMINAL.R()
    ukf_snapshot, ukn_snapshot = {}, {}
    ukf = ekf_like_steps(UKF(NOMINAL.dt, R), states, z, ukf_snapshot)
    ukn = ukn_steps(zero_init_weights(tmp_path), NOMINAL.dt, R, states, z, ukn_snapshot)
    for t, (expected, actual) in enumerate(zip(ukf, ukn, strict=True)):
        assert all(np.array_equal(e, a) for e, a in zip(expected, actual, strict=True)), f"step {t}"
        assert np.array_equal(ukf_snapshot["P"], ukn_snapshot["P"]), f"step {t}"


STAND_IN_SEED = 7
STAND_IN_HEAD_SCALE = 0.05
STAND_IN_GAIN_SCALE = 0.02
SATURATED_PRE_ACTIVATION = 40.0
FLOAT64_EPS = np.finfo(np.float64).eps
SEAM_OFFSET_RAD = 1e-6
SEAM_POSITION = np.array([-1000.0, 1e-3, 100.0, -50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HEAD_LAYERS = ("_diagonal.", "_lower.", "decoder.4.")


def torch_cholesky_multiplier(diagonal, lower):
    return ukn_network.cholesky_multiplier(torch.tensor(diagonal)[None], torch.tensor(lower)[None])[0].numpy()


def torch_scaled_covariance(base, base_factor, multiplier):
    return ukn_filter.scaled_covariance(
        torch.tensor(base), torch.tensor(base_factor), torch.tensor(multiplier)
    ).numpy()


def stand_in_network(gain_scale=STAND_IN_GAIN_SCALE):
    torch.manual_seed(STAND_IN_SEED)
    net = UnscentedKalmanNetwork(gain_scale=gain_scale).double()
    for name, parameter in net.named_parameters():
        if any(layer in name for layer in HEAD_LAYERS):
            torch.nn.init.uniform_(parameter, -STAND_IN_HEAD_SCALE, STAND_IN_HEAD_SCALE)
    return net.eval()


def numpy_weights(net):
    return {name: tensor.detach().numpy().copy() for name, tensor in net.state_dict().items()}


def saturated_gain_network(gain_scale, sign=1.0):
    net = UnscentedKalmanNetwork(gain_scale=gain_scale).double()
    torch.nn.init.constant_(net.gain_net.decoder[-1].bias, sign * SATURATED_PRE_ACTIVATION)
    return net


@pytest.mark.parametrize("multiplier", [numpy_forward.cholesky_multiplier, torch_cholesky_multiplier])
def test_variance_multiplier_saturates_at_fifty_times_the_baseline(multiplier):
    """MS-44: tanh(40) is 1 in float64 and exp(4) = 54.6 clips to 50, so each diagonal entry is sqrt(50) exactly."""
    A = multiplier(np.full(3, SATURATED_PRE_ACTIVATION), np.full(3, SATURATED_PRE_ACTIVATION))
    assert np.array_equal(np.diag(A), np.full(3, np.sqrt(50.0)))


@pytest.mark.parametrize("multiplier", [numpy_forward.cholesky_multiplier, torch_cholesky_multiplier])
def test_variance_multiplier_saturates_at_one_tenth_of_the_baseline(multiplier):
    """MS-44: exp(-4) = 0.018 clips to 0.1, so each diagonal entry is sqrt(0.1) exactly."""
    A = multiplier(np.full(3, -SATURATED_PRE_ACTIVATION), np.zeros(3))
    assert np.array_equal(np.diag(A), np.full(3, np.sqrt(0.1)))


@pytest.mark.parametrize("multiplier", [numpy_forward.cholesky_multiplier, torch_cholesky_multiplier])
def test_variance_multiplier_inside_the_clip_scales_variance_by_exp_four_tanh(multiplier):
    """MS-44: tanh = 1/2 gives variance x e²; rtol 4 eps covers tanh(atanh), exp and sqrt at one ulp each."""
    A = multiplier(np.full(3, np.arctanh(0.5)), np.zeros(3))
    np.testing.assert_allclose(np.diag(A), np.full(3, np.e), rtol=4 * FLOAT64_EPS, atol=0.0)


@pytest.mark.parametrize("multiplier", [numpy_forward.cholesky_multiplier, torch_cholesky_multiplier])
def test_variance_multiplier_off_diagonal_saturates_at_plus_and_minus_one_below_the_diagonal_only(multiplier):
    """MS-44: strictly lower entries are c_off tanh(.) with c_off = 1, row-major; the upper triangle stays 0."""
    lower = np.array([SATURATED_PRE_ACTIVATION, -SATURATED_PRE_ACTIVATION, 0.0])
    A = multiplier(np.zeros(3), lower)
    assert np.array_equal(A, np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]))


@pytest.mark.parametrize("scaled", [numpy_forward.scaled_covariance, torch_scaled_covariance])
def test_scaled_covariance_is_the_product_of_the_scaled_cholesky_factors_on_a_hand_worked_case(scaled):
    """MS-44: L = diag(2, 3), A = [[1, 0], [1, 1]] gives (L A)(L A)ᵀ = [[4, 6], [6, 18]], every step exact."""
    base, factor, A = np.diag([4.0, 9.0]), np.diag([2.0, 3.0]), np.array([[1.0, 0.0], [1.0, 1.0]])
    assert np.array_equal(scaled(base, factor, A), np.array([[4.0, 6.0], [6.0, 18.0]]))


@pytest.mark.parametrize("scaled", [numpy_forward.scaled_covariance, torch_scaled_covariance])
def test_scaled_covariance_at_the_identity_multiplier_is_the_cv_process_noise_bit_for_bit(scaled):
    """MS-44: cond(Q_cv) = 3e9, and L Lᵀ misses Q by 7e-21; the A = I term is exactly zero instead."""
    Q = UKF(NOMINAL.dt, NOMINAL.R()).Q
    assert np.array_equal(scaled(Q, np.linalg.cholesky(Q), np.eye(len(Q))), Q)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_gain_residual_saturates_each_column_at_c_k_times_that_column_of_the_ukf_gain(sign):
    """MS-46: column norms 5, 13 and 2, c_K = 1/2, tanh(±40) = ±1: ΔK's columns are ±2.5, ±6.5 and ±1."""
    K = np.zeros((STATE_DIM, MEAS_DIM))
    K[:2, 0], K[3:5, 1], K[8, 2] = [3.0, 4.0], [5.0, 12.0], -2.0
    weights = numpy_weights(saturated_gain_network(0.5, sign))
    network = numpy_forward.UnscentedKalmanNetwork(weights)
    delta_K, _ = network.gain_residual(
        np.ones(numpy_forward.GAIN_FEATURE_DIM), K, network.init_hidden()["gain"]
    )
    torch_delta_K, _ = saturated_gain_network(0.5, sign).gain_net(
        torch.ones(1, numpy_forward.GAIN_FEATURE_DIM, dtype=torch.float64), torch.tensor(K)[None], None
    )
    expected = sign * np.tile([2.5, 6.5, 1.0], (STATE_DIM, 1))
    assert np.array_equal(delta_K, expected)
    assert np.array_equal(torch_delta_K[0].detach().numpy(), expected)


def one_step(flt, x, P, z):
    flt.init_sequence(x, P)
    return flt.step(z)


def test_posterior_with_a_gain_residual_is_the_arbitrary_gain_joseph_form():
    """MS-47: P_UKF + ΔK S ΔKᵀ (eq 17) equals eq 16 at K_UKF + ΔK. The cross terms cancel only to rounding of
    terms as large as |K S Kᵀ|, so atol is 64 eps of that scale."""
    R = NOMINAL.R()
    x, P = np.array([800.0, 500.0, 100.0, 30.0, -10.0, 0.0, 0.0, 0.0, 0.0]), _P0.copy()
    z = h(x) + [2e-3, -1e-3, 5.0]
    flt = numpy_forward.UnscentedKalmanNetFilter(
        numpy_forward.UnscentedKalmanNetwork(numpy_weights(saturated_gain_network(0.25))), NOMINAL.dt, R
    )
    x_post, P_post, y, S = one_step(flt, x, P, z)
    ukf = UKF(NOMINAL.dt, R)
    dx, dz, _ = ukf.innovation(x, ukf._sigma_points(x, P), z)
    K_ukf, _, _ = ukf.analytic_update(P, dx, dz, R)
    cross = (dx.T * ukf.Wc) @ dz
    K = K_ukf + 0.25 * np.linalg.norm(K_ukf, axis=0)
    joseph = P - cross @ K.T - K @ cross.T + K @ S @ K.T
    np.testing.assert_array_equal(x_post, x + K @ y)
    np.testing.assert_allclose(P_post, joseph, rtol=0.0, atol=64 * FLOAT64_EPS * np.abs(K @ S @ K.T).max())


def seam_step(backend, azimuth_shift):
    """One step at azimuth π - 1e-6 with the measurement across the seam, at -π + 1e-6, plus azimuth_shift."""
    x, P = SEAM_POSITION, _P0.copy()
    z = np.array([-np.pi + SEAM_OFFSET_RAD + azimuth_shift, *h(x)[1:]])
    if backend == "numpy":
        flt = numpy_forward.UnscentedKalmanNetFilter(
            numpy_forward.UnscentedKalmanNetwork(numpy_weights(stand_in_network())), NOMINAL.dt, NOMINAL.R()
        )
        flt.init_sequence(x, P)
        flt.step(h(x))
        return flt.step(z)[:3]
    flt = UnscentedKalmanNetFilter(stand_in_network(), NOMINAL.dt, NOMINAL.R())
    flt.init_sequence(torch.tensor(x)[None], torch.tensor(P)[None])
    with torch.no_grad():
        flt.step(torch.tensor(h(x))[None])
        return [value[0].numpy() for value in flt.step(torch.tensor(z)[None])[:3]]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_a_measurement_across_the_azimuth_seam_gives_a_small_wrapped_innovation(backend):
    """MS-45: the prior sits at π - 1.001e-6 (px moved by -1 m) and the measurement at -π + 1e-6, so the wrapped
    innovation is +2.0e-6 rad; the sigma-point circular mean shifts it by 2.9e-8, inside the 1e-7 band. Unwrapped
    it would be -2π."""
    _, _, y = seam_step(backend, 0.0)
    assert y[AZIMUTH] == pytest.approx(2.0e-6, rel=0.0, abs=1e-7)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_shifting_the_measured_azimuth_by_two_pi_at_the_seam_leaves_the_step_unchanged(backend):
    """MS-45: every innovation-derived feature is wrapped and y enters NoiseNet as sin/cos, so z_az + 2π moves
    only by the rounding of the shift, one ulp of 2π (8.9e-16 rad), carried through K (|K| <= 1e3 m/rad) and the
    network: atol 1e-9 on x and P leaves a 1e3 amplification margin."""
    x, P, y = seam_step(backend, 0.0)
    x_shifted, P_shifted, y_shifted = seam_step(backend, 2.0 * np.pi)
    np.testing.assert_allclose(x_shifted, x, rtol=0.0, atol=SEAM_ATOL)
    np.testing.assert_allclose(P_shifted, P, rtol=0.0, atol=SEAM_ATOL)
    assert y_shifted[AZIMUTH] == pytest.approx(y[AZIMUTH], rel=0.0, abs=2 * np.spacing(2.0 * np.pi))


def test_numpy_unscented_kalmannet_matches_torch_at_every_step_of_the_overhead_pass():
    """MS-43: float64 summation order drifts x by 9.3e-13 at |x| <= 503 m and P by 1.3e-13 at |P| <= 5 over the
    200 steps (measured), so atol 1e-11 on x and 1e-12 on P leave ~10x and ~7x headroom."""
    states, z = fly_scenario(json.loads(OVERHEAD_PASS.read_text()))
    net = stand_in_network()
    numpy_filter = numpy_forward.UnscentedKalmanNetFilter(
        numpy_forward.UnscentedKalmanNetwork(numpy_weights(net)), NOMINAL.dt, NOMINAL.R()
    )
    numpy_filter.init_sequence(states[0] + _INIT_OFFSET, _P0.copy())
    torch_filter = UnscentedKalmanNetFilter(net, NOMINAL.dt, NOMINAL.R())
    torch_filter.init_sequence(torch.tensor(states[0] + _INIT_OFFSET)[None], torch.tensor(_P0.copy())[None])
    with torch.no_grad():
        for z_t in z:
            x, P, _, _ = numpy_filter.step(z_t)
            x_torch, P_torch, *_ = torch_filter.step(torch.tensor(z_t)[None])
            np.testing.assert_allclose(x, x_torch[0].numpy(), rtol=0.0, atol=PARITY_ATOL_STATE)
            np.testing.assert_allclose(P, P_torch[0].numpy(), rtol=0.0, atol=PARITY_ATOL_P)


def test_network_is_sized_as_the_papers_manoeuvring_target_example():
    """MS-45: d_G 32, d_N 25, d_e 32, H_G 128, H_N 64; heads emit A^Q (9 + 36), A^R (3 + 3) and ΔK (27)."""
    shapes = {
        name: tuple(tensor.shape)
        for name, tensor in UnscentedKalmanNetwork(gain_scale=1.0).state_dict().items()
    }
    assert shapes["gain_net.encoder.0.weight"] == (32,)
    assert shapes["noise_net.encoder.0.weight"] == (25,)
    assert shapes["gain_net.encoder.1.weight"] == (32, 32)
    assert shapes["noise_net.encoder.3.weight"] == (32, 32)
    assert shapes["gain_net.gru.weight_hh_l0"] == (3 * 128, 128)
    assert shapes["noise_net.gru.weight_hh_l0"] == (3 * 64, 64)
    assert shapes["noise_net.process_adapter.weight"] == (25, 25)
    head_widths = [shapes[f"noise_net.{head}.bias"][0] for head in HEADS]
    assert head_widths == [9, 36, 3, 3]
    assert shapes["gain_net.decoder.4.weight"] == (STATE_DIM * MEAS_DIM, 128)


def test_batched_training_filter_steps_each_trajectory_as_if_it_were_alone():
    """MS-43: training batches trajectories, so no tensor axis may mix them. Batched and single kernels round
    differently: measured 1.8e-11 on x at |x| <= 2.3e3 m and 2.2e-13 on P at |P| <= 5, so 2e-10 and 2e-12 (~10x)."""
    flights = [fly_scenario(json.loads(path.read_text())) for path in (OVERHEAD_PASS, BREAK_TURN)]
    steps = min(len(z) for _, z in flights)
    starts = torch.tensor(np.array([states[0] + _INIT_OFFSET for states, _ in flights]))
    measurements = torch.tensor(np.array([z[:steps] for _, z in flights]))
    net = stand_in_network()
    batched = UnscentedKalmanNetFilter(net, NOMINAL.dt, NOMINAL.R())
    batched.init_sequence(starts, torch.tensor(_P0).expand(2, -1, -1))
    alone = [UnscentedKalmanNetFilter(net, NOMINAL.dt, NOMINAL.R()) for _ in flights]
    for index, flt in enumerate(alone):
        flt.init_sequence(starts[index : index + 1], torch.tensor(_P0)[None])
    with torch.no_grad():
        for t in range(steps):
            x, P, *_ = batched.step(measurements[:, t])
            for index, flt in enumerate(alone):
                x_alone, P_alone, *_ = flt.step(measurements[index : index + 1, t])
                np.testing.assert_allclose(x[index], x_alone[0], rtol=0.0, atol=BATCH_ATOL_STATE)
                np.testing.assert_allclose(P[index], P_alone[0], rtol=0.0, atol=BATCH_ATOL_P)


@pytest.mark.parametrize("features", [numpy_forward.noise_features, lambda *a: torch_noise_features(*a)])
def test_noise_net_reads_the_measurement_as_azimuth_sine_and_cosine_then_elevation_and_range(features):
    """MS-45: [sin az, cos az, el, range, y, x, log diag P]: 4 + 3 + 9 + 9 = 25 inputs, no control term."""
    z, y = np.array([-3.0, 0.25, 1000.0]), np.array([1e-3, -2e-3, 4.0])
    x, P = np.arange(1.0, STATE_DIM + 1.0), np.diag(np.exp(np.arange(STATE_DIM, dtype=float)))
    row = features(z, y, x, P)
    expected = np.concatenate([[np.sin(-3.0), np.cos(-3.0), 0.25, 1000.0], y, x, np.arange(STATE_DIM, dtype=float)])
    np.testing.assert_allclose(row, expected, rtol=2 * FLOAT64_EPS, atol=0.0)


def torch_noise_features(z, y, x, P):
    as_batch = [torch.tensor(value)[None] for value in (z, y, x, P)]
    return ukn_filter.noise_features(*as_batch)[0].numpy()


def test_learned_process_and_measurement_noise_are_the_ones_the_filter_steps_with():
    """MS-44: process heads saturated high (A_Q = sqrt(50) I) and measurement heads low (A_R = sqrt(0.1) I): two
    UKN steps equal two filters/ukf.py steps run with those Q_t and R_t, bit for bit, and differ from the UKF's."""
    net = UnscentedKalmanNetwork(gain_scale=1.0).double()
    torch.nn.init.constant_(net.noise_net.process_diagonal.bias, SATURATED_PRE_ACTIVATION)
    torch.nn.init.constant_(net.noise_net.measurement_diagonal.bias, -SATURATED_PRE_ACTIVATION)
    R = NOMINAL.R()
    flt = numpy_forward.UnscentedKalmanNetFilter(
        numpy_forward.UnscentedKalmanNetwork(numpy_weights(net)), NOMINAL.dt, R
    )
    states, z = fly_scenario(json.loads(OVERHEAD_PASS.read_text()))
    x0 = states[0] + _INIT_OFFSET
    flt.init_sequence(x0, _P0.copy())
    flt.step(z[0])
    x, P, _, _ = flt.step(z[1])
    ukf = UKF(NOMINAL.dt, R)
    Q_t = numpy_forward.scaled_covariance(ukf.Q, np.linalg.cholesky(ukf.Q), np.sqrt(50.0) * np.eye(STATE_DIM))
    R_t = numpy_forward.scaled_covariance(R, np.linalg.cholesky(R), np.sqrt(0.1) * np.eye(MEAS_DIM))
    dx, dz, y = ukf.innovation(x0, ukf._sigma_points(x0, _P0), z[0])
    K, P_expected, _ = ukf.analytic_update(_P0, dx, dz, R_t)
    x_expected = x0 + K @ y
    x_pred, P_pred, pts_pred = ukf.predict(x_expected, P_expected, Q_t)
    dx, dz, y = ukf.innovation(x_pred, pts_pred, z[1])
    K, P_expected, _ = ukf.analytic_update(P_pred, dx, dz, R_t)
    assert np.array_equal(x, x_pred + K @ y)
    assert np.array_equal(P, P_expected)
    x_ukf, P_ukf, _, _ = ukf.step(*ukf.update(x0, _P0, ukf._sigma_points(x0, _P0), z[0])[:2], z[1])
    assert not np.array_equal(P, P_ukf)


def test_untrained_torch_unscented_kalmannet_tracks_the_overhead_pass_like_the_ukf():
    """MS-43: the training backbone at zero init is filters/ukf.py up to the numpy-vs-torch budget (atol 1e-11 on
    x, 1e-12 on P, as the parity test derives)."""
    states, z = fly_scenario(json.loads(OVERHEAD_PASS.read_text()))
    ukf_snapshot = {}
    ukf = ekf_like_steps(UKF(NOMINAL.dt, NOMINAL.R()), states, z, ukf_snapshot)
    flt = UnscentedKalmanNetFilter(UnscentedKalmanNetwork(gain_scale=1.0).double(), NOMINAL.dt, NOMINAL.R())
    flt.init_sequence(torch.tensor(states[0] + _INIT_OFFSET)[None], torch.tensor(_P0)[None])
    with torch.no_grad():
        for truth, z_t, (error, _, _) in zip(states, z, ukf, strict=True):
            x, P, *_ = flt.step(torch.tensor(z_t)[None])
            np.testing.assert_allclose(x[0].numpy(), truth + error, rtol=0.0, atol=PARITY_ATOL_STATE)
            np.testing.assert_allclose(P[0].numpy(), ukf_snapshot["P"], rtol=0.0, atol=PARITY_ATOL_P)


def test_eval_reports_unscented_kalmannet_rmse_anees_nis_and_coverage_on_every_regime(imm_tuning, tmp_path, monkeypatch):  # noqa: F811
    """MS-43: UKN sits beside EKF, UKF and IMM in results_table.csv on nominal, the four shifted regimes and the
    scenarios, and in the high-MC calibration file, each row its own evaluation."""
    data_dir, out, weights = tmp_path / "datasets", tmp_path / "results", tmp_path / "ukn.npz"
    toy = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1", "--mc-test", "2", "--n-highmc", "1"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy, "--mc-highmc", "2"], cwd=REPO, check=True)
    export_npz(stand_in_network().state_dict(), weights)
    eval_argv = ["eval.py", "--data-dir", str(data_dir), "--imm-tuning", str(imm_tuning), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*eval_argv, "--bkn-weights", str(tmp_path / "missing.npz"), "--ukn-weights", str(weights)])
    monkeypatch.chdir(REPO)

    bench_eval.main()

    with open(out / "results_table.csv") as f:
        ukn = {row["regime"]: row for row in csv.DictReader(f) if row["filter"] == "UKN"}
    highmc = {row["filter"]: row for row in json.loads((out / "anees_highmc.json").read_text())["results"]}
    assert list(ukn) == ["nominal", "high_maneuver", "high_noise", "low_noise", "heavy_tailed", "scenarios"]
    assert list(highmc) == ["EKF", "UKF", "IMM", "UKN"]
    states, measurements = bench_eval._load(data_dir / "test.npz")
    nominal = bench_eval.evaluate_dataset(*bench_eval.ukn_entry(load_weights(weights), NOMINAL.dt, NOMINAL.R()), states, measurements)
    consistency = bench_eval.summarize_consistency(nominal)
    assert float(ukn["nominal"]["pos_rmse"]) == bench_eval.summarize_rmse(nominal)[0]
    for key in ("anees_mean", "anis_mean", "coverage_68", "coverage_95"):
        assert float(ukn["nominal"][key]) == consistency[key]


def test_latency_bench_times_the_numpy_unscented_kalmannet_step(tmp_path, monkeypatch):
    """MS-43: the UKN figure is the page's numpy step plus the NEES/NIS scoring, on the exported weights."""
    from scripts import bench_latency

    nominal = {"dt": NOMINAL.dt, "sigma_az_rad": 0.001, "sigma_el_rad": 0.001, "sigma_range_m": 2.0}
    (tmp_path / "manifest.json").write_text(json.dumps({"regimes": {"nominal": nominal}}))
    tuning = tmp_path / "imm_tuning.json"
    tuning.write_text(json.dumps({"best": json.loads((REPO / "web" / "imm_tuning.json").read_text())["best"]}))
    weights = tmp_path / "ukn.npz"
    export_npz(stand_in_network().state_dict(), weights)
    stepped = []
    monkeypatch.setattr(bench_latency, "ukn_steps", lambda w, *args: stepped.append(w) or ukn_steps(w, *args))
    out = tmp_path / "latency.json"
    argv = ["bench_latency.py", "--data-dir", str(tmp_path), "--imm-tuning", str(tuning), "--ukn-weights", str(weights),
            "--n-warmup", "1", "--n-measure", "2", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    bench_latency.main()

    timed = json.loads(out.read_text())["results_ms"]
    assert timed["UKN-numpy"]["median_ms"] > 0.0
    assert [sorted(w) for w in stepped] == [sorted(load_weights(weights))]
