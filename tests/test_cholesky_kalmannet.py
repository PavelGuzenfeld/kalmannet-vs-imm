import csv
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from data.config import NOMINAL, REGIMES
from data.trajectory_gen import fly_scenario
from filters.knet import ckn_network, numpy_forward
from filters.knet.ckn_filter import CholeskyKalmanNetFilter
from filters.knet.ckn_network import CholeskyKalmanNetwork
from filters.knet.numpy_forward import load_weights
from filters.measurement_model import H_jacobian, h, innovation
from filters.motion_models import cv_model
from filters.state import AZIMUTH, MEAS_DIM, STATE_DIM
from filters.torch_ops import export_npz
from scripts import eval as bench_eval
from scripts.eval import _INIT_OFFSET, ckn_steps
from scripts.train_knet import _INIT_PERTURB_STD
from tests.test_numpy_forward import REPO
from tests.test_replay import imm_tuning  # noqa: F401
from tests.test_unscented_kalmannet import numpy_weights

BREAK_TURN = REPO / "data" / "scenarios" / "break_turn_9g.json"
OVERHEAD_PASS = REPO / "data" / "scenarios" / "overhead_pass.json"
STAND_IN_SEED = 7
FLOAT64_EPS = np.finfo(np.float64).eps
SEAM_OFFSET_RAD = 1e-6
SEAM_POSITION = np.array([-1000.0, 1e-3, 100.0, -50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
SIGMA_AZ_RAD = 1e-3
SEAM_ATOL = 1e-9
PARITY_ATOL_STATE = 1e-9
PARITY_P_RELATIVE_TO_MAX = 1e-11
PARITY_RTOL_NIS = 1e-11
BATCH_ATOL_STATE = 2e-9
BATCH_P_RELATIVE_TO_MAX = 2e-11
HAND_WORKED_X = np.array([800.0, 500.0, 100.0, 30.0, -10.0, 0.0, 0.0, 0.0, 0.0])
PRIOR_OUTPUT = np.array([2.0**-6, 2.0**-8, 2.0**-5, *([2.0**-7] * 42)])
INNOVATION_OUTPUT = np.array([1.0, 0.25, 0.5, -0.125, 0.0, 2.0])


def stand_in_network():
    torch.manual_seed(STAND_IN_SEED)
    return CholeskyKalmanNetwork(NOMINAL.R()).double().eval()


def constant_output_network():
    """Decoder output layers at zero weight and fixed bias: L₁ and L₂ are the PDEL of PRIOR/INNOVATION_OUTPUT."""
    net = stand_in_network()
    for branch, output in ((net.prior, PRIOR_OUTPUT), (net.innovation, INNOVATION_OUTPUT)):
        torch.nn.init.zeros_(branch.decoder[-1].weight)
        branch.decoder[-1].bias.data = torch.tensor(output)
    return net


def numpy_filter(net, R=None):
    weights = numpy_weights(net)
    R = NOMINAL.R() if R is None else R
    return numpy_forward.CholeskyKalmanNetFilter(numpy_forward.CholeskyKalmanNetwork(weights), NOMINAL.dt, R)


def test_every_posterior_covariance_on_the_break_turn_is_symmetric_positive_definite():
    """MS-52: the posterior is carried as a QR factor of the Joseph form, so its Cholesky succeeds at every step."""
    states, z = fly_scenario(json.loads(BREAK_TURN.read_text()))
    snapshot = {}
    stepped = 0
    for _ in ckn_steps(numpy_weights(stand_in_network()), NOMINAL.dt, NOMINAL.R(), states, z, snapshot):
        P = snapshot["P"]
        assert np.array_equal(P, P.T), f"step {stepped}"
        np.linalg.cholesky(P)
        stepped += 1
    assert stepped == len(z)


@pytest.mark.parametrize(
    "factor",
    [numpy_forward.lower_triangular_factor, lambda e, q: ckn_network.lower_triangular_factor(torch.tensor(e)[None], q)[0].numpy()],
)
def test_pdel_fills_the_lower_triangle_row_by_row_and_takes_the_absolute_diagonal_plus_epsilon(factor):
    """MS-51: eq 4 with p(a) = |a| + 1e-8 on the diagonal only; off-diagonal entries keep their sign."""
    expected = np.array([[2.0 + 1e-8, 0.0, 0.0], [-3.0, 4.0 + 1e-8, 0.0], [5.0, -6.0, 0.5 + 1e-8]])
    assert np.array_equal(factor(np.array([-2.0, -3.0, 4.0, 5.0, -6.0, -0.5]), 3), expected)


def test_hand_worked_step_gains_with_split_covariances_and_returns_the_joseph_posterior_and_nis():
    """MS-51, MS-52: K = G¹ Hᵀ G², x = x̂ + K y, P = (I − KH) G¹ (I − KH)ᵀ + K R Kᵀ and NIS = yᵀ G² y. Products
    associate differently here, so x is held to 64 eps of |x̂| + |K||y|, P to 64 eps of max|P| (QR against the
    direct sum) and NIS to 64 eps relative."""
    R = NOMINAL.R()
    flt = numpy_filter(constant_output_network(), R)
    flt.init_sequence(HAND_WORKED_X)
    z = h(HAND_WORKED_X) + [2e-3, -1e-3, 5.0]
    x, P_factor, nis = flt.step(z)
    D_x, D_z = np.array(_INIT_PERTURB_STD), np.sqrt(np.diag(R))
    G1 = np.diag(D_x) @ numpy_forward.lower_triangular_factor(PRIOR_OUTPUT, STATE_DIM)
    G1 = G1 @ G1.T
    G2 = np.diag(1.0 / D_z) @ numpy_forward.lower_triangular_factor(INNOVATION_OUTPUT, MEAS_DIM)
    G2 = G2 @ G2.T
    F, _ = cv_model(NOMINAL.dt)
    x_prior = F @ HAND_WORKED_X
    H, y = H_jacobian(x_prior), innovation(z, x_prior)
    K = G1 @ H.T @ G2
    I_KH = np.eye(STATE_DIM) - K @ H
    joseph = I_KH @ G1 @ I_KH.T + K @ R @ K.T
    rounding = 64 * FLOAT64_EPS * (np.abs(x_prior) + np.abs(K) @ np.abs(y))
    assert np.all(np.abs(x - (x_prior + K @ y)) <= rounding)
    np.testing.assert_allclose(P_factor @ P_factor.T, joseph, rtol=0.0, atol=64 * FLOAT64_EPS * np.abs(joseph).max())
    assert nis == pytest.approx(y @ G2 @ y, rel=64 * FLOAT64_EPS)
    assert not np.allclose(P_factor @ P_factor.T, G1, rtol=1e-3)


def test_untrained_factors_are_the_diagonal_init_times_the_identity_in_each_branchs_units():
    """MS-51: output layer bias 2⁻⁶ on L₁'s diagonal and 1 on L₂'s, zero off the diagonal, so G¹ = 2⁻¹² D_x² and
    G² = D_z⁻² up to the U(±1e-3) weights, the largest of 4,860 past 0.99e-3; D_x is σ_init, D_z the nominal σ."""
    torch.manual_seed(STAND_IN_SEED)
    net = CholeskyKalmanNetwork(NOMINAL.R())
    rows, columns = torch.tril_indices(STATE_DIM, STATE_DIM)
    assert torch.equal(net.prior.decoder[-1].bias, (rows == columns) * 2.0**-6)
    rows, columns = torch.tril_indices(MEAS_DIM, MEAS_DIM)
    assert torch.equal(net.innovation.decoder[-1].bias, (rows == columns) * 1.0)
    assert 0.99e-3 < net.prior.decoder[-1].weight.abs().max() <= 1e-3
    assert torch.equal(net.state_scale, torch.as_tensor(_INIT_PERTURB_STD))
    assert torch.equal(net.measurement_scale, torch.sqrt(torch.diagonal(torch.as_tensor(NOMINAL.R()))))


def test_network_is_sized_by_the_reference_formulas_at_nine_states_and_three_measurements():
    """MS-51: H1 = (m + n)·80 = 960, H2 = m·n·4 = 108, a 2-layer GRU of 2(m² + n²) = 180, inputs 48 and 36, PDEL
    entries 45 and 6."""
    shapes = {name: tuple(t.shape) for name, t in CholeskyKalmanNetwork(NOMINAL.R()).state_dict().items()}
    assert shapes["prior.encoder.0.weight"] == (960, 48)
    assert shapes["innovation.encoder.0.weight"] == (960, 36)
    for branch in ("prior", "innovation"):
        assert shapes[f"{branch}.gru.weight_ih_l1"] == (3 * 180, 180)
        assert shapes[f"{branch}.decoder.0.weight"] == (108, 180)
    assert shapes["prior.decoder.2.weight"] == (45, 108)
    assert shapes["innovation.decoder.2.weight"] == (6, 108)


def recorded_features(flt):
    features = []
    branch = flt.net.branch

    def record(name, values, hidden):
        features.append((name, values))
        return branch(name, values, hidden)

    flt.net.branch = record
    return features


def seam_steps(backend, azimuth_shift):
    """From SEAM_POSITION (azimuth π − 1e-6): z across the seam at −π + 1e-6 plus azimuth_shift, then h(x₀).
    Returns x and P after both, and the innovation branch's features at each step."""
    x0 = SEAM_POSITION
    measurements = [np.array([-np.pi + SEAM_OFFSET_RAD + azimuth_shift, *h(x0)[1:]]), h(x0)]
    if backend == "numpy":
        flt = numpy_filter(stand_in_network())
        features = recorded_features(flt)
        flt.init_sequence(x0)
        for z in measurements:
            x, P_factor, _ = flt.step(z)
        return x, P_factor @ P_factor.T, [values for name, values in features if name == "innovation"]
    flt = CholeskyKalmanNetFilter(stand_in_network(), NOMINAL.dt, NOMINAL.R())
    flt.init_sequence(torch.tensor(x0)[None])
    captured = []
    innovation_branch = flt.net.innovation.forward
    flt.net.innovation.forward = lambda values, hidden: captured.append(values) or innovation_branch(values, hidden)
    with torch.no_grad():
        for z in measurements:
            x, P, _ = flt.step(torch.tensor(z)[None])
    return x[0].numpy(), P[0].numpy(), [values[0].numpy() for values in captured]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_innovation_and_measurement_difference_features_wrap_across_the_azimuth_seam(backend):
    """MS-51: the first prior is at px = −1001 m, azimuth π − 1e-3/1001, and z at −π + 1e-6: the wrapped innovation
    is +1.999000999e-6 rad. The second z − z_previous is h(x₀) − z = −2e-6 rad. Both read in σ_az = 1e-3 units,
    to 8 ulp of π (four roundings near π) over σ_az. Unwrapped, either would be ∓6283."""
    _, _, features = seam_steps(backend, 0.0)
    rounding = 8 * np.spacing(np.pi) / SIGMA_AZ_RAD
    assert features[0][AZIMUTH] == pytest.approx((1e-6 + 1e-3 / 1001) / SIGMA_AZ_RAD, rel=0.0, abs=rounding)
    assert features[1][MEAS_DIM + AZIMUTH] == pytest.approx(-2e-6 / SIGMA_AZ_RAD, rel=0.0, abs=rounding)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_shifting_the_measured_azimuth_by_two_pi_at_the_seam_leaves_the_step_unchanged(backend):
    """MS-51: z_az + 2π moves only by one ulp of 2π (8.9e-16 rad), 8.9e-13 in σ_az units, carried through the
    network and K (|K| ≤ 1e3 m/rad): atol 1e-9 on x and P leaves a 1e3 margin."""
    x, P, _ = seam_steps(backend, 0.0)
    x_shifted, P_shifted, _ = seam_steps(backend, 2.0 * np.pi)
    np.testing.assert_allclose(x_shifted, x, rtol=0.0, atol=SEAM_ATOL)
    np.testing.assert_allclose(P_shifted, P, rtol=0.0, atol=SEAM_ATOL)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_linearisation_feature_is_continuous_across_the_azimuth_seam(backend):
    """MS-51: h(x̂) − H x̂ = [az, el, 0] jumps by 2π at the seam; as [sin az, cos az, el] the two sides at
    az = ±(π − 1e-6) differ by 2e-6 in sin only, to 2 ulp of π for the rounding of each azimuth."""
    above, below = SEAM_POSITION.copy(), SEAM_POSITION.copy()
    below[1] = -below[1]
    if backend == "numpy":
        features = [numpy_forward.linearisation_feature(x) for x in (above, below)]
    else:
        from filters.knet.ckn_filter import linearisation_feature

        features = [linearisation_feature(torch.tensor(x)[None])[0].numpy() for x in (above, below)]
    assert features[0][0] - features[1][0] == pytest.approx(2e-6, rel=0.0, abs=2 * np.spacing(np.pi))
    assert features[0][1:] == pytest.approx(features[1][1:], rel=0.0, abs=0.0)


def test_numpy_cholesky_kalmannet_matches_torch_at_every_step_of_the_overhead_pass():
    """MS-51, MS-52: the numpy P is a QR factor, torch sums the Joseph Gram matrices; both run the same network.
    Measured over the 200 steps: x 7.4e-11 at |x| ≤ 1.3e3 m, P 8.0e-13 and NIS 7.9e-13 relative; atol 1e-9 on x,
    1e-11 of max|P| and rtol 1e-11 on NIS leave 12x or more."""
    states, z = fly_scenario(json.loads(OVERHEAD_PASS.read_text()))
    net = stand_in_network()
    flt = numpy_filter(net)
    flt.init_sequence(states[0] + _INIT_OFFSET)
    torch_filter = CholeskyKalmanNetFilter(net, NOMINAL.dt, NOMINAL.R())
    torch_filter.init_sequence(torch.tensor(states[0] + _INIT_OFFSET)[None])
    with torch.no_grad():
        for z_t in z:
            x, P_factor, nis = flt.step(z_t)
            x_torch, P_torch, nis_torch = torch_filter.step(torch.tensor(z_t)[None])
            np.testing.assert_allclose(x, x_torch[0].numpy(), rtol=0.0, atol=PARITY_ATOL_STATE)
            P = P_factor @ P_factor.T
            np.testing.assert_allclose(P, P_torch[0].numpy(), rtol=0.0, atol=PARITY_P_RELATIVE_TO_MAX * np.abs(P).max())
            assert nis_torch.shape == (1,)
            assert nis == pytest.approx(nis_torch[0].item(), rel=PARITY_RTOL_NIS)


def test_batched_training_filter_steps_each_trajectory_as_if_it_were_alone():
    """MS-51: training batches trajectories, so no tensor axis may mix them. Batched and single kernels round
    differently: measured 1.4e-10 on x at |x| ≤ 2.3e3 m and 1.5e-12 of max|P| on P, so 2e-9 and 2e-11 (~13x)."""
    flights = [fly_scenario(json.loads(path.read_text())) for path in (OVERHEAD_PASS, BREAK_TURN)]
    steps = min(len(z) for _, z in flights)
    starts = torch.tensor(np.array([states[0] + _INIT_OFFSET for states, _ in flights]))
    measurements = torch.tensor(np.array([z[:steps] for _, z in flights]))
    net = stand_in_network()
    batched = CholeskyKalmanNetFilter(net, NOMINAL.dt, NOMINAL.R())
    batched.init_sequence(starts)
    alone = [CholeskyKalmanNetFilter(net, NOMINAL.dt, NOMINAL.R()) for _ in flights]
    for index, flt in enumerate(alone):
        flt.init_sequence(starts[index : index + 1])
    with torch.no_grad():
        for t in range(steps):
            x, P, _ = batched.step(measurements[:, t])
            for index, flt in enumerate(alone):
                x_alone, P_alone, _ = flt.step(measurements[index : index + 1, t])
                np.testing.assert_allclose(x[index], x_alone[0], rtol=0.0, atol=BATCH_ATOL_STATE)
                atol = BATCH_P_RELATIVE_TO_MAX * P_alone[0].abs().max().item()
                np.testing.assert_allclose(P[index], P_alone[0], rtol=0.0, atol=atol)


def test_eval_reports_cholesky_kalmannet_rmse_anees_nis_and_coverage_on_every_regime(imm_tuning, tmp_path, monkeypatch):  # noqa: F811
    """MS-52: CKN sits beside EKF, UKF and IMM in results_table.csv on nominal, the four shifted regimes (each under
    its own R, checked on high_noise) and the scenarios, and in the high-MC calibration file, each row its own
    evaluation."""
    data_dir, out, weights = tmp_path / "datasets", tmp_path / "results", tmp_path / "ckn.npz"
    toy = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1", "--mc-test", "2", "--n-highmc", "1"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy, "--mc-highmc", "2"], cwd=REPO, check=True)
    export_npz(stand_in_network().state_dict(), weights)
    missing = str(tmp_path / "missing.npz")
    eval_argv = ["eval.py", "--data-dir", str(data_dir), "--imm-tuning", str(imm_tuning), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*eval_argv, "--bkn-weights", missing, "--ukn-weights", missing, "--ckn-weights", str(weights)])
    monkeypatch.chdir(REPO)

    bench_eval.main()

    with open(out / "results_table.csv") as f:
        ckn = {row["regime"]: row for row in csv.DictReader(f) if row["filter"] == "CKN"}
    highmc = {row["filter"]: row for row in json.loads((out / "anees_highmc.json").read_text())["results"]}
    assert list(ckn) == ["nominal", "high_maneuver", "high_noise", "low_noise", "heavy_tailed", "scenarios"]
    assert list(highmc) == ["EKF", "UKF", "IMM", "CKN"]
    states, measurements = bench_eval._load(data_dir / "test.npz")
    nominal = bench_eval.evaluate_dataset(*bench_eval.ckn_entry(load_weights(weights), NOMINAL.dt, NOMINAL.R()), states, measurements)
    consistency = bench_eval.summarize_consistency(nominal)
    assert float(ckn["nominal"]["pos_rmse"]) == bench_eval.summarize_rmse(nominal)[0]
    for key in ("anees_mean", "anis_mean", "coverage_68", "coverage_95"):
        assert float(ckn["nominal"][key]) == consistency[key]
    high_noise = REGIMES["high_noise"]
    states, measurements = bench_eval._load(data_dir / "test_high_noise.npz")
    shifted = bench_eval.evaluate_dataset(*bench_eval.ckn_entry(load_weights(weights), high_noise.dt, high_noise.R()), states, measurements)
    assert float(ckn["high_noise"]["pos_rmse"]) == bench_eval.summarize_rmse(shifted)[0]
    assert float(ckn["high_noise"]["anis_mean"]) == bench_eval.summarize_consistency(shifted)["anis_mean"]


def test_latency_bench_times_the_numpy_cholesky_kalmannet_step(tmp_path, monkeypatch):
    """MS-52: the CKN figure is the page's numpy step plus the NEES/NIS scoring, on the exported weights."""
    from scripts import bench_latency

    nominal = {"dt": NOMINAL.dt, "sigma_az_rad": 0.001, "sigma_el_rad": 0.001, "sigma_range_m": 2.0}
    (tmp_path / "manifest.json").write_text(json.dumps({"regimes": {"nominal": nominal}}))
    tuning = tmp_path / "imm_tuning.json"
    tuning.write_text(json.dumps({"best": json.loads((REPO / "web" / "imm_tuning.json").read_text())["best"]}))
    weights = tmp_path / "ckn.npz"
    export_npz(stand_in_network().state_dict(), weights)
    stepped = []
    monkeypatch.setattr(bench_latency, "ckn_steps", lambda w, *args: stepped.append(w) or ckn_steps(w, *args))
    out = tmp_path / "latency.json"
    argv = ["bench_latency.py", "--data-dir", str(tmp_path), "--imm-tuning", str(tuning), "--ckn-weights", str(weights),
            "--n-warmup", "1", "--n-measure", "2", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    bench_latency.main()

    timed = json.loads(out.read_text())["results_ms"]
    assert timed["CKN-numpy"]["median_ms"] > 0.0
    assert [sorted(w) for w in stepped] == [sorted(load_weights(weights))]


def test_eval_starts_cholesky_kalmannet_from_the_offset_truth_as_kalmannet_and_rkn_start():
    """MS-52: step 0 predicts from x₀ + _INIT_OFFSET, not from the truth or the other side of it."""
    states, z = fly_scenario(json.loads(BREAK_TURN.read_text()))
    weights = numpy_weights(stand_in_network())
    error, _, _ = next(ckn_steps(weights, NOMINAL.dt, NOMINAL.R(), states, z))
    flt = numpy_forward.CholeskyKalmanNetFilter(numpy_forward.CholeskyKalmanNetwork(weights), NOMINAL.dt, NOMINAL.R())
    flt.init_sequence(states[0] + _INIT_OFFSET)
    assert np.array_equal(error, flt.step(z[0])[0] - states[0])


def test_snapshot_covariance_is_the_one_the_nees_was_scored_with():
    """MS-52: eᵀ P⁻¹ e on the snapshot P equals the NEES whitened through P's factor, to 1e-9 relative, the solve
    against the squared factor rounding at cond(P) ≤ 1e6 here times eps."""
    states, z = fly_scenario(json.loads(BREAK_TURN.read_text()))
    snapshot = {}
    steps = ckn_steps(numpy_weights(stand_in_network()), NOMINAL.dt, NOMINAL.R(), states, z, snapshot)
    for _, (error, nees, _) in zip(range(50), steps):
        assert error @ np.linalg.solve(snapshot["P"], error) == pytest.approx(nees, rel=1e-9)
