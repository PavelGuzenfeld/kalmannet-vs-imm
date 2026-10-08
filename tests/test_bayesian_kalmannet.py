import csv
import json
import subprocess
import sys

import numpy as np
import torch

from data.config import NOMINAL
from filters.knet import numpy_forward
from filters.knet.numpy_forward import load_weights
from filters.torch_ops import export_npz
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.measurement_model import H_jacobian, innovation
from filters.motion_models import cv_model
from filters.state import STATE_DIM
from scripts import eval as bench_eval
from scripts.eval import SINGULAR_P_NEES, _INIT_OFFSET, bkn_filter, bkn_steps, knet_steps
from scripts.eval_knet import _INIT_OFFSET as TORCH_INIT_OFFSET, _priors
from scripts.tune_bkn import selected
from tests.test_numpy_forward import REPO, SEQUENCE_ATOL_STATE, exported, flight, knet_stand_in  # noqa: F401
from tests.test_replay import imm_tuning  # noqa: F401

NON_POWER_OF_TWO_REALISATIONS = 3
FULL_RANK_REALISATIONS = STATE_DIM + 1
TRAJECTORY_SEED = 8
STAND_IN_DROPOUT_P = 0.2
DROPOUT_DRAWS = 10_000
DROPOUT_FRACTION_4_SIGMA = 4 * np.sqrt(STAND_IN_DROPOUT_P * (1 - STAND_IN_DROPOUT_P) / DROPOUT_DRAWS)
MASK_EFFECT_FLOOR_M = 1.0
MEAN_SUMMATION_ATOL = 4 * np.spacing(2048.0)
SPREAD_SUMMATION_ATOL = 2 * MEAN_SUMMATION_ATOL


class NumpyMaskDropout(torch.nn.Module):
    """Torch-side dropout drawing its masks from a numpy Generator, so both backends see the same masks."""

    def __init__(self, dropout):
        super().__init__()
        self.numpy_dropout = dropout

    def forward(self, x):
        return torch.from_numpy(self.numpy_dropout(x.numpy()))


def stand_in_bkn(weights, seed=TRAJECTORY_SEED, dropout_p=STAND_IN_DROPOUT_P, realisations=FULL_RANK_REALISATIONS):
    return bkn_filter(weights, NOMINAL.dt, NOMINAL.R(), seed, dropout_p, realisations)


def bkn_run(flt, states, z, snapshot=None):
    return [tuple(np.atleast_1d(v).copy() for v in step) for step in bkn_steps(flt, NOMINAL.dt, NOMINAL.R(), states, z, snapshot)]


def test_bayesian_kalmannet_without_dropout_tracks_the_committed_scenario_exactly_like_kalmannet(flight, tmp_path):  # noqa: F811
    """MS-39: at p = 0 every realisation is vanilla KalmanNet, so the mean is its estimate bit for bit and P is 0."""
    states, z = flight
    weights = exported(knet_stand_in(), tmp_path)
    R = NOMINAL.R()
    flt = bkn_filter(weights, NOMINAL.dt, R, TRAJECTORY_SEED, dropout_p=0.0, realisations=NON_POWER_OF_TWO_REALISATIONS)
    snapshot = {}

    knet = knet_steps(weights, NOMINAL.dt, R, states, z)
    bkn = bkn_steps(flt, NOMINAL.dt, R, states, z, snapshot)
    for step, (knet_err, (bkn_err, *_)) in enumerate(zip(knet, bkn, strict=True)):
        np.testing.assert_array_equal(bkn_err, knet_err, err_msg=f"step {step}")
        np.testing.assert_array_equal(snapshot["P"], np.zeros((STATE_DIM, STATE_DIM)), err_msg=f"step {step}")


def test_ensemble_covariance_of_two_estimates_divides_the_spread_by_j_not_j_minus_1():
    """MS-39, eq 22b: estimates 2 apart give deviations of +-1, so every entry is (1 + 1) / 2 = 1."""
    low = np.arange(STATE_DIM, dtype=float)

    mean, P = numpy_forward.ensemble_mean_and_covariance(np.array([low, low + 2.0]))

    np.testing.assert_array_equal(mean, low + 1.0)
    np.testing.assert_array_equal(P, np.ones((STATE_DIM, STATE_DIM)))


def test_ensemble_of_three_estimates_has_their_mean_and_the_squared_deviations_over_three():
    """MS-39, eq 22a/22b: px at 0, 1, 5 has mean 2 and variance (4 + 1 + 9) / 3; the other states agree, so 0."""
    estimates = np.zeros((3, STATE_DIM))
    estimates[:, 0] = [0.0, 1.0, 5.0]
    expected_P = np.zeros((STATE_DIM, STATE_DIM))
    expected_P[0, 0] = 14.0 / 3.0

    mean, P = numpy_forward.ensemble_mean_and_covariance(estimates)

    np.testing.assert_array_equal(mean, np.eye(STATE_DIM)[0] * 2.0)
    np.testing.assert_array_equal(P, expected_P)


def test_ensemble_of_three_equal_estimates_averages_to_that_estimate_exactly():
    """MS-39: (0.1 + 0.1 + 0.1) / 3 rounds to 0.1 + 1 ulp, so a plain mean would leave P at ulp level, not 0."""
    estimates = np.full((3, STATE_DIM), 0.1)

    mean, P = numpy_forward.ensemble_mean_and_covariance(estimates)

    np.testing.assert_array_equal(mean, estimates[0])
    np.testing.assert_array_equal(P, np.zeros((STATE_DIM, STATE_DIM)))


def test_dropout_zeroes_a_fraction_p_of_the_units_and_scales_the_rest_by_one_over_1_minus_p():
    """MS-39: torch.nn.Dropout's convention at p = 0.2: 1/0.8 rounds to 1.25 exactly. 10,000 Bernoulli(0.2) draws:
    the zeroed fraction's sigma is 0.004."""
    dropout = numpy_forward.bernoulli_dropout(numpy_forward.dropout_rng(TRAJECTORY_SEED), STAND_IN_DROPOUT_P)

    dropped = dropout(np.ones(DROPOUT_DRAWS))

    assert set(np.unique(dropped)) == {0.0, 1.25}
    assert abs(np.mean(dropped == 0.0) - STAND_IN_DROPOUT_P) < DROPOUT_FRACTION_4_SIGMA


def test_dropout_stream_is_reproducible_and_is_not_the_measurement_noise_stream_of_the_same_seed():
    """MS-39: the page seeds both from the scenario seed; sharing a stream would move every filter's measurements."""
    draws = numpy_forward.dropout_rng(TRAJECTORY_SEED).random(4)

    np.testing.assert_array_equal(draws, numpy_forward.dropout_rng(TRAJECTORY_SEED).random(4))
    assert not np.any(draws == np.random.default_rng(TRAJECTORY_SEED).random(4))


def test_torch_and_numpy_gain_networks_given_the_same_masks_track_the_committed_scenario_alike(flight, tmp_path):  # noqa: F811
    """MS-39, MS-10: dropout sits at the same seven FC outputs in both backends, drawn in the same order.
    atol 1e-10 is test_numpy_forward's float64 summation-order budget over the same 296 steps."""
    states, z = flight
    net = knet_stand_in()
    weights = exported(net, tmp_path)
    net.dropout = NumpyMaskDropout(numpy_forward.bernoulli_dropout(numpy_forward.dropout_rng(TRAJECTORY_SEED), STAND_IN_DROPOUT_P))
    priors = _priors(NOMINAL.dt, NOMINAL.R())
    torch_filter = KalmanNetFilter(net, NOMINAL.dt)
    torch_filter.init_sequence(torch.tensor(states[None, 0]) + TORCH_INIT_OFFSET, *priors)
    numpy_dropout = numpy_forward.bernoulli_dropout(numpy_forward.dropout_rng(TRAJECTORY_SEED), STAND_IN_DROPOUT_P)
    numpy_filter = numpy_forward.KalmanNetFilter(numpy_forward.KalmanGainGRU(weights, dropout=numpy_dropout), NOMINAL.dt)
    numpy_filter.init_sequence(states[0] + _INIT_OFFSET, *(prior.numpy() for prior in priors))
    unmasked = numpy_forward.KalmanNetFilter(numpy_forward.KalmanGainGRU(weights), NOMINAL.dt)
    unmasked.init_sequence(states[0] + _INIT_OFFSET, *(prior.numpy() for prior in priors))

    mask_effect_m = 0.0
    with torch.no_grad():
        for z_t in z:
            x_torch = torch_filter.step(torch.tensor(z_t)[None])[0].numpy()
            x_numpy = numpy_filter.step(z_t)
            np.testing.assert_allclose(x_numpy, x_torch, rtol=0.0, atol=SEQUENCE_ATOL_STATE)
            mask_effect_m = max(mask_effect_m, np.linalg.norm(x_numpy[:3] - unmasked.step(z_t)[:3]))

    assert mask_effect_m > MASK_EFFECT_FLOOR_M


def test_bayesian_kalmannet_replays_bit_for_bit_from_the_same_seed_and_differs_from_another(flight, tmp_path):  # noqa: F811
    """MS-39: one dropout Generator per trajectory, so a replay of the page's seed redraws the page's masks."""
    states, z = flight
    weights = exported(knet_stand_in(), tmp_path)

    first = bkn_run(stand_in_bkn(weights), states, z)
    again = bkn_run(stand_in_bkn(weights), states, z)
    other = bkn_run(stand_in_bkn(weights, seed=TRAJECTORY_SEED + 1), states, z)

    for step_first, step_again in zip(first, again, strict=True):
        for value_first, value_again in zip(step_first, step_again, strict=True):
            np.testing.assert_array_equal(value_first, value_again)
    assert not np.array_equal(first[-1][0], other[-1][0])


def test_ten_realisations_whiten_the_error_through_p_where_p_0_scores_the_singular_sentinel(flight, tmp_path):  # noqa: F811
    """MS-39: rank P <= J - 1, so 10 realisations are the fewest that can whiten a 9-state error."""
    states, z = flight
    weights = exported(knet_stand_in(), tmp_path)

    spread = bkn_run(stand_in_bkn(weights), states[:50], z[:50])
    collapsed = bkn_run(stand_in_bkn(weights, dropout_p=0.0), states[:50], z[:50])

    assert all(np.isfinite(nees[0]) and nees[0] != SINGULAR_P_NEES for _, nees, _ in spread)
    assert all(nees[0] == SINGULAR_P_NEES for _, nees, _ in collapsed)


def test_innovation_covariance_propagates_the_previous_steps_spread_and_starts_at_r(flight, tmp_path):  # noqa: F811
    """MS-40: S_t = H_t F P_(t-1) Fᵀ H_tᵀ + R, with P_0 = 0, so the first step's S is R itself."""
    states, z = flight
    flt = stand_in_bkn(exported(knet_stand_in(), tmp_path))
    F, Q = cv_model(NOMINAL.dt)
    x0 = states[0] + _INIT_OFFSET
    flt.init_sequence(x0, Q, np.eye(STATE_DIM) * 100.0, NOMINAL.R())

    x1, P1, y1, S1 = flt.step(z[0])
    x2, P2, y2, S2 = flt.step(z[1])

    np.testing.assert_array_equal(S1, NOMINAL.R())
    np.testing.assert_array_equal(y1, innovation(z[0], F @ x0))
    H2 = H_jacobian(F @ x1)
    np.testing.assert_array_equal(S2, H2 @ F @ P1 @ F.T @ H2.T + NOMINAL.R())
    assert not np.array_equal(P1, P2)


def test_dropout_adds_no_weights_so_a_bayesian_checkpoint_loads_into_the_kalmannet_forward_pass():
    """MS-39: f, h and the GRU structure stay KalmanNet's; the exported .npz keys are the same."""
    bayesian, vanilla = KalmanGainGRU(dropout_p=STAND_IN_DROPOUT_P).state_dict(), KalmanGainGRU().state_dict()

    assert {name: w.shape for name, w in bayesian.items()} == {name: w.shape for name, w in vanilla.items()}


def test_dropout_hits_the_seven_fc_relu_outputs_in_forward_order_and_never_the_gain(tmp_path):
    """MS-39: FC5, FC6, FC1, FC7, FC2's hidden layer, FC3, FC4 (m = 9, n = 3, in_mult 5, out_mult 40)."""
    widths = []

    def recording(x):
        widths.append(x.shape)
        return x

    net = numpy_forward.KalmanGainGRU(exported(knet_stand_in(), tmp_path), dropout=recording)
    hidden = (np.zeros(81), np.zeros(81), np.zeros(9))
    net.forward(np.ones(3), np.ones(3), np.ones(9), np.ones(9), hidden)

    assert widths == [(45,), (45,), (9,), (30,), (90 * 40,), (81,), (81,)]


def test_ensemble_filter_output_is_the_mean_and_spread_of_j_independent_kalmannets_from_the_shared_prior(flight, tmp_path):  # noqa: F811
    """MS-39, eq 22a/22b: J KalmanNets drawing masks from one stream in realisation order, re-anchored at the mean
    each step, give the filter's x and P. A plain mean and the anchored one differ only in rounding: 4 ulp at
    |x| < 2048 m is 1.8e-12; P then moves by at most 2 x |deviation| (< 1 m) x 1.8e-12."""
    states, z = flight
    weights = exported(knet_stand_in(), tmp_path)
    _, Q = cv_model(NOMINAL.dt)
    priors = (Q, np.eye(STATE_DIM) * 100.0, NOMINAL.R())
    x0 = states[0] + _INIT_OFFSET
    flt = stand_in_bkn(weights)
    flt.init_sequence(x0, *priors)
    dropout = numpy_forward.bernoulli_dropout(numpy_forward.dropout_rng(TRAJECTORY_SEED), STAND_IN_DROPOUT_P)
    by_hand = [numpy_forward.KalmanNetFilter(numpy_forward.KalmanGainGRU(weights, dropout=dropout), NOMINAL.dt)
               for _ in range(FULL_RANK_REALISATIONS)]
    for realisation in by_hand:
        realisation.init_sequence(x0, *priors)

    for z_t in z[:3]:
        x, P, _, _ = flt.step(z_t)
        estimates = np.array([realisation.step(z_t) for realisation in by_hand])
        mean = estimates.mean(axis=0)
        np.testing.assert_allclose(x, mean, rtol=0.0, atol=MEAN_SUMMATION_ATOL)
        spread = (estimates - mean).T @ (estimates - mean) / FULL_RANK_REALISATIONS
        np.testing.assert_allclose(P, spread, rtol=0.0, atol=SPREAD_SUMMATION_ATOL)
        assert np.linalg.matrix_rank(P) == STATE_DIM
        for realisation in by_hand:
            realisation.x_posterior = x


def test_each_trajectory_draws_its_masks_from_one_generator_seeded_once(flight, tmp_path, monkeypatch):  # noqa: F811
    """MS-39: the seed is per trajectory, not per step: one dropout_rng call however many steps run."""
    states, z = flight
    seeded = []
    monkeypatch.setattr(numpy_forward, "dropout_rng", lambda seed: seeded.append(seed) or np.random.default_rng(seed))

    bkn_run(stand_in_bkn(exported(knet_stand_in(), tmp_path)), states[:5], z[:5])

    assert seeded == [TRAJECTORY_SEED]


def tuning_row(dropout_p, realisations, pos_rmse, diverged_fraction, anees_mean):
    return dict(
        dropout_p=dropout_p, realisations=realisations, pos_rmse=pos_rmse, diverged_fraction=diverged_fraction,
        anees_mean=anees_mean,
    )


def test_tuning_picks_the_anees_nearest_9_among_runs_as_accurate_and_as_stable_as_the_best():
    """MS-39: rows past 10% RMSE or 5 points of diverged fraction drop out even at ANEES 9; 9/2 and 18 tie in
    log ratio, so the smaller J wins."""
    rows = [
        tuning_row(0.1, 10, 1.0, 0.0, 18.0),
        tuning_row(0.1, 20, 1.0, 0.0, 4.5),
        tuning_row(0.2, 10, 1.25, 0.0, 9.0),
        tuning_row(0.3, 10, 1.0, 0.25, 9.0),
        tuning_row(0.2, 40, 1.1, 0.05, 40.0),
    ]

    assert selected(rows) == rows[0]


def test_eval_reports_bayesian_kalmannet_rmse_anees_nis_and_coverage_on_every_regime(imm_tuning, tmp_path, monkeypatch):  # noqa: F811
    """MS-39, MS-40: BKN sits beside IMM in results_table.csv on nominal, the four shifted regimes and the
    scenarios, and in the high-MC calibration file, with its own NEES and NIS."""
    data_dir, out, weights = tmp_path / "datasets", tmp_path / "results", tmp_path / "bkn.npz"
    toy = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1", "--mc-test", "2", "--n-highmc", "1"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy, "--mc-highmc", "2"], cwd=REPO, check=True)
    export_npz(knet_stand_in().state_dict(), weights)
    eval_argv = ["eval.py", "--data-dir", str(data_dir), "--imm-tuning", str(imm_tuning), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", [*eval_argv, "--bkn-weights", str(weights)])
    monkeypatch.chdir(REPO)

    bench_eval.main()

    with open(out / "results_table.csv") as f:
        bkn = {row["regime"]: row for row in csv.DictReader(f) if row["filter"] == "BKN"}
    highmc = {row["filter"]: row for row in json.loads((out / "anees_highmc.json").read_text())["results"]}
    assert list(bkn) == ["nominal", "high_maneuver", "high_noise", "low_noise", "heavy_tailed", "scenarios"]
    for row in bkn.values():
        assert all(row[key] != "" for key in ("pos_rmse", "anees_mean", "anis_mean", "coverage_68", "coverage_95"))
    assert list(highmc) == ["EKF", "UKF", "IMM", "BKN"]
    states, measurements = bench_eval._load(data_dir / "test.npz")
    nominal = bench_eval.evaluate_dataset(*bench_eval.bkn_entry(load_weights(weights), NOMINAL.dt, NOMINAL.R()), states, measurements)
    consistency = bench_eval.summarize_consistency(nominal)
    assert float(bkn["nominal"]["pos_rmse"]) == bench_eval.summarize_rmse(nominal)[0]
    for key in ("anees_mean", "anis_mean", "coverage_68", "coverage_95"):
        assert float(bkn["nominal"][key]) == consistency[key]


def test_ensemble_runs_the_20_realisations_validation_picked(tmp_path):
    """MS-39: J = 20 from scripts/tune_bkn.py on val.npz (write-up sec. 1.7); every consumer builds through bkn_filter."""
    flt = bkn_filter(exported(knet_stand_in(), tmp_path), NOMINAL.dt, NOMINAL.R(), TRAJECTORY_SEED)

    assert len(flt.realisations) == 20


def test_tuning_keeps_a_run_exactly_at_the_rmse_slack_of_the_best():
    """MS-39: 1.1 x 2.0 is 2.2 exactly in binary, so the edge row is inside the 10% band and wins on ANEES."""
    rows = [tuning_row(0.1, 10, 2.0, 0.0, 18.0), tuning_row(0.2, 10, 2.2, 0.0, 9.0)]

    assert selected(rows) == rows[1]


def test_tuning_keeps_a_run_exactly_at_the_diverged_slack_of_the_fewest():
    """MS-39: 0 + 0.05 is 0.05 exactly, so the edge row is inside the 5-point band and wins on ANEES."""
    rows = [tuning_row(0.1, 10, 1.0, 0.0, 18.0), tuning_row(0.2, 10, 1.0, 0.05, 9.0)]

    assert selected(rows) == rows[1]


def test_tuning_script_scores_every_j_for_each_checkpoint_on_val_under_the_nominal_r(tmp_path, monkeypatch):
    """MS-39, MS-40: rows come from val.npz, the dropout p each checkpoint's training log records, and nominal R."""
    from scripts import tune_bkn

    data_dir, weights = tmp_path / "datasets", tmp_path / "bkn" / "best.npz"
    toy = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1", "--mc-test", "1", "--n-highmc", "1"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy, "--mc-highmc", "1"], cwd=REPO, check=True)
    weights.parent.mkdir()
    export_npz(knet_stand_in().state_dict(), weights)
    (weights.parent / "training_log.json").write_text(json.dumps({"hyperparams": {"dropout_p": STAND_IN_DROPOUT_P}}))
    out = tmp_path / "tuning.json"
    argv = ["tune_bkn.py", str(weights), "--data-dir", str(data_dir), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(REPO)

    tune_bkn.main()

    tuning = json.loads(out.read_text())
    states, measurements = bench_eval._load(data_dir / "val.npz")
    expected = [
        tune_bkn.validation_row(weights, realisations, NOMINAL.dt, NOMINAL.R(), states, measurements)
        for realisations in (10, 20, 40)
    ]
    assert tuning["rows"] == json.loads(json.dumps(expected))
    assert tuning["best"] == json.loads(json.dumps(selected(expected)))


def test_latency_bench_times_the_kalmannet_and_the_20_realisation_ensemble_forward_passes(tmp_path, monkeypatch):
    """MS-39: the BKN figure is J numpy KalmanNet passes plus the NEES/NIS scoring the page runs each step."""
    from scripts import bench_latency

    nominal = {"dt": NOMINAL.dt, "sigma_az_rad": 0.001, "sigma_el_rad": 0.001, "sigma_range_m": 2.0}
    (tmp_path / "manifest.json").write_text(json.dumps({"regimes": {"nominal": nominal}}))
    tuning = tmp_path / "imm_tuning.json"
    tuning.write_text(json.dumps({"best": json.loads((REPO / "web" / "imm_tuning.json").read_text())["best"]}))
    weights = tmp_path / "bkn.npz"
    export_npz(knet_stand_in().state_dict(), weights)
    built = []
    monkeypatch.setattr(
        bench_latency, "bkn_steps", lambda flt, *args: built.append(flt) or bkn_steps(flt, *args)
    )
    out = tmp_path / "latency.json"
    argv = ["bench_latency.py", "--data-dir", str(tmp_path), "--imm-tuning", str(tuning), "--bkn-weights", str(weights),
            "--n-warmup", "1", "--n-measure", "2", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    bench_latency.main()

    timed = json.loads(out.read_text())["results_ms"]
    assert {"KalmanNet-numpy", "BKN-numpy-J20"} <= timed.keys()
    assert [len(flt.realisations) for flt in built] == [20]
    assert all(timed[name]["median_ms"] > 0.0 for name in ("KalmanNet-numpy", "BKN-numpy-J20"))
