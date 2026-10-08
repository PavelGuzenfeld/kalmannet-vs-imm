import csv
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from data.config import NOMINAL
from filters.knet import numpy_forward
from filters.knet.gain_network import ConcreteDropout, KalmanGainGRU
from filters.knet.knet_filter import BayesianKalmanNetFilter, KalmanNetFilter
from filters.knet.numpy_forward import DROPOUT_LAYERS, load_weights
from filters.state import ACC, STATE_DIM
from filters.torch_ops import export_npz
from scripts import eval as bench_eval
from scripts import train_knet, tune_bkn
from tests.test_numpy_forward import REPO, exported, knet_stand_in
from tests.test_replay import imm_tuning  # noqa: F401

P_INIT_LOW, P_INIT_HIGH = 0.5, 0.8
P_MOVED_FLOOR = 1e-4


def test_eq23_training_learns_one_p_per_dropout_layer_and_the_ensemble_runs_at_the_learned_p(tmp_path, monkeypatch):
    """MS-39, MS-50: --learned-p keeps a checkpoint only once beta is ramped up; seven p drawn in
    [0.5, 0.8] each move > 1e-4 (76 Adam steps of ~3e-4 in logit, far above a round trip's ulp); validation and
    evaluation run the ensemble at exactly those p."""
    data_dir, out = tmp_path / "datasets", tmp_path / "bkn_eq23"
    toy = ["--n-train", "2", "--n-val", "1", "--n-test", "1", "--n-sweep", "1", "--mc-test", "1", "--n-highmc", "1"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy, "--mc-highmc", "1"], cwd=REPO, check=True)
    argv = ["train_knet.py", "--data-dir", str(data_dir), "--out", str(out), "--epochs", "2", "--batch-size", "2",
            "--learned-p", "--realisations", "2", "--beta", "0.5", "--beta-ramp-epochs", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.chdir(REPO)

    train_knet.main()

    checkpoint = torch.load(out / "best.pt", weights_only=True)
    export_npz(checkpoint, out / "best.npz")
    weights = numpy_forward.load_weights(out / "best.npz")
    log = json.loads((out / "training_log.json").read_text())
    assert log["best_epoch"] == 1
    assert [epoch["beta"] for epoch in log["history"]] == [0.25, 0.5]
    initial = log["initial_dropout_p"]
    learned = numpy_forward.learned_dropout_p(weights)
    assert list(initial) == list(DROPOUT_LAYERS)
    assert all(P_INIT_LOW <= p <= P_INIT_HIGH for p in initial.values())
    assert np.all(np.abs(learned - list(initial.values())) > P_MOVED_FLOOR)
    states, measurements = bench_eval._load(data_dir / "val.npz")
    row = tune_bkn.validation_row(out / "best.npz", 10, NOMINAL.dt, NOMINAL.R(), states, measurements)
    expected = bench_eval.evaluate_dataset(
        *bench_eval.bkn_entry(weights, NOMINAL.dt, NOMINAL.R(), learned, 10), states, measurements
    )
    assert row["dropout_p"] == learned.tolist()
    assert row["pos_rmse"] == bench_eval.summarize_rmse(expected)[0]
    assert row["anees_mean"] == bench_eval.summarize_consistency(expected)["anees_mean"]


DROPPED_FRACTION_DRAWS = 10_000
CONCRETE_P = 0.25
DROPPED_FRACTION_4_SIGMA = 4 * np.sqrt(CONCRETE_P * (1 - CONCRETE_P) / DROPPED_FRACTION_DRAWS)
EVEN_P = 0.5
DROPPED_UNITS = {"FC5": 45, "FC6": 45, "FC1": 9, "FC7": 30, "FC2": 3600, "FC3": 81, "FC4": 81}
CONSUMER_COLUMNS = {
    "FC5": ("GRU_Q.weight_ih_l0", 0, 45),
    "FC6": ("GRU_Sigma.weight_ih_l0", 81, 126),
    "FC1": ("GRU_S.weight_ih_l0", 0, 9),
    "FC7": ("GRU_S.weight_ih_l0", 9, 39),
    "FC2": ("FC2.2.weight", 0, 3600),
    "FC3": ("FC4.0.weight", 81, 162),
    "FC4": ("GRU_Sigma.weight_hh_l0", 0, 81),
}
ENTROPY_SUM_RTOL = 8 * 2.0**-53


def zeroed(net):
    with torch.no_grad():
        for parameter in net.parameters():
            if parameter.dim() > 1:
                parameter.zero_()
    return net


def concrete_net(p=EVEN_P):
    return zeroed(KalmanGainGRU(learned_p_init=[p] * len(DROPOUT_LAYERS)).double())


def test_each_dropout_layers_regularised_weights_are_the_columns_its_dropped_units_feed():
    """MS-50: eq 24's θ_λ per [47] is the next layer's input weights for the dropped units; FC4's
    units become the Sigma GRU's hidden state, so its columns are that GRU's hidden-to-hidden weights."""
    for layer, (name, start, stop) in CONSUMER_COLUMNS.items():
        net = concrete_net()
        with torch.no_grad():
            net.get_parameter(name)[:, start:stop] = 1.0

        sums = {other: theta.sum().item() for other, theta in net.dropped_unit_weights().items()}

        rows = net.get_parameter(name).shape[0]
        assert sums == {other: (rows * (stop - start) if other == layer else 0.0) for other in DROPOUT_LAYERS}, layer


def test_kl_weight_term_is_c1_times_the_squared_dropped_unit_weights_over_1_minus_p():
    """MS-50, eq 24 with c2 = 0: FC2's 27 x 3600 output weights at 0.5 square to 24,300; p = 0.5 doubles it."""
    net = concrete_net()
    with torch.no_grad():
        net.FC2[2].weight.fill_(0.5)

    assert net.kl_regulariser(c1=1.0, c2=0.0).item() == 48_600.0


def test_kl_entropy_term_is_c2_times_the_bernoulli_entropy_per_dropped_unit_and_rewards_spread():
    """MS-50, eq 24/25 with c1 = 0: H(0.5) = ln 2 per dropped unit, 3,891 units, subtracted.
    rtol: seven products and seven sums, each rounded to half an ulp."""
    units = sum(DROPPED_UNITS.values())

    regulariser = concrete_net().kl_regulariser(c1=0.0, c2=1.0).item()

    assert regulariser == pytest.approx(-units * math.log(2.0), rel=ENTROPY_SUM_RTOL)
    assert {layer: theta.shape[1] for layer, theta in concrete_net().dropped_unit_weights().items()} == DROPPED_UNITS


def test_concrete_mask_keeps_the_lowest_uniform_draw_and_drops_the_highest_without_nan(monkeypatch):
    """MS-50, [47]: a unit drops when u > 1 - p; at temperature 0.1 the extreme draws 0 and
    1 - 2⁻⁵³ saturate the mask to exactly 1 and 0, and eps keeps both logs finite."""
    draws = torch.tensor([0.0, 1.0 - 2.0**-53], dtype=torch.float64)
    monkeypatch.setattr(torch, "rand_like", lambda x: draws)

    out = ConcreteDropout(CONCRETE_P)(torch.ones(2, dtype=torch.float64))

    assert out.tolist() == [1.0 / (1.0 - CONCRETE_P), 0.0]


def test_concrete_mask_at_p_0_keeps_every_unit_and_at_p_next_to_1_drops_it_without_nan(monkeypatch):
    """MS-50: eps keeps log p and log(1 - p) finite when the learned p saturates; u = 0.5."""
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.full_like(x, 0.5))
    x = torch.ones(1, dtype=torch.float64)

    assert ConcreteDropout(0.0)(x).tolist() == [1.0]
    assert ConcreteDropout(1.0 - 2.0**-30)(x).tolist() == [0.0]


def test_concrete_dropout_drops_a_fraction_p_of_the_units_and_passes_gradient_to_p():
    """MS-50: the relaxed mask passes 1/2 exactly when u = 1 - p, so P(dropped) = p.
    10,000 draws at p = 0.25: the fraction's sigma is 0.0043."""
    torch.manual_seed(0)
    dropout = ConcreteDropout(CONCRETE_P)

    out = dropout(torch.ones(DROPPED_FRACTION_DRAWS, dtype=torch.float64))
    out.sum().backward()

    dropped = (out < 0.5 / (1.0 - CONCRETE_P)).double().mean().item()
    assert abs(dropped - CONCRETE_P) < DROPPED_FRACTION_4_SIGMA
    assert dropout.p_logit.grad.item() != 0.0


def test_eq20_loss_blends_the_weighted_mse_and_the_absolute_second_moment_gap_by_beta():
    """MS-50: on ax the loss weight is 1/0.5² = 4. e = 1 with Σ = 0.25 gives MSE 4 and M2 4 x 0.75 = 3."""
    truth = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    mean = truth.clone()
    mean[0, ACC.start] = 1.0
    variance = torch.zeros_like(mean)
    variance[0, ACC.start] = 0.25

    assert train_knet.empirical_loss(mean, variance, truth, beta=0.25).item() == 0.75 * 4.0 + 0.25 * 3.0


def test_second_moment_term_penalises_an_overdispersed_spread_as_much_as_an_underdispersed_one():
    """MS-50, eq 19's absolute value: Σ = 2.25 over e² = 1 costs 4 x 1.25 = 5, not -5."""
    truth = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    mean = truth.clone()
    mean[0, ACC.start] = 1.0
    variance = torch.zeros_like(mean)
    variance[0, ACC.start] = 2.25

    assert train_knet.empirical_loss(mean, variance, truth, beta=1.0).item() == 5.0


def test_training_ensemble_returns_mean_and_spread_and_reanchors_every_realisation_at_the_mean(monkeypatch):
    """MS-39, MS-50: px at 0, 1, 5 across J = 3 has mean 2 and spread (4 + 1 + 9) / 3; every
    realisation's next prediction starts from that mean, as Algorithm 1 line 6 and the numpy filter do."""
    estimates = torch.zeros(3, STATE_DIM, dtype=torch.float64)
    estimates[:, 0] = torch.tensor([0.0, 1.0, 5.0])
    monkeypatch.setattr(KalmanNetFilter, "step", lambda self, z: estimates)
    flt = BayesianKalmanNetFilter(concrete_net(), NOMINAL.dt, realisations=3)

    mean, variance = flt.step(torch.zeros(1, 3, dtype=torch.float64))

    expected_mean = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    expected_mean[0, 0] = 2.0
    expected_variance = torch.zeros(1, STATE_DIM, dtype=torch.float64)
    expected_variance[0, 0] = 14.0 / 3.0
    assert torch.equal(mean, expected_mean)
    assert torch.equal(variance, expected_variance)
    assert torch.equal(flt.x_posterior, expected_mean.repeat(3, 1))


def test_numpy_dropout_takes_one_p_per_layer_in_forward_order_and_wraps_to_the_first(tmp_path):
    """MS-50: p 0.5 on FC6 only: its units are 0 or 2, every other layer passes through; the
    eighth call is the next realisation's FC5."""
    per_layer = [0.0, EVEN_P, 0.0, 0.0, 0.0, 0.0, 0.0]
    dropout = numpy_forward.bernoulli_dropout(numpy_forward.dropout_rng(0), per_layer)

    outputs = [dropout(np.ones(DROPPED_UNITS[layer])) for layer in (*DROPOUT_LAYERS, "FC5")]

    assert set(np.unique(outputs[1])) == {0.0, 2.0}
    for index in (0, 2, 3, 4, 5, 6, 7):
        np.testing.assert_array_equal(outputs[index], np.ones_like(outputs[index]))


def test_learned_p_reads_each_layers_logit_in_forward_order_and_is_absent_at_a_fixed_p(tmp_path):
    """MS-50: logits 0, -inf and +inf are p = 0.5, 0 and 1 exactly."""
    logits = [0.0, -np.inf, np.inf, 0.0, -np.inf, np.inf, 0.0]
    weights = {f"dropout.{layer}.p_logit": np.array([logit]) for layer, logit in zip(DROPOUT_LAYERS, logits)}

    np.testing.assert_array_equal(numpy_forward.learned_dropout_p(weights), [0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5])
    assert numpy_forward.learned_dropout_p(exported(knet_stand_in(), tmp_path)) is None


def test_eval_runs_a_concrete_dropout_checkpoint_at_its_learned_p(imm_tuning, tmp_path, monkeypatch):  # noqa: F811
    """MS-50: results_table.csv's BKN row for eq 23 weights is the ensemble at the learned p, not at
    the fixed BKN_DROPOUT_P, and at the J that --bkn-realisations names."""
    data_dir, out, weights = tmp_path / "datasets", tmp_path / "results", tmp_path / "bkn.npz"
    toy = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1", "--mc-test", "2", "--n-highmc", "1"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy, "--mc-highmc", "2"], cwd=REPO, check=True)
    net = knet_stand_in()
    net.dropout = KalmanGainGRU(learned_p_init=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]).double().dropout
    export_npz(net.state_dict(), weights)
    monkeypatch.setattr(sys, "argv", ["eval.py", "--data-dir", str(data_dir), "--imm-tuning", str(imm_tuning),
                                      "--out", str(out), "--bkn-weights", str(weights), "--bkn-realisations", "10"])
    monkeypatch.chdir(REPO)

    bench_eval.main()

    with open(out / "results_table.csv") as f:
        nominal = next(row for row in csv.DictReader(f) if row["filter"] == "BKN" and row["regime"] == "nominal")
    loaded = load_weights(weights)
    states, measurements = bench_eval._load(data_dir / "test.npz")
    expected = bench_eval.evaluate_dataset(
        *bench_eval.bkn_entry(loaded, NOMINAL.dt, NOMINAL.R(), numpy_forward.learned_dropout_p(loaded), 10),
        states,
        measurements,
    )
    assert float(nominal["pos_rmse"]) == bench_eval.summarize_rmse(expected)[0]
    assert float(nominal["anees_mean"]) == bench_eval.summarize_consistency(expected)["anees_mean"]
    fixed_p = bench_eval.evaluate_dataset(
        *bench_eval.bkn_entry(loaded, NOMINAL.dt, NOMINAL.R(), bench_eval.BKN_DROPOUT_P, 10), states, measurements
    )
    assert float(nominal["anees_mean"]) != bench_eval.summarize_consistency(fixed_p)["anees_mean"]
