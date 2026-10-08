import csv
import json
import warnings

import numpy as np
import pytest
from scipy.stats import chi2

from filters.metrics import (
    anees_over_counted,
    anees_with_bounds,
    counted_mask,
    counted_span,
    divergence_steps,
    divergence_summary,
    empirical_coverage,
    fraction_inside_bounds,
    nees,
    nis,
    position_rmse,
    velocity_rmse,
)
from scripts import eval as bench_eval
from scripts import eval_knet, eval_rkn


def test_position_and_velocity_rmse_known_values():
    x_true = np.zeros((2, 9))
    x_est = np.zeros((2, 9))
    x_est[0, 0] = 3.0
    x_est[0, 1] = 4.0  # position error norm 5 at step 0, 0 at step 1
    rmse = position_rmse(x_true, x_est)
    assert np.isclose(rmse, np.sqrt((25 + 0) / 2))

    x_est_v = np.zeros((2, 9))
    x_est_v[:, 3] = 3.0
    x_est_v[:, 4] = 4.0
    assert np.isclose(velocity_rmse(x_true, x_est_v), 5.0)


def test_nees_matches_hand_computation_for_identity_covariance():
    e = np.array([1.0, 2.0, 3.0])
    x_true = e
    x_est = np.zeros(3)
    P = np.eye(3)
    value = nees(x_true, x_est, P, idx_slice=slice(None))
    assert np.isclose(value, 1.0 + 4.0 + 9.0)


def test_nis_matches_hand_computation():
    y = np.array([2.0, 0.0])
    S = np.diag([4.0, 1.0])
    value = nis(y, S)
    assert np.isclose(value, 1.0)  # 2^2/4 + 0


def test_anees_bounds_match_scipy_chi2_directly():
    # A well-calibrated filter has E[NEES] = dim (the chi-square mean), not 1
    # -- use that as the "perfectly consistent" synthetic value.
    N, dim = 10, 6
    nees_values = np.full((N, 5), float(dim))
    anees, (low, high) = anees_with_bounds(nees_values, dim=dim, mc_axis=0)
    assert np.allclose(anees, float(dim))
    assert np.isclose(low, chi2.ppf(0.025, df=N * dim) / N)
    assert np.isclose(high, chi2.ppf(0.975, df=N * dim) / N)
    assert low < dim < high  # a perfectly consistent filter should pass


def test_fraction_inside_bounds_counts_correctly():
    values = np.array([0.5, 1.0, 5.0, 10.0])
    frac = fraction_inside_bounds(values, (0.9, 5.5))
    assert np.isclose(frac, 0.5)  # 1.0 and 5.0 are inside


def test_empirical_coverage_all_inside_gives_full_coverage():
    dim = 3
    threshold_95 = chi2.ppf(0.95, df=dim)
    values = np.full(100, threshold_95 * 0.5)
    cov = empirical_coverage(values, dim=dim, levels=(0.68, 0.95))
    assert cov[0.95] == 1.0


BENCH_STEPS = 30
DIVERGES_AT = 25
WITHIN_LIMIT_M = [3.0, 4.0, 0.0]
PAST_LIMIT_M = [300.0, 0.0, 0.0]
LAST_COUNTED_M = [19.0, 7.0, 0.0]


class ScriptedPositionError:
    """Estimates its scripted position error against a zero truth, with unit covariances; counts its steps."""

    def __init__(self, errors_m):
        self.errors_m, self.steps = errors_m, 0

    def update(self, x, P, z):
        return self.step(x, P, z)

    def step(self, x, P, z):
        estimate = np.zeros(9)
        estimate[:3] = self.errors_m[self.steps]
        self.steps += 1
        return estimate, np.eye(9), np.zeros(3), np.eye(3)


def test_benchmark_stops_one_of_four_runs_at_its_divergence_step_and_drops_that_step_from_the_rmse():
    """Ten steps past burn-in per run at 25 m^2, but the diverging run's last counted step sits at 410 m^2: the
    35 counted squares sum to 36 * 35, and that step's ANEES cell reads (25 + 410) / 2 among nineteen at 25."""
    within = [WITHIN_LIMIT_M] * BENCH_STEPS
    diverging = within[: DIVERGES_AT - 1] + [LAST_COUNTED_M, PAST_LIMIT_M] + within[DIVERGES_AT + 1 :]
    filters = [ScriptedPositionError(errors_m) for errors_m in (within, within, within, diverging)]
    made = iter(filters)

    result = bench_eval.evaluate_dataset(
        lambda: next(made), bench_eval.run_ekf_like, np.zeros((2, BENCH_STEPS, 9)), np.zeros((2, 2, BENCH_STEPS, 3))
    )

    assert bench_eval.summarize_rmse(result) == (6.0, 0.0)
    assert bench_eval.summarize_consistency(result)["anees_mean"] == (19 * 25.0 + (25.0 + 410.0) / 2) / 20
    assert divergence_summary(divergence_steps(result["err"])) == {
        "diverged_fraction": 0.25,
        "median_divergence_step": float(DIVERGES_AT),
    }
    assert [flt.steps for flt in filters] == [BENCH_STEPS] * 3 + [DIVERGES_AT + 1]


def two_sided_95(runs, dim):
    alpha = 1 - 0.95
    return chi2.ppf(alpha / 2, df=runs * dim) / runs, chi2.ppf(1 - alpha / 2, df=runs * dim) / runs


def test_anees_cell_averages_only_its_counted_runs_and_is_tested_against_the_bounds_for_their_count():
    """Run 1 stops after step 0 and run 0 after step 1, so step 0 has two counted runs, step 1 one, step 2 none.
    Per trajectory: the edges themselves, inside; then just past each edge; half the eight cells lie inside."""
    dim, uncounted = 3, 1e6
    (low2, high2), (low1, high1) = two_sided_95(2, dim), two_sided_95(1, dim)
    cells = [
        (low2, low1),
        (high2, high1),
        (np.nextafter(low2, 0), np.nextafter(high1, np.inf)),
        (np.nextafter(high2, np.inf), np.nextafter(low1, 0)),
    ]
    nees = np.array([[[two_runs, one_run, uncounted], [two_runs, uncounted, uncounted]] for two_runs, one_run in cells])
    counted = np.array([[[True, True, False], [True, False, False]]] * len(cells))

    anees, inside = anees_over_counted(nees, counted, dim=dim, mc_axis=1)

    np.testing.assert_array_equal(anees, np.ravel(cells))
    assert inside == 0.5


def test_counted_mask_marks_each_runs_steps_before_its_divergence_step():
    err = np.zeros((2, 4, 9))
    err[0, 2, 0] = PAST_LIMIT_M[0]
    err[0, 3] = np.nan

    np.testing.assert_array_equal(counted_mask(err), [[True, True, False, False], [True, True, True, True]])


def test_counted_span_of_a_stopped_run_runs_from_its_burn_in_to_its_divergence_step():
    err = np.zeros((5, 9))
    err[4, 1] = PAST_LIMIT_M[0]

    assert counted_span(err, burn_in=1) == slice(1, 4)


def test_consistency_cells_average_over_the_monte_carlo_runs_with_bounds_for_the_run_count():
    """Averaged over the two runs each step reads 0, 0 and 30 (NIS 15): outside the two-run bands. Averaged over
    the three steps instead they would read 10 and 5, inside the three-step bands."""
    result = {
        "err": np.zeros((1, 2, 3, 9)),
        "nees": np.array([[[0.0, 0.0, 30.0]] * 2]),
        "nis": np.array([[[0.0, 0.0, 15.0]] * 2]),
    }

    consistency = bench_eval.summarize_consistency(result, burn_in=0)

    assert (consistency["anees_fraction_inside_95"], consistency["anis_fraction_inside_95"]) == (0.0, 0.0)
    assert (consistency["anees_bounds"], consistency["anis_bounds"]) == (two_sided_95(2, 9), two_sided_95(2, 3))


def test_divergence_summary_is_the_diverged_share_of_runs_and_the_median_over_the_diverged_ones_only():
    assert divergence_summary([np.nan, 3.0, 10.0, 4.0]) == {"diverged_fraction": 0.75, "median_divergence_step": 4.0}
    assert divergence_summary([2.0, 5.0]) == {"diverged_fraction": 1.0, "median_divergence_step": 3.5}
    assert divergence_summary([np.nan, np.nan]) == {"diverged_fraction": 0.0, "median_divergence_step": None}


def stopped_run(nees_values, diverges_at):
    """One run, burn-in 0: zero error up to a 300 m divergence step, carrying the given NEES (NIS alike)."""
    err = np.zeros((1, 1, len(nees_values), 9))
    err[0, 0, diverges_at, 0] = PAST_LIMIT_M[0]
    nees_values = np.array([[nees_values]])
    return {"err": err, "nees": nees_values, "nis": nees_values}


def test_coverage_counts_only_the_steps_before_the_divergence_step():
    """Counted NEES 1, 12 and 30 against the 68% (10.4) and 95% (16.9) chi-square(9) levels: one and two of three.
    The divergence step's 9e4 would make them one and two of four."""
    result = stopped_run([1.0, 12.0, 30.0, 9e4], diverges_at=3)

    for summary in (bench_eval.summarize_consistency(result, burn_in=0), eval_rkn.summarize(result, burn_in=0)):
        assert (summary["coverage_68"], summary["coverage_95"]) == (1 / 3, 2 / 3)


class ScriptedIMM:
    """IMM-shaped: estimates its scripted position error against a zero truth; counts its steps."""

    def __init__(self, errors_m):
        self.position, self.steps = ScriptedPositionError(errors_m), 0

    def init_state(self, x, P):
        return {"mu": np.full(4, 0.25)}

    def step(self, state, z):
        x, P, y, S = self.position.step(None, None, z)
        self.steps += 1
        return x, P, state, y, S


def test_imm_run_is_stepped_through_its_divergence_step_and_no_further():
    within = [WITHIN_LIMIT_M] * BENCH_STEPS
    imm = ScriptedIMM(within[:DIVERGES_AT] + [PAST_LIMIT_M] + within[DIVERGES_AT + 1 :])

    err, _, _ = bench_eval.run_imm(imm, np.zeros((BENCH_STEPS, 9)), np.zeros((BENCH_STEPS, 3)))

    assert (imm.steps, len(err)) == (DIVERGES_AT + 1, DIVERGES_AT + 1)


def test_scenarios_row_cuts_each_flight_before_its_divergence_step_and_pools_the_rest():
    """Flight one counts steps 20-23 at 25 m^2 and step 24 at 190 m^2, flight two ten steps at 25 m^2: 540 over 15
    is 36. Keeping flight one's 300 m divergence step would not."""
    within = [WITHIN_LIMIT_M] * BENCH_STEPS
    diverging = within[: DIVERGES_AT - 1] + [[9.0, 10.0, 3.0], PAST_LIMIT_M] + within[DIVERGES_AT + 1 :]
    made = iter([ScriptedPositionError(diverging), ScriptedPositionError(within)])
    flights = [(np.zeros((BENCH_STEPS, 9)), np.zeros((BENCH_STEPS, 3)))] * 2

    row = bench_eval.scenarios_row("EKF", lambda: next(made), bench_eval.run_ekf_like, flights)

    assert (row["pos_rmse"], row["vel_rmse"]) == (6.0, 0.0)
    assert (row["diverged_fraction"], row["median_divergence_step"]) == (0.5, float(DIVERGES_AT))


def test_an_error_off_only_in_altitude_counts_toward_divergence():
    err = np.zeros((1, 3, 9))
    err[0, 1, 2] = 201.0

    assert divergence_steps(err).tolist() == [1.0]
    assert counted_span(err[0], burn_in=0) == slice(0, 1)


def test_every_run_diverging_inside_its_burn_in_reads_nan_metrics_without_a_warning():
    within = [WITHIN_LIMIT_M] * BENCH_STEPS
    early = within[:5] + [PAST_LIMIT_M] + within[6:]
    made = iter([ScriptedPositionError(early) for _ in range(4)])
    result = bench_eval.evaluate_dataset(
        lambda: next(made), bench_eval.run_ekf_like, np.zeros((2, BENCH_STEPS, 9)), np.zeros((2, 2, BENCH_STEPS, 3))
    )
    knet_runs = [result["err"][0, 0, :6]]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rmse = bench_eval.summarize_rmse(result)
        consistency = bench_eval.summarize_consistency(result)
        rkn = eval_rkn.summarize({"err": result["err"], "nees": result["nees"]})
        knet = eval_knet.regime_row("nominal", knet_runs)

    read = [*rmse, knet["pos_rmse"], knet["vel_rmse"], rkn["pos_rmse"], rkn["anees_mean"], rkn["coverage_95"]]
    read += [consistency[key] for key in ("anees_mean", "anees_fraction_inside_95", "anis_mean", "coverage_68")]
    assert all(np.isnan(value) for value in read)
    assert divergence_summary(divergence_steps(result["err"]))["diverged_fraction"] == 1.0


def test_coverage_counts_a_nees_exactly_at_each_level_as_inside():
    levels = (0.68, 0.95)
    at_levels = np.array([chi2.ppf(level, df=9) for level in levels])

    assert empirical_coverage(at_levels, dim=9, levels=levels) == {0.68: 0.5, 0.95: 1.0}


def test_velocity_rmse_is_the_root_mean_square_of_the_velocity_error_norm():
    result = stopped_run([1.0, 1.0], diverges_at=1)
    result["err"][0, 0, 0, 3:6] = [3.0, 4.0, 0.0]

    assert bench_eval.summarize_rmse(result, burn_in=0)[1] == 5.0
    assert eval_rkn.summarize(result, burn_in=0)["vel_rmse"] == 5.0


def test_counted_mask_latches_at_the_first_step_past_200_m_though_the_error_comes_back_under():
    err = np.zeros((1, 5, 9))
    err[0, 1, 0] = PAST_LIMIT_M[0]
    err[0, 3, 0] = PAST_LIMIT_M[0]

    np.testing.assert_array_equal(counted_mask(err), [[True, False, False, False, False]])


def test_counted_mask_keeps_a_step_at_exactly_200_m_and_drops_the_first_float_above_it():
    err = np.zeros((2, 4, 9))
    err[0, 2, 0] = 200.0
    err[1, 2, 0] = np.nextafter(200.0, np.inf)

    np.testing.assert_array_equal(counted_mask(err), [[True] * 4, [True, True, False, False]])


@pytest.mark.parametrize(
    ("error_m", "pulled"), [(200.0, BENCH_STEPS), (np.nextafter(200.0, np.inf), 3)], ids=["at_limit", "past_limit"]
)
def test_eval_run_is_stopped_only_by_an_error_past_200_m_not_by_one_at_it(error_m, pulled):
    errors_m = [WITHIN_LIMIT_M] * 2 + [[error_m, 0.0, 0.0]] + [WITHIN_LIMIT_M] * (BENCH_STEPS - 3)
    flt = ScriptedPositionError(errors_m)

    err, _, _ = bench_eval.run_ekf_like(flt, np.zeros((BENCH_STEPS, 9)), np.zeros((BENCH_STEPS, 3)))

    assert (flt.steps, len(err)) == (pulled, pulled)


def test_nis_averages_only_the_counted_steps_of_a_run_stopped_part_way():
    """Counted NIS reads 3 in the 95% band for df 3; the stopped step's 9e4 would drop the inside share to 3/4."""
    err = np.zeros((1, 1, 4, 9))
    err[0, 0, 3, 0] = PAST_LIMIT_M[0]
    result = {"err": err, "nees": np.full((1, 1, 4), 9.0), "nis": np.array([[[3.0, 3.0, 3.0, 9e4]]])}

    consistency = bench_eval.summarize_consistency(result, burn_in=0)

    assert (consistency["anis_mean"], consistency["anis_fraction_inside_95"]) == (3.0, 1.0)


def diverging_and_healthy_runs():
    err = np.zeros((1, 2, BENCH_STEPS, 9))
    err[0, 0, DIVERGES_AT, 0] = PAST_LIMIT_M[0]
    err[0, 0, DIVERGES_AT + 1 :] = np.nan
    ones = np.ones((1, 2, BENCH_STEPS))
    return {"err": err, "nees": ones, "nis": ones}


DIVERGENCE_COLUMNS = {"diverged_fraction": 0.5, "median_divergence_step": float(DIVERGES_AT)}


def test_knet_regime_row_of_a_nominal_regime_carries_the_divergence_columns():
    runs = [np.zeros((DIVERGES_AT + 1, 9)), np.zeros((BENCH_STEPS, 9))]
    runs[0][DIVERGES_AT, 0] = PAST_LIMIT_M[0]

    row = eval_knet.regime_row("nominal", runs)

    assert {key: row[key] for key in DIVERGENCE_COLUMNS} == DIVERGENCE_COLUMNS


def test_rkn_summary_of_a_nominal_regime_carries_the_divergence_columns():
    summary = eval_rkn.summarize(diverging_and_healthy_runs())

    assert {key: summary[key] for key in DIVERGENCE_COLUMNS} == DIVERGENCE_COLUMNS


def test_eval_writes_the_divergence_columns_on_the_nominal_and_sweep_regime_rows(tmp_path, monkeypatch):
    regime = {"dt": 0.02, "sigma_az_rad": 0.001, "sigma_el_rad": 0.001, "sigma_range_m": 2.0}
    data_dir, out, scenarios = tmp_path / "datasets", tmp_path / "results", tmp_path / "scenarios"
    for directory in (data_dir, scenarios):
        directory.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps({"regimes": {"nominal": regime, "high_noise": regime}}))
    (data_dir / "test_high_noise.npz").touch()
    tuning = tmp_path / "imm_tuning.json"
    tuning.write_text(json.dumps({"best": {
        "omega_ct_deg": 15.0, "p_stay": 0.97, "ca_sigma_a": 3.0, "ca_tau": 4.0, "cv_sigma_a": 0.05, "ct_sigma_a": 0.3,
    }}))
    monkeypatch.setattr(bench_eval, "_load", lambda path: (np.zeros((1, BENCH_STEPS, 9)), np.zeros((1, 2, BENCH_STEPS, 3))))
    monkeypatch.setattr(bench_eval, "evaluate_dataset", lambda *_: diverging_and_healthy_runs())
    argv = ["eval.py", "--data-dir", str(data_dir), "--imm-tuning", str(tuning), "--scenario-dir", str(scenarios), "--out", str(out),
            "--bkn-weights", str(tmp_path / "missing.npz"), "--ukn-weights", str(tmp_path / "missing.npz")]
    monkeypatch.setattr("sys.argv", argv)

    bench_eval.main()

    with open(out / "results_table.csv") as f:
        rows = list(csv.DictReader(f))
    assert [row["regime"] for row in rows] == ["nominal"] * 3 + ["high_noise"] * 3
    assert {(row["diverged_fraction"], row["median_divergence_step"]) for row in rows} == {("0.5", "25.0")}
