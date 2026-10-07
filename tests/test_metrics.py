import numpy as np
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
    within = [WITHIN_LIMIT_M] * BENCH_STEPS
    diverging = within[:DIVERGES_AT] + [PAST_LIMIT_M] + within[DIVERGES_AT + 1 :]
    filters = [ScriptedPositionError(errors_m) for errors_m in (within, within, within, diverging)]
    made = iter(filters)

    result = bench_eval.evaluate_dataset(
        lambda: next(made), bench_eval.run_ekf_like, np.zeros((2, BENCH_STEPS, 9)), np.zeros((2, 2, BENCH_STEPS, 3))
    )

    assert bench_eval.summarize_rmse(result) == (5.0, 0.0)
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
