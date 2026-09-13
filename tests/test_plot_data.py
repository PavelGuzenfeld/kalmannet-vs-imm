import csv
import json
import math
from unittest.mock import patch

from filters.plot_data import coverage_series, rmse_series_by_regime


def test_rmse_series_by_regime_selects_the_matching_filter_row():
    rows = [
        {"filter": "EKF", "regime": "nominal", "pos_rmse": "1.5"},
        {"filter": "IMM", "regime": "nominal", "pos_rmse": "0.4"},
    ]
    ys = rmse_series_by_regime(rows, ["nominal"], filter_name="IMM")
    assert ys == [0.4]


def test_rmse_series_by_regime_fills_a_missing_regime_with_nan():
    rows = [{"filter": "EKF", "regime": "nominal", "pos_rmse": "1.0"}]
    ys = rmse_series_by_regime(rows, ["nominal", "high_noise"], filter_name="EKF")
    assert len(ys) == 2
    assert ys[0] == 1.0
    assert math.isnan(ys[1])


def test_rmse_series_by_regime_follows_the_requested_regime_order():
    # Rows deliberately out of order, so an implementation that just walks
    # rows (rather than regime_order) would emit them reversed.
    rows = [
        {"filter": "EKF", "regime": "high_noise", "pos_rmse": "2.0"},
        {"filter": "EKF", "regime": "nominal", "pos_rmse": "1.0"},
    ]
    ys = rmse_series_by_regime(rows, ["nominal", "high_noise"], filter_name="EKF")
    assert ys == [1.0, 2.0]


def test_rmse_series_by_regime_ignores_filter_field_when_filter_name_is_none():
    rows = [
        {"filter": "EKF", "regime": "nominal", "pos_rmse": "1.5"},
        {"filter": "IMM", "regime": "nominal", "pos_rmse": "2.5"},
    ]
    # filter_name=None must not silently prefer one real filter over another;
    # first-row-wins is the same rule a present filter_name uses.
    ys = rmse_series_by_regime(rows, ["nominal"])
    assert ys == [1.5]


def test_rmse_series_by_regime_takes_the_first_match_when_several_rows_tie():
    rows = [
        {"filter": "EKF", "regime": "nominal", "pos_rmse": "1.0"},
        {"filter": "EKF", "regime": "nominal", "pos_rmse": "2.0"},
    ]
    ys = rmse_series_by_regime(rows, ["nominal"], filter_name="EKF")
    assert ys == [1.0]


def test_rmse_series_by_regime_is_nan_when_the_regime_has_no_row_for_that_filter():
    # The regime is present, but only for a different filter -- must not fall
    # back to it and silently plot the wrong filter's number.
    rows = [{"filter": "UKF", "regime": "nominal", "pos_rmse": "4.0"}]
    ys = rmse_series_by_regime(rows, ["nominal"], filter_name="EKF")
    assert math.isnan(ys[0])


def test_rmse_series_by_regime_reads_the_requested_column():
    rows = [{"filter": "EKF", "regime": "nominal", "pos_rmse": "1.0", "vel_rmse": "9.0"}]
    ys = rmse_series_by_regime(rows, ["nominal"], filter_name="EKF", value_col="vel_rmse")
    assert ys == [9.0]


def test_coverage_series_preserves_result_order_without_rkn():
    results = [
        {"filter": "EKF", "coverage_68": 0.62, "coverage_95": 0.91},
        {"filter": "IMM", "coverage_68": 0.70, "coverage_95": 0.96},
    ]
    filters, cov68, cov95 = coverage_series(results)
    assert filters == ["EKF", "IMM"]
    assert cov68 == [0.62, 0.70]
    assert cov95 == [0.91, 0.96]


def test_coverage_series_appends_rkn_without_disturbing_existing_entries():
    results = [{"filter": "EKF", "coverage_68": 0.62, "coverage_95": 0.91}]
    rkn_highmc = {"coverage_68": 0.71, "coverage_95": 0.94}
    filters, cov68, cov95 = coverage_series(results, rkn_highmc)
    assert filters == ["EKF", "RKN"]
    assert cov68 == [0.62, 0.71]
    assert cov95 == [0.91, 0.94]


def test_coverage_series_does_not_mutate_its_results_argument():
    results = [{"filter": "EKF", "coverage_68": 0.62, "coverage_95": 0.91}]
    before = json.dumps(results)
    coverage_series(results, rkn_highmc={"coverage_68": 0.71, "coverage_95": 0.94})
    assert json.dumps(results) == before


def test_make_plots_uses_the_shared_selection_functions_not_a_private_copy():
    import scripts.make_plots as make_plots

    assert make_plots.rmse_series_by_regime is rmse_series_by_regime
    assert make_plots.coverage_series is coverage_series


def test_plot_accuracy_vs_maneuver_intensity_actually_calls_rmse_series_by_regime(tmp_path):
    from scripts import make_plots

    results_csv = tmp_path / "results_table.csv"
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filter", "regime", "pos_rmse"])
        w.writeheader()
        w.writerow({"filter": "EKF", "regime": "nominal", "pos_rmse": "3.0"})

    with patch.object(make_plots, "rmse_series_by_regime", wraps=make_plots.rmse_series_by_regime) as spy:
        make_plots.plot_accuracy_vs_maneuver_intensity(
            tmp_path, str(results_csv), str(tmp_path / "missing_knet.csv")
        )

    # A reimplementation that never calls the shared function -- private
    # copy, or inline filtering -- would leave this unexercised.
    assert spy.call_count == 3
    assert {c.kwargs["filter_name"] for c in spy.call_args_list} == {"EKF", "UKF", "IMM"}


def test_plot_coverage_actually_calls_coverage_series(tmp_path):
    from scripts import make_plots

    highmc = tmp_path / "anees_highmc.json"
    highmc.write_text(json.dumps({"results": [{"filter": "EKF", "coverage_68": 0.6, "coverage_95": 0.9}]}))

    with patch.object(make_plots, "coverage_series", wraps=make_plots.coverage_series) as spy:
        make_plots.plot_coverage(tmp_path, str(highmc))

    spy.assert_called_once()
    assert spy.call_args.args[0] == [{"filter": "EKF", "coverage_68": 0.6, "coverage_95": 0.9}]
    assert spy.call_args.args[1] is None
