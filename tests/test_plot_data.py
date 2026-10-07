import csv
import json
import math
from unittest.mock import patch

import numpy as np

from filters.plot_data import coverage_series, imm_kwargs_from_tuning, latency_series, rmse_series_by_regime
from filters.state import POS


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
    assert len(ys) == 1
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


def test_imm_kwargs_from_tuning_converts_the_turn_rate_to_radians():
    """MS-12: all six tuned values reach the IMM constructor."""
    kwargs = imm_kwargs_from_tuning(
        {"omega_ct_deg": 15.0, "p_stay": 0.97, "ca_sigma_a": 3.0, "ca_tau": 4.0, "cv_sigma_a": 0.1, "ct_sigma_a": 0.5}
    )
    assert kwargs == {
        "omega_ct": math.radians(15.0),
        "p_stay": 0.97,
        "ca_sigma_a": 3.0,
        "ca_tau": 4.0,
        "cv_sigma_a": 0.1,
        "ct_sigma_a": 0.5,
    }


def test_latency_series_reads_names_params_and_median_in_dict_order():
    data = {
        "EKF": {"median_ms": 0.5},
        "KalmanNet-vanilla": {"median_ms": 1.2, "param_count": 4096},
    }
    names, params, latency = latency_series(data)
    assert names == ["EKF", "KalmanNet-vanilla"]
    assert params == [0, 4096]
    assert latency == [0.5, 1.2]


def test_make_plots_uses_the_shared_selection_functions_not_a_private_copy():
    import scripts.make_plots as make_plots
    from filters.metrics import per_step_rmse

    assert make_plots.rmse_series_by_regime is rmse_series_by_regime
    assert make_plots.coverage_series is coverage_series
    assert make_plots.imm_kwargs_from_tuning is imm_kwargs_from_tuning
    assert make_plots.latency_series is latency_series
    assert make_plots.per_step_rmse is per_step_rmse


_ACCURACY_REGIME_ORDER = ["low_noise", "nominal", "high_noise", "heavy_tailed", "high_maneuver"]


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_plot_accuracy_vs_maneuver_intensity_actually_calls_rmse_series_by_regime(tmp_path):
    from scripts import make_plots

    results_csv = tmp_path / "results_table.csv"
    _write_csv(results_csv, ["filter", "regime", "pos_rmse"],
               [{"filter": "EKF", "regime": "nominal", "pos_rmse": "3.0"}])

    with patch.object(make_plots, "rmse_series_by_regime", wraps=make_plots.rmse_series_by_regime) as spy:
        make_plots.plot_accuracy_vs_maneuver_intensity(
            tmp_path, str(results_csv), str(tmp_path / "missing_knet.csv")
        )

    # A reimplementation that never calls the shared function -- private
    # copy, or inline filtering -- would leave this unexercised.
    assert spy.call_count == 3
    # Order matters: it's the plotted legend order, not just the call set.
    assert [c.kwargs["filter_name"] for c in spy.call_args_list] == ["EKF", "UKF", "IMM"]
    # Every call must walk the same 5-regime x-axis, or two filters could be
    # plotted against misaligned regime positions.
    assert all(c.args[1] == _ACCURACY_REGIME_ORDER for c in spy.call_args_list)
    # And it must be fed the row actually read from results_csv, not some
    # other source.
    first_call_rows = spy.call_args_list[0].args[0]
    assert first_call_rows == [{"filter": "EKF", "regime": "nominal", "pos_rmse": "3.0"}]
    assert (tmp_path / "accuracy_vs_intensity.png").exists()


def _write_accuracy_csvs(tmp_path, knet_value="5.0", rkn_value="4.0"):
    results_csv = tmp_path / "results_table.csv"
    _write_csv(results_csv, ["filter", "regime", "pos_rmse"],
               [{"filter": "EKF", "regime": "nominal", "pos_rmse": "3.0"}])
    knet_csv = tmp_path / "knet_test_rmse.csv"
    _write_csv(knet_csv, ["regime", "pos_rmse"], [{"regime": "nominal", "pos_rmse": knet_value}])
    rkn_csv = tmp_path / "rkn_test_rmse.csv"
    _write_csv(rkn_csv, ["regime", "pos_rmse"], [{"regime": "nominal", "pos_rmse": rkn_value}])
    return results_csv, knet_csv, rkn_csv


def test_plot_accuracy_vs_maneuver_intensity_plots_the_knet_and_rkn_branches_when_present(tmp_path):
    from scripts import make_plots

    results_csv, knet_csv, rkn_csv = _write_accuracy_csvs(tmp_path)

    with patch.object(make_plots, "rmse_series_by_regime", wraps=make_plots.rmse_series_by_regime) as spy:
        make_plots.plot_accuracy_vs_maneuver_intensity(
            tmp_path, str(results_csv), str(knet_csv), rkn_csv_path=str(rkn_csv)
        )

    # EKF, UKF, IMM, knet, rkn -- the knet/rkn branches only fire when their
    # CSVs exist, which the always-missing-path test above never exercises.
    assert spy.call_count == 5
    knet_call, rkn_call = spy.call_args_list[3], spy.call_args_list[4]
    assert knet_call.args[0] == [{"regime": "nominal", "pos_rmse": "5.0"}]
    assert rkn_call.args[0] == [{"regime": "nominal", "pos_rmse": "4.0"}]
    # Exactly (rows, regime_order) positionally, no filter_name at all --
    # `"filter_name" not in call.kwargs` alone would miss a filter_name
    # passed positionally as a 3rd arg.
    assert len(knet_call.args) == 2 and not knet_call.kwargs
    assert len(rkn_call.args) == 2 and not rkn_call.kwargs
    assert all(c.args[1] == _ACCURACY_REGIME_ORDER for c in spy.call_args_list)
    assert (tmp_path / "accuracy_vs_intensity.png").exists()


def test_plot_accuracy_vs_maneuver_intensity_plots_only_the_knet_branch_when_rkn_csv_is_absent(tmp_path):
    from scripts import make_plots

    results_csv, knet_csv, _ = _write_accuracy_csvs(tmp_path)

    with patch.object(make_plots, "rmse_series_by_regime", wraps=make_plots.rmse_series_by_regime) as spy:
        make_plots.plot_accuracy_vs_maneuver_intensity(tmp_path, str(results_csv), str(knet_csv))

    # knet_rows and rkn_rows gate independently -- a knet-only run must not
    # also require (or silently skip) the rkn branch.
    assert spy.call_count == 4
    knet_call = spy.call_args_list[3]
    assert knet_call.args[0] == [{"regime": "nominal", "pos_rmse": "5.0"}]
    assert len(knet_call.args) == 2 and not knet_call.kwargs
    assert all(c.args[1] == _ACCURACY_REGIME_ORDER for c in spy.call_args_list)


def test_plot_accuracy_vs_maneuver_intensity_draws_no_rkn_line_when_rkn_csv_is_absent(tmp_path):
    from scripts import make_plots

    results_csv, knet_csv, _ = _write_accuracy_csvs(tmp_path)

    real_subplots = make_plots.plt.subplots
    captured = {}

    def _capture_subplots(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    with patch.object(make_plots.plt, "subplots", side_effect=_capture_subplots):
        make_plots.plot_accuracy_vs_maneuver_intensity(tmp_path, str(results_csv), str(knet_csv))

    labels = {line.get_label() for line in captured["ax"].lines}
    assert "RKN" not in labels


def test_plot_accuracy_vs_maneuver_intensity_plots_only_the_rkn_branch_when_knet_csv_is_absent(tmp_path):
    from scripts import make_plots

    results_csv, _, rkn_csv = _write_accuracy_csvs(tmp_path)

    with patch.object(make_plots, "rmse_series_by_regime", wraps=make_plots.rmse_series_by_regime) as spy:
        make_plots.plot_accuracy_vs_maneuver_intensity(
            tmp_path, str(results_csv), str(tmp_path / "missing_knet.csv"), rkn_csv_path=str(rkn_csv)
        )

    # The other direction of the same independent-gating requirement: an
    # implementation that nests the rkn branch inside "if knet_rows:" would
    # silently drop it here.
    assert spy.call_count == 4
    rkn_call = spy.call_args_list[3]
    assert rkn_call.args[0] == [{"regime": "nominal", "pos_rmse": "4.0"}]
    assert len(rkn_call.args) == 2 and not rkn_call.kwargs
    assert all(c.args[1] == _ACCURACY_REGIME_ORDER for c in spy.call_args_list)


def test_plot_accuracy_vs_maneuver_intensity_draws_no_knet_line_when_knet_csv_is_absent(tmp_path):
    from scripts import make_plots

    results_csv, _, rkn_csv = _write_accuracy_csvs(tmp_path)

    real_subplots = make_plots.plt.subplots
    captured = {}

    def _capture_subplots(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    with patch.object(make_plots.plt, "subplots", side_effect=_capture_subplots):
        make_plots.plot_accuracy_vs_maneuver_intensity(
            tmp_path, str(results_csv), str(tmp_path / "missing_knet.csv"), rkn_csv_path=str(rkn_csv)
        )

    labels = {line.get_label() for line in captured["ax"].lines}
    assert "KalmanNet-vanilla" not in labels


def test_plot_accuracy_vs_maneuver_intensity_draws_the_actual_knet_and_rkn_values(tmp_path):
    from scripts import make_plots

    results_csv, knet_csv, rkn_csv = _write_accuracy_csvs(tmp_path)

    real_subplots = make_plots.plt.subplots
    captured = {}

    def _capture_subplots(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    with patch.object(make_plots.plt, "subplots", side_effect=_capture_subplots):
        make_plots.plot_accuracy_vs_maneuver_intensity(
            tmp_path, str(results_csv), str(knet_csv), rkn_csv_path=str(rkn_csv)
        )

    # A call-spy alone can't tell a plot that discards rmse_series_by_regime's
    # return value from one that actually draws it -- read the rendered line.
    lines_by_label = {line.get_label(): line for line in captured["ax"].lines}
    knet_line, rkn_line = lines_by_label["KalmanNet-vanilla"], lines_by_label["RKN"]
    knet_y, rkn_y = knet_line.get_ydata(), rkn_line.get_ydata()
    nominal_idx = _ACCURACY_REGIME_ORDER.index("nominal")
    assert knet_y[nominal_idx] == 5.0
    assert rkn_y[nominal_idx] == 4.0
    # Every other regime has no row in either single-row CSV -- and both
    # lines' x-axis must be the same 5-regime order, not just its length.
    assert list(knet_line.get_xdata()) == _ACCURACY_REGIME_ORDER
    assert list(rkn_line.get_xdata()) == _ACCURACY_REGIME_ORDER
    for idx in range(len(_ACCURACY_REGIME_ORDER)):
        if idx == nominal_idx:
            continue
        assert math.isnan(knet_y[idx])
        assert math.isnan(rkn_y[idx])


def test_plot_coverage_actually_calls_coverage_series(tmp_path):
    from scripts import make_plots

    highmc = tmp_path / "anees_highmc.json"
    highmc.write_text(json.dumps({"results": [{"filter": "EKF", "coverage_68": 0.6, "coverage_95": 0.9}]}))

    with patch.object(make_plots, "coverage_series", wraps=make_plots.coverage_series) as spy:
        make_plots.plot_coverage(tmp_path, str(highmc))

    spy.assert_called_once()
    assert spy.call_args.args[0] == [{"filter": "EKF", "coverage_68": 0.6, "coverage_95": 0.9}]
    assert spy.call_args.args[1] is None
    assert (tmp_path / "coverage.png").exists()


def test_plot_coverage_forwards_the_rkn_result_when_present(tmp_path):
    from scripts import make_plots

    highmc = tmp_path / "anees_highmc.json"
    highmc.write_text(json.dumps({"results": [{"filter": "EKF", "coverage_68": 0.6, "coverage_95": 0.9}]}))
    rkn_eval = tmp_path / "rkn_eval.json"
    rkn_eval.write_text(json.dumps({"highmc": {"coverage_68": 0.71, "coverage_95": 0.94}}))

    with patch.object(make_plots, "coverage_series", wraps=make_plots.coverage_series) as spy:
        make_plots.plot_coverage(tmp_path, str(highmc), rkn_eval_json_path=str(rkn_eval))

    assert spy.call_args.args[1] == {"coverage_68": 0.71, "coverage_95": 0.94}


def test_plot_latency_vs_parameters_actually_calls_latency_series(tmp_path):
    from scripts import make_plots

    latency_json = tmp_path / "latency.json"
    latency_json.write_text(json.dumps({
        "results_ms": {
            "EKF": {"median_ms": 0.5},
            "KalmanNet-vanilla": {"median_ms": 1.2, "param_count": 4096},
        }
    }))

    with patch.object(make_plots, "latency_series", wraps=make_plots.latency_series) as spy:
        make_plots.plot_latency_vs_parameters(tmp_path, str(latency_json))

    spy.assert_called_once_with({
        "EKF": {"median_ms": 0.5},
        "KalmanNet-vanilla": {"median_ms": 1.2, "param_count": 4096},
    })
    assert (tmp_path / "latency_vs_parameters.png").exists()


def test_plot_latency_vs_parameters_draws_the_values_latency_series_returns(tmp_path):
    from scripts import make_plots

    latency_json = tmp_path / "latency.json"
    latency_json.write_text(json.dumps({
        "results_ms": {
            "EKF": {"median_ms": 0.5},
            "KalmanNet-vanilla": {"median_ms": 1.2, "param_count": 4096},
        }
    }))

    real_subplots = make_plots.plt.subplots
    captured = {}

    def _capture_subplots(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    with patch.object(make_plots.plt, "subplots", side_effect=_capture_subplots):
        make_plots.plot_latency_vs_parameters(tmp_path, str(latency_json))

    # A plot that calls latency_series but discards its return value (and
    # recomputes something else inline) would still pass the call-spy test.
    offsets = captured["ax"].collections[0].get_offsets()
    assert list(offsets[:, 0]) == [0, 4096]
    assert list(offsets[:, 1]) == [0.5, 1.2]
    # The "names" column must reach the point annotations too, in order.
    assert [t.get_text() for t in captured["ax"].texts] == ["EKF", "KalmanNet-vanilla"]


def _straight_line_trajectory(x0, T, dt):
    states = np.zeros((T, 9))
    for t in range(T):
        states[t, :3] = x0[:3] + t * dt * x0[3:6]
        states[t, 3:6] = x0[3:6]
    return states


def _measure_exactly(states):
    from filters.measurement_model import h

    return np.array([h(x) for x in states])


def test_plot_trajectory_overlay_draws_the_imm_built_from_the_tuned_kwargs(tmp_path):
    """MS-12."""
    from filters.imm import IMM
    from scripts import make_plots

    dt, T = 0.1, 4
    R = np.diag([0.001**2, 0.001**2, 2.0**2])
    x0 = np.array([3000.0, 500.0, 200.0, 50.0, -20.0, 5.0, 0.0, 0.0, 0.0])
    states = _straight_line_trajectory(x0, T, dt)[None, :, :]
    measurements = _measure_exactly(states[0])[None, None, :, :]

    imm_best = {
        "omega_ct_deg": 12.0, "p_stay": 0.9, "ca_sigma_a": 2.0, "ca_tau": 3.0, "cv_sigma_a": 0.1, "ct_sigma_a": 0.5,
    }
    imm_tuning = tmp_path / "imm_tuning.json"
    imm_tuning.write_text(json.dumps({"best": imm_best}))

    # Independently reproduce the tuned IMM's actual output, using the same
    # (already-tested) IMM class and _run_imm -- not a copy of make_plots'
    # inline formula, so this doesn't just check the code against itself.
    expected_imm = IMM(dt, R, **imm_kwargs_from_tuning(imm_best))
    expected_xs, _ = make_plots._run_imm(expected_imm, states[0], measurements[0, 0])

    real_subplots = make_plots.plt.subplots
    captured = {}

    def _capture_subplots(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    with patch.object(make_plots, "imm_kwargs_from_tuning", wraps=make_plots.imm_kwargs_from_tuning) as spy, \
         patch.object(make_plots.plt, "subplots", side_effect=_capture_subplots):
        make_plots.plot_trajectory_overlay(tmp_path, dt, R, states, measurements,
                                            imm_tuning_json_path=str(imm_tuning))

    spy.assert_called_once_with(imm_best)
    # The rendered IMM line must be the tuned IMM's own track, not a
    # differently-configured IMM or a second EKF run under the IMM label.
    imm_line = {line.get_label(): line for line in captured["ax"].lines}["IMM (tuned)"]
    assert np.array_equal(imm_line.get_xdata(), expected_xs[:, 0])
    assert np.array_equal(imm_line.get_ydata(), expected_xs[:, 1])
    assert (tmp_path / "trajectory_overlay.png").exists()


def test_plot_trajectory_overlay_draws_ground_truth_ekf_and_ukf_for_the_first_trajectory_only(tmp_path):
    """MS-9, MS-12."""
    from filters.ekf import EKF
    from filters.imm import IMM
    from filters.ukf import UKF
    from scripts import make_plots

    dt, T = 0.1, 4
    R = np.diag([0.001**2, 0.001**2, 2.0**2])
    x0 = np.array([3000.0, 500.0, 200.0, 50.0, -20.0, 5.0, 0.0, 0.0, 0.0])
    # A second, distinguishable trajectory alongside the first, so a line
    # drawn from the wrong trajectory (or from both) is caught rather than
    # passing by coincidence of a single-trajectory fixture.
    x1 = np.array([100.0, 100.0, 0.0, -30.0, 40.0, -5.0, 0.0, 0.0, 0.0])
    states = np.stack([_straight_line_trajectory(x0, T, dt), _straight_line_trajectory(x1, T, dt)])
    measurements = np.stack([_measure_exactly(s) for s in states])[:, None, :, :]

    imm_best = {
        "omega_ct_deg": 12.0, "p_stay": 0.9, "ca_sigma_a": 2.0, "ca_tau": 3.0, "cv_sigma_a": 0.1, "ct_sigma_a": 0.5,
    }
    imm_tuning = tmp_path / "imm_tuning.json"
    imm_tuning.write_text(json.dumps({"best": imm_best}))

    expected_ekf_xs, _ = make_plots._run_ekf_like(EKF(dt, R), states[0], measurements[0, 0])
    expected_ukf_xs, _ = make_plots._run_ekf_like(UKF(dt, R), states[0], measurements[0, 0])
    expected_imm = IMM(dt, R, **imm_kwargs_from_tuning(imm_best))
    expected_imm_xs, _ = make_plots._run_imm(expected_imm, states[0], measurements[0, 0])

    real_subplots = make_plots.plt.subplots
    captured = {}

    def _capture_subplots(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    with patch.object(make_plots.plt, "subplots", side_effect=_capture_subplots):
        make_plots.plot_trajectory_overlay(tmp_path, dt, R, states, measurements,
                                            imm_tuning_json_path=str(imm_tuning))

    lines = {line.get_label(): line for line in captured["ax"].lines}

    truth_line = lines["ground truth"]
    assert np.array_equal(truth_line.get_xdata(), states[0][:, 0])
    assert np.array_equal(truth_line.get_ydata(), states[0][:, 1])

    ekf_line = lines["EKF (CV-only)"]
    assert np.array_equal(ekf_line.get_xdata(), expected_ekf_xs[:, 0])
    assert np.array_equal(ekf_line.get_ydata(), expected_ekf_xs[:, 1])

    ukf_line = lines["UKF (CV-only)"]
    assert np.array_equal(ukf_line.get_xdata(), expected_ukf_xs[:, 0])
    assert np.array_equal(ukf_line.get_ydata(), expected_ukf_xs[:, 1])

    imm_line = lines["IMM (tuned)"]
    assert np.array_equal(imm_line.get_xdata(), expected_imm_xs[:, 0])
    assert np.array_equal(imm_line.get_ydata(), expected_imm_xs[:, 1])


def test_plot_rmse_vs_time_uses_the_shared_imm_kwargs_builder_and_draws_per_step_rmse(tmp_path):
    """MS-12."""
    from scripts import make_plots

    dt, T, n_traj = 0.1, 4, 2
    # One trajectory more than n_traj, so a call fed states[:n_traj] (correct)
    # is distinguishable from one fed the whole array (n_traj ignored).
    n_available = n_traj + 1
    R = np.diag([0.001**2, 0.001**2, 2.0**2])
    x0 = np.array([3000.0, 500.0, 200.0, 50.0, -20.0, 5.0, 0.0, 0.0, 0.0])
    states = np.stack([_straight_line_trajectory(x0, T, dt) for _ in range(n_available)])
    measurements = np.stack([_measure_exactly(s) for s in states])[:, None, :, :]

    imm_best = {
        "omega_ct_deg": 12.0, "p_stay": 0.9, "ca_sigma_a": 2.0, "ca_tau": 3.0, "cv_sigma_a": 0.1, "ct_sigma_a": 0.5,
    }
    imm_tuning = tmp_path / "imm_tuning.json"
    imm_tuning.write_text(json.dumps({"best": imm_best}))

    rmse_returns = []
    _real_per_step_rmse = make_plots.per_step_rmse

    def _capture_rmse(*a, **kw):
        r = _real_per_step_rmse(*a, **kw)
        rmse_returns.append(r)
        return r

    real_subplots = make_plots.plt.subplots
    captured = {}

    def _capture_subplots(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    with patch.object(make_plots, "imm_kwargs_from_tuning", wraps=make_plots.imm_kwargs_from_tuning) as kwargs_spy, \
         patch.object(make_plots, "per_step_rmse", side_effect=_capture_rmse) as rmse_spy, \
         patch.object(make_plots.plt, "subplots", side_effect=_capture_subplots):
        make_plots.plot_rmse_vs_time(tmp_path, dt, R, states, measurements, n_traj=n_traj,
                                      imm_tuning_json_path=str(imm_tuning))

    # The IMM factory is re-invoked once per trajectory (n_traj=2), so its
    # kwargs builder is too -- same as the pre-existing per-trajectory
    # IMM(...) construction this replaces.
    assert kwargs_spy.call_count == n_traj
    for call in kwargs_spy.call_args_list:
        assert call.args[0] == imm_best
    # One call per filter (EKF, UKF, IMM); a reimplementation of the RMSE
    # formula inline would leave this unexercised.
    assert rmse_spy.call_count == 3
    for call in rmse_spy.call_args_list:
        # Fed the ground truth for exactly the first n_traj trajectories
        # (not all n_available), sliced to position.
        assert np.array_equal(call.args[0], states[:n_traj])
        assert call.args[0].shape[0] == n_traj
        assert call.args[2] == POS

    # EKF, UKF and IMM must each be scored against their own estimates, not
    # three copies of the same run -- otherwise the label-to-return zip below
    # could match by coincidence rather than by which filter actually ran.
    estimates_seen = {call.args[1].tobytes() for call in rmse_spy.call_args_list}
    assert len(estimates_seen) == 3

    # Each filter's line must be exactly what per_step_rmse returned for it,
    # plotted against dt-scaled time -- a call-spy alone can't rule out a
    # plot that discards the return value and draws something else.
    lines_by_label = {line.get_label(): line for line in captured["ax"].lines}
    t_axis = np.arange(T) * dt
    for name, expected in zip(["EKF", "UKF", "IMM"], rmse_returns):
        assert np.array_equal(lines_by_label[name].get_ydata(), expected)
        assert np.array_equal(lines_by_label[name].get_xdata(), t_axis)
    assert (tmp_path / "rmse_vs_time.png").exists()


def test_latency_bench_builds_its_imm_from_all_six_tuned_values(monkeypatch, tmp_path):
    """MS-12: the latency figure is for the IMM the benchmark runs."""
    from filters.imm import IMM
    from scripts import bench_latency

    nominal = {"dt": 0.02, "sigma_az_rad": 0.001, "sigma_el_rad": 0.001, "sigma_range_m": 2.0}
    (tmp_path / "manifest.json").write_text(json.dumps({"regimes": {"nominal": nominal}}))
    tuned = {"omega_ct_deg": 20.0, "p_stay": 0.98, "ca_sigma_a": 1.5, "ca_tau": 3.0, "cv_sigma_a": 0.01, "ct_sigma_a": 0.7}
    tuning = tmp_path / "imm_tuning.json"
    tuning.write_text(json.dumps({"best": tuned}))
    built = []
    monkeypatch.setattr(bench_latency, "IMM", lambda dt, R, **kwargs: built.append(kwargs) or IMM(dt, R, **kwargs))
    argv = ["bench_latency.py", "--data-dir", str(tmp_path), "--imm-tuning", str(tuning),
            "--n-warmup", "1", "--n-measure", "1", "--out", str(tmp_path / "latency.json")]
    monkeypatch.setattr("sys.argv", argv)
    bench_latency.main()
    assert built == [{
        "omega_ct": math.radians(20.0), "p_stay": 0.98, "ca_sigma_a": 1.5, "ca_tau": 3.0,
        "cv_sigma_a": 0.01, "ct_sigma_a": 0.7,
    }]


RUNAWAY_DRIFT_M = 150.0
RUNAWAY_STEPS = 5
RUNAWAY_DIVERGENCE_STEP = 2
RUNAWAY_IMM_KWARGS = {
    "omega_ct_deg": 12.0, "p_stay": 0.9, "ca_sigma_a": 2.0, "ca_tau": 3.0, "cv_sigma_a": 0.1, "ct_sigma_a": 0.5,
}
RUNAWAY_POSITION_ERRORS_BEFORE_CUT = [np.sqrt(5.0), np.sqrt(152.0**2 + 1.0)]


class _RunawayEKF:
    drift_scale = 1.0

    def __init__(self, dt, R):
        pass

    def update(self, x, P, z):
        return x, np.eye(9), np.zeros(3), np.eye(3)

    def step(self, x, P, z):
        return x + z[0] * self.drift_scale * np.eye(9)[0], np.eye(9), np.zeros(3), np.eye(3)


def _runaway_ukf_class(drift_scale=1.0):
    from filters.ukf import UKF

    class _RunawayUKF(UKF):
        scale = drift_scale

        def __init__(self, dt, R):
            pass

        def _sigma_points(self, x, P):
            return None

        def update(self, x, P, pts, z):
            return x, np.eye(9), np.zeros(3), np.eye(3)

        def step(self, x, P, z):
            return x + z[0] * self.scale * np.eye(9)[0], np.eye(9), np.zeros(3), np.eye(3)

    return _RunawayUKF


class _RunawayIMM:
    drift_scale = 1.0

    def __init__(self, dt, R, **kwargs):
        pass

    def init_state(self, x, P):
        return {"x": x, "P": P, "mu": np.full(4, 0.25), "steps": 0}

    def step(self, state, z):
        x = state["x"] + (z[0] * self.drift_scale if state["steps"] else 0.0) * np.eye(9)[0]
        return x, np.eye(9), {**state, "x": x, "steps": state["steps"] + 1}, np.zeros(3), np.eye(3)


def _patch_runaway_classical_filters(make_module, ukf_drift_scale=1.0):
    return (
        patch.object(make_module, "EKF", _RunawayEKF),
        patch.object(make_module, "UKF", _runaway_ukf_class(ukf_drift_scale)),
        patch.object(make_module, "IMM", _RunawayIMM),
    )


def _runaway_flight(drifts=(RUNAWAY_DRIFT_M, RUNAWAY_DRIFT_M)):
    states = np.zeros((2, RUNAWAY_STEPS, 9))
    measurements = np.zeros((2, 1, RUNAWAY_STEPS, 3))
    measurements[:, 0, :, 0] = np.array(drifts)[:, None]
    return states, measurements


def _capture_axes(make_module):
    captured = {}
    real_subplots = make_module.plt.subplots

    def _subplots(*a, **kw):
        fig, ax = real_subplots(*a, **kw)
        captured["ax"] = ax
        return fig, ax

    return captured, patch.object(make_module.plt, "subplots", side_effect=_subplots)


def _run_with_runaway_filters(make_module, tmp_path, draw, ukf_drift_scale=1.0):
    ekf, ukf, imm = _patch_runaway_classical_filters(make_module, ukf_drift_scale)
    captured, subplots = _capture_axes(make_module)
    tuning = tmp_path / "imm_tuning.json"
    tuning.write_text(json.dumps({"best": RUNAWAY_IMM_KWARGS}))
    with ekf, ukf, imm, subplots:
        draw(str(tuning))
    return {line.get_label(): line.get_ydata() for line in captured["ax"].lines}


def test_trajectory_overlay_draws_no_filter_estimate_from_its_divergence_step_on(tmp_path):
    from scripts import make_plots

    dt, R = 0.1, np.eye(3)
    states, measurements = _runaway_flight()

    drawn = _run_with_runaway_filters(
        make_plots,
        tmp_path,
        lambda tuning: make_plots.plot_trajectory_overlay(
            tmp_path, dt, R, states, measurements, imm_tuning_json_path=tuning
        ),
    )

    for label in ["EKF (CV-only)", "UKF (CV-only)", "IMM (tuned)"]:
        assert np.isfinite(drawn[label][:RUNAWAY_DIVERGENCE_STEP]).all()
        assert np.isnan(drawn[label][RUNAWAY_DIVERGENCE_STEP:]).all()
    assert np.isfinite(drawn["ground truth"]).all()


def test_rmse_vs_time_averages_only_the_flights_still_running_and_ends_at_the_cut(tmp_path):
    from scripts import make_plots

    dt, R = 0.1, np.eye(3)
    states, measurements = _runaway_flight()

    drawn = _run_with_runaway_filters(
        make_plots,
        tmp_path,
        lambda tuning: make_plots.plot_rmse_vs_time(
            tmp_path, dt, R, states, measurements, n_traj=2, imm_tuning_json_path=tuning
        ),
    )

    for label in ["EKF", "UKF"]:
        assert list(drawn[label][:RUNAWAY_DIVERGENCE_STEP]) == RUNAWAY_POSITION_ERRORS_BEFORE_CUT
        assert np.isnan(drawn[label][RUNAWAY_DIVERGENCE_STEP:]).all()


def test_anees_vs_time_ends_at_the_cut(tmp_path, monkeypatch):
    from scripts import make_plots

    dt, R = 0.1, np.eye(3)
    states, measurements = _runaway_flight()
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "imm_tuning.json").write_text(json.dumps({"best": RUNAWAY_IMM_KWARGS}))
    monkeypatch.chdir(tmp_path)
    ekf, ukf, imm = _patch_runaway_classical_filters(make_plots)
    captured, subplots = _capture_axes(make_plots)

    with ekf, imm, subplots:
        make_plots.plot_nees_with_bounds(tmp_path, dt, R, states, measurements)

    anees = {line.get_label(): line.get_ydata() for line in captured["ax"].lines}
    for label in ["EKF ANEES(t)", "IMM ANEES(t)"]:
        assert list(anees[label][:RUNAWAY_DIVERGENCE_STEP]) == [5.0, 152.0**2 + 1.0]
        assert np.isnan(anees[label][RUNAWAY_DIVERGENCE_STEP:]).all()


SLOW_RUNAWAY_DRIFT_M = 60.0
SLOW_RUNAWAY_DIVERGENCE_STEP = 4
MIXED_DRIFTS = (RUNAWAY_DRIFT_M, SLOW_RUNAWAY_DRIFT_M)


def test_overlay_cuts_each_filter_at_its_own_divergence_step(tmp_path):
    from scripts import make_plots

    states, measurements = _runaway_flight()
    slow_ukf_scale = SLOW_RUNAWAY_DRIFT_M / RUNAWAY_DRIFT_M

    drawn = _run_with_runaway_filters(
        make_plots,
        tmp_path,
        lambda tuning: make_plots.plot_trajectory_overlay(
            tmp_path, 0.1, np.eye(3), states, measurements, imm_tuning_json_path=tuning
        ),
        ukf_drift_scale=slow_ukf_scale,
    )

    assert np.isnan(drawn["EKF (CV-only)"][RUNAWAY_DIVERGENCE_STEP:]).all()
    assert np.isfinite(drawn["EKF (CV-only)"][:RUNAWAY_DIVERGENCE_STEP]).all()
    assert np.isnan(drawn["IMM (tuned)"][RUNAWAY_DIVERGENCE_STEP:]).all()
    assert np.isfinite(drawn["UKF (CV-only)"][:SLOW_RUNAWAY_DIVERGENCE_STEP]).all()
    assert np.isnan(drawn["UKF (CV-only)"][SLOW_RUNAWAY_DIVERGENCE_STEP:]).all()


def test_rmse_between_two_flights_cuts_averages_only_the_flight_still_running(tmp_path):
    from scripts import make_plots

    states, measurements = _runaway_flight(MIXED_DRIFTS)
    expected = [
        np.sqrt(5.0),
        np.sqrt((152.0**2 + 1.0 + 62.0**2 + 1.0) / 2),
        np.sqrt(122.0**2 + 1.0),
        np.sqrt(182.0**2 + 1.0),
        np.nan,
    ]

    drawn = _run_with_runaway_filters(
        make_plots,
        tmp_path,
        lambda tuning: make_plots.plot_rmse_vs_time(
            tmp_path, 0.1, np.eye(3), states, measurements, n_traj=2, imm_tuning_json_path=tuning
        ),
    )

    for label in ["EKF", "UKF", "IMM"]:
        np.testing.assert_array_equal(drawn[label], expected)


def test_anees_between_two_flights_cuts_averages_only_the_flight_still_running(tmp_path, monkeypatch):
    from scripts import make_plots

    states, measurements = _runaway_flight(MIXED_DRIFTS)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "imm_tuning.json").write_text(json.dumps({"best": RUNAWAY_IMM_KWARGS}))
    monkeypatch.chdir(tmp_path)
    ekf, _, imm = _patch_runaway_classical_filters(make_plots)
    captured, subplots = _capture_axes(make_plots)

    with ekf, imm, subplots:
        make_plots.plot_nees_with_bounds(tmp_path, 0.1, np.eye(3), states, measurements)

    anees = {line.get_label(): line.get_ydata() for line in captured["ax"].lines}
    for label in ["EKF ANEES(t)", "IMM ANEES(t)"]:
        np.testing.assert_array_equal(
            anees[label], [5.0, (152.0**2 + 1.0 + 62.0**2 + 1.0) / 2, 122.0**2 + 1.0, 182.0**2 + 1.0, np.nan]
        )


def test_animation_records_each_filter_through_its_divergence_step_and_no_further(tmp_path, monkeypatch):
    import torch

    from filters.knet.gain_network import KalmanGainGRU
    from filters.knet.rkn_network import RKNGainCovNetwork
    from scripts import make_animation

    class _RunawayKalmanNet:
        def __init__(self, net, dt):
            self.steps = 0

        def init_sequence(self, x0, Q, Sigma, S):
            pass

        def step(self, z):
            x = torch.zeros(1, 9, dtype=torch.float64)
            x[0, 0] = RUNAWAY_DRIFT_M * self.steps
            self.steps += 1
            return x

    class _RunawayRKN(_RunawayKalmanNet):
        def init_sequence(self, x0, P0):
            pass

        def step(self, z):
            return super().step(z), torch.eye(9, dtype=torch.float64).unsqueeze(0)

    knet_ckpt, rkn_ckpt = tmp_path / "knet.pt", tmp_path / "rkn.pt"
    torch.save(KalmanGainGRU().double().state_dict(), knet_ckpt)
    torch.save(RKNGainCovNetwork(fc1_mult=1, hidden_mult=1, fc2_mult=1).double().state_dict(), rkn_ckpt)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "imm_tuning.json").write_text(json.dumps({"best": RUNAWAY_IMM_KWARGS}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(make_animation, "EKF", _RunawayEKF)
    monkeypatch.setattr(make_animation, "UKF", _runaway_ukf_class())
    monkeypatch.setattr(make_animation, "IMM", _RunawayIMM)
    monkeypatch.setattr(make_animation, "KalmanNetFilter", _RunawayKalmanNet)
    monkeypatch.setattr(make_animation, "RKNFilter", _RunawayRKN)

    est, err, nees, mus = make_animation.run_all_filters(
        np.zeros((RUNAWAY_STEPS, 9)), np.full((RUNAWAY_STEPS, 3), RUNAWAY_DRIFT_M), 0.1, np.eye(3), knet_ckpt, rkn_ckpt, (1, 1, 1)
    )

    pulled = RUNAWAY_DIVERGENCE_STEP + 1
    for name in make_animation.FILTERS:
        assert np.isfinite(err[name][:pulled]).all() and err[name][RUNAWAY_DIVERGENCE_STEP] > 200.0
        assert np.isnan(err[name][pulled:]).all()
        assert np.isnan(est[name][pulled:]).all()
