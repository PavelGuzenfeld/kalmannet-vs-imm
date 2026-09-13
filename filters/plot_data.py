"""Pure data-selection helpers behind the figures in scripts/make_plots.py --
split out so the CSV-column and filter-to-series selection logic is gated and
tested like the rest of filters/, leaving make_plots.py itself as pixel-only
rendering (see the waiver in .mutation-gate-waivers.toml)."""

import numpy as np


def imm_kwargs_from_tuning(imm_best):
    """IMM constructor kwargs from an imm_tuning.json "best" dict -- the only
    place that converts its degrees turn-rate to radians."""
    return {
        "omega_ct": np.deg2rad(imm_best["omega_ct_deg"]),
        "p_stay": imm_best["p_stay"],
        "ca_sigma_a": imm_best["ca_sigma_a"],
        "ca_tau": imm_best["ca_tau"],
    }


def latency_series(data):
    """(names, param_counts, median_ms) columns from a latency.json
    "results_ms" dict, one entry per filter name; param_count defaults to 0
    for the classical filters, which carry none."""
    names = list(data.keys())
    params = [data[n].get("param_count", 0) for n in names]
    latency = [data[n]["median_ms"] for n in names]
    return names, params, latency


def rmse_series_by_regime(rows, regime_order, filter_name=None, value_col="pos_rmse"):
    """One value per entry in regime_order, pulled from the row matching that
    regime (and filter_name, when given) in rows -- a list of dict-like CSV
    records. NaN for a regime with no matching row."""
    ys = []
    for regime in regime_order:
        match = [
            r for r in rows
            if r["regime"] == regime and (filter_name is None or r["filter"] == filter_name)
        ]
        ys.append(float(match[0][value_col]) if match else float("nan"))
    return ys


def coverage_series(results, rkn_highmc=None):
    """(filter names, coverage@68, coverage@95) from a list of per-filter
    high-MC eval result dicts, with an "RKN" entry appended when rkn_highmc
    (a single result dict) is given."""
    filters = [r["filter"] for r in results]
    cov68 = [r["coverage_68"] for r in results]
    cov95 = [r["coverage_95"] for r in results]
    if rkn_highmc is not None:
        filters.append("RKN")
        cov68.append(rkn_highmc["coverage_68"])
        cov95.append(rkn_highmc["coverage_95"])
    return filters, cov68, cov95
