#!/usr/bin/env python3
"""Evaluate every filter over the full test set (RMSE, ANEES/coverage on the
high-MC subset, NIS) and the robustness sweep regimes, with identical metric
code for every filter -- per the brief's fairness requirement. Writes
results/results_table.csv (one row per filter x regime) and
results/anees_highmc.json (the headline calibration numbers).
"""

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from data.trajectory_gen import fly_scenario
from filters.ekf import EKF
from filters.imm import IMM
from filters.knet import numpy_forward
from filters.flags import until_diverged
from filters.metrics import (
    anees_over_counted,
    counted_mean,
    anees_with_bounds,
    counted_mask,
    counted_span,
    divergence_steps,
    divergence_summary,
    empirical_coverage,
)
from filters.motion_models import cv_model
from filters.plot_data import imm_kwargs_from_tuning
from filters.state import POS, STATE_DIM, VEL
from filters.ukf import UKF

_INIT_OFFSET = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_P0 = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
LEARNED_P0_VARIANCE = 100.0
DIVERGED_ERROR = 1e5
SINGULAR_P_NEES = 1e10
BKN_DROPOUT_P = 0.2
BKN_REALISATIONS = 20
_BURN_IN = 20  # steps skipped for RMSE/consistency stats, letting initial-condition error settle


def _load(path):
    d = np.load(path)
    return d["states"], d["measurements"]


def _scored(x, P, y, S, truth):
    e = x - truth
    return e, e @ np.linalg.solve(P, e), y @ np.linalg.solve(S, y)


def _remember(snapshot, **fields):
    if snapshot is not None:
        snapshot.update(fields)


def _stacked(steps):
    err, nees, nis = zip(*steps)
    return np.array(err), np.array(nees), np.array(nis)


def ekf_like_steps(flt, x_true, z, snapshot=None):
    """Yields (err[9], nees, nis) per step as x_true and z yield, live-drivable; see run_ekf_like.
    snapshot, if given, is overwritten each step with the latest P (imm_steps adds mu, rkn_steps P)."""
    for t, (truth, z_t) in enumerate(zip(x_true, z)):
        if t == 0:
            x, P = truth + _INIT_OFFSET, _P0.copy()
            if isinstance(flt, UKF):
                pts_pred = flt._sigma_points(x, P)
                x, P, y, S = flt.update(x, P, pts_pred, z_t)
            else:
                x, P, y, S = flt.update(x, P, z_t)
        else:
            x, P, y, S = flt.step(x, P, z_t)
        _remember(snapshot, P=P)
        yield _scored(x, P, y, S, truth)


def run_ekf_like(flt, x_true, z):
    """flt: EKF or UKF instance ((x,P,z)->x,P,y,S signature). x_true: [T,9],
    z: [T,3]. Returns (err[T',9], nees[T'], nis[T']), T' <= T as the run stops at its divergence step."""
    return _stacked(until_diverged(ekf_like_steps(flt, x_true, z)))


def imm_steps(imm, x_true, z, snapshot=None):
    for t, (truth, z_t) in enumerate(zip(x_true, z)):
        if t == 0:
            state = imm.init_state(truth + _INIT_OFFSET, _P0.copy())
        x, P, state, y, S = imm.step(state, z_t)
        _remember(snapshot, P=P, mu=state["mu"])
        yield _scored(x, P, y, S, truth)


def run_imm(imm, x_true, z):
    return _stacked(until_diverged(imm_steps(imm, x_true, z)))


def knet_steps(weights, dt, R, x_true, z):
    """Yields err[9] per step of the numpy KalmanNet, priors as scripts/eval_knet._priors; live-drivable."""
    _, Q = cv_model(dt)
    flt = numpy_forward.KalmanNetFilter(numpy_forward.KalmanGainGRU(weights), dt)
    for t, (truth, z_t) in enumerate(zip(x_true, z)):
        if t == 0:
            flt.init_sequence(truth + _INIT_OFFSET, Q, np.eye(STATE_DIM) * LEARNED_P0_VARIANCE, R)
        yield flt.step(z_t) - truth


def error_and_nees(x, P_factor, truth):
    """NEES whitened through the lower-triangular factor of P = P_factor P_factor^T: never negative, unlike a
    solve against the squared P once cond(P) passes 1/eps. Also used by scripts/eval_rkn."""
    e = np.nan_to_num(x - truth, nan=DIVERGED_ERROR, posinf=DIVERGED_ERROR, neginf=-DIVERGED_ERROR)
    try:
        whitened = solve_triangular(P_factor, e, lower=True, check_finite=False)
        return e, whitened @ whitened
    except np.linalg.LinAlgError:
        return e, SINGULAR_P_NEES


def bkn_filter(weights, dt, R, trajectory_seed, dropout_p=BKN_DROPOUT_P, realisations=BKN_REALISATIONS):
    rng = numpy_forward.dropout_rng(trajectory_seed)
    return numpy_forward.BayesianKalmanNetFilter(weights, dt, R, dropout_p, rng, realisations)


def bkn_steps(flt, dt, R, x_true, z, snapshot=None):
    """Yields (err[9], nees, nis) per step of the numpy Bayesian KalmanNet, priors as knet_steps; live-drivable.
    A P of rank < 9 (J <= 9 realisations) scores SINGULAR_P_NEES."""
    _, Q = cv_model(dt)
    for t, (truth, z_t) in enumerate(zip(x_true, z)):
        if t == 0:
            flt.init_sequence(truth + _INIT_OFFSET, Q, np.eye(STATE_DIM) * LEARNED_P0_VARIANCE, R)
        x, P, y, S = flt.step(z_t)
        _remember(snapshot, P=P)
        try:
            P_factor = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            P_factor = np.zeros_like(P)
        yield *error_and_nees(x, P_factor, truth), y @ np.linalg.solve(S, y)


def bkn_entry(weights, dt, R, dropout_p=BKN_DROPOUT_P, realisations=BKN_REALISATIONS):
    """(factory, run_fn) for evaluate_dataset; each run's dropout is seeded by its index in the dataset."""
    trajectory_seeds = itertools.count()
    return (
        lambda: bkn_filter(weights, dt, R, next(trajectory_seeds), dropout_p, realisations),
        lambda flt, x_true, z: _stacked(until_diverged(bkn_steps(flt, dt, R, x_true, z))),
    )


def ukn_steps(weights, dt, R, x_true, z, snapshot=None):
    """Yields (err[9], nees, nis) per step of the numpy Unscented KalmanNet, started as the UKF; live-drivable."""
    flt = numpy_forward.UnscentedKalmanNetFilter(numpy_forward.UnscentedKalmanNetwork(weights), dt, R)
    for t, (truth, z_t) in enumerate(zip(x_true, z)):
        if t == 0:
            flt.init_sequence(truth + _INIT_OFFSET, _P0.copy())
        x, P, y, S = flt.step(z_t)
        _remember(snapshot, P=P)
        yield _scored(x, P, y, S, truth)


def ukn_entry(weights, dt, R):
    return (
        lambda: weights,
        lambda flt_weights, x_true, z: _stacked(until_diverged(ukn_steps(flt_weights, dt, R, x_true, z))),
    )


def ckn_steps(weights, dt, R, x_true, z, snapshot=None):
    """Yields (err[9], nees, nis) per step of the numpy Cholesky-KalmanNet, NEES through its own factor of P."""
    flt = numpy_forward.CholeskyKalmanNetFilter(numpy_forward.CholeskyKalmanNetwork(weights), dt, R)
    for t, (truth, z_t) in enumerate(zip(x_true, z)):
        if t == 0:
            flt.init_sequence(truth + _INIT_OFFSET)
        x, P_factor, nis = flt.step(z_t)
        _remember(snapshot, P=P_factor @ P_factor.T)
        yield *error_and_nees(x, P_factor, truth), nis


def ckn_entry(weights, dt, R):
    return (
        lambda: weights,
        lambda flt_weights, x_true, z: _stacked(until_diverged(ckn_steps(flt_weights, dt, R, x_true, z))),
    )


def rkn_steps(weights, dt, x_true, z, snapshot=None):
    """Yields (err[9], nees) per step of the numpy RKN; live-drivable."""
    flt = numpy_forward.RKNFilter(numpy_forward.RKNGainCovNetwork(weights), dt)
    for t, (truth, z_t) in enumerate(zip(x_true, z)):
        if t == 0:
            flt.init_sequence(truth + _INIT_OFFSET, np.eye(STATE_DIM) * LEARNED_P0_VARIANCE)
        x, P = flt.step(z_t)
        _remember(snapshot, P=P)
        yield error_and_nees(x, flt.S, truth)


def evaluate_dataset(filter_factory, run_fn, states, measurements):
    """states: [N,T,9], measurements: [N,mc,T,3]. Returns dict of aggregated
    arrays: err [N,mc,T,9], nees [N,mc,T], nis [N,mc,T], NaN past a run's divergence step."""
    N, mc, T, _ = measurements.shape
    err = np.full((N, mc, T, STATE_DIM), np.nan)
    nees = np.full((N, mc, T), np.nan)
    nis = np.full((N, mc, T), np.nan)
    for i in range(N):
        for m in range(mc):
            flt = filter_factory()
            e, ne, ni = run_fn(flt, states[i], measurements[i, m])
            err[i, m, : len(e)] = e
            nees[i, m, : len(e)] = ne
            nis[i, m, : len(e)] = ni
    return {"err": err, "nees": nees, "nis": nis}


def summarize_rmse(result, burn_in=_BURN_IN):
    err = result["err"][:, :, burn_in:, :][counted_mask(result["err"])[:, :, burn_in:]]
    pos_rmse = float(np.sqrt(counted_mean(np.sum(err[..., POS] ** 2, axis=-1))))
    vel_rmse = float(np.sqrt(counted_mean(np.sum(err[..., VEL] ** 2, axis=-1))))
    return pos_rmse, vel_rmse


def summarize_consistency(result, dim=STATE_DIM, meas_dim=3, burn_in=_BURN_IN):
    counted = counted_mask(result["err"])[:, :, burn_in:]
    nees = result["nees"][:, :, burn_in:]
    nis = result["nis"][:, :, burn_in:]
    mc = nees.shape[1]

    _, bounds = anees_with_bounds(nees, dim=dim, mc_axis=1)
    anees, anees_pass_frac = anees_over_counted(nees, counted, dim=dim, mc_axis=1)

    _, nis_bounds = anees_with_bounds(nis, dim=meas_dim, mc_axis=1)
    anis, anis_pass_frac = anees_over_counted(nis, counted, dim=meas_dim, mc_axis=1)

    coverage = empirical_coverage(nees[counted], dim=dim, levels=(0.68, 0.95))

    return {
        "anees_mean": float(counted_mean(anees)),
        "anees_bounds": bounds,
        "anees_fraction_inside_95": anees_pass_frac,
        "anis_mean": float(counted_mean(anis)),
        "anis_bounds": nis_bounds,
        "anis_fraction_inside_95": anis_pass_frac,
        "coverage_68": coverage[0.68],
        "coverage_95": coverage[0.95],
        "n_mc": mc,
    }


def scenarios_row(name, factory, run_fn, flights):
    """flights: (states [T,9], z [T,3]) pairs of any lengths, each cut before its divergence step and pooled."""
    runs = [run_fn(factory(), states, z) for states, z in flights]
    result = {
        key: np.concatenate([run[i][counted_span(run[0], _BURN_IN)] for run in runs])[None, None]
        for i, key in enumerate(("err", "nees", "nis"))
    }
    pos_rmse, vel_rmse = summarize_rmse(result, burn_in=0)
    divergence = divergence_summary([divergence_steps(run[0]) for run in runs])
    return {"filter": name, "regime": "scenarios", "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, **divergence,
            **summarize_consistency(result, burn_in=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--imm-tuning", default="results/imm_tuning.json")
    ap.add_argument("--scenario-dir", default="data/scenarios")
    ap.add_argument("--bkn-weights", type=Path, default=Path("results/bkn/best.npz"))
    ap.add_argument("--bkn-realisations", type=int, default=BKN_REALISATIONS)
    ap.add_argument("--ukn-weights", type=Path, default=Path("results/ukn/best.npz"))
    ap.add_argument("--ckn-weights", type=Path, default=Path("results/ckn/best.npz"))
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    dt = manifest["regimes"]["nominal"]["dt"]

    def R_for(regime_name):
        r = manifest["regimes"][regime_name]
        return np.diag([r["sigma_az_rad"] ** 2, r["sigma_el_rad"] ** 2, r["sigma_range_m"] ** 2])

    imm_tuning = json.loads(Path(args.imm_tuning).read_text())
    best = imm_tuning["best"]
    print(f"using tuned IMM: {best}")

    bkn_weights = numpy_forward.load_weights(args.bkn_weights) if args.bkn_weights.exists() else None
    ukn_weights = numpy_forward.load_weights(args.ukn_weights) if args.ukn_weights.exists() else None
    ckn_weights = numpy_forward.load_weights(args.ckn_weights) if args.ckn_weights.exists() else None

    def make_filters(regime_name):
        R = R_for(regime_name)
        filters = {
            "EKF": (lambda: EKF(dt, R), run_ekf_like),
            "UKF": (lambda: UKF(dt, R), run_ekf_like),
            "IMM": (
                lambda: IMM(dt, R, **imm_kwargs_from_tuning(best)),
                run_imm,
            ),
        }
        if bkn_weights is not None:
            learned = numpy_forward.learned_dropout_p(bkn_weights)
            dropout_p = BKN_DROPOUT_P if learned is None else learned
            filters["BKN"] = bkn_entry(bkn_weights, dt, R, dropout_p, args.bkn_realisations)
        if ukn_weights is not None:
            filters["UKN"] = ukn_entry(ukn_weights, dt, R)
        if ckn_weights is not None:
            filters["CKN"] = ckn_entry(ckn_weights, dt, R)
        return filters

    rows = []

    # --- Main test set: RMSE with the full 200-trajectory / mc_test set ---
    test_states, test_meas = _load(data_dir / "test.npz")
    filters = make_filters("nominal")
    main_results = {}
    for name, (factory, run_fn) in filters.items():
        print(f"evaluating {name} on test.npz ({test_states.shape[0]} traj x {test_meas.shape[1]} mc)...", flush=True)
        result = evaluate_dataset(factory, run_fn, test_states, test_meas)
        main_results[name] = result
        pos_rmse, vel_rmse = summarize_rmse(result)
        cons = summarize_consistency(result)
        divergence = divergence_summary(divergence_steps(result["err"]))
        rows.append({"filter": name, "regime": "nominal", "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, **divergence, **cons})
        print(f"  {name}: pos_rmse={pos_rmse:.2f}m vel_rmse={vel_rmse:.2f}m/s "
              f"ANEES={cons['anees_mean']:.2f} (bounds {cons['anees_bounds'][0]:.2f}-{cons['anees_bounds'][1]:.2f}) "
              f"frac_in_95={cons['anees_fraction_inside_95']:.3f}")

    # --- High-MC subset: the tight-bound ANEES/NIS headline numbers ---
    highmc_states, highmc_meas = _load(data_dir / "test_highmc.npz")
    highmc_rows = []
    for name, (factory, run_fn) in filters.items():
        print(f"evaluating {name} on test_highmc.npz ({highmc_states.shape[0]} traj x {highmc_meas.shape[1]} mc)...", flush=True)
        result = evaluate_dataset(factory, run_fn, highmc_states, highmc_meas)
        cons = summarize_consistency(result)
        highmc_rows.append({"filter": name, **cons})
        print(f"  {name}: ANEES={cons['anees_mean']:.2f} (bounds {cons['anees_bounds'][0]:.2f}-{cons['anees_bounds'][1]:.2f}) "
              f"frac_in_95={cons['anees_fraction_inside_95']:.3f} coverage68={cons['coverage_68']:.3f} coverage95={cons['coverage_95']:.3f}")

    # --- Robustness sweep: RMSE per shifted regime ---
    for regime_name in manifest["regimes"]:
        if regime_name == "nominal":
            continue
        sweep_path = data_dir / f"test_{regime_name}.npz"
        if not sweep_path.exists():
            continue
        sweep_states, sweep_meas = _load(sweep_path)
        filters_r = make_filters(regime_name)
        print(f"evaluating robustness sweep regime '{regime_name}' ({sweep_states.shape[0]} traj)...", flush=True)
        for name, (factory, run_fn) in filters_r.items():
            result = evaluate_dataset(factory, run_fn, sweep_states, sweep_meas)
            pos_rmse, vel_rmse = summarize_rmse(result)
            cons = summarize_consistency(result)
            divergence = divergence_summary(divergence_steps(result["err"]))
            rows.append({"filter": name, "regime": regime_name, "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, **divergence, **cons})
            print(f"  {name}: pos_rmse={pos_rmse:.2f}m vel_rmse={vel_rmse:.2f}m/s frac_in_95={cons['anees_fraction_inside_95']:.3f}")

    flights = [fly_scenario(json.loads(p.read_text())) for p in sorted(Path(args.scenario_dir).glob("*.json"))]
    if flights:
        print(f"evaluating scenarios regime ({len(flights)} recorded flights, nominal R)...", flush=True)
        for name, (factory, run_fn) in filters.items():
            row = scenarios_row(name, factory, run_fn, flights)
            rows.append(row)
            print(f"  {name}: pos_rmse={row['pos_rmse']:.2f}m vel_rmse={row['vel_rmse']:.2f}m/s "
                  f"frac_in_95={row['anees_fraction_inside_95']:.3f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import csv
    fieldnames = ["filter", "regime", "pos_rmse", "vel_rmse", "diverged_fraction", "median_divergence_step",
                  "anees_mean", "anees_bounds",
                  "anees_fraction_inside_95", "anis_mean", "anis_bounds", "anis_fraction_inside_95",
                  "coverage_68", "coverage_95", "n_mc"]
    with open(out / "results_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    (out / "anees_highmc.json").write_text(json.dumps({
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "n_highmc_trajectories": int(highmc_states.shape[0]),
        "mc_per_trajectory": int(highmc_meas.shape[1]),
        "results": highmc_rows,
    }, indent=2))

    print(f"\nwrote {out}/results_table.csv and {out}/anees_highmc.json")


if __name__ == "__main__":
    main()
