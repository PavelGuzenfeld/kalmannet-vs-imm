#!/usr/bin/env python3
"""Coordinate search for IMM hyperparameters (filters/imm_tuning.py) on the
validation set ONLY; every candidate and the final choice are recorded to
results/imm_tuning.json, the test set is never touched, and an edge selection exits non-zero unless allowed."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from filters.imm import IMM
from filters.imm_tuning import IMM_PINNED, IMM_SEARCH_GRIDS, IMM_SEARCH_START, coordinate_search, edge_axes
from filters.plot_data import imm_kwargs_from_tuning

MAX_EXTENSIONS_PER_AXIS = 3


def _load_split(path):
    d = np.load(path)
    return d["states"], d["measurements"]


def run_imm(states, measurements, dt, R, **imm_kwargs):
    """states: [N,T,9], measurements: [N,1,T,3] (single MC realization).
    Mean position RMSE over steps 20..T, skipping the initial-condition
    transient."""
    imm = IMM(dt, R, **imm_kwargs)
    N, T, _ = states.shape
    all_err = []
    for i in range(N):
        x0 = states[i, 0].copy()
        x0[3:] += np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        P0 = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        state = imm.init_state(x0, P0)
        errs = np.zeros(T)
        for t in range(T):
            x_out, _, state, _, _ = imm.step(state, measurements[i, 0, t])
            errs[t] = np.linalg.norm(x_out[:3] - states[i, t, :3])
        all_err.append(errs[20:])
    all_err = np.concatenate(all_err)
    return float(np.sqrt(np.mean(all_err ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--out", default="results/imm_tuning.json")
    ap.add_argument("--n-search", type=int, default=30, help="val trajectories used during the search")
    ap.add_argument(
        "--allow-edge-selection", action="store_true",
        help="write a selection left on a grid edge instead of refusing; for plumbing smoke runs only",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    dt = manifest["regimes"]["nominal"]["dt"]
    R = np.diag([
        manifest["regimes"]["nominal"]["sigma_az_rad"] ** 2,
        manifest["regimes"]["nominal"]["sigma_el_rad"] ** 2,
        manifest["regimes"]["nominal"]["sigma_range_m"] ** 2,
    ])

    val_states, val_meas = _load_split(data_dir / "val.npz")
    search_states, search_meas = val_states[: args.n_search], val_meas[: args.n_search]

    def score(candidate):
        rmse = run_imm(search_states, search_meas, dt, R, **imm_kwargs_from_tuning({**IMM_PINNED, **candidate}))
        print(f"  {candidate} -> {rmse:.3f} m", flush=True)
        return rmse

    print(f"coordinate search from {IMM_SEARCH_START}, {args.n_search} val trajectories each", flush=True)
    t0 = time.time()
    best, grids, evaluated = coordinate_search(IMM_SEARCH_GRIDS, IMM_SEARCH_START, score, MAX_EXTENSIONS_PER_AXIS)
    print(f"search took {time.time() - t0:.1f}s")
    on_edge = edge_axes(grids, best)
    if on_edge and not args.allow_edge_selection:
        raise SystemExit(f"selection sits on a grid edge on axes {on_edge}: {best}")
    best = {**IMM_PINNED, **best, "val_pos_rmse": evaluated[tuple(sorted(best.items()))]}
    print(f"best (search subset): {best}")

    full_rmse = run_imm(val_states, val_meas, dt, R, **imm_kwargs_from_tuning(best))
    print(f"best config, full val set ({val_states.shape[0]} traj) RMSE: {full_rmse:.2f} m")

    results = sorted(
        ({**IMM_PINNED, **dict(key), "val_pos_rmse": rmse} for key, rmse in evaluated.items()), key=lambda r: r["val_pos_rmse"]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "n_search_trajectories": args.n_search,
        "n_full_val_trajectories": int(val_states.shape[0]),
        "grid": grids,
        "edge_axes": on_edge,
        "all_candidates": results,
        "best": best,
        "best_full_val_pos_rmse": full_rmse,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
