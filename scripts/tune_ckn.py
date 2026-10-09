#!/usr/bin/env python3
"""Pick Cholesky-KalmanNet's β on val.npz only (#136), through the page's numpy forward, one checkpoint per β of the
grid {0.01, 0.05, 0.25} fixed before training (MS-53). Rule as scripts/tune_bkn.py, for calibration: of the runs within
DIVERGED_SLACK of the fewest diverged and RMSE_SLACK of the best position RMSE, the ANEES closest to 9 in log ratio."""

import argparse
import json
from pathlib import Path

import numpy as np

from filters.knet import numpy_forward
from filters.metrics import divergence_steps, divergence_summary
from filters.state import STATE_DIM
from scripts.eval import _load, ckn_entry, evaluate_dataset, summarize_consistency, summarize_rmse
from scripts.train_ukn import nominal_R
from scripts.tune_bkn import DIVERGED_SLACK, RMSE_SLACK


def validation_row(weights_npz, dt, R, states, measurements):
    result = evaluate_dataset(*ckn_entry(numpy_forward.load_weights(weights_npz), dt, R), states, measurements)
    pos_rmse, vel_rmse = summarize_rmse(result)
    consistency = summarize_consistency(result)
    return {
        "weights": str(weights_npz),
        "beta": json.loads((weights_npz.parent / "training_log.json").read_text())["hyperparams"]["beta"],
        "pos_rmse": pos_rmse,
        "vel_rmse": vel_rmse,
        **divergence_summary(divergence_steps(result["err"])),
        **{key: consistency[key] for key in ("anees_mean", "anis_mean", "coverage_68", "coverage_95")},
    }


def selected(rows):
    """The smaller β on a tie: the released one, 0.05, wins unless calibration separates them."""
    fewest_diverged = min(row["diverged_fraction"] for row in rows)
    best_rmse = min(row["pos_rmse"] for row in rows)
    candidates = [
        row
        for row in rows
        if row["diverged_fraction"] <= fewest_diverged + DIVERGED_SLACK and row["pos_rmse"] <= RMSE_SLACK * best_rmse
    ]
    return min(candidates, key=lambda row: (abs(np.log(row["anees_mean"] / STATE_DIM)), row["beta"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path, nargs="+", help="one exported best.npz per trained β, its training_log.json beside it")
    ap.add_argument("--data-dir", type=Path, default=Path("datasets"))
    ap.add_argument("--out", type=Path, default=Path("results/ckn_tuning.json"))
    args = ap.parse_args()

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    states, measurements = _load(args.data_dir / "val.npz")
    rows = []
    for weights_npz in args.weights:
        rows.append(validation_row(weights_npz, manifest["regimes"]["nominal"]["dt"], nominal_R(manifest), states, measurements))
        print(json.dumps(rows[-1]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "best": selected(rows)}))
    print(f"best: {selected(rows)}")


if __name__ == "__main__":
    main()
