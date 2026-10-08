#!/usr/bin/env python3
"""Pick Bayesian KalmanNet's dropout p and ensemble size J on val.npz only (#135), through the page's numpy forward.
Rule: of the runs within DIVERGED_SLACK of the fewest diverged and RMSE_SLACK of the best position RMSE, the ANEES
closest to the state dimension in log ratio; the smaller J on a tie."""

import argparse
import json
from pathlib import Path

import numpy as np

from filters.knet import numpy_forward
from filters.metrics import divergence_steps, divergence_summary
from filters.state import STATE_DIM
from scripts.eval import _load, bkn_entry, evaluate_dataset, summarize_consistency, summarize_rmse

RMSE_SLACK = 1.1
DIVERGED_SLACK = 0.05


def validation_row(weights_npz, realisations, dt, R, states, measurements):
    weights = numpy_forward.load_weights(weights_npz)
    learned = numpy_forward.learned_dropout_p(weights)
    if learned is None:
        dropout_p = json.loads((weights_npz.parent / "training_log.json").read_text())["hyperparams"]["dropout_p"]
    else:
        dropout_p = learned.tolist()
    entry = bkn_entry(weights, dt, R, dropout_p, realisations)
    result = evaluate_dataset(*entry, states, measurements)
    pos_rmse, vel_rmse = summarize_rmse(result)
    consistency = summarize_consistency(result)
    return {
        "weights": str(weights_npz),
        "dropout_p": dropout_p,
        "realisations": realisations,
        "pos_rmse": pos_rmse,
        "vel_rmse": vel_rmse,
        **divergence_summary(divergence_steps(result["err"])),
        **{key: consistency[key] for key in ("anees_mean", "anis_mean", "coverage_68", "coverage_95")},
    }


def selected(rows):
    fewest_diverged = min(row["diverged_fraction"] for row in rows)
    best_rmse = min(row["pos_rmse"] for row in rows)
    candidates = [
        row
        for row in rows
        if row["diverged_fraction"] <= fewest_diverged + DIVERGED_SLACK and row["pos_rmse"] <= RMSE_SLACK * best_rmse
    ]
    return min(candidates, key=lambda row: (abs(np.log(row["anees_mean"] / STATE_DIM)), row["realisations"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path, nargs="+", help="one exported best.npz per trained p")
    ap.add_argument("--realisations", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--data-dir", type=Path, default=Path("datasets"))
    ap.add_argument("--out", type=Path, default=Path("results/bkn_tuning.json"))
    args = ap.parse_args()

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    nominal = manifest["regimes"]["nominal"]
    R = np.diag([nominal["sigma_az_rad"] ** 2, nominal["sigma_el_rad"] ** 2, nominal["sigma_range_m"] ** 2])
    states, measurements = _load(args.data_dir / "val.npz")

    rows = []
    for weights_npz in args.weights:
        for realisations in args.realisations:
            rows.append(validation_row(weights_npz, realisations, nominal["dt"], R, states, measurements))
            print(json.dumps(rows[-1]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "best": selected(rows)}))
    print(f"best: {selected(rows)}")


if __name__ == "__main__":
    main()
