#!/usr/bin/env python3
"""Evaluate the trained vanilla KalmanNet checkpoint on the same test.npz
(all mc realizations) and robustness-sweep sets the classical filters are
judged on, so its number in the results table is directly comparable. No
covariance is reported (see issue #15 sec. 1.3 -- state dim 9 > measurement
dim 3, so vanilla KalmanNet has no recoverable Sigma): RMSE only.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data.config import NOMINAL
from data.trajectory_gen import fly_scenario
from filters.flags import until_diverged
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.metrics import counted_mean, counted_span, divergence_steps, divergence_summary
from filters.motion_models import cv_model
from filters.state import STATE_DIM

_INIT_OFFSET = torch.tensor([2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
_BURN_IN = 20


def _priors(dt, R):
    _, Q = cv_model(dt)
    return (
        torch.tensor(Q, dtype=torch.float64),
        torch.eye(STATE_DIM, dtype=torch.float64) * 100.0,
        torch.tensor(R, dtype=torch.float64),
    )


def flight_errors(net, dt, R, x_true, z):
    """x_true: [T,9], z: [T,3]. Returns the errors [T',9], T' <= T as the run stops at its divergence step."""
    prior_Q, prior_Sigma, prior_S = _priors(dt, R)
    x_true = torch.tensor(x_true, dtype=torch.float64).unsqueeze(0)
    z = torch.tensor(z, dtype=torch.float64).unsqueeze(0)
    flt = KalmanNetFilter(net, dt)
    flt.init_sequence(x_true[:, 0, :] + _INIT_OFFSET, prior_Q, prior_Sigma, prior_S)
    with torch.no_grad():
        steps = ((flt.step(z[:, t, :])[0].numpy() - x_true[0, t].numpy(),) for t in range(z.shape[1]))
        return np.array([err for err, in until_diverged(steps)])


def regime_row(regime_name, runs):
    """runs: each flight's errors as flight_errors returns them; RMSE over their counted steps."""
    pos_rmse, vel_rmse = position_velocity_rmse(np.concatenate([err[counted_span(err, _BURN_IN)] for err in runs]))
    print(f"{regime_name}: pos_rmse={pos_rmse:.2f}m vel_rmse={vel_rmse:.2f}m/s", flush=True)
    divergence = divergence_summary([divergence_steps(err) for err in runs])
    return {"filter": "KalmanNet-vanilla", "regime": regime_name, "pos_rmse": pos_rmse, "vel_rmse": vel_rmse, **divergence}


def position_velocity_rmse(errors):
    """errors: [n,9]. Returns (pos_rmse, vel_rmse)."""
    return (
        float(np.sqrt(counted_mean(np.sum(errors[:, :3] ** 2, axis=-1)))),
        float(np.sqrt(counted_mean(np.sum(errors[:, 3:6] ** 2, axis=-1)))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--checkpoint", default="results/knet_vanilla/best.pt")
    ap.add_argument("--scenario-dir", default="data/scenarios")
    ap.add_argument("--out", default="results/knet_test_rmse.csv")
    args = ap.parse_args()

    torch.set_num_threads(1)
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    net = KalmanGainGRU().double()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    net.eval()

    rows = []
    for regime_name in manifest["regimes"]:
        path = data_dir / ("test.npz" if regime_name == "nominal" else f"test_{regime_name}.npz")
        if not path.exists():
            continue
        d = np.load(path)
        states, measurements = d["states"], d["measurements"]
        r = manifest["regimes"][regime_name]
        R = np.diag([r["sigma_az_rad"] ** 2, r["sigma_el_rad"] ** 2, r["sigma_range_m"] ** 2])
        dt = r["dt"]

        runs = [
            flight_errors(net, dt, R, states[i], measurements[i, m])
            for i in range(measurements.shape[0])
            for m in range(measurements.shape[1])
        ]
        rows.append(regime_row(regime_name, runs))

    flights = [fly_scenario(json.loads(p.read_text())) for p in sorted(Path(args.scenario_dir).glob("*.json"))]
    if flights:
        rows.append(regime_row("scenarios", [flight_errors(net, NOMINAL.dt, NOMINAL.R(), *flight) for flight in flights]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filter", "regime", "pos_rmse", "vel_rmse", "diverged_fraction", "median_divergence_step"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
