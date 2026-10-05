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
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
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
    """x_true: [T,9], z: [T,3]. Returns the errors [T - burn-in, 9]."""
    prior_Q, prior_Sigma, prior_S = _priors(dt, R)
    x_true = torch.tensor(x_true, dtype=torch.float64).unsqueeze(0)
    z = torch.tensor(z, dtype=torch.float64).unsqueeze(0)
    flt = KalmanNetFilter(net, dt)
    flt.init_sequence(x_true[:, 0, :] + _INIT_OFFSET, prior_Q, prior_Sigma, prior_S)
    errors = []
    with torch.no_grad():
        for t in range(z.shape[1]):
            x_upd = flt.step(z[:, t, :])
            if t >= _BURN_IN:
                err = (x_upd[0] - x_true[0, t]).numpy()
                errors.append(np.nan_to_num(err, nan=1e5, posinf=1e5, neginf=-1e5))
    return np.array(errors)


def position_velocity_rmse(errors):
    """errors: [n,9]. Returns (pos_rmse, vel_rmse)."""
    return (
        float(np.sqrt(np.mean(np.sum(errors[:, :3] ** 2, axis=-1)))),
        float(np.sqrt(np.mean(np.sum(errors[:, 3:6] ** 2, axis=-1)))),
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

        errors = [
            flight_errors(net, dt, R, states[i], measurements[i, m])
            for i in range(measurements.shape[0])
            for m in range(measurements.shape[1])
        ]
        pos_rmse, vel_rmse = position_velocity_rmse(np.concatenate(errors))
        print(f"{regime_name}: pos_rmse={pos_rmse:.2f}m vel_rmse={vel_rmse:.2f}m/s", flush=True)
        rows.append({"filter": "KalmanNet-vanilla", "regime": regime_name, "pos_rmse": pos_rmse, "vel_rmse": vel_rmse})

    flights = [fly_scenario(json.loads(p.read_text())) for p in sorted(Path(args.scenario_dir).glob("*.json"))]
    if flights:
        errors = np.concatenate([flight_errors(net, NOMINAL.dt, NOMINAL.R(), states, z) for states, z in flights])
        pos_rmse, vel_rmse = position_velocity_rmse(errors)
        print(f"scenarios: pos_rmse={pos_rmse:.2f}m vel_rmse={vel_rmse:.2f}m/s", flush=True)
        rows.append({"filter": "KalmanNet-vanilla", "regime": "scenarios", "pos_rmse": pos_rmse, "vel_rmse": vel_rmse})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filter", "regime", "pos_rmse", "vel_rmse"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
