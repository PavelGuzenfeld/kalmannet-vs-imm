#!/usr/bin/env python3
"""Truncated-BPTT Cholesky-KalmanNet training on eq 5-7, (1 − β)·MSE + β·L1 over every entry of e eᵀ − P (MS-53).
As the reference: Adam at 5e-4 on one branch at a time, alternating every step, gradient norm clipped to 1. As
train_ukn.py: non-finite chunks are skipped, and the checkpoint is the best validation position RMSE."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from filters.knet.ckn_filter import CholeskyKalmanNetFilter
from filters.knet.ckn_network import CholeskyKalmanNetwork
from filters.state import POS, STATE_DIM
from scripts.eval import _INIT_OFFSET, _P0
from scripts.train_knet import _LOSS_WEIGHT
from scripts.train_ukn import DIVERGED_ERROR, _load_split, nominal_R

RELEASED_BETA = 0.05


def ckn_loss(error, P, beta):
    """Eq 5-7 with the reference's reductions: MSE averaged over states, L1 summed over all 81 entries, both averaged
    over the batch; entry (i, j) weighted by sqrt(w_i w_j), w the MS-10 loss weights, so both terms are unitless."""
    mse = torch.mean(error**2 * _LOSS_WEIGHT)
    scale = torch.sqrt(_LOSS_WEIGHT)
    outer = error.unsqueeze(-1) * error.unsqueeze(-2)
    covariance = torch.mean(torch.sum(torch.abs((outer - P) * scale[:, None] * scale[None, :]), dim=(-2, -1)))
    return (1.0 - beta) * mse + beta * covariance


def run_training_epoch(net, dt, R, states, measurements, args, rng, optimizers, step_count):
    N, T, _ = states.shape
    order = rng.permutation(N)
    total_loss, total_chunks = 0.0, 0
    P0_factor = torch.linalg.cholesky(torch.tensor(_P0, dtype=torch.float64))
    branches = (net.prior, net.innovation)
    for start in range(0, N, args.batch_size):
        idx = order[start : start + args.batch_size]
        x_true = torch.tensor(states[idx], dtype=torch.float64)
        z = torch.tensor(measurements[idx, 0], dtype=torch.float64)
        x0 = x_true[:, 0] + torch.randn(len(idx), STATE_DIM, dtype=torch.float64) @ P0_factor.T
        flt = CholeskyKalmanNetFilter(net, dt, R)
        flt.init_sequence(x0)
        loss_accum, chunk_steps = torch.zeros((), dtype=torch.float64), 0
        for t in range(T):
            x, P, _ = flt.step(z[:, t])
            loss_accum = loss_accum + ckn_loss(x - x_true[:, t], P, args.beta)
            chunk_steps += 1
            if chunk_steps == args.trunc_len or t == T - 1:
                active = step_count % len(optimizers)
                loss = loss_accum / chunk_steps
                net.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(branches[active].parameters(), max_norm=1.0)
                if torch.isfinite(loss) and torch.isfinite(grad_norm):
                    optimizers[active].step()
                    total_loss += min(loss.item(), 1e6)
                    total_chunks += 1
                step_count += 1
                flt.detach_state()
                loss_accum, chunk_steps = torch.zeros((), dtype=torch.float64), 0
    return total_loss / max(total_chunks, 1), step_count


def validation_position_rmse(net, dt, R, states, measurements, batch_size):
    """Started as eval.py starts CKN (fixed offset); non-finite errors read as DIVERGED_ERROR."""
    squared = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            x_true = torch.tensor(states[start : start + batch_size], dtype=torch.float64)
            z = torch.tensor(measurements[start : start + batch_size, 0], dtype=torch.float64)
            flt = CholeskyKalmanNetFilter(net, dt, R)
            flt.init_sequence(x_true[:, 0] + torch.tensor(_INIT_OFFSET))
            for t in range(x_true.shape[1]):
                error = torch.nan_to_num(
                    flt.step(z[:, t])[0] - x_true[:, t], nan=DIVERGED_ERROR, posinf=DIVERGED_ERROR, neginf=-DIVERGED_ERROR
                )
                squared.append((error[:, POS] ** 2).sum(-1).numpy())
    return float(np.sqrt(np.mean(np.concatenate(squared))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--out", default="results/ckn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--trunc-len", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=RELEASED_BETA, help="β of eq 5")
    ap.add_argument("--max-train", type=int, default=None, help="first N training trajectories, for a smoke run")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    dt, R = manifest["regimes"]["nominal"]["dt"], nominal_R(manifest)
    train_states, train_meas = (a[: args.max_train] for a in _load_split(data_dir / "train.npz"))
    val_states, val_meas = (a[: args.max_val] for a in _load_split(data_dir / "val.npz"))

    net = CholeskyKalmanNetwork(R).double()
    optimizers = [torch.optim.Adam(branch.parameters(), lr=args.lr) for branch in (net.prior, net.innovation)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, patience_left, step_count = float("inf"), -1, args.patience, 0
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        train_loss, step_count = run_training_epoch(
            net, dt, R, train_states, train_meas, args, rng, optimizers, step_count
        )
        val_rmse = validation_position_rmse(net, dt, R, val_states, val_meas, args.batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pos_rmse": val_rmse})
        print(json.dumps(history[-1]), f"elapsed={time.time() - t0:.0f}s", flush=True)
        if val_rmse < best_val:
            best_val, best_epoch, patience_left = val_rmse, epoch, args.patience
            torch.save(net.state_dict(), out / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    log = {
        "seed": args.seed,
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "data_manifest_seed": manifest["seed"],
        "hyperparams": vars(args),
        "param_count": sum(p.numel() for p in net.parameters()),
        "best_epoch": best_epoch,
        "best_val_pos_rmse": best_val,
        "wall_time_s": time.time() - t0,
        "history": history,
    }
    (out / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"done: best epoch {best_epoch}, val_pos_rmse={best_val:.4f}, wall_time={log['wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
