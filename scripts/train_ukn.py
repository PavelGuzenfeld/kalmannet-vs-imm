#!/usr/bin/env python3
"""Truncated-BPTT Unscented KalmanNet training on the eq 28 loss with eq 26-27 adaptive weights (MS-48); the ΔK term is
relative to ‖K_UKF‖_F², whose columns mix m/rad and m/m. As train_rkn.py: non-finite chunks are skipped, a failed
Cholesky drops the batch's remaining steps, and the checkpoint is the best validation position RMSE after warmup."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from filters.knet.ukn_filter import UnscentedKalmanNetFilter
from filters.knet.ukn_network import UnscentedKalmanNetwork
from filters.state import MEAS_DIM, POS, STATE_DIM
from scripts.eval import _INIT_OFFSET, _P0
from scripts.train_knet import _LOSS_WEIGHT

DIVERGED_ERROR = 1e5


def _load_split(path):
    d = np.load(path)
    return d["states"], d["measurements"]


def nominal_R(manifest):
    nominal = manifest["regimes"]["nominal"]
    return np.diag(
        [nominal["sigma_az_rad"] ** 2, nominal["sigma_el_rad"] ** 2, nominal["sigma_range_m"] ** 2]
    )


def calibration_loss(error, P, soft_clamp):
    """Eq 23: diagonal Gaussian NLL whose normalised squared error is soft-clamped by c."""
    variance = torch.diagonal(P, dim1=-2, dim2=-1)
    return 0.5 * torch.mean(
        soft_clamp * torch.log1p(error**2 / (soft_clamp * variance)) + torch.log(variance)
    )


def measurement_loss(y, S, dof):
    """Eq 24: Student-t innovation NLL with ν_df degrees of freedom."""
    nis = (y.unsqueeze(-2) @ torch.linalg.solve(S, y.unsqueeze(-1))).squeeze((-2, -1))
    return 0.5 * torch.mean(torch.logdet(S) + (dof + MEAS_DIM) * torch.log1p(nis / dof))


def relative_gain_loss(delta_K, K_ukf):
    return torch.mean(torch.linalg.matrix_norm(delta_K) ** 2 / torch.linalg.matrix_norm(K_ukf) ** 2)


def warmup_ramp(epoch, start, length):
    return float(np.clip((epoch - start + 1) / length, 0.0, 1.0))


def smoothed_discrepancy(previous, discrepancy, ema):
    """Eq 26, seeded with the first epoch's discrepancy."""
    return discrepancy if previous is None else ema * previous + (1.0 - ema) * discrepancy


def updated_weight(weight, smoothed_discrepancy, args):
    """Eq 27: multiplicative step towards the target discrepancy τ, clipped to [w_min, w_max]."""
    return float(
        np.clip(
            weight * np.exp(args.weight_rate * (smoothed_discrepancy - args.weight_target)),
            *args.weight_bounds,
        )
    )


class EpochDiagnostics:
    """Eq 26's inputs: epoch-pooled squared error against predicted variance, and mean NIS against n_y."""

    def __init__(self):
        self.squared_error = np.zeros(STATE_DIM)
        self.variance = np.zeros(STATE_DIM)
        self.nis = 0.0
        self.count = 0

    def add(self, error, P, y, S):
        with torch.no_grad():
            finite = torch.isfinite(error).all(1)
            self.squared_error += (error[finite] ** 2).sum(0).numpy()
            self.variance += torch.diagonal(P[finite], dim1=-2, dim2=-1).sum(0).numpy()
            self.nis += (
                (y[finite].unsqueeze(-2) @ torch.linalg.solve(S[finite], y[finite].unsqueeze(-1)))
                .sum()
                .item()
            )
            self.count += int(finite.sum())

    def discrepancies(self):
        calibration = float(np.mean(np.abs(np.log(self.squared_error / self.variance))))
        return calibration, abs(float(np.log(self.nis / max(self.count, 1) / MEAS_DIM)))


def run_training_epoch(net, dt, R, states, measurements, args, weights, rng, optimizer):
    N, T, _ = states.shape
    order = rng.permutation(N)
    diagnostics = EpochDiagnostics()
    total_loss, total_chunks, dropped_batches = 0.0, 0, 0
    P0 = torch.tensor(_P0, dtype=torch.float64)
    for start in range(0, N, args.batch_size):
        idx = order[start : start + args.batch_size]
        x_true = torch.tensor(states[idx], dtype=torch.float64)
        z = torch.tensor(measurements[idx, 0], dtype=torch.float64)
        x0 = (
            x_true[:, 0] + torch.randn(len(idx), STATE_DIM, dtype=torch.float64) @ torch.linalg.cholesky(P0).T
        )
        flt = UnscentedKalmanNetFilter(net, dt, R)
        flt.init_sequence(x0, P0.expand(len(idx), -1, -1))
        loss_accum, chunk_steps = torch.zeros((), dtype=torch.float64), 0
        for t in range(T):
            try:
                x, P, y, S, delta_K, K_ukf = flt.step(z[:, t])
            except torch.linalg.LinAlgError:
                dropped_batches += 1
                break
            error = x - x_true[:, t]
            diagnostics.add(error, P, y, S)
            loss_accum = loss_accum + (
                torch.mean(torch.sum(error**2 * _LOSS_WEIGHT, dim=-1))
                + weights["calibration"] * calibration_loss(error, P, args.calibration_soft_clamp)
                + weights["measurement"] * measurement_loss(y, S, args.student_dof)
                + args.gain_weight * relative_gain_loss(delta_K, K_ukf)
            )
            chunk_steps += 1
            if chunk_steps == args.trunc_len or t == T - 1:
                loss = loss_accum / chunk_steps
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                if torch.isfinite(loss) and torch.isfinite(grad_norm):
                    optimizer.step()
                    total_loss += min(loss.item(), 1e6)
                    total_chunks += 1
                flt.detach_state()
                loss_accum, chunk_steps = torch.zeros((), dtype=torch.float64), 0
    return total_loss / max(total_chunks, 1), diagnostics.discrepancies(), dropped_batches


def validation_position_rmse(net, dt, R, states, measurements, batch_size):
    """Started as eval.py starts the UKF (fixed offset, P0); non-finite errors read as DIVERGED_ERROR."""
    squared = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            x_true = torch.tensor(states[start : start + batch_size], dtype=torch.float64)
            z = torch.tensor(measurements[start : start + batch_size, 0], dtype=torch.float64)
            flt = UnscentedKalmanNetFilter(net, dt, R)
            batch = x_true.shape[0]
            flt.init_sequence(
                x_true[:, 0] + torch.tensor(_INIT_OFFSET), torch.tensor(_P0).expand(batch, -1, -1)
            )
            for t in range(x_true.shape[1]):
                try:
                    error = flt.step(z[:, t])[0] - x_true[:, t]
                except torch.linalg.LinAlgError:
                    error = torch.full_like(x_true[:, t], DIVERGED_ERROR)
                error = torch.nan_to_num(
                    error, nan=DIVERGED_ERROR, posinf=DIVERGED_ERROR, neginf=-DIVERGED_ERROR
                )
                squared.append((error[:, POS] ** 2).sum(-1).numpy())
    return float(np.sqrt(np.mean(np.concatenate(squared))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--out", default="results/ukn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--trunc-len", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gain-scale", type=float, default=0.3, help="c_K of eq 20, chosen on validation by scripts/tune_ukn.py")
    ap.add_argument("--gain-weight", type=float, default=1e-2, help="w_ΔK of eq 28, on ‖ΔK‖²/‖K_UKF‖²")
    ap.add_argument("--calibration-soft-clamp", type=float, default=9.0, help="c of eq 23")
    ap.add_argument("--student-dof", type=float, default=4.0, help="ν_df of eq 24")
    ap.add_argument("--ema", type=float, default=0.7, help="β of eq 26")
    ap.add_argument("--weight-rate", type=float, default=0.5, help="η of eq 27")
    ap.add_argument("--weight-target", type=float, default=0.1, help="τ of eq 27")
    ap.add_argument("--weight-bounds", type=float, nargs=2, default=[0.1, 10.0])
    ap.add_argument("--warmup-start", type=int, default=2, help="epochs of MSE alone")
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument(
        "--max-train", type=int, default=None, help="first N training trajectories, for a smoke run"
    )
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

    net = UnscentedKalmanNetwork(gain_scale=args.gain_scale).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    adaptive = {"calibration": 1.0, "measurement": 1.0}
    smoothed = {"calibration": None, "measurement": None}
    first_selectable = args.warmup_start + args.warmup_epochs - 1
    best_val, best_epoch, patience_left = float("inf"), -1, args.patience
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        ramp = warmup_ramp(epoch, args.warmup_start, args.warmup_epochs)
        weights = {name: ramp * value for name, value in adaptive.items()}
        train_loss, discrepancies, dropped_batches = run_training_epoch(
            net, dt, R, train_states, train_meas, args, weights, rng, optimizer
        )
        for name, discrepancy in zip(adaptive, discrepancies):
            smoothed[name] = smoothed_discrepancy(smoothed[name], discrepancy, args.ema)
            adaptive[name] = updated_weight(adaptive[name], smoothed[name], args)
        val_rmse = validation_position_rmse(net, dt, R, val_states, val_meas, args.batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_pos_rmse": val_rmse,
                "ramp": ramp,
                "dropped_batches": dropped_batches,
                "discrepancies": discrepancies,
                "weights": dict(adaptive),
            }
        )
        print(json.dumps(history[-1]), f"elapsed={time.time() - t0:.0f}s", flush=True)
        if epoch < first_selectable:
            continue
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
