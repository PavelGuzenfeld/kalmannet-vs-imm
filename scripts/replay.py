#!/usr/bin/env python3
"""Replay a data/scenarios/ flight natively through all five filters; print each flag's first trip step per filter.

--record writes the scripted seed scenario (a 9 g break turn in the benchmark envelope) to the given path first.
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data.config import NOMINAL
from data.fixed_wing import DEFAULT_PRESET, PRESETS
from data.trajectory_gen import fly_scenario
from filters.ekf import EKF
from filters.flags import DEFAULT_THRESHOLDS, first_trips
from filters.imm import IMM
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.plot_data import imm_kwargs_from_tuning
from filters.state import POS
from filters.ukf import UKF
from scripts.eval import run_ekf_like, run_imm
from scripts.eval_knet import _INIT_OFFSET, _priors
from scripts.eval_rkn import evaluate_dataset as rkn_errors_and_nees

FLAGS = ("track_loss", "overconfidence", "raw_error")
UNGATED_REASONS = {("RKN", "track_loss"): "RKN defines no innovation covariance"}
RKN_LAYER_MULTS = {"fc1_mult": 3, "hidden_mult": 3, "fc2_mult": 4}
SEED_SCENARIO_SEED = 8
SEED_BREAK_TURN_BANK_STEPS = 23
SEED_BREAK_TURN_PULL_STEPS = 150


def break_turn_controls():
    """Level, roll to ~83 deg at full rate, pull 9 g at full throttle for 3 s, roll out, level. 50 Hz steps."""
    roll_rate = PRESETS[DEFAULT_PRESET].max_roll_rate_rad_s
    level, pull = [0.0, 1.0, 0.6], [0.0, 9.0, 1.0]
    roll_in, roll_out = [roll_rate, 1.0, 1.0], [-roll_rate, 1.0, 1.0]
    bank, hold = SEED_BREAK_TURN_BANK_STEPS, SEED_BREAK_TURN_PULL_STEPS
    return [level] * 50 + [roll_in] * bank + [pull] * hold + [roll_out] * bank + [level] * 50


def seed_scenario():
    return {
        "seed": SEED_SCENARIO_SEED,
        "envelope": asdict(PRESETS[DEFAULT_PRESET]),
        "noise_regime": "nominal",
        "thresholds": DEFAULT_THRESHOLDS,
        "git_tag": os.environ.get("GIT_COMMIT", "unknown"),
        "controls": break_turn_controls(),
    }


def knet_errors(checkpoint, states, z, R):
    net = KalmanGainGRU().double()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    flt = KalmanNetFilter(net, NOMINAL.dt)
    flt.init_sequence(torch.tensor(states[None, 0]) + _INIT_OFFSET, *_priors(NOMINAL.dt, R))
    with torch.no_grad():
        estimates = torch.cat([flt.step(torch.tensor(z_t)[None]) for z_t in z]).numpy()
    return estimates - states


def rkn_run(checkpoint, states, z):
    net = RKNGainCovNetwork(**RKN_LAYER_MULTS).double()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    result = rkn_errors_and_nees(net, NOMINAL.dt, states[None], z[None, None])
    return result["err"][0, 0], result["nees"][0, 0]


def replay(scenario, imm_tuning, knet_checkpoint, rkn_checkpoint):
    """Filters assume nominal R whatever the scenario's noise regime. Returns {filter: trips dict or skip reason}."""
    states, z = fly_scenario(scenario)
    R, dt = NOMINAL.R(), NOMINAL.dt
    thresholds = scenario["thresholds"]

    def trips(err, nees=None, nis=None):
        return first_trips(np.linalg.norm(err[:, POS], axis=1), nees, nis, thresholds)

    out = {
        "EKF": trips(*run_ekf_like(EKF(dt, R), states, z)),
        "UKF": trips(*run_ekf_like(UKF(dt, R), states, z)),
    }
    if imm_tuning.exists():
        best = json.loads(imm_tuning.read_text())["best"]
        out["IMM"] = trips(*run_imm(IMM(dt, R, **imm_kwargs_from_tuning(best)), states, z))
    else:
        out["IMM"] = f"skipped, no tuning at {imm_tuning}"
    if knet_checkpoint.exists():
        out["KalmanNet"] = trips(knet_errors(knet_checkpoint, states, z, R))
    else:
        out["KalmanNet"] = f"skipped, no checkpoint at {knet_checkpoint}"
    if rkn_checkpoint.exists():
        out["RKN"] = trips(*rkn_run(rkn_checkpoint, states, z))
    else:
        out["RKN"] = f"skipped, no checkpoint at {rkn_checkpoint}"
    return out


def shown(name, trips, flag):
    if flag not in trips:
        reason = UNGATED_REASONS.get((name, flag))
        return f"n/a ({reason})" if reason else "n/a"
    return "-" if trips[flag] is None else str(trips[flag])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--imm-tuning", type=Path, default=Path("results/imm_tuning.json"))
    ap.add_argument("--knet-checkpoint", type=Path, default=Path("results/knet_vanilla/best.pt"))
    ap.add_argument("--rkn-checkpoint", type=Path, default=Path("results/rkn/best.pt"))
    ap.add_argument("--record", action="store_true", help="write the scripted seed scenario to SCENARIO first")
    args = ap.parse_args()

    scenario = seed_scenario() if args.record else json.loads(args.scenario.read_text())
    results = replay(scenario, args.imm_tuning, args.knet_checkpoint, args.rkn_checkpoint)
    for name, trips in results.items():
        if isinstance(trips, str):
            print(f"{name}: {trips}")
        else:
            print(f"{name}: " + " ".join(f"{flag}={shown(name, trips, flag)}" for flag in FLAGS))

    if args.record:
        scenario["tripped"] = {name: trips for name, trips in results.items() if not isinstance(trips, str)}
        args.scenario.parent.mkdir(parents=True, exist_ok=True)
        args.scenario.write_text(json.dumps(scenario) + "\n")


if __name__ == "__main__":
    main()
