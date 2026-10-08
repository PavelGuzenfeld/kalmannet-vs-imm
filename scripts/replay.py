#!/usr/bin/env python3
"""Replay a data/scenarios/ flight natively through all six filters; print each flag's first trip step per filter.

--record writes the scripted seed scenario (a 9 g break turn in the benchmark envelope) to the given path first.
"""

import argparse
import io
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
from filters.flags import (
    DEFAULT_THRESHOLDS,
    FLAGS,
    counted_steps,
    divergence_step,
    first_trips,
    shown,
    until_diverged,
)
from filters.imm import IMM
from filters.knet import numpy_forward
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.knet_filter import KalmanNetFilter
from filters.knet.rkn_network import RKNGainCovNetwork
from filters.plot_data import imm_kwargs_from_tuning
from filters.state import POS
from filters.torch_ops import export_npz
from filters.ukf import UKF
from scripts.eval import bkn_filter, bkn_steps, knet_steps, rkn_steps, run_ekf_like, run_imm, ukn_steps
from scripts.eval_knet import _INIT_OFFSET, _priors
from scripts.eval_rkn import evaluate_dataset as rkn_errors_and_nees

RKN_LAYER_MULTS = {"fc1_mult": 3, "hidden_mult": 3, "fc2_mult": 4}
SEED_SCENARIO_SEED = 8
SEED_BREAK_TURN_BANK_STEPS = 23
SEED_BREAK_TURN_PULL_STEPS = 150
OVERHEAD_SCENARIO_SEED = 4
OVERHEAD_START_M = [-500.0, 0.0, 300.0]
OVERHEAD_PASS_STEPS = 200


def break_turn_controls():
    """Level, roll to ~83 deg at full rate, pull 9 g at full throttle for 3 s, roll out, level. 50 Hz steps."""
    roll_rate = PRESETS[DEFAULT_PRESET].max_roll_rate_rad_s
    level, pull = [0.0, 1.0, 0.6, 0.0, 0.0, 0.0], [0.0, 9.0, 1.0, 0.0, 0.0, 0.0]
    roll_in, roll_out = [roll_rate, 1.0, 1.0, 0.0, 0.0, 0.0], [-roll_rate, 1.0, 1.0, 0.0, 0.0, 0.0]
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


def overhead_scenario():
    """MS-35: unaccelerated full-throttle pass at max speed; truth reaches rho = 0 exactly at step 100."""
    envelope = PRESETS[DEFAULT_PRESET]
    return {
        **seed_scenario(),
        "seed": OVERHEAD_SCENARIO_SEED,
        "start": {"position": OVERHEAD_START_M, "velocity": [envelope.max_speed_m_s, 0.0, 0.0]},
        "controls": [[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]] * OVERHEAD_PASS_STEPS,
    }


def knet_errors(checkpoint, states, z, R):
    net = KalmanGainGRU().double()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    flt = KalmanNetFilter(net, NOMINAL.dt)
    flt.init_sequence(torch.tensor(states[None, 0]) + _INIT_OFFSET, *_priors(NOMINAL.dt, R))
    with torch.no_grad():
        steps = ((flt.step(torch.tensor(z_t)[None])[0].numpy() - truth,) for truth, z_t in zip(states, z))
        return np.array([err for err, in until_diverged(steps)])


def rkn_run(checkpoint, states, z):
    net = RKNGainCovNetwork(**RKN_LAYER_MULTS).double()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    result = rkn_errors_and_nees(net, NOMINAL.dt, states[None], z[None, None])
    stepped = slice(counted_steps(np.linalg.norm(result["err"][0, 0, :, POS], axis=1)) + 1)
    return result["err"][0, 0, stepped], result["nees"][0, 0, stepped]


def numpy_weights(checkpoint):
    """Through the .npz the page loads, so the numpy backend never reads a tensor."""
    npz = io.BytesIO()
    export_npz(torch.load(checkpoint, map_location="cpu", weights_only=True), npz)
    npz.seek(0)
    return numpy_forward.load_weights(npz)


def knet_errors_numpy(checkpoint, states, z, R):
    steps = ((err,) for err in knet_steps(numpy_weights(checkpoint), NOMINAL.dt, R, states, z))
    return np.array([err for err, in until_diverged(steps)])


def rkn_run_numpy(checkpoint, states, z):
    err, nees = zip(*until_diverged(rkn_steps(numpy_weights(checkpoint), NOMINAL.dt, states, z)))
    return np.array(err), np.array(nees)


def bkn_run(checkpoint, seed, states, z, R):
    """The page's numpy Bayesian KalmanNet, dropout seeded from the scenario seed as the page seeds it."""
    flt = bkn_filter(numpy_weights(checkpoint), NOMINAL.dt, R, seed)
    err, nees, nis = zip(*until_diverged(bkn_steps(flt, NOMINAL.dt, R, states, z)))
    return np.array(err), np.array(nees), np.array(nis)


def ukn_run(checkpoint, states, z, R):
    err, nees, nis = zip(*until_diverged(ukn_steps(numpy_weights(checkpoint), NOMINAL.dt, R, states, z)))
    return np.array(err), np.array(nees), np.array(nis)


LEARNED_BACKENDS = {"torch": (knet_errors, rkn_run), "numpy": (knet_errors_numpy, rkn_run_numpy)}


def replay(
    scenario, imm_tuning, knet_checkpoint, rkn_checkpoint, learned_backend="torch", bkn_checkpoint=None, ukn_checkpoint=None
):
    """Filters assume nominal R whatever the scenario's noise regime. Returns {filter: trips dict or skip reason}."""
    states, z = fly_scenario(scenario)
    R, dt = NOMINAL.R(), NOMINAL.dt
    thresholds = scenario["thresholds"]
    knet_run, rkn_backend_run = LEARNED_BACKENDS[learned_backend]

    def trips(err, nees=None, nis=None):
        position_error_m = np.linalg.norm(err[:, POS], axis=1)
        found = first_trips(position_error_m, nees, nis, thresholds)
        step = divergence_step(position_error_m)
        return found if step is None else {**found, "diverged": step}

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
        out["KalmanNet"] = trips(knet_run(knet_checkpoint, states, z, R))
    else:
        out["KalmanNet"] = f"skipped, no checkpoint at {knet_checkpoint}"
    if rkn_checkpoint.exists():
        out["RKN"] = trips(*rkn_backend_run(rkn_checkpoint, states, z))
    else:
        out["RKN"] = f"skipped, no checkpoint at {rkn_checkpoint}"
    if bkn_checkpoint is not None and bkn_checkpoint.exists():
        out["BKN"] = trips(*bkn_run(bkn_checkpoint, scenario["seed"], states, z, R))
    else:
        out["BKN"] = f"skipped, no checkpoint at {bkn_checkpoint}"
    if ukn_checkpoint is not None and ukn_checkpoint.exists():
        out["UKN"] = trips(*ukn_run(ukn_checkpoint, states, z, R))
    else:
        out["UKN"] = f"skipped, no checkpoint at {ukn_checkpoint}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--imm-tuning", type=Path, default=Path("results/imm_tuning.json"))
    ap.add_argument("--knet-checkpoint", type=Path, default=Path("results/knet_vanilla/best.pt"))
    ap.add_argument("--rkn-checkpoint", type=Path, default=Path("results/rkn/best.pt"))
    ap.add_argument("--bkn-checkpoint", type=Path, default=Path("results/bkn/best.pt"))
    ap.add_argument("--ukn-checkpoint", type=Path, default=Path("results/ukn/best.pt"))
    ap.add_argument("--learned-backend", choices=LEARNED_BACKENDS, default="torch")
    ap.add_argument("--record", action="store_true", help="write the scripted seed scenario to SCENARIO first")
    ap.add_argument("--overhead", action="store_true", help="with --record, write the overhead pass instead")
    args = ap.parse_args()

    if args.record:
        scenario = overhead_scenario() if args.overhead else seed_scenario()
    else:
        scenario = json.loads(args.scenario.read_text())
    results = replay(
        scenario, args.imm_tuning, args.knet_checkpoint, args.rkn_checkpoint, args.learned_backend, args.bkn_checkpoint,
        args.ukn_checkpoint,
    )
    for name, trips in results.items():
        if isinstance(trips, str):
            print(f"{name}: {trips}")
        else:
            diverged = f" diverged={trips['diverged']}" if "diverged" in trips else ""
            print(f"{name}: " + " ".join(f"{flag}={shown(name, trips, flag)}" for flag in FLAGS) + diverged)

    if args.record:
        scenario["tripped"] = {name: trips for name, trips in results.items() if not isinstance(trips, str)}
        args.scenario.parent.mkdir(parents=True, exist_ok=True)
        args.scenario.write_text(json.dumps(scenario) + "\n")


if __name__ == "__main__":
    main()
