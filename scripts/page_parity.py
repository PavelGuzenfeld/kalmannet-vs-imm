#!/usr/bin/env python3
"""Native side of the page checks: `cases` writes scenarios plus replay.py's flags for the browser test;
`drift` compares a Pyodide trace (scripts/pyodide_trace.mjs) against the same stepping run natively."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from filters.flags import FLAGS, shown
from filters.knet.gain_network import KalmanGainGRU
from filters.torch_ops import export_npz
from scripts.replay import replay
from web.live_flight import trace

SCENARIOS_DIR = Path("data/scenarios")
IMM_TUNING = Path("web/imm_tuning.json")
NO_WEIGHTS = Path("/nonexistent")
TIGHT_THRESHOLDS = {
    "innovation_gate_nis": 3.0,
    "nees_upper": 10.0,
    "overconfident_steps": 5,
    "raw_error_m": 5.0,
}
TRIP_PROBE = ("EKF", "raw_error")
DRIFT_QUANTITIES = ("estimate_m", "nees", "nis")
DIVERGING_KNET_NPZ = "diverging_knet.npz"
DIVERGING_KNET_PT = "diverging_knet.pt"
DIVERGING_KNET_DIVERGENCE = "diverging_knet_divergence.json"
DIVERGING_KNET_SEED = 7
DIVERGING_KNET_WEIGHT_SCALE = 0.1


def native_flags(scenario):
    results = replay(scenario, IMM_TUNING, NO_WEIGHTS, NO_WEIGHTS)
    return {
        name: {flag: shown(name, trips, flag) for flag in FLAGS}
        for name, trips in results.items()
        if not isinstance(trips, str)
    }


def case_scenarios():
    break_turn = json.loads((SCENARIOS_DIR / "break_turn_9g.json").read_text())
    overhead = json.loads((SCENARIOS_DIR / "overhead_pass.json").read_text())
    trip_step = int(native_flags(break_turn)[TRIP_PROBE[0]][TRIP_PROBE[1]])
    return {
        "break_turn_9g": break_turn,
        "overhead_pass": overhead,
        "break_turn_9g_tight_thresholds": {**break_turn, "thresholds": TIGHT_THRESHOLDS},
        "break_turn_9g_one_step_before_ekf_raw_error_trip": {
            **break_turn,
            "controls": break_turn["controls"][:trip_step],
        },
        "break_turn_9g_at_ekf_raw_error_trip": {
            **break_turn,
            "controls": break_turn["controls"][: trip_step + 1],
        },
    }


def write_cases(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, scenario in case_scenarios().items():
        (out_dir / f"{name}.json").write_text(json.dumps(scenario) + "\n")
        expected = {"steps": len(scenario["controls"]), "flags": native_flags(scenario)}
        (out_dir / f"{name}.expected.json").write_text(json.dumps(expected) + "\n")
    write_diverging_knet(out_dir)


def write_diverging_knet(out_dir):
    """KalmanNet weights whose gain runs away on break_turn_9g, and the step replay sees them diverge."""
    torch.manual_seed(DIVERGING_KNET_SEED)
    net = KalmanGainGRU()
    for parameter in net.parameters():
        torch.nn.init.uniform_(parameter, -DIVERGING_KNET_WEIGHT_SCALE, DIVERGING_KNET_WEIGHT_SCALE)
    export_npz(net.double().state_dict(), out_dir / DIVERGING_KNET_NPZ)
    torch.save(net.double().state_dict(), out_dir / DIVERGING_KNET_PT)
    break_turn = json.loads((SCENARIOS_DIR / "break_turn_9g.json").read_text())
    knet = replay(break_turn, IMM_TUNING, out_dir / DIVERGING_KNET_PT, NO_WEIGHTS, "numpy")["KalmanNet"]
    (out_dir / DIVERGING_KNET_DIVERGENCE).write_text(json.dumps({"KalmanNet": knet["diverged"]}) + "\n")


def absent_or_difference(native, other):
    return 0.0 if native is None and other is None else float(np.max(np.abs(np.subtract(native, other))))


def max_abs_difference(native_steps, pyodide_steps, filter_name, quantity):
    return max(
        absent_or_difference(native[filter_name][quantity], other[filter_name][quantity])
        for native, other in zip(native_steps, pyodide_steps, strict=True)
    )


def drift_table(pyodide_traces):
    imm_best = json.loads(IMM_TUNING.read_text())["best"]
    rows = []
    for stem, pyodide_steps in sorted(pyodide_traces.items()):
        scenario = json.loads((SCENARIOS_DIR / f"{stem}.json").read_text())
        native_steps = json.loads(json.dumps(trace(scenario, imm_best, NO_WEIGHTS)))
        for filter_name in sorted({name for step in native_steps for name in step}):
            rows.append(
                (stem, filter_name)
                + tuple(
                    max_abs_difference(native_steps, pyodide_steps, filter_name, quantity)
                    for quantity in DRIFT_QUANTITIES
                )
            )
    return rows


def drift_markdown(rows):
    header = "| scenario | filter | " + " | ".join(f"max abs {q}" for q in DRIFT_QUANTITIES) + " |"
    divider = "|" + " --- |" * (2 + len(DRIFT_QUANTITIES))
    body = [
        f"| {stem} | {name} | " + " | ".join(f"{value:.3e}" for value in values) + " |"
        for stem, name, *values in rows
    ]
    return "\n".join(
        ["### Pyodide vs native, max per-step difference (estimate_m in metres)", header, divider, *body]
    )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cases").add_argument("out_dir", type=Path)
    commands.add_parser("drift").add_argument("pyodide_trace", type=Path)
    args = parser.parse_args()
    if args.command == "cases":
        write_cases(args.out_dir)
    else:
        print(drift_markdown(drift_table(json.loads(args.pyodide_trace.read_text()))))


if __name__ == "__main__":
    main()
