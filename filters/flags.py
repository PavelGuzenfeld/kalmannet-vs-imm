"""First step each scenario flag trips; None if it never does. Numpy-only, so the page can run it in Pyodide.

Quantiles are scipy chi2.ppf values pinned as literals; NaN trips every flag, so a diverged filter never passes.
"""

import numpy as np

INNOVATION_GATE_NIS_99_DOF3 = 11.344866730144373
NEES_UPPER_975_DOF9 = 19.02276779864163
OVERCONFIDENT_STEPS = 25
RAW_ERROR_M = 20.0

DEFAULT_THRESHOLDS = {
    "innovation_gate_nis": INNOVATION_GATE_NIS_99_DOF3,
    "nees_upper": NEES_UPPER_975_DOF9,
    "overconfident_steps": OVERCONFIDENT_STEPS,
    "raw_error_m": RAW_ERROR_M,
}


def _first_above(values, limit):
    above = np.flatnonzero(~(np.asarray(values) <= limit))
    return int(above[0]) if above.size else None


def first_track_loss_step(nis, gate=INNOVATION_GATE_NIS_99_DOF3):
    return _first_above(nis, gate)


def first_overconfidence_step(nees, upper=NEES_UPPER_975_DOF9, steps=OVERCONFIDENT_STEPS):
    run = 0
    for t, inside in enumerate(np.asarray(nees) <= upper):
        run = 0 if inside else run + 1
        if run >= steps:
            return t
    return None


def first_raw_error_step(position_error_m, limit_m=RAW_ERROR_M):
    return _first_above(position_error_m, limit_m)


def first_trips(position_error_m, nees=None, nis=None, thresholds=DEFAULT_THRESHOLDS):
    """A filter without an innovation (nis) or a covariance (nees) gets no flag for it."""
    trips = {}
    if nis is not None:
        trips["track_loss"] = first_track_loss_step(nis, thresholds["innovation_gate_nis"])
    if nees is not None:
        trips["overconfidence"] = first_overconfidence_step(
            nees, thresholds["nees_upper"], thresholds["overconfident_steps"]
        )
    trips["raw_error"] = first_raw_error_step(position_error_m, thresholds["raw_error_m"])
    return trips
