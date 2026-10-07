"""First step each scenario flag trips; None if it never does. Numpy-only, so the page can run it in Pyodide.

Quantiles are scipy chi2.ppf values pinned as literals; NaN trips every flag, so a diverged filter never passes.
"""

import numpy as np

from filters.state import POS

INNOVATION_GATE_NIS_99_DOF3 = 11.344866730144373
NEES_UPPER_975_DOF9 = 19.02276779864163
NEES_LOWER_025_DOF9 = 2.7003894999803584
ELLIPSOID_95_DOF3 = 7.814727903251179
OVERCONFIDENT_STEPS = 25
RAW_ERROR_M = 20.0
DIVERGED_POSITION_ERROR_M = 200.0
FLAGS = ("track_loss", "overconfidence", "raw_error")
UNGATED_REASONS = {("RKN", "track_loss"): "RKN defines no innovation covariance"}

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


def divergence_step(position_error_m):
    """A filter is stepped no further after this step, and metrics count only the steps before it."""
    return _first_above(position_error_m, DIVERGED_POSITION_ERROR_M)


def counted_steps(position_error_m):
    step = divergence_step(position_error_m)
    return len(position_error_m) if step is None else step


def until_diverged(steps):
    """steps yield tuples led by the state error [9]; the diverged step is the last one pulled."""
    for scored in steps:
        yield scored
        if divergence_step([np.linalg.norm(scored[0][POS])]) is not None:
            return


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


def shown(name, trips, flag):
    if flag not in trips:
        reason = UNGATED_REASONS.get((name, flag))
        return f"n/a ({reason})" if reason else "n/a"
    return "-" if trips[flag] is None else str(trips[flag])
