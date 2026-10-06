import json
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from data.config import NOMINAL, REGIMES
from data.fixed_wing import DEFAULT_PRESET, PRESETS, Envelope, fly_steps
from data.trajectory_gen import measure, scenario_start
from filters.ekf import EKF
from filters.flags import (
    DEFAULT_THRESHOLDS,
    ELLIPSOID_95_DOF3,
    FLAGS,
    NEES_LOWER_025_DOF9,
    UNGATED_REASONS,
    first_trips,
    shown,
)
from filters.imm import IMM
from filters.knet.numpy_forward import load_weights
from filters.plot_data import imm_kwargs_from_tuning
from filters.state import POS, VEL
from filters.ukf import UKF
from scripts.eval import ekf_like_steps, imm_steps, knet_steps, rkn_steps

NOISE_REGIMES = ["nominal", "high_noise", "low_noise", "heavy_tailed"]
FILTER_NAMES = ("EKF", "UKF", "IMM", "KalmanNet", "RKN")
LEARNED_WEIGHTS = {"KalmanNet": "results/knet_vanilla/best.npz", "RKN": "results/rkn/best.npz"}
SETUP_KEYS = ("seed", "envelope", "noise_regime", "start")
UNKNOWN_GIT_TAG = "unknown"
DIVERGED_RAW_ERROR_MULTIPLE = 100.0


def page_options():
    return {
        "presets": {name: asdict(envelope) for name, envelope in PRESETS.items()},
        "default_preset": DEFAULT_PRESET,
        "noise_regimes": NOISE_REGIMES,
        "default_thresholds": DEFAULT_THRESHOLDS,
        "nees_lower": NEES_LOWER_025_DOF9,
        "dt_s": NOMINAL.dt,
        "learned_weights": LEARNED_WEIGHTS,
    }


def _latest(read):
    while True:
        yield read()


def _unavailable_flags(name):
    missing = f"not available (no weights at {LEARNED_WEIGHTS[name]})"
    return {flag: shown(name, {}, flag) if (name, flag) in UNGATED_REASONS else missing for flag in FLAGS}


def _position_errors_m(err):
    return np.linalg.norm(np.array(err)[:, POS], axis=1)


def _ellipsoid(P):
    """95% position ellipsoid: semi-axes in m, unit axis directions as rows."""
    position_P = P[POS, POS]
    if not np.isfinite(position_P).all():
        return None
    variances, directions = np.linalg.eigh((position_P + position_P.T) / 2)
    return {
        "semi_axes_m": np.sqrt(np.clip(variances, 0.0, None) * ELLIPSOID_95_DOF3).tolist(),
        "axes": directions.T.tolist(),
    }


def _readout(truth, err, nees=None, nis=None, *, snapshot):
    return {
        "ellipsoid": _ellipsoid(snapshot["P"]) if "P" in snapshot else None,
        "mode_probabilities": snapshot["mu"].tolist() if "mu" in snapshot else None,
        "estimate_m": (truth[POS] + err[POS]).tolist(),
        "error_m": float(np.linalg.norm(err[POS])),
        "nees": None if nees is None else float(nees),
        "nis": None if nis is None else float(nis),
    }


def diverged(position_error_m, raw_error_m):
    """Diverged once any position error so far is non-finite or above DIVERGED_RAW_ERROR_MULTIPLE raw-error
    thresholds; the page keeps a diverged filter out of its plot scales and 3D framing."""
    return bool(
        not np.isfinite(position_error_m).all() or position_error_m.max() > DIVERGED_RAW_ERROR_MULTIPLE * raw_error_m
    )


def framing_radius_m(readouts):
    """Farthest reach from truth of any non-diverged estimate or its 95% ellipsoid, for the 3D view to frame."""
    return max(
        (
            readout["error_m"] + (max(readout["ellipsoid"]["semi_axes_m"]) if readout["ellipsoid"] else 0.0)
            for readout in readouts.values()
            if not readout["diverged"]
        ),
        default=0.0,
    )


class LiveFlight:
    """scripts/replay.py's flight advanced one control row at a time: same seed draws, filters on nominal R.
    A learned filter runs when weights_root holds its LEARNED_WEIGHTS .npz."""

    def __init__(self, setup, imm_best, weights_root):
        self.setup = {key: setup[key] for key in SETUP_KEYS if key in setup}
        self.thresholds = dict(setup["thresholds"])
        self.controls = []
        self._regime = REGIMES[setup["noise_regime"]]
        self._rng = np.random.default_rng(setup["seed"])
        start = scenario_start(setup, self._rng, self._regime)
        envelope = Envelope(**setup["envelope"])
        position, velocity, latest_control = (
            start["position"],
            start["velocity"],
            _latest(lambda: self.controls[-1]),
        )
        self._flight = fly_steps(envelope, position, velocity, 0.0, latest_control, self._regime.dt)
        self._truth, self._z = None, None
        truth, z = _latest(lambda: self._truth), _latest(lambda: self._z)
        R, dt = NOMINAL.R(), NOMINAL.dt
        snapshots = self._snapshots = defaultdict(dict)
        self._filters = {
            "EKF": ekf_like_steps(EKF(dt, R), truth, z, snapshots["EKF"]),
            "UKF": ekf_like_steps(UKF(dt, R), truth, z, snapshots["UKF"]),
            "IMM": imm_steps(IMM(dt, R, **imm_kwargs_from_tuning(imm_best)), truth, z, snapshots["IMM"]),
        }
        weights = {
            name: weights_root / path
            for name, path in LEARNED_WEIGHTS.items()
            if (weights_root / path).exists()
        }
        if "KalmanNet" in weights:
            knet = knet_steps(load_weights(weights["KalmanNet"]), dt, R, truth, z)
            self._filters["KalmanNet"] = ((err,) for err in knet)
        if "RKN" in weights:
            self._filters["RKN"] = rkn_steps(load_weights(weights["RKN"]), dt, truth, z, snapshots["RKN"])
        self._scores = {name: [] for name in self._filters}

    def advance(self, control):
        self.controls.append([float(value) for value in control])
        self._truth = next(self._flight)
        self._z = measure(self._rng, self._truth[None], self._regime)[0]
        for name, steps in self._filters.items():
            self._scores[name].append(next(steps))

    def trips(self):
        return {name: self._trips(*zip(*scores)) for name, scores in self._scores.items()}

    def _trips(self, err, *nees_and_nis):
        return first_trips(_position_errors_m(err), *nees_and_nis, thresholds=self.thresholds)

    def readouts(self):
        return {
            name: {
                **_readout(self._truth, *scores[-1], snapshot=self._snapshots[name]),
                "diverged": diverged(
                    _position_errors_m([err for err, *_ in scores]), self.thresholds["raw_error_m"]
                ),
            }
            for name, scores in self._scores.items()
        }

    def frame(self):
        trips = self.trips()
        flags = {name: {flag: shown(name, found, flag) for flag in FLAGS} for name, found in trips.items()}
        readouts = self.readouts()
        return {
            "step": len(self.controls),
            "truth_m": self._truth[POS].tolist(),
            "truth_velocity_m_s": self._truth[VEL].tolist(),
            "filters": readouts,
            "framing_radius_m": framing_radius_m(readouts),
            "trips": trips,
            "flags": {
                name: flags[name] if name in flags else _unavailable_flags(name) for name in FILTER_NAMES
            },
        }

    def scenario_json(self, git_tag=UNKNOWN_GIT_TAG):
        scenario = {
            **self.setup,
            "thresholds": self.thresholds,
            "git_tag": git_tag,
            "controls": self.controls,
        }
        return json.dumps({**scenario, "tripped": self.trips()}) + "\n"


def trace(scenario, imm_best, weights_root):
    flight = LiveFlight(scenario, imm_best, weights_root)
    steps = []
    for control in scenario["controls"]:
        flight.advance(control)
        steps.append(flight.readouts())
    return steps
