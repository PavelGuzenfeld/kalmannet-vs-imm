"""Piecewise-switching ground-truth trajectory generator (CV/CT/CA segments)
and the az/el/range measurement model applied on top, per the brief's
maneuvering-target data spec. Segment durations, motion kind, and per-segment
parameters (turn rate, Singer accel/tau) are all randomized so the test set
contains maneuvers no filter was tuned to a single fixed instance of.
"""

import numpy as np

from data.config import REGIMES
from data.fixed_wing import Envelope, fly
from filters.measurement_model import h, wrap_angle
from filters.motion_models import MOTION_LABELS, ca_model, ct_model, cv_model, step
from filters.state import POS, PX, PY, PZ, STATE_DIM, VEL, VX, VY, VZ


START_SPEC_KEYS = frozenset({"range_m", "bearing_deg", "altitude_m", "heading_deg", "speed_m_s"})


def _sample_initial_state(rng, regime, speed_range=None):
    r0 = rng.uniform(*regime.range0)
    az0 = rng.uniform(-np.pi, np.pi)
    el0 = rng.uniform(np.deg2rad(-10), np.deg2rad(60))
    px = r0 * np.cos(el0) * np.cos(az0)
    py = r0 * np.cos(el0) * np.sin(az0)
    pz = r0 * np.sin(el0)

    speed = rng.uniform(*(speed_range or regime.speed0))
    heading = rng.uniform(-np.pi, np.pi)
    climb = rng.uniform(np.deg2rad(-10), np.deg2rad(10))
    vx = speed * np.cos(climb) * np.cos(heading)
    vy = speed * np.cos(climb) * np.sin(heading)
    vz = speed * np.sin(climb)

    x0 = np.zeros(STATE_DIM)
    x0[PX], x0[PY], x0[PZ] = px, py, pz
    x0[VX], x0[VY], x0[VZ] = vx, vy, vz
    return x0


def _sample_segments(rng, T, regime):
    """List of (length, kind, params) covering all T steps."""
    segments = []
    remaining = T
    while remaining > 0:
        seg_len = int(rng.integers(regime.seg_len_range[0], regime.seg_len_range[1]))
        seg_len = min(seg_len, remaining)
        kind = rng.choice(["cv", "ct", "ca"], p=[0.4, 0.35, 0.25])
        if kind == "ct":
            mag = np.deg2rad(rng.uniform(*regime.ct_omega_deg_s_range))
            sign = rng.choice([-1.0, 1.0])
            params = {"omega": float(sign * mag)}
        elif kind == "ca":
            params = {"sigma_a": regime.ca_sigma_a, "tau": regime.ca_tau}
        else:
            params = {}
        segments.append((seg_len, kind, params))
        remaining -= seg_len
    return segments


def _fq_for(kind, params, dt, regime):
    if kind == "cv":
        return cv_model(dt, sigma_a=regime.cv_sigma_a)
    if kind == "ca":
        return ca_model(dt, sigma_a=params["sigma_a"], tau=params["tau"])
    if kind == "ct":
        return ct_model(dt, omega=params["omega"])
    raise ValueError(kind)


def generate_ground_truth(rng, regime):
    """Returns (states[T, 9] float64, labels[T] int8 in MOTION_LABELS)."""
    T, dt = regime.T, regime.dt
    x = _sample_initial_state(rng, regime)
    states = np.zeros((T, STATE_DIM))
    labels = np.zeros(T, dtype=np.int8)

    t = 0
    for seg_len, kind, params in _sample_segments(rng, T, regime):
        F, Q = _fq_for(kind, params, dt, regime)
        L = np.linalg.cholesky(Q + 1e-12 * np.eye(STATE_DIM))
        label = MOTION_LABELS[kind]
        for _ in range(seg_len):
            if t >= T:
                break
            states[t] = x
            labels[t] = label
            w = L @ rng.standard_normal(STATE_DIM)
            x = step(x, F) + w
            t += 1
    return states, labels


def measure(rng, states, regime):
    """Apply h() plus sensor noise to a ground-truth trajectory. Azimuth is
    re-wrapped after adding noise so a true target sitting near +-pi never
    leaks a measurement outside [-pi, pi)."""
    T = states.shape[0]
    z = np.zeros((T, 3))
    scale = np.array([regime.sigma_az_rad, regime.sigma_el_rad, regime.sigma_range_m])
    for t in range(T):
        z_true = h(states[t])
        if regime.heavy_tailed:
            noise = rng.standard_t(regime.heavy_tailed_dof, size=3) * scale
        else:
            noise = rng.normal(size=3) * scale
        z[t] = z_true + noise
        z[t, 0] = wrap_angle(z[t, 0])
    return z


def _start_speed_range(envelope, regime):
    """regime.speed0 clipped to the envelope (MS-19); a disjoint pair falls back to the whole envelope."""
    low = max(regime.speed0[0], envelope["min_speed_m_s"])
    high = min(regime.speed0[1], envelope["max_speed_m_s"])
    return (low, high) if low <= high else (envelope["min_speed_m_s"], envelope["max_speed_m_s"])


def _refuse_start_spec(spec, envelope):
    unknown = spec.keys() - START_SPEC_KEYS
    if unknown:
        raise ValueError(f"unknown start fields {sorted(unknown)}; the fields are {sorted(START_SPEC_KEYS)}")
    if spec.get("range_m", 0.0) < 0.0:
        raise ValueError(f"range {spec['range_m']} m is below 0 m")
    if spec.get("altitude_m", 0.0) < 0.0:
        raise ValueError(f"altitude {spec['altitude_m']} m is below the z = 0 plane")
    low, high = envelope["min_speed_m_s"], envelope["max_speed_m_s"]
    if not low <= spec.get("speed_m_s", low) <= high:
        raise ValueError(f"speed {spec['speed_m_s']} m/s is outside the envelope's {low} to {high} m/s")


def _with_start_spec(drawn, spec, envelope, sensor):
    """drawn is sensor-centred, the start is not: range is horizontal from the sensor, altitude is above z = 0.
    Bearing and heading are degrees clockwise from north; a set heading or speed flies level."""
    _refuse_start_spec(spec, envelope)
    position, velocity = drawn[POS] + sensor, drawn[VEL]
    if spec.keys() & {"range_m", "bearing_deg", "altitude_m"}:
        ground_range = spec.get("range_m", np.hypot(drawn[PX], drawn[PY]))
        bearing = (
            np.deg2rad(spec["bearing_deg"]) if "bearing_deg" in spec else np.arctan2(drawn[PX], drawn[PY])
        )
        position = np.array(
            [
                sensor[0] + ground_range * np.sin(bearing),
                sensor[1] + ground_range * np.cos(bearing),
                spec.get("altitude_m", position[2]),
            ]
        )
    if spec.keys() & {"heading_deg", "speed_m_s"}:
        speed = spec.get("speed_m_s", np.linalg.norm(velocity))
        heading = (
            np.deg2rad(spec["heading_deg"]) if "heading_deg" in spec else np.arctan2(velocity[0], velocity[1])
        )
        velocity = np.array([speed * np.sin(heading), speed * np.cos(heading), 0.0])
    return {"position": position, "velocity": velocity}


def sensor_position(scenario):
    """East, north, up in m of the fixed sensor; the origin when the scenario has no "sensor"."""
    return np.asarray(scenario.get("sensor", {}).get("position", (0.0, 0.0, 0.0)), dtype=np.float64)


def scenario_start(scenario, rng, regime):
    """The scenario's recorded "start", or else one drawn from rng around the sensor, before any measurement noise.
    A partial "start_spec" fills its unset fields from that same draw."""
    start = scenario.get("start")
    if start is None:
        x0 = _sample_initial_state(rng, regime, _start_speed_range(scenario["envelope"], regime))
        start = _with_start_spec(
            x0, scenario.get("start_spec", {}), scenario["envelope"], sensor_position(scenario)
        )
    return start


def sensor_centred(start, sensor):
    """The one place a flight leaves the world frame: truth, measurements and every filter run from here on."""
    return np.asarray(start["position"], dtype=np.float64) - sensor, start["velocity"]


def fly_scenario(scenario):
    """Sensor-centred (states[T, 9], measurements[T, 3]) of a data/scenarios/ flight: its "start" or else a seeded
    one, wings level."""
    regime = REGIMES[scenario["noise_regime"]]
    rng = np.random.default_rng(scenario["seed"])
    position, velocity = sensor_centred(scenario_start(scenario, rng, regime), sensor_position(scenario))
    controls = np.asarray(scenario["controls"], dtype=np.float64)
    envelope_changes = {step: Envelope(**fields) for step, fields in scenario.get("envelope_changes", [])}
    envelope = Envelope(**scenario["envelope"])
    states = fly(envelope, position, velocity, 0.0, controls, regime.dt, envelope_changes)
    return states, measure(rng, states, regime)
