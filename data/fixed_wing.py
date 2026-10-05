"""Point-mass fixed-wing truth model. ENU frame, z up, no wind.

Independent of filters/ so the truth it flies is never the filters' own motion model (MS-27).
"""

from dataclasses import dataclass

import numpy as np

from data.config import NOMINAL

STANDARD_GRAVITY_M_S2 = 9.80665
MAX_LIFT_TO_DRAG = 10.0
UP = np.array([0.0, 0.0, 1.0])

ROLL_RATE, LOAD_FACTOR, THROTTLE = 0, 1, 2


@dataclass(frozen=True)
class Envelope:
    """min_speed is stall (1 g lift); max_speed is the level full-throttle equilibrium."""

    min_speed_m_s: float
    max_speed_m_s: float
    max_load_factor: float
    max_roll_rate_rad_s: float

    @property
    def best_glide_speed_m_s(self) -> float:
        """Geometric mean of the speed range: a simplification until per-airframe polars are needed."""
        return float(np.sqrt(self.min_speed_m_s * self.max_speed_m_s))


DEFAULT_PRESET = "benchmark"
PRESETS = {
    "benchmark": Envelope(NOMINAL.speed0[0], NOMINAL.speed0[1], 9.0, float(np.deg2rad(180.0))),
    "small aerial object": Envelope(12.0, 45.0, 4.0, float(np.deg2rad(120.0))),
    "light aircraft": Envelope(25.0, 70.0, 3.8, float(np.deg2rad(60.0))),
    "jet": Envelope(70.0, 300.0, 9.0, float(np.deg2rad(240.0))),
}


def _drag_m_s2(envelope: Envelope, speed_m_s: float, load_factor: float) -> float:
    speed_ratio = speed_m_s / envelope.best_glide_speed_m_s
    return STANDARD_GRAVITY_M_S2 / (2.0 * MAX_LIFT_TO_DRAG) * (speed_ratio**2 + (load_factor / speed_ratio) ** 2)


def _lift_direction(unit_velocity: np.ndarray, bank_rad: float) -> np.ndarray:
    wings_level_up = UP - unit_velocity[2] * unit_velocity
    wings_level_up /= np.linalg.norm(wings_level_up)
    right_wing = np.cross(unit_velocity, wings_level_up)
    return np.cos(bank_rad) * wings_level_up + np.sin(bank_rad) * right_wing


def fly(
    envelope: Envelope,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    bank_rad: float,
    controls: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """controls[T, 3]: roll rate rad/s, load factor g, throttle in [0, 1]. Flight path must stay off vertical.

    Returns states[T, 9] = px py pz vx vy vz ax ay az; row t's acceleration carries it to row t + 1.
    """
    max_thrust_m_s2 = _drag_m_s2(envelope, envelope.max_speed_m_s, 1.0)
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    states = np.zeros((len(controls), 9))
    for t, (roll_rate, load_factor_command, throttle) in enumerate(controls):
        max_roll_rate = envelope.max_roll_rate_rad_s
        bank_rad = bank_rad + np.clip(roll_rate, -max_roll_rate, max_roll_rate) * dt_s
        speed = np.linalg.norm(velocity)
        unit_velocity = velocity / speed
        available_load_factor = min(envelope.max_load_factor, (speed / envelope.min_speed_m_s) ** 2)
        load_factor = np.clip(load_factor_command, -available_load_factor, available_load_factor)
        tangential = throttle * max_thrust_m_s2 - _drag_m_s2(envelope, speed, load_factor)
        acceleration = (
            tangential * unit_velocity
            + load_factor * STANDARD_GRAVITY_M_S2 * _lift_direction(unit_velocity, bank_rad)
            - STANDARD_GRAVITY_M_S2 * UP
        )
        next_velocity = velocity + acceleration * dt_s
        next_speed = np.linalg.norm(next_velocity)
        next_velocity *= max(envelope.min_speed_m_s, next_speed) / next_speed
        acceleration = (next_velocity - velocity) / dt_s
        states[t] = np.concatenate((position, velocity, acceleration))
        position = position + velocity * dt_s + 0.5 * acceleration * dt_s**2
        velocity = next_velocity
    return states
