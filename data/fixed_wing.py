"""Point-mass fixed-wing truth model. ENU frame, z up, no wind.

Independent of filters/ so the truth it flies is never the filters' own motion model (MS-27).
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from data.config import NOMINAL

STANDARD_GRAVITY_M_S2 = 9.80665
MAX_LIFT_TO_DRAG = 10.0
UP = np.array([0.0, 0.0, 1.0])
NORTH = np.array([0.0, 1.0, 0.0])

ROLL_RATE, LOAD_FACTOR, THROTTLE, RUDDER, SPEED_BRAKE, AFTERBURNER = range(6)
NEUTRAL_ROW = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)


def full_control_row(control: Sequence[float]) -> tuple[float, ...]:
    """A 3-column row, as recorded before #107, flies with every new input neutral."""
    return (*control, *NEUTRAL_ROW[len(control) :])


@dataclass(frozen=True)
class Envelope:
    """min_speed is stall (1 g lift); max_speed is the level full-throttle equilibrium.
    max_negative_load_factor is signed; None, as recorded before #107, pushes to -max_load_factor."""

    min_speed_m_s: float
    max_speed_m_s: float
    max_load_factor: float
    max_roll_rate_rad_s: float
    max_side_load_factor: float = 0.0
    max_negative_load_factor: float | None = None
    brake_drag_multiplier: float = 1.0
    afterburner_thrust_multiplier: float = 1.0

    @property
    def best_glide_speed_m_s(self) -> float:
        """Geometric mean of the speed range: a simplification until per-airframe polars are needed."""
        return float(np.sqrt(self.min_speed_m_s * self.max_speed_m_s))


DEFAULT_PRESET = "benchmark"
PRESETS = {
    "benchmark": Envelope(NOMINAL.speed0[0], NOMINAL.speed0[1], 9.0, float(np.deg2rad(180.0)), 2.0, -4.5, 3.0, 1.5),
    "small aerial object": Envelope(12.0, 45.0, 4.0, float(np.deg2rad(120.0)), 0.5, -2.0, 2.0, 1.0),
    "light aircraft": Envelope(25.0, 70.0, 3.8, float(np.deg2rad(60.0)), 0.3, -1.5, 2.0, 1.0),
    "jet": Envelope(70.0, 300.0, 9.0, float(np.deg2rad(240.0)), 1.0, -3.0, 3.0, 1.5),
}


def _drag_m_s2(envelope: Envelope, speed_m_s: float, load_factor: float) -> float:
    speed_ratio = speed_m_s / envelope.best_glide_speed_m_s
    return STANDARD_GRAVITY_M_S2 / (2.0 * MAX_LIFT_TO_DRAG) * (speed_ratio**2 + (load_factor / speed_ratio) ** 2)


def _perpendicular_unit(reference: np.ndarray, unit_velocity: np.ndarray) -> np.ndarray:
    perpendicular = reference - (reference @ unit_velocity) * unit_velocity
    return perpendicular / np.linalg.norm(perpendicular)


def _initial_wings_level_up(unit_velocity: np.ndarray) -> np.ndarray:
    """Straight up or down has no horizon, so wings level there means lift toward north."""
    reference = UP if np.linalg.norm(np.cross(UP, unit_velocity)) > 0.0 else NORTH
    return _perpendicular_unit(reference, unit_velocity)


def _lift_direction(unit_velocity: np.ndarray, wings_level_up: np.ndarray, bank_rad: float) -> np.ndarray:
    right_wing = np.cross(unit_velocity, wings_level_up)
    return np.cos(bank_rad) * wings_level_up + np.sin(bank_rad) * right_wing


def fly_steps(
    envelope: Envelope,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    bank_rad: float,
    controls: Iterable[Sequence[float]],
    dt_s: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yields (state row, unit lift direction) per control row as `controls` yields it; see fly."""
    max_thrust_m_s2 = _drag_m_s2(envelope, envelope.max_speed_m_s, 1.0)
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    wings_level_up = _initial_wings_level_up(velocity / np.linalg.norm(velocity))
    push_limit = (
        envelope.max_load_factor
        if envelope.max_negative_load_factor is None
        else -envelope.max_negative_load_factor
    )
    for control in controls:
        roll_rate, load_factor_command, throttle, rudder, speed_brake, afterburner = full_control_row(control)
        max_roll_rate = envelope.max_roll_rate_rad_s
        bank_rad = bank_rad + np.clip(roll_rate, -max_roll_rate, max_roll_rate) * dt_s
        speed = np.linalg.norm(velocity)
        unit_velocity = velocity / speed
        wings_level_up = _perpendicular_unit(wings_level_up, unit_velocity)
        lift_available = (speed / envelope.min_speed_m_s) ** 2
        available_load_factor = min(envelope.max_load_factor, lift_available)
        load_factor = np.clip(load_factor_command, -min(push_limit, lift_available), available_load_factor)
        side_limit = min(envelope.max_side_load_factor, lift_available)
        side_load_factor = np.clip(rudder, -side_limit, side_limit)
        throttle = np.clip(throttle, 0.0, 1.0)
        thrust_multiplier = envelope.afterburner_thrust_multiplier if afterburner and throttle == 1.0 else 1.0
        drag_multiplier = envelope.brake_drag_multiplier if speed_brake else 1.0
        drag = _drag_m_s2(envelope, speed, load_factor) * drag_multiplier
        tangential = throttle * max_thrust_m_s2 * thrust_multiplier - drag
        lift_direction = _lift_direction(unit_velocity, wings_level_up, bank_rad)
        acceleration = (
            tangential * unit_velocity
            + load_factor * STANDARD_GRAVITY_M_S2 * lift_direction
            + side_load_factor * STANDARD_GRAVITY_M_S2 * np.cross(unit_velocity, lift_direction)
            - STANDARD_GRAVITY_M_S2 * UP
        )
        next_velocity = velocity + acceleration * dt_s
        next_speed = np.linalg.norm(next_velocity)
        next_velocity *= max(envelope.min_speed_m_s, next_speed) / next_speed
        acceleration = (next_velocity - velocity) / dt_s
        yield np.concatenate((position, velocity, acceleration)), lift_direction
        position = position + velocity * dt_s + 0.5 * acceleration * dt_s**2
        velocity = next_velocity


def fly(
    envelope: Envelope,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    bank_rad: float,
    controls: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """controls[T, 3 or 6]: roll rate rad/s, load factor g, throttle [0, 1], rudder g to the right, brake, burner.
    bank_rad is from wings level at the start; wings level then turns with the flight path, not the horizon.
    Returns states[T, 9] = px py pz vx vy vz ax ay az; row t's acceleration carries it to row t + 1."""
    states = [state for state, _ in fly_steps(envelope, position_m, velocity_m_s, bank_rad, controls, dt_s)]
    return np.array(states).reshape(len(states), 9)
