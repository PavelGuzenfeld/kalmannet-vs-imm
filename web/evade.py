from collections.abc import Callable, Iterator

import numpy as np

from data.fixed_wing import NORTH, STANDARD_GRAVITY_M_S2, UP, Envelope, _initial_wings_level_up
from web.manoeuvres import Row

TURN_TIME_CONSTANT_S = 0.5
JINK_OFF_BEAM_RAD = float(np.deg2rad(5.0))
JINK_INTERVAL_S = (1.0, 3.0)
CLIMB_ANGLE_RAD = float(np.deg2rad(15.0))
CLIMB_INTERVAL_S = (2.0, 5.0)
FORCED_DIVE_SPEED_OVER_MIN_SPEED = 1.5


def beam_direction(position_m, velocity_m_s, jink, climb):
    """Unit velocity across the line of sight from the sensor at position_m's origin, on the side the target leans to.
    climb tilts it up or down within the beam plane; jink tilts it off the beam, toward or away from the sensor."""
    line_of_sight = position_m / np.linalg.norm(position_m)
    across = np.cross(UP, line_of_sight)
    if not np.linalg.norm(across):
        across = np.cross(NORTH, line_of_sight)
    across /= np.linalg.norm(across)
    side = 1.0 if velocity_m_s @ across >= 0.0 else -1.0
    over = np.cross(line_of_sight, across)
    direction = (
        side * np.cos(CLIMB_ANGLE_RAD) * across
        + climb * np.sin(CLIMB_ANGLE_RAD) * over
        + jink * np.tan(JINK_OFF_BEAM_RAD) * line_of_sight
    )
    return direction / np.linalg.norm(direction)


def steer_row(envelope: Envelope, dt_s, velocity_m_s, lift_direction, direction) -> Row:
    """Rolls the lift toward the pull that turns the velocity onto direction and holds off gravity, at full throttle.
    The lift is perpendicular to the velocity, so the pull's along-track part never reaches the row."""
    speed = np.linalg.norm(velocity_m_s)
    forward = velocity_m_s / speed
    pull_g = direction * speed / (TURN_TIME_CONSTANT_S * STANDARD_GRAVITY_M_S2) + UP
    roll_rad = np.arctan2(forward @ np.cross(lift_direction, pull_g), lift_direction @ pull_g)
    max_roll_rate = envelope.max_roll_rate_rad_s
    push_limit = (
        -envelope.max_load_factor
        if envelope.max_negative_load_factor is None
        else envelope.max_negative_load_factor
    )
    return (
        float(np.clip(roll_rad / dt_s, -max_roll_rate, max_roll_rate)),
        float(np.clip(pull_g @ lift_direction, push_limit, envelope.max_load_factor)),
        1.0,
        0.0,
        0.0,
        0.0,
    )


def alternating_signs(rng: np.random.Generator, interval_s, dt_s) -> Iterator[float]:
    """One sign per row: a random first sign, flipped after a uniformly drawn whole number of rows in interval_s."""
    sign = float(rng.choice((-1.0, 1.0)))
    low, high = (round(seconds / dt_s) for seconds in interval_s)
    while True:
        for _ in range(rng.integers(low, high, endpoint=True)):
            yield sign
        sign = -sign


def climb_within_speed_limits(envelope: Envelope, speed_m_s, climb):
    """Trades energy back into the speed range: climbs at or over max speed, dives when near stall."""
    if speed_m_s >= envelope.max_speed_m_s:
        return 1.0
    if speed_m_s <= FORCED_DIVE_SPEED_OVER_MIN_SPEED * envelope.min_speed_m_s:
        return -1.0
    return climb


def evade_rows(
    read_envelope: Callable[[], Envelope], dt_s: float, rng: np.random.Generator, read_truth: Callable
) -> Iterator[Row]:
    """Control rows that keep the velocity across the sensor's line of sight, jinking and climbing on rng's schedule.
    read_truth returns the target's own sensor-centred position, velocity, and lift direction (None before row 0)."""
    for jink, climb in zip(
        alternating_signs(rng, JINK_INTERVAL_S, dt_s), alternating_signs(rng, CLIMB_INTERVAL_S, dt_s)
    ):
        envelope = read_envelope()
        position, velocity, lift = read_truth()
        speed = np.linalg.norm(velocity)
        if lift is None:
            lift = _initial_wings_level_up(velocity / speed)
        climb = climb_within_speed_limits(envelope, speed, climb)
        yield steer_row(envelope, dt_s, velocity, lift, beam_direction(position, velocity, jink, climb))
