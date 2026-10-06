"""data/fixed_wing.py's flight as of 4c68643, before the wider control row (#107), frozen as the bit-for-bit
reference for 3-column scenarios; a stored trajectory cannot match exactly across runners (#55)."""

import numpy as np

STANDARD_GRAVITY_M_S2 = 9.80665
MAX_LIFT_TO_DRAG = 10.0
UP = np.array([0.0, 0.0, 1.0])
NORTH = np.array([0.0, 1.0, 0.0])


def _drag_m_s2(envelope, speed_m_s, load_factor):
    speed_ratio = speed_m_s / envelope.best_glide_speed_m_s
    return STANDARD_GRAVITY_M_S2 / (2.0 * MAX_LIFT_TO_DRAG) * (speed_ratio**2 + (load_factor / speed_ratio) ** 2)


def _perpendicular_unit(reference, unit_velocity):
    perpendicular = reference - (reference @ unit_velocity) * unit_velocity
    return perpendicular / np.linalg.norm(perpendicular)


def _initial_wings_level_up(unit_velocity):
    reference = UP if np.linalg.norm(np.cross(UP, unit_velocity)) > 0.0 else NORTH
    return _perpendicular_unit(reference, unit_velocity)


def _lift_direction(unit_velocity, wings_level_up, bank_rad):
    right_wing = np.cross(unit_velocity, wings_level_up)
    return np.cos(bank_rad) * wings_level_up + np.sin(bank_rad) * right_wing


def fly_steps(envelope, position_m, velocity_m_s, bank_rad, controls, dt_s):
    max_thrust_m_s2 = _drag_m_s2(envelope, envelope.max_speed_m_s, 1.0)
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    wings_level_up = _initial_wings_level_up(velocity / np.linalg.norm(velocity))
    for roll_rate, load_factor_command, throttle in controls:
        max_roll_rate = envelope.max_roll_rate_rad_s
        bank_rad = bank_rad + np.clip(roll_rate, -max_roll_rate, max_roll_rate) * dt_s
        speed = np.linalg.norm(velocity)
        unit_velocity = velocity / speed
        wings_level_up = _perpendicular_unit(wings_level_up, unit_velocity)
        available_load_factor = min(envelope.max_load_factor, (speed / envelope.min_speed_m_s) ** 2)
        load_factor = np.clip(load_factor_command, -available_load_factor, available_load_factor)
        tangential = np.clip(throttle, 0.0, 1.0) * max_thrust_m_s2 - _drag_m_s2(envelope, speed, load_factor)
        acceleration = (
            tangential * unit_velocity
            + load_factor * STANDARD_GRAVITY_M_S2 * _lift_direction(unit_velocity, wings_level_up, bank_rad)
            - STANDARD_GRAVITY_M_S2 * UP
        )
        next_velocity = velocity + acceleration * dt_s
        next_speed = np.linalg.norm(next_velocity)
        next_velocity *= max(envelope.min_speed_m_s, next_speed) / next_speed
        acceleration = (next_velocity - velocity) / dt_s
        yield np.concatenate((position, velocity, acceleration))
        position = position + velocity * dt_s + 0.5 * acceleration * dt_s**2
        velocity = next_velocity


def fly(envelope, position_m, velocity_m_s, bank_rad, controls, dt_s):
    states = list(fly_steps(envelope, position_m, velocity_m_s, bank_rad, controls, dt_s))
    return np.array(states).reshape(len(states), 9)
