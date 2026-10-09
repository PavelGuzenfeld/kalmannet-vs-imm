import numpy as np

from data.fixed_wing import (
    AFTERBURNER,
    LOAD_FACTOR,
    ROLL_RATE,
    SPEED_BRAKE,
    STANDARD_GRAVITY_M_S2,
    THROTTLE,
    Envelope,
    drag_m_s2,
)

HEADING, ALTITUDE, SPEED, BANK = "heading", "altitude", "speed", "bank"
HOLD_AXES = {
    HEADING: (ROLL_RATE,),
    ALTITUDE: (LOAD_FACTOR,),
    SPEED: (THROTTLE, SPEED_BRAKE, AFTERBURNER),
    BANK: (ROLL_RATE,),
}
HEADING_TIME_CONSTANT_S = 2.0
MAX_HOLD_BANK_RAD = float(np.deg2rad(30.0))
ALTITUDE_NATURAL_FREQUENCY_RAD_S = 0.5
SPEED_TIME_CONSTANT_S = 2.0


def heading_rad(velocity_m_s):
    """Clockwise from north, so a positive bank turns it up."""
    return float(np.arctan2(velocity_m_s[0], velocity_m_s[1]))


def _wrapped_rad(angle_rad):
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


class Holds:
    """Overwrites the held axes of each full control row.
    bank_rad is tracked the way data/fixed_wing.fly_steps tracks it, which does not report it."""

    def __init__(self, envelope: Envelope, dt_s: float, bank_rad: float = 0.0, sensor_altitude_m: float = 0.0):
        self.envelope = envelope
        self.dt_s = dt_s
        self.bank_rad = bank_rad
        self.sensor_altitude_m = sensor_altitude_m
        self.targets = {}

    def toggle(self, name, position_m, velocity_m_s):
        if name in self.targets:
            del self.targets[name]
            return
        self.disengage_axes(HOLD_AXES[name])
        self.targets[name] = {
            HEADING: heading_rad(velocity_m_s),
            ALTITUDE: float(position_m[2]),
            SPEED: float(np.linalg.norm(velocity_m_s)),
            BANK: self.bank_rad,
        }[name]

    def disengage_axes(self, axes):
        self.targets = {
            name: target for name, target in self.targets.items() if not set(HOLD_AXES[name]) & set(axes)
        }

    def steer(self, control, position_m, velocity_m_s):
        row = [float(value) for value in control]
        if ALTITUDE in self.targets:
            row[LOAD_FACTOR] = self._load_factor(position_m, velocity_m_s)
        if HEADING in self.targets:
            row[ROLL_RATE] = self._roll_rate_to(self._heading_bank_rad(velocity_m_s))
        if BANK in self.targets:
            row[ROLL_RATE] = self._roll_rate_to(self.targets[BANK])
        if SPEED in self.targets:
            row[THROTTLE], row[SPEED_BRAKE], row[AFTERBURNER] = self._speed_controls(
                velocity_m_s, row[LOAD_FACTOR]
            )
        max_roll_rate = self.envelope.max_roll_rate_rad_s
        self.bank_rad = self.bank_rad + np.clip(row[ROLL_RATE], -max_roll_rate, max_roll_rate) * self.dt_s
        return row

    def _roll_rate_to(self, bank_rad):
        max_roll_rate = self.envelope.max_roll_rate_rad_s
        return float(np.clip((bank_rad - self.bank_rad) / self.dt_s, -max_roll_rate, max_roll_rate))

    def _heading_bank_rad(self, velocity_m_s):
        turn_rate = _wrapped_rad(self.targets[HEADING] - heading_rad(velocity_m_s)) / HEADING_TIME_CONSTANT_S
        bank = np.arctan(turn_rate * np.linalg.norm(velocity_m_s) / STANDARD_GRAVITY_M_S2)
        return float(np.clip(bank, -MAX_HOLD_BANK_RAD, MAX_HOLD_BANK_RAD))

    def _target_altitude_m(self):
        """Sensor-centred; the envelope's min_altitude_m is world altitude."""
        floor_m = self.envelope.min_altitude_m
        if floor_m is None:
            return self.targets[ALTITUDE]
        return max(self.targets[ALTITUDE], floor_m - self.sensor_altitude_m)

    def _load_factor(self, position_m, velocity_m_s):
        envelope = self.envelope
        omega = ALTITUDE_NATURAL_FREQUENCY_RAD_S
        altitude_error_m = self._target_altitude_m() - position_m[2]
        climb_acceleration = omega**2 * altitude_error_m - 2.0 * omega * velocity_m_s[2]
        flight_path_rad = np.arcsin(velocity_m_s[2] / np.linalg.norm(velocity_m_s))
        lift_up = max(np.cos(self.bank_rad) * np.cos(flight_path_rad), 1.0 / envelope.max_load_factor)
        load_factor = (1.0 + climb_acceleration / STANDARD_GRAVITY_M_S2) / lift_up
        push_limit = (
            -envelope.max_load_factor
            if envelope.max_negative_load_factor is None
            else envelope.max_negative_load_factor
        )
        return float(np.clip(load_factor, push_limit, envelope.max_load_factor))

    def _speed_controls(self, velocity_m_s, load_factor):
        """Throttle, brake and afterburner for the thrust the speed error asks for; brake and burner are on/off."""
        speed = np.linalg.norm(velocity_m_s)
        max_thrust = drag_m_s2(self.envelope, self.envelope.max_speed_m_s, 1.0)
        along_track = (self.targets[SPEED] - speed) / SPEED_TIME_CONSTANT_S
        climb = STANDARD_GRAVITY_M_S2 * velocity_m_s[2] / speed
        thrust = along_track + climb + drag_m_s2(self.envelope, speed, load_factor)
        throttle = float(np.clip(thrust / max_thrust, 0.0, 1.0))
        return throttle, float(thrust < 0.0), float(thrust > max_thrust)
