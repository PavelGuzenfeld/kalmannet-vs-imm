import math
from collections.abc import Callable, Iterator

import numpy as np

from data.fixed_wing import LOAD_FACTOR, NEUTRAL_ROW, ROLL_RATE, RUDDER, THROTTLE, Envelope

MANOEUVRES = ("break turn", "reversal", "split-S", "jink", "weave", "barrel roll")

BREAK_TURN_DEGREES = 90.0
JINK_HALF_PERIOD_S = 1.0
JINK_HALF_PERIODS = 4
WEAVE_BANK_RAD = np.pi / 3.0
WEAVE_HOLD_S = 3.0
BARREL_ROLL_RATE_FRACTION = 0.5
BARREL_ROLL_LOAD_FACTOR = 2.0

Row = tuple[float, ...]


def _row(roll_rate=0.0, load_factor=1.0, rudder=0.0) -> Row:
    row = list(NEUTRAL_ROW)
    row[ROLL_RATE], row[LOAD_FACTOR], row[THROTTLE], row[RUDDER] = roll_rate, load_factor, 1.0, rudder
    return tuple(row)


class _Pilot:
    """Tracks the bank it has commanded and the flight-path angle turned since entry, from the latest velocity."""

    def __init__(self, envelope: Envelope, dt_s: float, entry_velocity, read_velocity: Callable):
        self.envelope, self.dt_s, self.read_velocity = envelope, dt_s, read_velocity
        self.bank_rad = 0.0
        self.turned_rad = 0.0
        self._velocity = np.asarray(entry_velocity, dtype=np.float64)

    def _observe(self):
        velocity = np.asarray(self.read_velocity(), dtype=np.float64)
        sine = np.linalg.norm(np.cross(self._velocity, velocity))
        self.turned_rad += float(np.arctan2(sine, self._velocity @ velocity))
        self._velocity = velocity

    def _fly(self, row: Row) -> Iterator[Row]:
        yield row
        self._observe()

    def roll_to(self, bank_rad, load_factor=1.0, rate_fraction=1.0) -> Iterator[Row]:
        max_step_rad = self.envelope.max_roll_rate_rad_s * rate_fraction * self.dt_s
        roll_rad = bank_rad - self.bank_rad
        steps = math.ceil(abs(roll_rad) / max_step_rad)
        self.bank_rad = bank_rad
        for _ in range(steps):
            yield from self._fly(_row(roll_rate=roll_rad / (steps * self.dt_s), load_factor=load_factor))

    def pull_until_turned(self, target_rad) -> Iterator[Row]:
        while self.turned_rad < target_rad:
            yield from self._fly(_row(load_factor=self.envelope.max_load_factor))

    def level_off(self) -> Iterator[Row]:
        while self._velocity[2] < 0.0:
            yield from self._fly(_row(load_factor=self.envelope.max_load_factor))

    def hold(self, duration_s, load_factor=1.0, rudder=0.0) -> Iterator[Row]:
        for _ in range(round(duration_s / self.dt_s)):
            yield from self._fly(_row(load_factor=load_factor, rudder=rudder))


def _break_turn(pilot: _Pilot, degrees: float) -> Iterator[Row]:
    level_bank_rad = float(np.arccos(1.0 / pilot.envelope.max_load_factor))
    yield from pilot.roll_to(level_bank_rad)
    yield from pilot.pull_until_turned(np.radians(degrees))
    yield from pilot.roll_to(0.0)


def _split_s(pilot: _Pilot) -> Iterator[Row]:
    yield from pilot.roll_to(np.pi)
    yield from pilot.pull_until_turned(np.pi)


def _reversal(pilot: _Pilot) -> Iterator[Row]:
    yield from pilot.pull_until_turned(np.pi)
    yield from pilot.roll_to(np.pi)
    yield from pilot.level_off()


def _jink(pilot: _Pilot) -> Iterator[Row]:
    for half_period in range(JINK_HALF_PERIODS):
        side_load_factor = pilot.envelope.max_side_load_factor
        rudder = (side_load_factor, -side_load_factor)[half_period % 2]
        yield from pilot.hold(JINK_HALF_PERIOD_S, rudder=rudder)


def _weave(pilot: _Pilot) -> Iterator[Row]:
    load_factor = 1.0 / float(np.cos(WEAVE_BANK_RAD))
    for swing, hold_s in enumerate((WEAVE_HOLD_S / 2, WEAVE_HOLD_S, WEAVE_HOLD_S, WEAVE_HOLD_S / 2)):
        yield from pilot.roll_to((WEAVE_BANK_RAD, -WEAVE_BANK_RAD)[swing % 2], load_factor=load_factor)
        yield from pilot.hold(hold_s, load_factor=load_factor)
    yield from pilot.roll_to(0.0)


def _barrel_roll(pilot: _Pilot) -> Iterator[Row]:
    yield from pilot.roll_to(
        2.0 * np.pi, load_factor=BARREL_ROLL_LOAD_FACTOR, rate_fraction=BARREL_ROLL_RATE_FRACTION
    )
    yield from pilot.level_off()


def manoeuvre_rows(name: str, envelope: Envelope, dt_s: float, entry_velocity, read_velocity: Callable) -> Iterator[Row]:
    """Control rows for one named manoeuvre, flown to the right of the entry heading. Segment lengths come from
    the latest velocity, so read_velocity must return the newest flown state's velocity."""
    pilot = _Pilot(envelope, dt_s, entry_velocity, read_velocity)
    flown = {
        "break turn": lambda: _break_turn(pilot, BREAK_TURN_DEGREES),
        "reversal": lambda: _reversal(pilot),
        "split-S": lambda: _split_s(pilot),
        "jink": lambda: _jink(pilot),
        "weave": lambda: _weave(pilot),
        "barrel roll": lambda: _barrel_roll(pilot),
    }
    return flown[name]()
