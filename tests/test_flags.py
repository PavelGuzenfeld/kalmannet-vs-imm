import numpy as np
import pytest
from scipy.stats import chi2

from filters.flags import (
    DEFAULT_THRESHOLDS,
    ELLIPSOID_95_DOF3,
    INNOVATION_GATE_NIS_99_DOF3,
    NEES_LOWER_025_DOF9,
    NEES_UPPER_975_DOF9,
    OVERCONFIDENT_STEPS,
    RAW_ERROR_M,
    counted_steps,
    divergence_step,
    first_overconfidence_step,
    first_raw_error_step,
    first_track_loss_step,
    first_trips,
    shown,
    until_diverged,
)

INSIDE_NEES = 9.0
OUTSIDE_NEES = np.nextafter(NEES_UPPER_975_DOF9, np.inf)


def nees_run(*segments):
    """segments: (count, value) pairs laid end to end."""
    return np.concatenate([np.full(count, value) for count, value in segments])


def test_pinned_quantiles_are_the_scipy_values_for_the_pinned_scipy():
    """Exact equality: the literals are scipy 1.14.1's own output, the version requirements.txt pins."""
    assert INNOVATION_GATE_NIS_99_DOF3 == chi2.ppf(0.99, df=3)
    assert NEES_UPPER_975_DOF9 == chi2.ppf(0.975, df=9)


def test_ticket_defaults_are_the_recorded_thresholds():
    assert DEFAULT_THRESHOLDS == {
        "innovation_gate_nis": INNOVATION_GATE_NIS_99_DOF3,
        "nees_upper": NEES_UPPER_975_DOF9,
        "overconfident_steps": 25,
        "raw_error_m": 20.0,
    }
    assert (OVERCONFIDENT_STEPS, RAW_ERROR_M) == (25, 20.0)


def test_nis_exactly_at_the_99_percent_gate_is_not_track_loss():
    assert first_track_loss_step(np.full(5, INNOVATION_GATE_NIS_99_DOF3)) is None


def test_nis_just_above_the_99_percent_gate_is_track_loss_at_that_step():
    nis = np.array([1.0, INNOVATION_GATE_NIS_99_DOF3, np.nextafter(INNOVATION_GATE_NIS_99_DOF3, np.inf), 1.0])
    assert first_track_loss_step(nis) == 2


def test_track_loss_gate_is_the_one_passed_in():
    assert first_track_loss_step(np.array([4.0, 5.0, 6.0]), gate=5.0) == 2


def test_24_consecutive_nees_steps_above_the_bound_are_not_overconfidence():
    assert first_overconfidence_step(nees_run((3, INSIDE_NEES), (24, OUTSIDE_NEES), (3, INSIDE_NEES))) is None


def test_25th_consecutive_nees_step_above_the_bound_trips_overconfidence_at_that_step():
    assert first_overconfidence_step(nees_run((3, INSIDE_NEES), (25, OUTSIDE_NEES))) == 27


def test_nees_exactly_at_the_bound_breaks_the_overconfident_run():
    nees = nees_run((24, OUTSIDE_NEES), (1, NEES_UPPER_975_DOF9), (24, OUTSIDE_NEES))
    assert first_overconfidence_step(nees) is None


def test_nees_far_below_the_band_is_not_overconfidence():
    assert first_overconfidence_step(np.zeros(40)) is None


def test_overconfidence_run_length_and_bound_are_the_ones_passed_in():
    assert first_overconfidence_step(np.array([1.0, 3.0, 3.0, 3.0]), upper=2.0, steps=2) == 2


def test_position_error_of_exactly_20_m_is_not_raw_error():
    assert first_raw_error_step(np.full(5, 20.0)) is None


def test_position_error_just_above_20_m_is_raw_error_at_that_step():
    assert first_raw_error_step(np.array([0.0, 20.0, np.nextafter(20.0, np.inf)])) == 2


def test_raw_error_limit_is_the_one_passed_in():
    assert first_raw_error_step(np.array([1.0, 2.0, 3.0]), limit_m=2.0) == 2


def test_a_diverged_nan_estimate_trips_every_flag():
    nan_run = np.full(OVERCONFIDENT_STEPS, np.nan)
    assert first_track_loss_step(nan_run) == 0
    assert first_overconfidence_step(nan_run) == OVERCONFIDENT_STEPS - 1
    assert first_raw_error_step(nan_run) == 0


def test_filter_without_covariance_is_flagged_on_raw_error_only():
    assert first_trips(np.array([0.0, 25.0])) == {"raw_error": 1}


def test_first_trips_reads_each_threshold_from_the_recorded_set():
    thresholds = {"innovation_gate_nis": 2.0, "nees_upper": 3.0, "overconfident_steps": 2, "raw_error_m": 1.0}
    error_m = np.array([0.0, 0.5, 1.5, 0.0])
    nees = np.array([4.0, 0.0, 4.0, 4.0])
    nis = np.array([0.0, 0.0, 0.0, 2.5])

    trips = first_trips(error_m, nees=nees, nis=nis, thresholds=thresholds)

    assert trips == {"track_loss": 3, "overconfidence": 3, "raw_error": 2}


def test_shown_reads_a_flag_that_never_tripped_as_a_dash_and_a_tripped_one_as_its_step():
    trips = {"track_loss": None, "raw_error": 0}

    assert [shown("EKF", trips, flag) for flag in ("track_loss", "raw_error")] == ["-", "0"]


def test_band_and_ellipsoid_quantiles_are_the_scipy_values_for_the_pinned_scipy():
    assert NEES_LOWER_025_DOF9 == chi2.ppf(0.025, df=9)
    assert ELLIPSOID_95_DOF3 == chi2.ppf(0.95, df=3)


def test_position_error_of_exactly_200_m_has_not_diverged():
    assert divergence_step(np.full(5, 200.0)) is None


def test_position_error_just_above_200_m_diverges_at_that_step():
    assert divergence_step(np.array([0.0, 200.0, np.nextafter(200.0, np.inf), 1.0])) == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_position_error_diverges_at_that_step(bad):
    assert divergence_step(np.array([1.0, bad, 3.0])) == 1


def test_counted_steps_are_the_ones_before_the_divergence_step_or_all_of_them():
    assert (counted_steps(np.array([1.0, 2.0, 250.0, 1.0])), counted_steps(np.array([1.0, 2.0]))) == (2, 2)


def scored(position_error_m):
    err = np.zeros(9)
    err[0] = position_error_m
    return (err,)


def test_a_filter_is_pulled_through_its_divergence_step_and_no_further():
    pulled = []

    def steps():
        for position_error_m in (1.0, 200.0, 201.0, 1.0):
            pulled.append(position_error_m)
            yield scored(position_error_m)

    yielded = [err[0] for err, in until_diverged(steps())]

    assert yielded == pulled == [1.0, 200.0, 201.0]


def test_a_filter_whose_error_stays_at_200_m_is_pulled_to_the_end():
    assert len(list(until_diverged(scored(200.0) for _ in range(4)))) == 4


def test_a_filter_off_only_in_altitude_is_pulled_no_further_once_past_200_m():
    err = np.zeros(9)
    err[2] = 201.0

    assert len(list(until_diverged((err,) for _ in range(3)))) == 1
