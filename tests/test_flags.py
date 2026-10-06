import numpy as np
from scipy.stats import chi2

from filters.flags import (
    DEFAULT_THRESHOLDS,
    INNOVATION_GATE_NIS_99_DOF3,
    NEES_UPPER_975_DOF9,
    OVERCONFIDENT_STEPS,
    RAW_ERROR_M,
    first_overconfidence_step,
    first_raw_error_step,
    first_track_loss_step,
    first_trips,
    shown,
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
