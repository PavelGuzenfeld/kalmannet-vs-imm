import argparse
import math

import numpy as np
import pytest
import torch

from filters.state import MEAS_DIM, STATE_DIM
from scripts.train_ukn import (
    EpochDiagnostics,
    calibration_loss,
    measurement_loss,
    nominal_R,
    relative_gain_loss,
    smoothed_discrepancy,
    updated_weight,
    warmup_ramp,
)
from scripts.tune_ukn import selected

LOG_RTOL = 4 * np.finfo(np.float64).eps
WEIGHT_ARGS = argparse.Namespace(weight_rate=0.5, weight_target=0.1, weight_bounds=[0.1, 10.0])


def batch(value, size):
    return torch.full((1, size), value, dtype=torch.float64)


def test_calibration_loss_at_a_squared_error_of_c_times_the_variance_is_c_log_two_plus_log_variance():
    """MS-48: e = 6, P = 4 I, c = 9: e²/(c P) = 1, so eq 23 is (9 ln 2 + ln 4)/2 = 5.5 ln 2."""
    loss = calibration_loss(batch(6.0, STATE_DIM), 4.0 * torch.eye(STATE_DIM, dtype=torch.float64)[None], 9.0)
    assert loss.item() == pytest.approx(5.5 * math.log(2.0), rel=LOG_RTOL, abs=0.0)


def test_measurement_loss_is_half_of_log_det_s_plus_the_student_t_log_term():
    """MS-48: S = 2 I, y = (4, 0, 0), ν_df = 4: NIS 8, so eq 24 is (3 ln 2 + 7 ln 3)/2."""
    y = torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float64)
    loss = measurement_loss(y, 2.0 * torch.eye(MEAS_DIM, dtype=torch.float64)[None], 4.0)
    assert loss.item() == pytest.approx((3 * math.log(2.0) + 7 * math.log(3.0)) / 2, rel=LOG_RTOL, abs=0.0)


def test_gain_regulariser_is_the_squared_norm_ratio_of_the_residual_to_the_ukf_gain():
    """MS-48: ‖ΔK‖_F = 3 against ‖K_UKF‖_F = 6 gives 1/4."""
    delta_K = torch.zeros(1, STATE_DIM, MEAS_DIM, dtype=torch.float64)
    K = torch.zeros(1, STATE_DIM, MEAS_DIM, dtype=torch.float64)
    delta_K[0, 0, 0], K[0, 4, 2] = 3.0, 6.0
    assert relative_gain_loss(delta_K, K).item() == 0.25


@pytest.mark.parametrize("epoch, ramp", [(1, 0.0), (2, 1 / 3), (3, 2 / 3), (4, 1.0), (9, 1.0)])
def test_warmup_ramp_rises_linearly_over_its_epochs_after_the_mse_only_ones(epoch, ramp):
    """MS-48: two MSE-only epochs, then r = 1/3, 2/3, 1."""
    assert warmup_ramp(epoch, start=2, length=3) == ramp


def test_first_smoothed_discrepancy_is_the_discrepancy_itself():
    """MS-48: eq 26 seeded at the first epoch."""
    assert smoothed_discrepancy(None, 0.75, 0.5) == 0.75


def test_smoothed_discrepancy_is_the_ema_of_the_previous_and_current():
    """MS-48: β = 1/4: 0.25 · 2 + 0.75 · 6 = 5."""
    assert smoothed_discrepancy(2.0, 6.0, 0.25) == 5.0


@pytest.mark.parametrize(
    "weight, discrepancy, expected",
    [(1.0, 0.1, 1.0), (1.0, 0.1 + 2 * math.log(2.0), 2.0), (8.0, 1.0, 10.0), (0.15, -1.0, 0.1)],
)
def test_adaptive_weight_steps_by_exp_eta_times_the_excess_discrepancy_within_its_bounds(
    weight, discrepancy, expected
):
    """MS-48: eq 27 with η = 1/2, τ = 0.1, w in [0.1, 10]."""
    assert updated_weight(weight, discrepancy, WEIGHT_ARGS) == pytest.approx(expected, rel=LOG_RTOL, abs=0.0)


def test_epoch_discrepancies_compare_pooled_squared_error_with_variance_and_nis_with_its_dimension():
    """MS-48: two finite rows with e_i² = e·P_ii (P_ii = 1..9) give g_cal = 1; NIS 3e² against n_y = 3 gives
    g_meas = 2. The third row is non-finite and skipped."""
    diagnostics = EpochDiagnostics()
    variance = torch.arange(1.0, STATE_DIM + 1.0, dtype=torch.float64)
    error = torch.sqrt(math.e * variance).expand(3, -1).clone()
    error[2, 0] = math.inf
    P = torch.diag(variance).expand(3, -1, -1)
    y = torch.tensor([[math.sqrt(6.0) * math.e, 0.0, 0.0]], dtype=torch.float64).expand(3, -1)
    diagnostics.add(error, P, y, 2.0 * torch.eye(MEAS_DIM, dtype=torch.float64).expand(3, -1, -1))
    calibration, measurement = diagnostics.discrepancies()
    assert calibration == pytest.approx(1.0, rel=LOG_RTOL, abs=0.0)
    assert measurement == pytest.approx(2.0, rel=LOG_RTOL, abs=0.0)


def test_measurement_loss_of_a_batch_is_the_mean_of_its_rows():
    """MS-48: a zero innovation with S = I adds 0, halving the (3 ln 2 + 7 ln 3)/2 of the other row."""
    y = torch.tensor([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    S = torch.stack(
        [2.0 * torch.eye(MEAS_DIM, dtype=torch.float64), torch.eye(MEAS_DIM, dtype=torch.float64)]
    )
    loss = measurement_loss(y, S, 4.0)
    assert loss.item() == pytest.approx((3 * math.log(2.0) + 7 * math.log(3.0)) / 4, rel=LOG_RTOL, abs=0.0)


def test_nominal_r_is_the_diagonal_of_the_manifest_sigmas_squared():
    """MS-11: sigmas 0.5 rad, 0.25 rad and 3 m give R = diag(0.25, 0.0625, 9), exact in binary."""
    manifest = {"regimes": {"nominal": {"sigma_az_rad": 0.5, "sigma_el_rad": 0.25, "sigma_range_m": 3.0}}}
    assert np.array_equal(nominal_R(manifest), np.diag([0.25, 0.0625, 9.0]))


def tuning_row(gain_scale, pos_rmse, diverged_fraction, anees_mean):
    return dict(gain_scale=gain_scale, pos_rmse=pos_rmse, diverged_fraction=diverged_fraction, anees_mean=anees_mean)


def test_tuning_drops_a_run_past_ten_percent_of_the_best_rmse_even_at_anees_nine():
    """MS-46: c_K is chosen on validation; 2.25 m is past 1.1 x 2.0 m, so its ANEES of 9 cannot win."""
    rows = [tuning_row(0.1, 2.0, 0.0, 100.0), tuning_row(0.3, 2.25, 0.0, 9.0)]
    assert selected(rows) == rows[0]


def test_tuning_keeps_a_run_exactly_at_the_rmse_slack_and_picks_its_nearer_anees():
    """MS-46: 1.1 x 2.0 is 2.2 exactly in binary, so the edge row is inside the band and wins on ANEES."""
    rows = [tuning_row(0.1, 2.0, 0.0, 18.0), tuning_row(0.3, 2.2, 0.0, 9.0)]
    assert selected(rows) == rows[1]


def test_tuning_drops_a_run_more_than_five_points_of_divergence_above_the_fewest():
    """MS-46: 0.0625 diverged is past 0 + 0.05, so that run drops out whatever its ANEES."""
    rows = [tuning_row(0.1, 2.0, 0.0, 100.0), tuning_row(0.3, 2.0, 0.0625, 9.0)]
    assert selected(rows) == rows[0]


def test_tuning_breaks_an_anees_tie_in_log_ratio_towards_the_smaller_gain_scale():
    """MS-46: 9/2 and 18 sit at the same |log(ANEES/9)|, so the smaller c_K wins."""
    rows = [tuning_row(1.0, 2.0, 0.0, 18.0), tuning_row(0.3, 2.0, 0.0, 4.5)]
    assert selected(rows) == rows[1]


def test_tuning_keeps_a_run_exactly_at_the_diverged_slack_of_the_fewest():
    """MS-46: 0 + 0.05 is 0.05 exactly, so the edge row is inside the 5-point band and wins on ANEES."""
    rows = [tuning_row(0.1, 2.0, 0.0, 18.0), tuning_row(0.3, 2.0, 0.05, 9.0)]
    assert selected(rows) == rows[1]


def test_tuning_measures_anees_against_the_state_dimension_of_nine():
    """MS-46: ANEES 9 sits at log ratio 0 and ANEES 1 at log 9, so 9 wins even with the larger c_K."""
    rows = [tuning_row(0.1, 2.0, 0.0, 1.0), tuning_row(0.3, 2.0, 0.0, 9.0)]
    assert selected(rows) == rows[1]
