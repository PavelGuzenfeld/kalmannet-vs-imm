import json
import sys

import numpy as np
import pytest

from data.trajectory_gen import fly_scenario
from filters.flags import DIVERGED_POSITION_ERROR_M
from scripts.page_parity import (
    DIVERGING_KNET_DIVERGENCE,
    DIVERGING_KNET_NPZ,
    DRIFT_QUANTITIES,
    EVADE_CASE,
    EVADE_ENGAGE_STEP,
    IMM_TUNING,
    MAX_G_LOWERED_CASE,
    NO_WEIGHTS,
    PILOT_ROW,
    SCENARIOS_DIR,
    TIGHT_THRESHOLDS,
    drift_markdown,
    drift_table,
    evasive_flight,
    absent_or_difference,
    main,
    max_abs_difference,
    write_cases,
)
from web.live_flight import LEARNED_WEIGHTS, LiveFlight, trace

KILOMETRE_ROUNDING_ATOL_M = 1e-9
MID_PULL_STEP = 100
LOWERED_MAX_LOAD_FACTOR = 3.0


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("cases")
    write_cases(out_dir)
    return out_dir


def case_flags(cases, name):
    return json.loads((cases / f"{name}.expected.json").read_text())["flags"]


def test_case_one_step_before_the_ekf_raw_error_trip_reads_untripped_and_at_it_reads_the_trip_step(cases):
    full = case_flags(cases, "break_turn_9g")["EKF"]["raw_error"]
    before = json.loads(
        (cases / "break_turn_9g_one_step_before_ekf_raw_error_trip.expected.json").read_text()
    )
    at = json.loads((cases / "break_turn_9g_at_ekf_raw_error_trip.expected.json").read_text())

    assert (before["steps"], before["flags"]["EKF"]["raw_error"]) == (int(full), "-")
    assert (at["steps"], at["flags"]["EKF"]["raw_error"]) == (int(full) + 1, full)


def test_tight_threshold_case_records_its_thresholds_in_the_scenario_and_trips_earlier(cases):
    scenario = json.loads((cases / "break_turn_9g_tight_thresholds.json").read_text())

    assert scenario["thresholds"] == TIGHT_THRESHOLDS
    tight, default = (
        case_flags(cases, name)["EKF"] for name in ("break_turn_9g_tight_thresholds", "break_turn_9g")
    )
    assert all(
        int(tight[flag]) < int(default[flag]) for flag in ("track_loss", "overconfidence", "raw_error")
    )


def test_case_expectations_leave_out_filters_replay_skips(cases):
    assert set(case_flags(cases, "break_turn_9g")) == {"EKF", "UKF", "IMM"}


def test_trace_has_one_readout_per_control_and_ends_on_the_page_frame_readouts(cases):
    scenario = json.loads((cases / "break_turn_9g_at_ekf_raw_error_trip.json").read_text())
    best = json.loads(IMM_TUNING.read_text())["best"]
    flight = LiveFlight(scenario, best, NO_WEIGHTS)
    for control in scenario["controls"]:
        flight.advance(control)

    steps = trace(scenario, best, NO_WEIGHTS)

    assert len(steps) == len(scenario["controls"])
    assert steps[-1] == flight.frame()["filters"]


def test_max_abs_difference_takes_the_largest_gap_over_steps_and_components_wherever_it_occurs():
    native = [{"EKF": {"estimate_m": [0.0, 0.0, 0.0]}}] * 3
    other = [
        {"EKF": {"estimate_m": [0.0, 0.0, 4.0]}},
        {"EKF": {"estimate_m": [0.0, 0.5, 0.0]}},
        {"EKF": {"estimate_m": [1.0, 0.0, 0.0]}},
    ]

    assert max_abs_difference(native, other, "EKF", "estimate_m") == 4.0


def test_absent_readout_on_both_sides_is_zero_difference():
    step = [{"IMM": {"nis": None}}]

    assert max_abs_difference(step, step, "IMM", "nis") == 0.0


def test_drift_markdown_has_one_row_per_scenario_and_filter_with_a_column_per_quantity():
    table = drift_markdown([("break_turn_9g", "EKF", 1.0, 2.0, 3.0)]).splitlines()

    assert table[-1] == "| break_turn_9g | EKF | 1.000e+00 | 2.000e+00 | 3.000e+00 |"
    assert len(table[1].split("|")) == 3 + len(DRIFT_QUANTITIES) + 1


def test_tight_thresholds_are_the_specified_values():
    assert TIGHT_THRESHOLDS == {
        "innovation_gate_nis": 3.0,
        "nees_upper": 10.0,
        "overconfident_steps": 5,
        "raw_error_m": 5.0,
    }


def test_readout_present_on_one_side_only_is_an_error_not_zero_drift():
    with pytest.raises(TypeError):
        absent_or_difference(None, [1.0])


def overhead_pyodide_stand_in():
    scenario = json.loads((SCENARIOS_DIR / "overhead_pass.json").read_text())
    best = json.loads(IMM_TUNING.read_text())["best"]
    return json.loads(json.dumps({"overhead_pass": trace(scenario, best, NO_WEIGHTS)}))


def test_drift_table_has_a_row_per_filter_and_zero_drift_against_an_identical_trace():
    rows = drift_table(overhead_pyodide_stand_in())

    assert rows == [("overhead_pass", name, 0.0, 0.0, 0.0) for name in ("EKF", "IMM", "UKF")]


def test_drift_table_reports_a_one_metre_vertical_estimate_shift_in_one_filter_only():
    shifted = overhead_pyodide_stand_in()
    for step in shifted["overhead_pass"]:
        step["UKF"]["estimate_m"][2] += 1.0

    rows = drift_table(shifted)

    estimates = {name: estimate for _, name, estimate, *_ in rows}
    assert (estimates["EKF"], estimates["IMM"]) == (0.0, 0.0)
    assert estimates["UKF"] == pytest.approx(1.0, abs=KILOMETRE_ROUNDING_ATOL_M)


def test_drift_markdown_separator_row_has_a_cell_per_column():
    table = drift_markdown([("break_turn_9g", "EKF", 1.0, 2.0, 3.0)]).splitlines()

    assert table[2] == "| --- | --- | --- | --- | --- |"


def test_main_drift_prints_the_table_for_a_trace_file(tmp_path, monkeypatch, capsys):
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps(overhead_pyodide_stand_in()))
    monkeypatch.setattr(sys, "argv", ["page_parity.py", "drift", str(trace_file)])

    main()

    assert (
        capsys.readouterr().out.splitlines()[-1]
        == "| overhead_pass | UKF | 0.000e+00 | 0.000e+00 | 0.000e+00 |"
    )


def test_main_cases_writes_the_scenario_and_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["page_parity.py", "cases", str(tmp_path)])

    main()

    assert (tmp_path / "overhead_pass.json").exists() and (tmp_path / "overhead_pass.expected.json").exists()


def diverging_knet_flight(cases, tmp_path):
    knet_npz = tmp_path / LEARNED_WEIGHTS["KalmanNet"]
    knet_npz.parent.mkdir(parents=True)
    knet_npz.write_bytes((cases / DIVERGING_KNET_NPZ).read_bytes())
    scenario = json.loads((cases / "break_turn_9g.json").read_text())
    flight = LiveFlight(scenario, json.loads(IMM_TUNING.read_text())["best"], tmp_path)
    frames = []
    for control in scenario["controls"]:
        flight.advance(control)
        frames.append(flight.frame())
    return frames


def test_page_stops_a_diverged_filter_at_the_step_native_replay_reports_and_shows_no_estimate_after_it(cases, tmp_path):
    native_step = json.loads((cases / DIVERGING_KNET_DIVERGENCE).read_text())["KalmanNet"]
    frames = diverging_knet_flight(cases, tmp_path)
    knet = [frame["filters"]["KalmanNet"] for frame in frames]

    assert 0 < native_step < len(frames) - 1
    assert (knet[native_step - 1]["diverged"], knet[native_step]["diverged"]) == (False, True)
    assert knet[native_step]["diverged_step"] == native_step
    assert knet[native_step]["error_m"] > DIVERGED_POSITION_ERROR_M
    assert all(readout["estimate_m"] is None and readout["error_m"] is None for readout in knet[native_step + 1 :])
    assert frames[-1]["flags"]["KalmanNet"] == frames[native_step]["flags"]["KalmanNet"]
    assert frames[-1]["trips"]["KalmanNet"] == frames[native_step]["trips"]["KalmanNet"]
    assert {name: readout["diverged"] for name, readout in frames[-1]["filters"].items()} == {
        "EKF": False,
        "UKF": False,
        "IMM": False,
        "KalmanNet": True,
    }


def test_max_g_lowered_mid_pull_case_flies_the_page_truth_and_leaves_the_break_turn_after_its_step(
    cases,
):
    """MS-27: the page applies the recorded envelope change at the row replay applies it."""
    scenario = json.loads((cases / f"{MAX_G_LOWERED_CASE}.json").read_text())
    flight = LiveFlight(scenario, json.loads(IMM_TUNING.read_text())["best"], NO_WEIGHTS)
    page_truth = []
    for control in scenario["controls"]:
        flight.advance(control)
        page_truth.append(flight.frame()["truth_m"] + flight.frame()["truth_velocity_m_s"])

    states, _ = fly_scenario(scenario)
    break_turn, _ = fly_scenario(json.loads((cases / "break_turn_9g.json").read_text()))

    assert [[step, envelope["max_load_factor"]] for step, envelope in scenario["envelope_changes"]] == [
        [MID_PULL_STEP, LOWERED_MAX_LOAD_FACTOR]
    ]
    assert np.array(page_truth).tobytes() == states[:, :6].tobytes()
    assert json.loads(flight.scenario_json())["envelope_changes"] == scenario["envelope_changes"]
    after = MID_PULL_STEP + 1
    assert states[:after, :6].tobytes() == break_turn[:after, :6].tobytes()
    assert not np.array_equal(states[after, :6], break_turn[after, :6])


def test_exported_evasive_flight_replays_to_the_flags_and_final_truth_the_live_flight_showed(cases):
    flight = evasive_flight(json.loads((SCENARIOS_DIR / "break_turn_9g.json").read_text()))
    scenario = json.loads((cases / f"{EVADE_CASE}.json").read_text())

    live_flags = flight.frame()["flags"]
    assert scenario["controls"] == json.loads(flight.scenario_json())["controls"]
    assert scenario["controls"][:EVADE_ENGAGE_STEP] == [PILOT_ROW] * EVADE_ENGAGE_STEP
    assert PILOT_ROW not in scenario["controls"][EVADE_ENGAGE_STEP:]
    assert case_flags(cases, EVADE_CASE) == {name: live_flags[name] for name in ("EKF", "UKF", "IMM")}
    assert fly_scenario(scenario)[0][-1, :6].tolist() == flight.frame()["truth_m"] + flight.frame()["truth_velocity_m_s"]


def test_evasive_case_flies_level_at_the_sensor_from_4_km_and_engages_evade_after_one_second_of_eight(cases):
    scenario = json.loads((cases / f"{EVADE_CASE}.json").read_text())
    controls = scenario["controls"]

    assert scenario["start"] == {"position": [0.0, 4000.0, 300.0], "velocity": [0.0, -110.0, 0.0]}
    assert controls[0] == [0.0, 1.0, 0.6, 0.0, 0.0, 0.0]
    assert len(controls) == 400
    assert [row == PILOT_ROW for row in controls].index(False) == 50
