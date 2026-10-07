import json
import sys
from pathlib import Path

import numpy as np
import pytest

from filters.imm_tuning import IMM_SEARCH_GRIDS, IMM_SEARCH_START, coordinate_search, edge_axes
from scripts import tune_imm

MAX_EXTENSIONS = 3


def test_selection_on_the_bottom_grid_value_is_an_edge_of_that_axis():
    assert edge_axes({"ca_tau": [1.0, 2.0, 4.0], "p_stay": [0.95, 0.97, 0.99]}, {"ca_tau": 1.0, "p_stay": 0.97}) == [
        "ca_tau"
    ]


def test_selection_on_the_top_grid_value_is_an_edge_of_that_axis():
    assert edge_axes({"ca_tau": [1.0, 2.0, 4.0], "p_stay": [0.95, 0.97, 0.99]}, {"ca_tau": 2.0, "p_stay": 0.99}) == [
        "p_stay"
    ]


def test_selection_strictly_inside_every_grid_has_no_edge():
    assert edge_axes({"ca_tau": [1.0, 2.0, 4.0], "p_stay": [0.95, 0.97, 0.99]}, {"ca_tau": 2.0, "p_stay": 0.97}) == []


def _tiny_val_split(data_dir):
    nominal = {"dt": 0.02, "sigma_az_rad": 0.001, "sigma_el_rad": 0.001, "sigma_range_m": 2.0}
    (data_dir / "manifest.json").write_text(json.dumps({"regimes": {"nominal": nominal}}))
    np.savez(data_dir / "val.npz", states=np.zeros((1, 1, 9)), measurements=np.zeros((1, 1, 1, 3)))


def _tune(monkeypatch, tmp_path, rmse_of_kwargs, *flags):
    _tiny_val_split(tmp_path)
    out = tmp_path / "imm_tuning.json"
    monkeypatch.setattr(tune_imm, "run_imm", lambda states, meas, dt, R, **kwargs: rmse_of_kwargs(kwargs))
    argv = ["tune_imm.py", "--data-dir", str(tmp_path), "--out", str(out), "--n-search", "1", *flags]
    monkeypatch.setattr(sys, "argv", argv)
    tune_imm.main()
    return out


def test_tune_imm_refuses_to_write_a_selection_left_on_an_edge_and_names_the_axis(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match=r"on axes \[.ct_sigma_a.\]:"):
        _tune(monkeypatch, tmp_path, lambda kwargs: -kwargs["ct_sigma_a"])
    assert not (tmp_path / "imm_tuning.json").exists()


def test_tune_imm_records_the_edge_axes_when_an_edge_selection_is_allowed(monkeypatch, tmp_path):
    out = _tune(monkeypatch, tmp_path, lambda kwargs: -kwargs["ct_sigma_a"], "--allow-edge-selection")
    tuning = json.loads(out.read_text())
    assert tuning["edge_axes"] == ["ct_sigma_a"]
    assert tuning["best"]["ct_sigma_a"] == tuning["grid"]["ct_sigma_a"][-1]
    assert len(tuning["grid"]["ct_sigma_a"]) == len(IMM_SEARCH_GRIDS["ct_sigma_a"]) + MAX_EXTENSIONS


def test_tune_imm_writes_an_interior_selection_with_no_edge_axes(monkeypatch, tmp_path):
    out = _tune(monkeypatch, tmp_path, lambda kwargs: (kwargs["ca_tau"] - 2.0) ** 2)
    tuning = json.loads(out.read_text())
    assert tuning["edge_axes"] == []
    assert {axis: tuning["best"][axis] for axis in IMM_SEARCH_START} == IMM_SEARCH_START


def test_cv_sigma_a_is_pinned_at_the_truth_value_in_every_scored_and_written_candidate(monkeypatch, tmp_path):
    scored_cv = set()
    out = _tune(monkeypatch, tmp_path, lambda kwargs: scored_cv.add(kwargs["cv_sigma_a"]) or (kwargs["ca_tau"] - 2.0) ** 2)
    tuning = json.loads(out.read_text())
    assert scored_cv == {0.05}
    assert tuning["best"]["cv_sigma_a"] == 0.05
    assert "cv_sigma_a" not in tuning["grid"]
    assert {c["cv_sigma_a"] for c in tuning["all_candidates"]} == {0.05}


def test_search_space_tunes_ct_sigma_a_alongside_the_ms12_axes():
    """MS-12 initial grids; edge axes extend past them at run time."""
    assert IMM_SEARCH_GRIDS == {
        "p_stay": [0.95, 0.97, 0.99, 0.995, 0.998],
        "omega_ct_deg": [15.0, 20.0, 25.0, 30.0, 40.0],
        "ca_sigma_a": [1.0, 2.0, 4.0],
        "ca_tau": [1.0, 2.0, 4.0],
        "ct_sigma_a": [0.1, 0.3, 1.0],
    }
    assert IMM_SEARCH_START == {
        "p_stay": 0.99, "omega_ct_deg": 25.0, "ca_sigma_a": 2.0, "ca_tau": 2.0, "ct_sigma_a": 0.3,
    }


def test_every_initial_grid_increases_strictly_with_the_start_strictly_inside():
    for axis, grid in IMM_SEARCH_GRIDS.items():
        assert grid == sorted(set(grid)), axis
        assert grid[0] < IMM_SEARCH_START[axis] < grid[-1], axis


def test_tune_imm_defaults_to_the_30_trajectory_validation_subset_of_the_protocol(monkeypatch, tmp_path):
    _tiny_val_split(tmp_path)
    out = tmp_path / "imm_tuning.json"
    monkeypatch.setattr(tune_imm, "run_imm", lambda states, meas, dt, R, **kwargs: 0.0)
    monkeypatch.setattr(sys, "argv", ["tune_imm.py", "--data-dir", str(tmp_path), "--out", str(out)])
    tune_imm.main()
    assert json.loads(out.read_text())["n_search_trajectories"] == 30


def test_minimum_above_the_top_edge_is_reached_by_geometric_extension():
    best, grids, _ = coordinate_search(
        {"ca_tau": [1.0, 2.0, 4.0]}, {"ca_tau": 2.0}, lambda c: (c["ca_tau"] - 16.0) ** 2, MAX_EXTENSIONS
    )
    assert best == {"ca_tau": 16.0}
    assert grids["ca_tau"] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def test_minimum_below_the_bottom_edge_is_reached_by_geometric_extension():
    best, grids, _ = coordinate_search(
        {"ca_tau": [1.0, 2.0, 4.0]}, {"ca_tau": 2.0}, lambda c: (c["ca_tau"] - 0.5) ** 2, MAX_EXTENSIONS
    )
    assert best == {"ca_tau": 0.5}
    assert grids["ca_tau"] == [0.25, 0.5, 1.0, 2.0, 4.0]


def test_p_stay_extends_geometrically_in_one_minus_p_so_it_never_reaches_one():
    best, grids, _ = coordinate_search(
        {"p_stay": [0.5, 0.75]}, {"p_stay": 0.5}, lambda c: (c["p_stay"] - 0.875) ** 2, MAX_EXTENSIONS
    )
    assert best == {"p_stay": 0.875}
    assert grids["p_stay"] == [0.5, 0.75, 0.875, 0.9375]


def test_extension_stops_at_the_cap_and_leaves_the_selection_on_the_edge():
    best, grids, _ = coordinate_search(
        {"ca_tau": [1.0, 2.0, 4.0]}, {"ca_tau": 2.0}, lambda c: -c["ca_tau"], MAX_EXTENSIONS
    )
    assert grids["ca_tau"] == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    assert best == {"ca_tau": 32.0}
    assert edge_axes(grids, best) == ["ca_tau"]


def test_each_axis_settles_at_its_own_minimum_with_the_others_held():
    def separable(c):
        return (c["cv_sigma_a"] - 0.1) ** 2 + (c["ct_sigma_a"] - 0.3) ** 2

    best, _, evaluated = coordinate_search(
        {"cv_sigma_a": [0.05, 0.1, 0.2], "ct_sigma_a": [0.1, 0.3, 1.0]},
        {"cv_sigma_a": 0.05, "ct_sigma_a": 1.0},
        separable,
        MAX_EXTENSIONS,
    )
    assert best == {"cv_sigma_a": 0.1, "ct_sigma_a": 0.3}
    assert evaluated[(("ct_sigma_a", 0.3), ("cv_sigma_a", 0.1))] == 0.0


def test_extension_leaves_the_callers_grids_untouched():
    grids = {"ca_tau": [1.0, 2.0, 4.0]}
    coordinate_search(grids, {"ca_tau": 2.0}, lambda c: -c["ca_tau"], MAX_EXTENSIONS)
    assert grids == {"ca_tau": [1.0, 2.0, 4.0]}


def test_a_tie_with_the_current_value_does_not_move_the_selection():
    best, _, _ = coordinate_search({"ca_tau": [1.0, 2.0, 4.0]}, {"ca_tau": 2.0}, lambda c: 0.0, MAX_EXTENSIONS)
    assert best == {"ca_tau": 2.0}


def test_a_coupled_axis_is_revisited_after_a_later_axis_moves():
    best, _, _ = coordinate_search(
        {"a": [0.5, 1.0, 2.0, 4.0, 8.0], "b": [0.5, 1.0, 2.0, 4.0, 8.0]},
        {"a": 1.0, "b": 1.0},
        lambda c: (c["a"] - c["b"]) ** 2 + 2.0 * (c["b"] - 2.0) ** 2,
        MAX_EXTENSIONS,
    )
    assert best == {"a": 2.0, "b": 2.0}


def test_every_axis_left_on_an_edge_is_named_not_only_the_first():
    assert edge_axes({"ca_tau": [1.0, 2.0, 4.0], "p_stay": [0.95, 0.97, 0.99]}, {"ca_tau": 1.0, "p_stay": 0.99}) == [
        "ca_tau",
        "p_stay",
    ]


def test_web_imm_tuning_json_holds_the_ms12_selected_values():
    """MS-12: spec #1 selection (CV sigma_a pinned at 0.05); the Pyodide page must fly the benchmarked IMM."""
    committed = json.loads((Path(__file__).resolve().parents[1] / "web" / "imm_tuning.json").read_text())

    assert committed == {
        "best": {
            "p_stay": 0.99,
            "omega_ct_deg": 25.0,
            "ca_sigma_a": 2.0,
            "ca_tau": 2.0,
            "ct_sigma_a": 1.0,
            "cv_sigma_a": 0.05,
        }
    }
