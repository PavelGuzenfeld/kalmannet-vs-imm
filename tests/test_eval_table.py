import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from filters.torch_ops import export_npz
from filters.knet.gain_network import KalmanGainGRU
from filters.knet.rkn_network import RKNGainCovNetwork
from scripts import eval as bench_eval
from scripts import eval_knet, eval_rkn, eval_table
from tests.test_bayesian_kalmannet import knet_stand_in
from tests.test_eval_learned_scenarios import RKN_MULTIPLIERS, zeroed
from tests.test_numpy_forward import REPO
from tests.test_replay import imm_tuning  # noqa: F401

REGIMES = ["nominal", "high_maneuver", "high_noise", "low_noise", "heavy_tailed", "scenarios"]
FILTERS = ["IMM", "KalmanNet-vanilla", "RKN", "BKN"]
RKN_JSON_KEYS = ("anees_mean", "anees_fraction_inside_95", "coverage_68", "coverage_95", "n_mc")
EVALUATOR_CSVS = ("results_table.csv", "knet_test_rmse.csv", "rkn_test_rmse.csv")


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run(monkeypatch, main, *argv):
    monkeypatch.setattr(sys, "argv", [main.__module__, *map(str, argv)])
    main()


@pytest.fixture
def evaluator_outputs(imm_tuning, tmp_path, monkeypatch):  # noqa: F811
    data_dir, results = tmp_path / "datasets", tmp_path / "results"
    results.mkdir()
    toy = ["--n-train", "1", "--n-val", "1", "--n-test", "1", "--n-sweep", "1", "--mc-test", "2", "--n-highmc", "1"]
    subprocess.run([sys.executable, "scripts/make_dataset.py", "--out", data_dir, *toy, "--mc-highmc", "2"], cwd=REPO, check=True)
    bkn_weights = tmp_path / "bkn.npz"
    export_npz(knet_stand_in().state_dict(), bkn_weights)
    knet = zeroed(KalmanGainGRU(), tmp_path / "knet.pt")
    rkn = zeroed(RKNGainCovNetwork(**RKN_MULTIPLIERS), tmp_path / "rkn.pt")
    monkeypatch.chdir(REPO)
    run(monkeypatch, bench_eval.main, "--data-dir", data_dir, "--imm-tuning", imm_tuning, "--bkn-weights", bkn_weights, "--out", results)
    run(monkeypatch, eval_knet.main, "--data-dir", data_dir, "--checkpoint", knet, "--out", results / "knet_test_rmse.csv")
    run(monkeypatch, eval_rkn.main, "--data-dir", data_dir, "--checkpoint", rkn, "--out", results / "rkn_eval.json",
        "--out-csv", results / "rkn_test_rmse.csv")
    return results


def test_one_table_holds_imm_knet_rkn_and_bkn_on_every_regime_with_each_evaluators_own_values(evaluator_outputs, monkeypatch):
    """MS-42: every row is its evaluator's own, RKN's consistency cells coming from rkn_eval.json."""
    out = evaluator_outputs / "eval_table.csv"

    run(monkeypatch, eval_table.main, "--results-dir", evaluator_outputs, "--out", out)

    table = read_rows(out)
    assert [(row["regime"], row["filter"]) for row in table] == [(regime, name) for regime in REGIMES for name in FILTERS]
    per_evaluator = [row for name in EVALUATOR_CSVS for row in read_rows(evaluator_outputs / name) if row["filter"] in FILTERS]
    assert len(per_evaluator) == len(table)
    for row in per_evaluator:
        combined = next(r for r in table if (r["regime"], r["filter"]) == (row["regime"], row["filter"]))
        assert {key: combined[key] for key in row} == row
    assert all(row["anees_mean"] == "" for row in table if row["filter"] == "KalmanNet-vanilla")
    assert all(row["anees_mean"] != "" for row in table if row["filter"] in ("IMM", "BKN"))
    rkn_json = json.loads((evaluator_outputs / "rkn_eval.json").read_text())
    for row in (r for r in table if r["filter"] == "RKN"):
        summary = rkn_json[row["regime"]]
        assert {key: row[key] for key in RKN_JSON_KEYS} == {key: str(summary[key]) for key in RKN_JSON_KEYS}
        assert row["anees_bounds"] == str(tuple(np.float64(bound) for bound in summary["anees_bounds"]))


def test_a_listed_filter_missing_from_its_evaluator_output_fails_naming_it(evaluator_outputs, monkeypatch):
    """MS-42: an evaluator output lacking a listed filter is an error, never a silently thinner table."""
    (evaluator_outputs / "rkn_test_rmse.csv").write_text("filter,regime,pos_rmse\n")

    with pytest.raises(SystemExit, match="RKN"):
        run(monkeypatch, eval_table.main, "--results-dir", evaluator_outputs, "--out", evaluator_outputs / "eval_table.csv")
