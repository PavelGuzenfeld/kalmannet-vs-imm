#!/usr/bin/env python3
"""Assemble one results table from the evaluators' own outputs, recomputing nothing; unreported columns stay
empty. RKN's consistency columns come from its JSON, its bounds written as the CSV evaluators write theirs.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

CONSISTENCY_KEYS = ("anees_mean", "anees_bounds", "anees_fraction_inside_95", "coverage_68", "coverage_95", "n_mc")


class Source(NamedTuple):
    csv_name: str
    filter_name: str
    consistency_json: str | None = None


SOURCES = (
    Source("results_table.csv", "IMM"),
    Source("knet_test_rmse.csv", "KalmanNet-vanilla"),
    Source("rkn_test_rmse.csv", "RKN", "rkn_eval.json"),
    Source("results_table.csv", "BKN"),
    Source("results_table.csv", "UKN"),
)


def read_source(results_dir, source):
    with open(results_dir / source.csv_name, newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row["filter"] == source.filter_name]
    if not rows:
        raise SystemExit(f"{source.csv_name} has no rows for {source.filter_name}")
    if source.consistency_json is None:
        return reader.fieldnames, rows
    summaries = json.loads((results_dir / source.consistency_json).read_text())
    return [*reader.fieldnames, *CONSISTENCY_KEYS], [{**row, **consistency_cells(summaries[row["regime"]])} for row in rows]


def consistency_cells(summary):
    cells = {key: str(summary[key]) for key in CONSISTENCY_KEYS}
    cells["anees_bounds"] = str(tuple(np.float64(bound) for bound in summary["anees_bounds"]))
    return cells


def assemble(results_dir, sources=SOURCES):
    fieldnames, rows = [], []
    for source in sources:
        source_fields, source_rows = read_source(results_dir, source)
        fieldnames += [name for name in source_fields if name not in fieldnames]
        rows += source_rows
    filter_order = [source.filter_name for source in sources]
    regime_order = list(dict.fromkeys(row["regime"] for row in rows))
    rows.sort(key=lambda row: (regime_order.index(row["regime"]), filter_order.index(row["filter"])))
    return fieldnames, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/eval_table.csv"))
    args = ap.parse_args()

    fieldnames, rows = assemble(args.results_dir)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
