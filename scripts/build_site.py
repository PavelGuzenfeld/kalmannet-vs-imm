#!/usr/bin/env python3
"""Assemble the Pages site in the repo layout web/worker.js fetches from (../data, ../filters, ...)."""

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["web", "data", "filters"]
SOURCE_FILES = ["scripts/eval.py"]
WEIGHTS_FILE = "best.npz"
ROOT_REDIRECT = '<!doctype html><meta http-equiv="refresh" content="0; url=web/">\n'


def build_site(tag: str, results: Path, out: Path) -> None:
    ignore_bytecode = shutil.ignore_patterns("__pycache__")
    for name in SOURCE_DIRS:
        shutil.copytree(REPO / name, out / name, ignore=ignore_bytecode)
    for name in SOURCE_FILES:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, out / name)
    for weights in results.glob(f"*/{WEIGHTS_FILE}"):
        target = out / "results" / weights.parent.name / WEIGHTS_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(weights, target)
    (out / "web" / "version.json").write_text(json.dumps({"tag": tag}))
    (out / "index.html").write_text(ROOT_REDIRECT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "_site")
    args = ap.parse_args()
    build_site(args.tag, args.results, args.out)


if __name__ == "__main__":
    main()
