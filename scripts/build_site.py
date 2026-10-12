#!/usr/bin/env python3
"""Assemble the Pages site in the repo layout web/worker.js fetches from (../data, ../filters, ...)."""

import argparse
import json
import shutil
from pathlib import Path
from string import Template

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ["web", "data", "filters"]
SOURCE_FILES = ["scripts/eval.py", "docs/view.html", "docs/write-up.md", "docs/source-notes.md", "docs/reproduce.md"]
WEIGHTS_FILE = "best.npz"
LANDING_PAGE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KalmanNet vs IMM</title>
<style>
  body { font: 16px/1.5 system-ui, sans-serif; margin: 0 auto; padding: 16px; max-width: 40rem; color: #1d1d1f; background: #fafafa; }
</style>
</head>
<body>
<h1>KalmanNet vs IMM</h1>
<ul>
<li><a href="web/">Tracking test ground</a></li>
<li><a href="docs/view.html?doc=write-up.md">Write-up</a></li>
<li><a href="docs/view.html?doc=source-notes.md">Source notes and references</a></li>
<li><a href="docs/view.html?doc=reproduce.md">Reproduce it</a></li>
</ul>
<p>built from $tag</p>
</body>
</html>
""")


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
    (out / "index.html").write_text(LANDING_PAGE.substitute(tag=tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "_site")
    args = ap.parse_args()
    build_site(args.tag, args.results, args.out)


if __name__ == "__main__":
    main()
