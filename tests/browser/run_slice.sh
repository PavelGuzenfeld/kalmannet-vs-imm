#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
work="$(mktemp -d)"
network="slice-page-views-$$"
server="$network-server"
trap 'docker rm -f "$server" >/dev/null 2>&1 || true; docker network rm "$network" >/dev/null 2>&1 || true; rm -rf "$work"' EXIT

for scenario in break_turn_9g overhead_pass; do
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl -v "$PWD":/work -w /work \
    kalmannet-bench:cpu python scripts/replay.py "data/scenarios/$scenario.json" \
    --imm-tuning web/imm_tuning.json > "$work/$scenario.txt"
done

docker network create "$network" >/dev/null
docker run -d --rm --name "$server" --network "$network" -v "$PWD":/srv:ro -w /srv python:3.12-slim \
  python -m http.server 8000 >/dev/null
docker run --rm --network "$network" --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/tests/browser":/t:ro -v "$work":/replay:ro mcr.microsoft.com/playwright/python:v1.48.0-jammy \
  sh -c 'pip install -q --user playwright==1.48.0 && python /t/slice_page_views.py "$0" /replay' "http://$server:8000"
