#!/usr/bin/env bash
# Build (if needed) and run a command inside the kalmannet-bench CPU image,
# with this repo mounted at /work so results/data persist on the host.
#
#   ./docker_run.sh pytest -q
#   ./docker_run.sh python eval.py
set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="kalmannet-bench:cpu"
# The container mounts the work tree without .git, so the commit has to be
# resolved here on the host and handed in -- every results artifact logs it,
# per the brief's reproducibility requirement.
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

docker build --network=host -q -f "$REPO_ROOT/docker/Dockerfile" -t "$IMAGE" "$REPO_ROOT" >/dev/null

exec docker run --rm -e GIT_COMMIT="$GIT_COMMIT" -v "$REPO_ROOT:/work" "$IMAGE" "$@"
