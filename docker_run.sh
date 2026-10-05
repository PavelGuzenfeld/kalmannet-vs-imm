#!/usr/bin/env bash
# Build (if needed) and run a command inside the kalmannet-bench CPU image,
# with this repo mounted at /work so results/data persist on the host.
#
#   ./docker_run.sh pytest -q
#   ./docker_run.sh python eval.py
set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="kalmannet-bench:cpu"
# The slim image ships no git binary, so the commit is resolved on the host and
# handed in -- every results artifact logs it, per the reproducibility brief.
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

[ -n "${BENCH_IMAGE_PREBUILT:-}" ] || docker build --network=host -q -f "$REPO_ROOT/docker/Dockerfile" -t "$IMAGE" "$REPO_ROOT" >/dev/null

# --user keeps generated results, caches and .coverage owned by the caller
# instead of root; MPLCONFIGDIR because that uid has no home inside the image.
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl \
    -e GIT_COMMIT="$GIT_COMMIT" -v "$REPO_ROOT:/work" "$IMAGE" "$@"
