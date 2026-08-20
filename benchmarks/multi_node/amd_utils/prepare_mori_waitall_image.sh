#!/usr/bin/env bash
# Prepare a node-local engine image with ROCm/mori #341 baked in.
set -euo pipefail

BASE_IMAGE=${1:?base image required}
OUT_IMAGE=${2:?output image required}
MORI_SOURCE=${3:?mori source directory required}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CNAME="mori341_build_${SLURM_JOB_ID:-manual}_$(hostname -s)"

if docker ps >/dev/null 2>&1; then
    CTR=(docker)
elif podman ps >/dev/null 2>&1; then
    CTR=(podman)
else
    CTR=(sudo docker)
fi

cleanup() {
    "${CTR[@]}" rm -f "$CNAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if "${CTR[@]}" image inspect "$OUT_IMAGE" >/dev/null 2>&1 &&
   "${CTR[@]}" run --rm -i --entrypoint python3 "$OUT_IMAGE" - <<'PY'
from mori.io import IOEngine
assert hasattr(IOEngine, "wait_all")
PY
then
    echo "[mori-image] reuse $OUT_IMAGE on $(hostname -s)"
    exit 0
fi

test -s "$MORI_SOURCE/src/io/engine.cpp"
test -s "$HERE/mori_build_inside.sh"
cleanup

echo "[mori-image] build $OUT_IMAGE from $BASE_IMAGE on $(hostname -s)"
"${CTR[@]}" run -d --name "$CNAME" --network host \
    -v "$MORI_SOURCE":/mori-src:ro \
    -v "$HERE/mori_build_inside.sh":/tmp/mori_build_inside.sh:ro \
    --entrypoint bash "$BASE_IMAGE" -c 'sleep infinity' >/dev/null

# Keep the compile transcript with the run: this build happens on the compute
# node, so without a copy on shared storage a failure leaves only whatever the
# srun stream happened to interleave.
BUILD_LOG="${BENCHMARK_LOGS_DIR:-/tmp}/server_logs/slurm_job-${SLURM_JOB_ID:-manual}/mori_build_$(hostname -s).log"
mkdir -p "$(dirname "$BUILD_LOG")"
if ! "${CTR[@]}" exec "$CNAME" bash /tmp/mori_build_inside.sh 2>&1 | tee "$BUILD_LOG"; then
    echo "[mori-image] FAILED to build $OUT_IMAGE on $(hostname -s); see $BUILD_LOG" >&2
    tail -40 "$BUILD_LOG" >&2 || true
    exit 1
fi
"${CTR[@]}" commit \
    --change 'LABEL mori.source=ROCm/mori@f7e6ac6863c53821bc7afb91a578cc6ce38fcad0' \
    --change 'LABEL mori.waitall=true' \
    "$CNAME" "$OUT_IMAGE" >/dev/null

"${CTR[@]}" run --rm -i --entrypoint python3 "$OUT_IMAGE" - <<'PY'
import mori
from mori.io import IOEngine
print("[mori-image] version=", getattr(mori, "__version__", "?"))
print("[mori-image] wait_all=", hasattr(IOEngine, "wait_all"))
assert hasattr(IOEngine, "wait_all")
PY
echo "MORI_IMAGE_DONE $OUT_IMAGE on $(hostname -s)"
