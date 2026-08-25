#!/usr/bin/env bash
set -euo pipefail

PARTITION="compute"
EXCLUDED_NODES="smci300x-ccs-aus-e04-19,smci300x-ccs-aus-e07-03"
LOCAL_ROOT="/raid/inferencex"
SQUASH_DIR="$LOCAL_ROOT/squash"
HF_HUB_CACHE_MOUNT="$LOCAL_ROOT/models/hub"
AIPERF_MMAP_CACHE_MOUNT="$LOCAL_ROOT/aiperf-mmap-cache"

export HF_HUB_CACHE_MOUNT
export AIPERF_DATASET_MMAP_CACHE_DIR="/aiperf_mmap_cache"
export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

IMAGE_SLUG="${IMAGE//\//_}"
IMAGE_SLUG="${IMAGE_SLUG//:/_}"
IMAGE_SLUG="${IMAGE_SLUG//@/_}"
IMAGE_SLUG="${IMAGE_SLUG//#/_}"
SQUASH_FILE="$SQUASH_DIR/${IMAGE_SLUG}.sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

set -x

JOB_ID=$(set +o pipefail; salloc \
    --partition="$PARTITION" \
    --exclude="$EXCLUDED_NODES" \
    --gres="gpu:$GPU_COUNT" \
    --cpus-per-task=128 \
    --time=180 \
    --no-shell \
    --job-name="$RUNNER_NAME" 2>&1 \
    | tee /dev/stderr \
    | grep -oP 'Granted job allocation \K[0-9]+')

if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: salloc failed to allocate a job" >&2
    exit 1
fi

export PORT=$((40000 + (JOB_ID % 10000)))
COMPUTE_TMPDIR="$LOCAL_ROOT/tmp/${UID}/${JOB_ID}"
export XDG_RUNTIME_DIR="$COMPUTE_TMPDIR/runtime"
export XDG_CACHE_HOME="$COMPUTE_TMPDIR/xdg-cache"
export TRITON_CACHE_DIR="$COMPUTE_TMPDIR/triton-cache"

cleanup() {
    local rc=$?
    # shellcheck disable=SC2016 # $1 is expanded by the compute-node shell.
    srun --jobid="$JOB_ID" bash -c 'rm -rf -- "$1"' bash "$COMPUTE_TMPDIR" >/dev/null 2>&1 || true
    scancel "$JOB_ID" >/dev/null 2>&1 || true
    exit "$rc"
}
trap cleanup EXIT

# Everything except the GitHub Actions workspace is node-local NVMe. The
# workspace is the sole NFS mount exposed to the benchmark container.
# shellcheck disable=SC2016 # Positional parameters expand on the compute node.
srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c '
    set -euo pipefail
    mkdir -p \
        "$1" "$2" "$3" "$4" "$5" "$6"
    chmod 700 "$4"
' bash \
    "$SQUASH_DIR" \
    "$HF_HUB_CACHE_MOUNT" \
    "$AIPERF_MMAP_CACHE_MOUNT" \
    "$XDG_RUNTIME_DIR" \
    "$XDG_CACHE_HOME" \
    "$TRITON_CACHE_DIR"

# Squash images are imported independently on each compute node and retained
# on that node's /raid. The lock only serializes imports on the selected node.
# shellcheck disable=SC2016 # Positional parameters expand on the compute node.
srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c '
    set -euo pipefail
    export TMPDIR="$4"
    exec 9>"$2"
    flock -w 1800 9 || { echo "Failed to acquire lock for $1" >&2; exit 1; }
    if unsquashfs -l "$1" >/dev/null 2>&1; then
        echo "Squash file already exists and is valid, skipping import"
    else
        rm -f "$1"
        enroot import -o "$1" "docker://$3"
    fi
' bash "$SQUASH_FILE" "$LOCK_FILE" "$IMAGE" "$COMPUTE_TMPDIR"

srun --jobid="$JOB_ID" \
    --job-name="$RUNNER_NAME" \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_MOUNT:/aiperf_mmap_cache,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    --container-writable \
    --container-remap-root \
    --container-workdir=/workspace/ \
    --no-container-entrypoint \
    --export=ALL \
    bash "benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh"
