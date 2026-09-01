#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_CACHE_MOUNT="/raid/inferencex/models/hub"
export AIPERF_MMAP_CACHE_MOUNT="/raid/inferencex/aiperf-mmap-cache"
export AIPERF_DATASET_MMAP_CACHE_DIR="/aiperf_mmap_cache"

PARTITION="compute"
SQUASH_FILE="/raid/inferencex/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

set -x

JOB_ID=$(set +o pipefail; salloc \
    --partition="$PARTITION" \
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
trap 'scancel "$JOB_ID" 2>/dev/null || true' EXIT

# Use flock to serialize concurrent imports to the same node-local squash file.
srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c "
    set -euo pipefail
    exec 9>\"$LOCK_FILE\"
    flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE' >&2; exit 1; }
    if unsquashfs -l \"$SQUASH_FILE\" >/dev/null 2>&1; then
        echo 'Squash file already exists and is valid, skipping import'
    else
        rm -f \"$SQUASH_FILE\"
        enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
    fi
"

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

scancel "$JOB_ID"
