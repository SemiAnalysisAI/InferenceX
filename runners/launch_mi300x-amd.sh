#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars IS_MULTINODE
set -eo pipefail

export HF_HUB_CACHE_MOUNT="/raid/inferencex/models/hub"
export AIPERF_MMAP_CACHE_MOUNT="/raid/inferencex/aiperf-mmap-cache"
export AIPERF_DATASET_MMAP_CACHE_DIR="/aiperf_mmap_cache"

PARTITION="compute"
SQUASH_FILE="/raid/inferencex/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

# DSv4.1 Flash AgentX creates runtime directories next to the repository, which
# must not land under /workspace; mount the checkout at /ix like the other
# dsv41flash launchers and rewrite the caller's RESULT_DIR to match.
CONTAINER_REPO=/workspace
if [[ "$MODEL" == "deepseek-ai/DeepSeek-V4.1-Flash" ]]; then
    CONTAINER_REPO=/ix
    export INFMAX_CONTAINER_WORKSPACE="$CONTAINER_REPO"
    case "${RESULT_DIR:-}" in
        /workspace/*) export RESULT_DIR="/ix/${RESULT_DIR#/workspace/}" ;;
    esac
fi

# A cold 511 GB checkpoint download plus a one-hour AgentX arm does not fit the
# 180-minute default allocation; give DSv4.1 Flash the MI325X launcher's 480.
SALLOC_TIME=180
[[ "$CONTAINER_REPO" == /ix ]] && SALLOC_TIME=480

check_env_vars GPU_COUNT

set -x

JOB_ID=$(set +o pipefail; salloc \
    --partition="$PARTITION" \
    --gres="gpu:$GPU_COUNT" \
    --cpus-per-task=128 \
    --time="$SALLOC_TIME" \
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

# Concurrent jobs import to the same node-local squash file; serialize them.
srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c "
    set -eo pipefail
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
    --container-mounts="$GITHUB_WORKSPACE:$CONTAINER_REPO/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_MOUNT:/aiperf_mmap_cache,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    --container-writable \
    --container-remap-root \
    --container-workdir="$CONTAINER_REPO/" \
    --no-container-entrypoint \
    --export=ALL \
    bash "benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh"

scancel "$JOB_ID"
