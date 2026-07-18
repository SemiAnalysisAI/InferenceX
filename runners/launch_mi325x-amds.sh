#!/usr/bin/env bash
set -euo pipefail

if [[ "${IS_MULTINODE:-false}" == "true" ]]; then
    exec bash "$(dirname "$0")/launch_mi325x-amds_multinode.sh"
fi

export HF_HUB_CACHE_MOUNT="/raid/hf-hub-cache/"

PARTITION="compute"
SQUASH_FILE="/raid/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

# Route spec-decoding=mtp configs to the _mtp benchmark script (parity with
# the h200 launchers, which have carried SPEC_SUFFIX since #392).
SPEC_SUFFIX=$([[ "${SPEC_DECODING:-}" == "mtp" ]] && printf '_mtp' || printf '')

export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

set -x

JOB_ID=$(set +o pipefail; salloc --partition=$PARTITION --gres=gpu:$GPU_COUNT --cpus-per-task=256 --time=480 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job" >&2
    exit 1
fi

export PORT=$(( 40000 + (JOB_ID % 10000) ))
export XDG_CACHE_HOME="/tmp/xdg-cache-$JOB_ID"
export TRITON_CACHE_DIR="/tmp/triton-cache-$JOB_ID"

trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT

# Use flock to serialize concurrent imports to the same squash file
srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c "
    set -euo pipefail
    exec 9>\"$LOCK_FILE\"
    flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE' >&2; exit 1; }
    if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
        echo 'Squash file already exists and is valid, skipping import'
    else
        rm -f \"$SQUASH_FILE\"
        enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
    fi
"
srun --jobid="$JOB_ID" \
--container-image="$SQUASH_FILE" \
--container-mounts="$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi325x${SPEC_SUFFIX}.sh

scancel $JOB_ID
