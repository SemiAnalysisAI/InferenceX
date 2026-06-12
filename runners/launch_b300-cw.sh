#!/usr/bin/env bash

# Launches single-node benchmarks on the b300-cw (CoreWeave B300) cluster.
# Adapted from launch_h200-cw.sh (same CoreWeave salloc/pyxis pattern) with
# the benchmark-script selection logic from launch_b300-nv.sh. The runner
# lives on the Slurm login node; jobs are scheduled onto the 8xB300 compute
# nodes via salloc/srun.
#
# Multi-node (srt-slurm/dynamo) is not wired up yet on this cluster — adapt
# the IS_MULTINODE branch of launch_b300-nv.sh when needed.

export HF_HUB_CACHE_MOUNT="/mnt/vast/hf_hub_cache"
export AIPERF_MMAP_CACHE_HOST_PATH="/mnt/vast/aiperf_mmap_cache"
export PORT=8888

if [[ "$IS_MULTINODE" == "true" ]]; then
    echo "Multi-node benchmarks are not yet supported on b300-cw." >&2
    exit 1
fi

PARTITION="b300"
SQUASH_FILE="/mnt/vast/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
# Prefer a framework-tagged script (e.g. minimaxm2.5_fp4_b300_trt.sh) so models
# with multiple inference engines can coexist; fall back to the historical
# name without an engine suffix (`_trt` for trt, bare for everyone else)
# for scripts that haven't been retagged yet.
BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_b300"
BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    LEGACY_FW_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
    BENCH_SCRIPT="${BENCH_BASE}${LEGACY_FW_SUFFIX}${SPEC_SUFFIX}.sh"
fi

if [[ -n "${BENCH_SCRIPT_OVERRIDE:-}" ]]; then
    BENCH_SCRIPT="$BENCH_SCRIPT_OVERRIDE"
fi

set -x

JOB_ID=$(salloc --partition=$PARTITION --gres=gpu:b300:$TP --time="${SALLOC_TIME_LIMIT:-180}" --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

# Use flock to serialize concurrent imports to the same squash file
srun --jobid=$JOB_ID --job-name="$RUNNER_NAME" bash -c "
    exec 9>\"$LOCK_FILE\"
    flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
    if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
        echo 'Squash file already exists and is valid, skipping import'
    else
        rm -f \"$SQUASH_FILE\"
        enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
    fi
"
CONTAINER_IMAGE=$(realpath $SQUASH_FILE)

srun --jobid=$JOB_ID \
--container-image=$CONTAINER_IMAGE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
--container-mount-home \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
bash "$BENCH_SCRIPT"

scancel $JOB_ID
