#!/usr/bin/env bash

# Poll until the SLURM job really exits so the NFS-backed $GITHUB_WORKSPACE
# mount flushes all of the container's writes back to the actions runner
# before the host-side status.txt check runs. A bare `scancel` returns
# immediately and leaves results/* invisible to the workflow.
scancel_sync() {
    local jobid=$1
    local timeout=${2:-600}
    local interval=5
    local start
    start=$(date +%s)
    scancel "$jobid" || true
    while [[ -n "$(squeue -j "$jobid" --noheader 2>/dev/null)" ]]; do
        local now
        now=$(date +%s)
        if (( now - start >= timeout )); then
            echo "[scancel_sync][WARN] job $jobid still present after ${timeout}s"
            return 1
        fi
        echo "[scancel_sync] waiting for job $jobid to exit ($((timeout-(now-start)))s remaining)"
        sleep "$interval"
    done
    echo "[scancel_sync] job $jobid exited"
    return 0
}

export HF_HUB_CACHE_MOUNT="/nfsdata/sa/gharunner/gharunners/hf-hub-cache/"
export PORT=8888

PARTITION="compute"
SQUASH_FILE="/nfsdata/sa/gharunner/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

set -x

JOB_ID=$(salloc --partition=$PARTITION --gres=gpu:$TP --cpus-per-task=256 --time=480 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

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
srun --jobid=$JOB_ID \
--container-image=$SQUASH_FILE \
--container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi325x.sh

set +x
scancel_sync $JOB_ID
set -x
