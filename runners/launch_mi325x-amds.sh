#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_CACHE_MOUNT="/nfsdata/sa/gharunner/gharunners/hf-hub-cache/"

PARTITION="compute"
SQUASH_FILE="/nfsdata/sa/gharunner/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

cleanup_stale_benchmark_logs() {
    if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
        sudo -n rm -rf "$GITHUB_WORKSPACE/benchmark_logs" 2>/dev/null || \
            rm -rf "$GITHUB_WORKSPACE/benchmark_logs" 2>/dev/null || true
    fi
}
cleanup_stale_benchmark_logs

set -x

# Exclude known-broken mi325x nodes:
#   chi-mi325x-pod1-121: enroot-aufs2ovlfs setcap fails on this node's NFS-backed
#                        squash dir; container image import never completes
#                        (root-caused via #1467/#1468/#1469 sweep failures).
JOB_ID=$(set +o pipefail; salloc --partition=$PARTITION --exclude=chi-mi325x-pod1-121.ord.vultr.cpe.ice.amd.com --gres=gpu:$TP --cpus-per-task=256 --time=480 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job" >&2
    exit 1
fi

export PORT=$(( 40000 + (JOB_ID % 10000) ))

trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; cleanup_stale_benchmark_logs; exit "$rc"' EXIT

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
--container-mounts="$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE" \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi325x.sh

scancel $JOB_ID
