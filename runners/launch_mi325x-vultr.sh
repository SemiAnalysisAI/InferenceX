#!/usr/bin/env bash
set -euo pipefail

# Pre-staged model weights / HF hub cache for the vultr mi325x fleet. Bind-mounted
# over the container-side HF_HUB_CACHE (/mnt/hf_hub_cache/); the bench scripts'
# `hf download "$MODEL"` resolves against the models--org--name caches already
# staged here (e.g. DeepSeek-R1-0528, Qwen3.5-397B-A17B-FP8, GLM-5-FP8) so weights
# are not re-downloaded from HF in CI.
export HF_HUB_CACHE_MOUNT="/nfsdata/sa/models/"

# enroot cache (import layer cache + the imported .sqsh images) for this fleet.
# Node-local ext4 present at the same path on every compute node; import and run
# happen in the same Slurm job on a single node, so node-local storage suffices.
export ENROOT_CACHE_PATH="/enroot/sa"
mkdir -p "$ENROOT_CACHE_PATH"

PARTITION="compute"
SQUASH_FILE="$ENROOT_CACHE_PATH/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
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
#   chi-mi325x-pod1-121: has a history of failing enroot container image import
#                        (root-caused via #1467/#1468/#1469 sweep failures);
#                        excluded for the same reason as the amds fleet.
#   chi-mi325x-pod1-027: fails SLURM resume/boot — salloc grants an allocation then
#                        relinquishes it with "Something is wrong with the boot of the
#                        nodes" (run 27454108525), which gated the whole sweep at the
#                        canary; excluded until the node is repaired.
JOB_ID=$(set +o pipefail; salloc --partition=$PARTITION --exclude=chi-mi325x-pod1-121.ord.vultr.cpe.ice.amd.com,chi-mi325x-pod1-027.ord.vultr.cpe.ice.amd.com --gres=gpu:$TP --cpus-per-task=256 --time=480 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

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
