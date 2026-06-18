#!/usr/bin/env bash

set -euo pipefail

export HF_HUB_CACHE_MOUNT="/raid/hf-hub-cache/"
export PORT=8888
# Pyxis intentionally filters ENROOT_* overrides during container setup. Its
# XDG defaults must point at node-local storage rather than the NFS home mount.
export XDG_DATA_HOME="/enroot/enroot-data-user-gharunner"
export XDG_CACHE_HOME="/enroot/enroot-cache-group-gharunner"

PARTITION="compute"
IMAGE_KEY="$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')"
SHARED_SQUASH_FILE="/home/gharunner/gharunners/squash/${IMAGE_KEY}.sqsh"

# Route spec-decoding=mtp configs to the _mtp benchmark script (parity with
# the h200 launchers, which have carried SPEC_SUFFIX since #392).
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
BENCHMARK_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh"

set -x

# Exclude known-bad nodes; let Slurm pick from anything else:
#   chi-mi300x-049: persistent /nvme_home disk-full
NODE_ARGS=(--exclude=chi-mi300x-049)
if [ "${PROFILE:-0}" = "1" ] && [ -n "${M3_PROFILE_NODE:-}" ]; then
    NODE_ARGS=(--nodelist="$M3_PROFILE_NODE")
fi
JOB_ID=$(
    salloc \
        --partition="$PARTITION" \
        "${NODE_ARGS[@]}" \
        --gres=gpu:"$TP" \
        --cpus-per-task=256 \
        --time=180 \
        --no-shell \
        --job-name="$RUNNER_NAME" \
        2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+'
)

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job"
    exit 1
fi

cleanup() {
    if [ -n "${JOB_ID:-}" ]; then
        scancel "$JOB_ID" || true
    fi
}
trap cleanup EXIT

if [ "${PROFILE:-0}" = "1" ]; then
    # The MI300X controller workspace and squash cache are not guaranteed to
    # be mounted on compute nodes. Stage the small checkout into node-local
    # storage and keep the image there so profiling does not depend on those
    # controller-only paths.
    PROFILE_WORKSPACE="/tmp/inferencex-profile-workspace-${JOB_ID}"
    SQUASH_DIR="/tmp/inferencex-squash"
    SQUASH_FILE="${SQUASH_DIR}/${IMAGE_KEY}.sqsh"

    srun --jobid="$JOB_ID" --chdir=/tmp bash -c '
        set -euo pipefail
        rm -rf "$1"
        mkdir -p "$1"
    ' _ "$PROFILE_WORKSPACE"
    tar \
        --exclude=.git \
        --exclude='profile_*.trace.json.gz' \
        --exclude='profile_summary_*.json' \
        -C "$GITHUB_WORKSPACE" -cf - . \
        | srun --jobid="$JOB_ID" --chdir=/tmp \
            tar -C "$PROFILE_WORKSPACE" -xf -

    # The compute nodes cannot import images because their unprivileged enroot
    # setup cannot preserve file capabilities. Reuse the controller-built
    # squash image and stage it once into the node-local cache instead.
    if ! unsquashfs -l "$SHARED_SQUASH_FILE" > /dev/null 2>&1; then
        echo "Controller squash image is missing or invalid: $SHARED_SQUASH_FILE" >&2
        exit 1
    fi
    if srun --jobid="$JOB_ID" --chdir=/tmp \
        unsquashfs -l "$SQUASH_FILE" > /dev/null 2>&1; then
        echo "Node-local squash image already exists and is valid: $SQUASH_FILE"
    else
        echo "Staging $(du -h "$SHARED_SQUASH_FILE" | cut -f1) squash image to compute node"
        cat "$SHARED_SQUASH_FILE" \
            | srun --jobid="$JOB_ID" --chdir=/tmp bash -c '
                set -euo pipefail
                squash_file="$1"
                squash_tmp="${squash_file}.tmp"
                mkdir -p "$(dirname "$squash_file")"
                rm -f "$squash_tmp"
                cat > "$squash_tmp"
                unsquashfs -l "$squash_tmp" > /dev/null
                mv -f "$squash_tmp" "$squash_file"
            ' _ "$SQUASH_FILE"
    fi

    run_status=0
    srun --jobid="$JOB_ID" \
        --chdir=/tmp \
        --container-image="$SQUASH_FILE" \
        --container-mounts="$PROFILE_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
        --no-container-mount-home \
        --container-writable \
        --container-remap-root \
        --container-workdir=/workspace/ \
        --no-container-entrypoint --export=ALL \
        bash -c '
            set -euo pipefail
            export HOME=/tmp/inferencex-container-home
            export XDG_DATA_HOME=/tmp/inferencex-container-data
            export XDG_CACHE_HOME=/tmp/inferencex-container-cache
            mkdir -p "$HOME" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
            if [[ "${MODEL_PREFIX:-}" == "minimaxm3" && "${FRAMEWORK:-}" == "vllm" ]]; then
                python3 /workspace/utils/patch_vllm_mi300x_rank0_profiler.py
            fi
            exec bash "$1"
        ' _ "$BENCHMARK_SCRIPT" || run_status=$?

    copy_status=0
    srun --jobid="$JOB_ID" --chdir=/tmp \
        tar -C "$PROFILE_WORKSPACE" -cf - . \
        | tar -C "$GITHUB_WORKSPACE" -xf - || copy_status=$?
    srun --jobid="$JOB_ID" --chdir=/tmp rm -rf "$PROFILE_WORKSPACE" || true

    if [ "$copy_status" -ne 0 ]; then
        echo "Failed to copy MI300X profile outputs back to the runner" >&2
        exit "$copy_status"
    fi
    if [ "$run_status" -ne 0 ]; then
        echo "MI300X profile container exited with status $run_status" >&2
        exit "$run_status"
    fi
else
    SQUASH_FILE="$SHARED_SQUASH_FILE"
    LOCK_FILE="${SQUASH_FILE}.lock"

    # Use flock to serialize concurrent imports to the same squash file.
    srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c "
        exec 9>\"$LOCK_FILE\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
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
        bash -c '
            exec bash "$1"
        ' _ "$BENCHMARK_SCRIPT"
fi
