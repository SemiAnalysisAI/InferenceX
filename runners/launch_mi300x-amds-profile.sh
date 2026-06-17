#!/usr/bin/env bash

set -euo pipefail

export HF_HUB_CACHE_MOUNT="/raid/hf-hub-cache/"
export PORT=8888

PARTITION="compute"
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
BENCHMARK_SCRIPT="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh"
SOURCE_WORKSPACE="$GITHUB_WORKSPACE"
JOB_ID=""

cleanup() {
    if [ -n "$JOB_ID" ]; then
        scancel "$JOB_ID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

JOB_ID="$(
    salloc \
        --partition="$PARTITION" \
        --nodelist=chi-mi300x-057 \
        --gres="gpu:$TP" \
        --cpus-per-task=256 \
        --time=180 \
        --no-shell \
        --job-name="$RUNNER_NAME" 2>&1 \
        | tee /dev/stderr \
        | grep -oP 'Granted job allocation \K[0-9]+'
)"
if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job" >&2
    exit 1
fi

# The Actions controller workspace is not mounted on compute nodes. Pick a
# node-local writable root and stream the checked-out source through Slurm.
REMOTE_BASE="$(
    srun --quiet --jobid="$JOB_ID" --ntasks=1 --chdir=/tmp bash -c '
        set -e
        for parent in /raid/gharunner /raid /tmp; do
            candidate="${parent}/inferencex-profile-${UID}"
            if mkdir -p "$candidate" 2>/dev/null && test -w "$candidate"; then
                printf "%s\n" "$candidate"
                exit 0
            fi
        done
        echo "No writable profile storage found" >&2
        exit 1
    '
)"
REMOTE_WORKSPACE="${REMOTE_BASE}/workspace-${JOB_ID}"
REMOTE_SQUASH_DIR="${REMOTE_BASE}/squash"
REMOTE_CACHE_DIR="${REMOTE_BASE}/enroot-cache"
REMOTE_DATA_DIR="${REMOTE_BASE}/enroot-data"
REMOTE_RUNTIME_DIR="${REMOTE_BASE}/enroot-runtime"
REMOTE_TEMP_DIR="${REMOTE_BASE}/enroot-temp"
REMOTE_HOME="${REMOTE_BASE}/home"
SQUASH_FILE="${REMOTE_SQUASH_DIR}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"

srun --quiet --jobid="$JOB_ID" --ntasks=1 --chdir=/tmp \
    mkdir -p \
        "$REMOTE_WORKSPACE" \
        "$REMOTE_SQUASH_DIR" \
        "$REMOTE_CACHE_DIR" \
        "$REMOTE_DATA_DIR" \
        "$REMOTE_RUNTIME_DIR" \
        "$REMOTE_TEMP_DIR" \
        "$REMOTE_HOME"
tar --exclude='.git' -C "$SOURCE_WORKSPACE" -cf - . \
    | srun --quiet --jobid="$JOB_ID" --ntasks=1 --chdir=/tmp \
        tar -xf - -C "$REMOTE_WORKSPACE"

srun --quiet --jobid="$JOB_ID" --ntasks=1 --chdir=/tmp bash -c "
    set -euo pipefail
    export HOME=\"$REMOTE_HOME\"
    export XDG_CACHE_HOME=\"$REMOTE_BASE/cache\"
    export ENROOT_CACHE_PATH=\"$REMOTE_CACHE_DIR\"
    export ENROOT_DATA_PATH=\"$REMOTE_DATA_DIR\"
    export ENROOT_RUNTIME_PATH=\"$REMOTE_RUNTIME_DIR\"
    export ENROOT_TEMP_PATH=\"$REMOTE_TEMP_DIR\"
    mkdir -p \
        \"\$HOME\" \
        \"\$XDG_CACHE_HOME\" \
        \"\$ENROOT_CACHE_PATH\" \
        \"\$ENROOT_DATA_PATH\" \
        \"\$ENROOT_RUNTIME_PATH\" \
        \"\$ENROOT_TEMP_PATH\"
    exec 9>\"$LOCK_FILE\"
    flock -w 600 9 || {
        echo 'Failed to acquire lock for $SQUASH_FILE' >&2
        exit 1
    }
    if unsquashfs -l \"$SQUASH_FILE\" >/dev/null 2>&1; then
        echo 'Squash file already exists and is valid, skipping import'
    else
        rm -f \"$SQUASH_FILE\"
        enroot import -o \"$SQUASH_FILE\" \"docker://$IMAGE\"
    fi
"

set +e
srun --jobid="$JOB_ID" \
    --ntasks=1 \
    --chdir=/tmp \
    --container-image="$SQUASH_FILE" \
    --container-mounts="$REMOTE_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    --no-container-mount-home \
    --container-remap-root \
    --container-workdir=/workspace/ \
    --no-container-entrypoint \
    --export=ALL \
    bash -c '
        overlay=/workspace/.vllm-profile-overlay
        rm -rf "$overlay"
        mkdir -p "$overlay"
        vllm_package="$(
            python3 -c "from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parent)"
        )"
        cp -a "$vllm_package" "$overlay/"
        export PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
        if [[ "${PROFILE:-0}" == "1" && "${MODEL_PREFIX:-}" == "minimaxm3" && "${FRAMEWORK:-}" == "vllm" ]]; then
            python3 /workspace/utils/patch_vllm_mi300x_rank0_profiler.py
        fi
        exec bash "$1"
    ' _ "$BENCHMARK_SCRIPT"
run_status=$?
set -e

artifacts=(
    "${RESULT_FILENAME}.json"
    "profile_${RESULT_FILENAME}.trace.json.gz"
    "server.log"
    "gpu_metrics.csv"
)
if [ -n "${MOE_DEBUG_LOG:-}" ]; then
    artifacts+=("$MOE_DEBUG_LOG")
fi
for artifact in "${artifacts[@]}"; do
    artifact_name="${artifact##*/}"
    remote_path="${REMOTE_WORKSPACE}/${artifact_name}"
    if srun --quiet --jobid="$JOB_ID" --ntasks=1 --chdir=/tmp \
        test -f "$remote_path"; then
        srun --quiet --jobid="$JOB_ID" --ntasks=1 --chdir=/tmp \
            cat "$remote_path" > "${SOURCE_WORKSPACE}/${artifact_name}"
    fi
done

srun --quiet --jobid="$JOB_ID" --ntasks=1 --chdir=/tmp \
    rm -rf "$REMOTE_WORKSPACE" || true
exit "$run_status"
