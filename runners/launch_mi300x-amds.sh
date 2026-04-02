#!/usr/bin/env bash

export HF_HUB_CACHE_MOUNT="/home/gharunner/gharunners/hf-hub-cache/"
export PORT=8888

# Local NVMe cache for model weights (set to empty to disable)
# MI300X nodes have local NVMe; /local-nvme must be set up
# via: sudo bash utils/setup_local_nvme.sh /local-nvme
export LOCAL_MODEL_CACHE_DIR="${LOCAL_MODEL_CACHE_DIR:-}"

PARTITION="compute"

# Detect benchmark subdir from where the script lives.
# Multi-node scripts include the framework suffix (e.g. _sglang-disagg.sh);
# single-node scripts do not (e.g. dsr1_fp8_mi300x.sh).
SCRIPT_NAME_WITH_FW="${EXP_NAME%%_*}_${PRECISION}_mi300x_${FRAMEWORK}.sh"
SCRIPT_NAME_BASE="${EXP_NAME%%_*}_${PRECISION}_mi300x.sh"
if [[ -f "benchmarks/multi_node/${SCRIPT_NAME_WITH_FW}" ]]; then
    BENCHMARK_SUBDIR="multi_node"
    SCRIPT_NAME="${SCRIPT_NAME_WITH_FW}"
elif [[ -f "benchmarks/single_node/${SCRIPT_NAME_BASE}" ]]; then
    BENCHMARK_SUBDIR="single_node"
    SCRIPT_NAME="${SCRIPT_NAME_BASE}"
else
    echo "ERROR: neither benchmarks/multi_node/${SCRIPT_NAME_WITH_FW} nor benchmarks/single_node/${SCRIPT_NAME_BASE} found"
    exit 1
fi

# =============================================================================
# Multi-node disaggregated path: sbatch + Docker via submit.sh
# =============================================================================
if [[ "$BENCHMARK_SUBDIR" == "multi_node" ]]; then

    scancel_sync() {
        local jobid=$1
        local timeout=${2:-600}
        local interval=10
        local start
        start=$(date +%s)

        echo "[scancel_sync] Requesting cancel of job $jobid"
        scancel "$jobid" || true

        while [[ -n "$(squeue -j "$jobid" --noheader 2>/dev/null)" ]]; do
            local now
            now=$(date +%s)
            if (( now - start >= timeout )); then
                echo "[scancel_sync][WARN] job $jobid still present after ${timeout}s"
                return 1
            fi
            echo "[scancel_sync] waiting for job $jobid to exit. $((timeout-(now-start))) secs remaining..."
            sleep "$interval"
        done
        echo "[scancel_sync] job $jobid exited"
        return 0
    }

    set -x

    export SLURM_ACCOUNT="$USER"
    export SLURM_PARTITION="$PARTITION"
    export SLURM_JOB_NAME="benchmark-sglang-disagg.job"

    export MODEL_PATH="${HF_HUB_CACHE_MOUNT%/}"

    # MODEL_YAML_KEY: top-level key in models.yaml for server config lookup.
    if [[ -z "${MODEL_YAML_KEY:-}" ]]; then
        export MODEL_YAML_KEY="${MODEL##*/}"
    fi

    # MODEL_NAME: relative path under MODEL_PATH for --model-path inside the container.
    # Auto-resolved from HF hub cache layout so no symlink is needed.
    if [[ -z "${MODEL_NAME:-}" ]]; then
        _HF_DIR="models--$(echo "${MODEL}" | tr '/' '--')"
        _SNAPSHOT=$(ls "${MODEL_PATH}/${_HF_DIR}/snapshots/" 2>/dev/null | sort | tail -1)
        if [[ -n "${_SNAPSHOT}" ]]; then
            export MODEL_NAME="${_HF_DIR}/snapshots/${_SNAPSHOT}"
        else
            export MODEL_NAME="${MODEL_YAML_KEY}"
        fi
    fi

    export GPUS_PER_NODE=8

    export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$GITHUB_WORKSPACE/benchmark_logs}"
    mkdir -p "$BENCHMARK_LOGS_DIR"
    # NFS-safe cleanup: use timeout to avoid hanging on stale NFS locks
    timeout --kill-after=5 30 sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    JOB_ID=$(bash "benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}")

    LOG_FILE="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID}.out"

    sleep 10

    while ! ls "$LOG_FILE" &>/dev/null; do
        if ! squeue -u "$USER" --noheader --format='%i' | grep -q "$JOB_ID"; then
            echo "ERROR: Job $JOB_ID failed before creating log file"
            scontrol show job "$JOB_ID"
            exit 1
        fi
        sleep 5
    done

    set +x

    (
        while squeue -u $USER --noheader --format='%i' | grep -q "$JOB_ID"; do
            sleep 10
        done
    ) &
    POLL_PID=$!

    tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

    wait $POLL_PID

    set -x

    cat > collect_latest_results.py <<'PY'
import os, sys
sgl_job_dir, isl, osl, nexp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for path in sorted([f"{sgl_job_dir}/logs/{name}/sglang_isl_{isl}_osl_{osl}" for name in os.listdir(f"{sgl_job_dir}/logs/") if os.path.isdir(f"{sgl_job_dir}/logs/{name}/sglang_isl_{isl}_osl_{osl}")], key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

    LOGS_DIR=$(python3 collect_latest_results.py "$BENCHMARK_LOGS_DIR" "$ISL" "$OSL" 1)
    if [ -z "$LOGS_DIR" ]; then
        echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
        exit 1
    fi

    echo "Found logs directory: $LOGS_DIR"
    ls -la "$LOGS_DIR"

    for result_file in $(find $LOGS_DIR -type f); do
        file_name=$(basename $result_file)
        if [ -f $result_file ]; then
            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
            echo "Found result file ${result_file}. Copying it to ${WORKSPACE_RESULT_FILE}"
            cp $result_file $WORKSPACE_RESULT_FILE
        fi
    done

    echo "All result files processed"
    set +x
    scancel_sync $JOB_ID
    set -x
    echo "Canceled the slurm job $JOB_ID"

    # NFS-safe cleanup: use timeout to avoid hanging on stale NFS locks
    timeout --kill-after=5 30 sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        ARTIFACT_DIR="$GITHUB_WORKSPACE/benchmark_artifacts"
        mkdir -p "$ARTIFACT_DIR"
        cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$ARTIFACT_DIR/" 2>/dev/null || true
        echo "Logs copied to $ARTIFACT_DIR for artifact upload"
    fi

# =============================================================================
# Single-node path: enroot via salloc + srun
# =============================================================================
else

    SQUASH_FILE="/home/gharunner/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
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
    bash benchmarks/single_node/${SCRIPT_NAME}

    scancel $JOB_ID

fi
