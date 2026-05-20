#!/usr/bin/env bash

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

# Exclude known-broken mi325x nodes:
#   chi-mi325x-pod1-121: enroot-aufs2ovlfs setcap fails on this node's NFS-backed
#                        squash dir; container image import never completes
#                        (root-caused via #1467/#1468/#1469 sweep failures).
export SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-chi-mi325x-pod1-021.ord.vultr.cpe.ice.amd.com,chi-mi325x-pod1-027.ord.vultr.cpe.ice.amd.com,chi-mi325x-pod1-028.ord.vultr.cpe.ice.amd.com,chi-mi325x-pod1-030.ord.vultr.cpe.ice.amd.com,chi-mi325x-pod1-121.ord.vultr.cpe.ice.amd.com}"

if [[ "$IS_MULTINODE" == "true" ]]; then
    set -x

    export SLURM_ACCOUNT="$USER"
    export SLURM_PARTITION="compute"
    export SLURM_JOB_NAME="benchmark-${FRAMEWORK}.job"

    export MODEL_NAME=${MODEL##*/}
    export MODEL_PATH="/nfsdata/sa/gharunner/gharunners/hf-hub-cache"
    export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
    export MORI_RDMA_TC=104

    export MODEL_DIR="$MODEL_PATH"
    export GPUS_PER_NODE=8

    export ISL="$ISL"
    export OSL="$OSL"

    export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$GITHUB_WORKSPACE/benchmark_logs}"
    mkdir -p "$BENCHMARK_LOGS_DIR"
    rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    cleanup_and_save_logs() {
        if [[ -n "${GITHUB_ACTIONS:-}" && -n "${JOB_ID:-}" ]]; then
            local art_dir="$GITHUB_WORKSPACE/benchmark_artifacts"
            mkdir -p "$art_dir"
            cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$art_dir/" 2>/dev/null || true
        fi
        local err_file="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID:-unknown}.err"
        if [[ -s "$err_file" ]]; then
            echo "=== Slurm job stderr ==="
            tail -100 "$err_file"
            echo "========================"
        fi
        rm -rf "$BENCHMARK_LOGS_DIR" 2>/dev/null || true
    }
    trap cleanup_and_save_logs EXIT

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_mi325x_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "sglang-disagg" ]] || [[ "$FRAMEWORK" == "vllm-disagg" ]]; then
        BENCHMARK_SUBDIR="multi_node"
    else
        BENCHMARK_SUBDIR="single_node"
    fi
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

    if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
        cat > collect_latest_results.py <<'PY'
import os, sys
sgl_job_dir, isl, osl, nexp, framework = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
for path in sorted([f"{sgl_job_dir}/logs/{name}/{framework}_isl_{isl}_osl_{osl}" for name in os.listdir(f"{sgl_job_dir}/logs/") if os.path.isdir(f"{sgl_job_dir}/logs/{name}/{framework}_isl_{isl}_osl_{osl}")], key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

        LOGS_DIR=$(python3 collect_latest_results.py "$BENCHMARK_LOGS_DIR" "$ISL" "$OSL" 1 "$FRAMEWORK")
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
    fi

    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        EVAL_DIR=$(find "$BENCHMARK_LOGS_DIR/logs" -type d -name eval_results 2>/dev/null | head -1)
        if [ -n "$EVAL_DIR" ] && [ -d "$EVAL_DIR" ]; then
            echo "Extracting eval results from $EVAL_DIR"
            shopt -s nullglob
            for eval_file in "$EVAL_DIR"/*; do
                [ -f "$eval_file" ] || continue
                cp "$eval_file" "$GITHUB_WORKSPACE/"
                echo "Copied eval artifact: $(basename "$eval_file")"
            done
            shopt -u nullglob
        else
            echo "WARNING: RUN_EVAL=true but no eval results found under $BENCHMARK_LOGS_DIR/logs"
        fi
    fi

    echo "All result files processed"
    set +x
    scancel_sync $JOB_ID
    set -x
    echo "Canceled the slurm job $JOB_ID"

    rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

else

    export HF_HUB_CACHE_MOUNT="/nfsdata/sa/gharunner/gharunners/hf-hub-cache/"
    export PORT=8888

    PARTITION="compute"
    SQUASH_FILE="/nfsdata/sa/gharunner/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    LOCK_FILE="${SQUASH_FILE}.lock"

    set -x

    EXCLUDE_OPT=()
    if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
        EXCLUDE_OPT=(--exclude "$SLURM_EXCLUDE_NODES")
    fi

    JOB_ID=$(salloc --partition=$PARTITION "${EXCLUDE_OPT[@]}" --gres=gpu:$TP --cpus-per-task=256 --time=480 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

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

    scancel $JOB_ID
fi
