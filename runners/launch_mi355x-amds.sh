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

if [[ "$IS_MULTINODE" == "true" ]]; then
    # This sets up the environment and launches multi-node benchmarks

    set -x

    # Set up environment variables for SLURM
    export SLURM_ACCOUNT="$USER"
    export SLURM_PARTITION="compute"
    export SLURM_JOB_NAME="benchmark-sglang-disagg.job"

    export MODEL_NAME="DeepSeek-R1"
    export MODEL_PATH="/it-share/data"
    export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
    export MORI_RDMA_TC=104

    # Set additional required env vars for multi_node scripts
    export MODEL_DIR="$MODEL_PATH"  # job.slurm uses MODEL_DIR
    export GPUS_PER_NODE=8          # MI355X has 8 GPUs (set to 4 for MI325X)

    export ISL="$ISL"
    export OSL="$OSL"

    # Logs go to BENCHMARK_LOGS_DIR (NFS-accessible, outside the repo tree)
    export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$GITHUB_WORKSPACE/benchmark_logs}"
    mkdir -p "$BENCHMARK_LOGS_DIR"
    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_mi355x_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "sglang-disagg" ]]; then
        BENCHMARK_SUBDIR="multi_node"
    else
        BENCHMARK_SUBDIR="single_node"
    fi
    JOB_ID=$(bash "benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}")

    # Wait for job to complete
    LOG_FILE="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID}.out"

    # Give slurm time to start the job and create log file
    sleep 10

    # Wait for log file to appear (also check job is still alive)
    while ! ls "$LOG_FILE" &>/dev/null; do
        if ! squeue -u "$USER" --noheader --format='%i' | grep -q "$JOB_ID"; then
            echo "ERROR: Job $JOB_ID failed before creating log file"
            scontrol show job "$JOB_ID"
            exit 1
        fi
        sleep 5
    done

    set +x

    # Poll for job completion in background
    (
        while squeue -u $USER --noheader --format='%i' | grep -q "$JOB_ID"; do
            sleep 10
        done
    ) &
    POLL_PID=$!

    # Tail the log file until job completes (-F follows by name, polls instead of inotify for NFS)
    tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

    wait $POLL_PID

    set -x

    # FIXME: The below is bad and is a result of the indirection of the ways in which
    # Dynamo jobs are launched. In a follow-up PR, the location of the result file should not
    # depend on the runner, it should always be in the same spot in the GH workspace.

    # Process results from all configurations

    # search for "FRAMEWORK_DIFF_IF_STATEMENT #3" for this if-statement
    # Find the latest log directory that contains the data

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

    # Result JSON are contained within the result directory
    for result_file in $(find $LOGS_DIR -type f); do
        # result_file should directly be isl_ISL_osl_OSL_concurrency_CONC_req_rate_R_gpus_N_ctx_M_gen_N.json
        file_name=$(basename $result_file)
        if [ -f $result_file ]; then
            # Copy the result file to workspace with a unique name
            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
            echo "Found result file ${result_file}. Copying it to ${WORKSPACE_RESULT_FILE}"
            cp $result_file $WORKSPACE_RESULT_FILE
        fi
    done

    # Extract eval results if eval was requested
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        EVAL_DIR="$(dirname "$LOGS_DIR")/eval_results"
        if [ -d "$EVAL_DIR" ]; then
            echo "Extracting eval results from $EVAL_DIR"
            for eval_file in "$EVAL_DIR"/*; do
                [ -f "$eval_file" ] && cp "$eval_file" "$GITHUB_WORKSPACE/"
                echo "Copied eval artifact: $(basename "$eval_file")"
            done
        else
            echo "WARNING: RUN_EVAL=true but no eval results found at $EVAL_DIR"
        fi
    fi

    echo "All result files processed"
    # Use sync scancel to ensure nfs file handle is released in time
    set +x
    scancel_sync $JOB_ID
    set -x
    echo "Canceled the slurm job $JOB_ID"

    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    # Upload logs as artifact if running in GitHub Actions
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        ARTIFACT_DIR="$GITHUB_WORKSPACE/benchmark_artifacts"
        mkdir -p "$ARTIFACT_DIR"
        cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$ARTIFACT_DIR/" 2>/dev/null || true
        echo "Logs copied to $ARTIFACT_DIR for artifact upload"
    fi

else

    export HF_HUB_CACHE_MOUNT="/var/lib/hf-hub-cache/"
    export PORT_OFFSET=${RUNNER_NAME: -1}
    export PORT=$(( 8888 + ${PORT_OFFSET} ))
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

    PARTITION="compute"
    SQUASH_FILE="/var/lib/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    set -x
    salloc --partition=$PARTITION --gres=gpu:$TP --cpus-per-task=128 --time=180 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)

    srun --jobid=$JOB_ID bash -c "docker stop \$(docker ps -a -q)"

    if [[ "$FRAMEWORK" == "atom" ]]; then
        srun --jobid=$JOB_ID bash -c "rm $SQUASH_FILE"
    fi

    srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
    if ! srun --jobid=$JOB_ID bash -c "unsquashfs -l $SQUASH_FILE > /dev/null"; then
        echo "unsquashfs failed, removing $SQUASH_FILE and re-importing..."
        srun --jobid=$JOB_ID bash -c "rm -f $SQUASH_FILE"
        srun --jobid=$JOB_ID bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
    fi

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE \
        --container-mount-home \
        --container-writable \
        --container-workdir=/workspace/ \
        --no-container-entrypoint --export=ALL \
        bash benchmarks/single_node/${EXP_NAME%%_*}_${PRECISION}_mi355x${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh

    scancel $JOB_ID

    if ls gpucore.* 1> /dev/null 2>&1; then
        echo "gpucore files exist. not good"
        rm -f gpucore.*
    fi
fi
