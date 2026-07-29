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

    export MODEL_NAME=${MODEL##*/}
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

    # Ensure root-owned files are cleaned up even on early exit to prevent
    # EACCES errors when the next GH Actions job checks out on this runner.
    # Always preserve slurm logs as CI artifacts for debugging.
    # KEEP_LOGS=1 disables the trap entirely (local-debug knob).
    cleanup_and_save_logs() {
        if [[ -n "${GITHUB_ACTIONS:-}" && -n "${JOB_ID:-}" ]]; then
            local art_dir="$GITHUB_WORKSPACE/benchmark_artifacts"
            mkdir -p "$art_dir"
            cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$art_dir/" 2>/dev/null || true
        fi
        # Print .err inline so failures are visible in CI output
        local err_file="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID:-unknown}.err"
        if [[ -s "$err_file" ]]; then
            echo "=== Slurm job stderr ==="
            tail -100 "$err_file"
            echo "========================"
        fi
        sudo rm -rf "$BENCHMARK_LOGS_DIR" 2>/dev/null || true
    }
    if [[ "${KEEP_LOGS:-0}" == "1" ]]; then
        trap '' EXIT
    else
        trap cleanup_and_save_logs EXIT
    fi

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_mi355x_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "sglang-disagg" ]] || [[ "$FRAMEWORK" == "vllm-disagg" ]] || [[ "$FRAMEWORK" == "atom-disagg" ]]; then
        # Agentic recipes live under multi_node/agentic/ and export the
        # HiCache tunables (page-size, io-backend, ...); fixed-seq-len recipes
        # live at the multi_node/ root. Honor SCENARIO_SUBDIR so agentic-coding
        # configs pick the agentic recipe instead of the root one.
        if [[ "${SCENARIO_SUBDIR}" == "agentic/" ]]; then
            BENCHMARK_SUBDIR="multi_node/agentic"
        else
            BENCHMARK_SUBDIR="multi_node"
        fi
    else
        BENCHMARK_SUBDIR="single_node/fixed_seq_len"
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

    if [[ "${EVAL_ONLY:-false}" != "true" && "${IS_AGENTIC:-0}" != "1" ]]; then
        cat > collect_latest_results.py <<'PY'
import os, sys
job_dir, isl, osl, nexp, framework = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
logs_root = f"{job_dir}/logs/"
candidates = []
if os.path.isdir(logs_root):
    for name in os.listdir(logs_root):
        subdir = f"{logs_root}{name}/{framework}_isl_{isl}_osl_{osl}"
        if os.path.isdir(subdir):
            candidates.append(subdir)
for path in sorted(candidates, key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

        LOGS_DIR=$(python3 collect_latest_results.py "$BENCHMARK_LOGS_DIR" "$ISL" "$OSL" 1 "$FRAMEWORK")
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
    fi

    # Extract eval results if eval was requested
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        # Find eval_results in the slurm job logs directory
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

    # Stage agentic raw artifacts + server logs for the CI upload steps.
    # server_sglang.sh copies /run_logs/slurm_job-<id> to
    # $BENCHMARK_LOGS_DIR/logs/slurm_job-<id> on shared storage, and
    # trace_replay.sh writes each concurrency's aiperf artifacts under
    # agentic/conc_<N>/ (mirroring agentic_srt.sh). benchmark-multinode-tmpl.yml
    # uploads them from $GITHUB_WORKSPACE/LOGS/agentic/conc_*/... plus a
    # multinode_server_logs.tar.gz, so preserve the conc_<N>/ nesting here
    # before the logs dir is removed below. The agg result JSON is already
    # written straight to the mounted workspace by the existing agentic
    # aggregation module.
    if [[ "${IS_AGENTIC:-0}" == "1" ]]; then
        JOB_LOGS_DIR="$BENCHMARK_LOGS_DIR/logs/slurm_job-${JOB_ID}"
        if [ -d "$JOB_LOGS_DIR" ]; then
            # trace_replay.sh always nests artifacts under agentic/conc_<N>/.
            # Copy the whole agentic/ tree so the conc_<N>/ subdirs are
            # preserved for the LOGS/agentic/conc_*/... upload globs.
            AGENTIC_SRC="$JOB_LOGS_DIR/agentic"
            if [ -d "$AGENTIC_SRC" ] && find "$AGENTIC_SRC" -mindepth 1 -maxdepth 1 -type d -name 'conc_*' -print -quit 2>/dev/null | grep -q .; then
                echo "Staging agentic raw artifacts from $AGENTIC_SRC"
                mkdir -p "$GITHUB_WORKSPACE/LOGS/agentic"
                cp -r "$AGENTIC_SRC"/. "$GITHUB_WORKSPACE/LOGS/agentic/"
                # The source artifacts are created inside the container as root
                # (--container-remap-root), so depending on how the runner
                # invokes this script the copies can end up root-owned and/or
                # read-only (aiperf/server_sglang make some dirs mode 0555). If
                # the staged tree isn't owned+writable by the runner user, the
                # next checkout's `git clean` fails with
                #   EACCES: permission denied, rmdir '.../LOGS/agentic'.
                # chown to the invoking user (the same one that runs git clean)
                # via sudo (already passwordless here for rm -rf). The follow-up
                # chmod uses a+rwX (not just u+rwX): the *next* job against this
                # same $GITHUB_WORKSPACE may be picked up by a different runner
                # process running as a different OS user, which would otherwise
                # fall outside the owner bits and still fail the same
                # `git clean` with EACCES despite the chown above.
                sudo chown -R "$(id -u):$(id -g)" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                chmod -R a+rwX "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                ls -laR "$GITHUB_WORKSPACE/LOGS/agentic"
            else
                echo "WARNING: no agentic conc_*/ artifacts found under $JOB_LOGS_DIR/agentic"
            fi
            # Server/router/prefill/decode logs for the multinode_server_logs_* artifact.
            if tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$JOB_LOGS_DIR" . 2>/dev/null; then
                echo "Created multinode_server_logs.tar.gz"
            else
                echo "WARNING: failed to create multinode_server_logs.tar.gz"
            fi
        else
            echo "WARNING: agentic staging skipped; $JOB_LOGS_DIR not found"
        fi
    fi

    echo "All result files processed"
    # Use sync scancel to ensure nfs file handle is released in time
    set +x
    scancel_sync $JOB_ID
    set -x
    echo "Canceled the slurm job $JOB_ID"

    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    # Log preservation and cleanup handled by EXIT trap (cleanup_and_save_logs)

else

    export HF_HUB_CACHE_MOUNT="/var/lib/hf-hub-cache/"
    export AIPERF_MMAP_CACHE_HOST_PATH="/it-share/aiperf-cache/"
    export PORT_OFFSET=${RUNNER_NAME: -1}
    export PORT=$(( 8888 + ${PORT_OFFSET} ))
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

    PARTITION="compute"
    SQUASH_FILE="/var/lib/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    LOCK_FILE="${SQUASH_FILE}.lock"

    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    set -x
    salloc --partition=$PARTITION --gres=gpu:$GPU_COUNT --exclusive --cpus-per-task=128 --time=500 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)

    srun --jobid=$JOB_ID bash -c "docker stop \$(docker ps -a -q)"

    # Use flock to serialize concurrent imports to the same squash file
    srun --jobid=$JOB_ID bash -c "
        exec 9>\"$LOCK_FILE\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
        if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import'
        else
            rm -f \"$SQUASH_FILE\"
            enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
        fi
    "

    export VLLM_CACHE_ROOT="/it-share/gharunners/.cache/vllm"
        #--container-mount-home \

    if [[ "$FRAMEWORK" == "atom" ]] || [[ "$FRAMEWORK" == "sglang" ]]; then
        SLRUM_HOME_MOUNT=""
    else
        SLRUM_HOME_MOUNT=" --container-mount-home "
    fi

    # to prevent reading outdated saved model. use a fresh model from hf repo
    if [[ ("$FRAMEWORK" == "vllm" || "$FRAMEWORK" == "atom") ]] && [[ "$MODEL" == "deepseek-ai/DeepSeek-V4-Pro" ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    # MiniMax-M3 weights are not staged on the node-local /var/lib NVMe cache;
    # they are pre-downloaded once to the NFS share instead. Covers both the
    # MiniMaxAI MXFP8 checkpoint and the amd MXFP4 atom checkpoint.
    if [[ "$MODEL" == MiniMaxAI/MiniMax-M3* || "$MODEL" == amd/MiniMax-M3* ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_mi355x"
    SCRIPT_FW="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    SCRIPT_FALLBACK="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
    if [[ -f "$SCRIPT_FW" ]]; then
        BENCHMARK_SCRIPT="$SCRIPT_FW"
    else
        BENCHMARK_SCRIPT="$SCRIPT_FALLBACK"
    fi

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        $SLRUM_HOME_MOUNT \
        --container-writable \
        --container-workdir=/workspace/ \
        --container-remap-root \
        --no-container-entrypoint --export=ALL,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash "$BENCHMARK_SCRIPT"

    scancel $JOB_ID

    if ls gpucore.* 1> /dev/null 2>&1; then
        echo "gpucore files exist. not good"
        rm -f gpucore.*
    fi
fi
