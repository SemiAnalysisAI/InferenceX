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

    echo "=== InferenceX runner identity ==="
    echo "github.sha=${GITHUB_SHA:-unset}"
    echo "git rev-parse HEAD=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git status --short:"
    git status --short 2>/dev/null || true
    echo "git diff --stat HEAD:"
    git diff --stat HEAD 2>/dev/null || true
    echo "CONTAINER_IMAGE=${IMAGE:-${CONTAINER_IMAGE:-unset}}"
    echo "FRAMEWORK=${FRAMEWORK:-unset} MODEL=${MODEL:-unset} RESULT_FILENAME=${RESULT_FILENAME:-unset}"
    echo "=================================="

    # Ensure root-owned files are cleaned up even on early exit to prevent
    # EACCES errors when the next GH Actions job checks out on this runner.
    # Always preserve slurm logs as CI artifacts for debugging.
    # KEEP_LOGS=1 disables the trap entirely (local-debug knob).
    collect_multinode_debug_logs() {
        [[ -n "${GITHUB_ACTIONS:-}" ]] || return 0
        local art_dir="$GITHUB_WORKSPACE/benchmark_artifacts"
        mkdir -p "$art_dir"

        if [[ -n "${JOB_ID:-}" ]]; then
            cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$art_dir/" 2>/dev/null || true

            local stdout_path stderr_path
            stdout_path=$(scontrol show job "$JOB_ID" 2>/dev/null | awk -F= '/StdOut=/{print $2; exit}')
            stderr_path=$(scontrol show job "$JOB_ID" 2>/dev/null | awk -F= '/StdErr=/{print $2; exit}')
            [[ -n "$stdout_path" ]] && cp -f "$stdout_path" "$art_dir/" 2>/dev/null || true
            [[ -n "$stderr_path" ]] && cp -f "$stderr_path" "$art_dir/" 2>/dev/null || true
        fi

        if [[ -d "$BENCHMARK_LOGS_DIR/logs" ]]; then
            mkdir -p "$art_dir/logs"
            cp -a "$BENCHMARK_LOGS_DIR/logs/." "$art_dir/logs/" 2>/dev/null || true
        fi

        find "$BENCHMARK_LOGS_DIR" -maxdepth 5 -type f \( \
            -name 'server_*.log' -o \
            -name 'prefill_*.log' -o \
            -name 'decode_*.log' -o \
            -name 'lmcache_*.log' -o \
            -name 'mc_lmcache_pd_proxy.log' \
        \) -exec cp -f {} "$art_dir/" \; 2>/dev/null || true
    }

    dump_missing_slurm_log_debug() {
        local jobid="$1"
        local expected_log="$2"
        echo "ERROR: Job $jobid ended before expected Slurm stdout appeared: $expected_log" >&2
        echo "=== squeue for current user ===" >&2
        squeue -u "$USER" 2>&1 || true
        echo "=== scontrol show job $jobid ===" >&2
        scontrol show job "$jobid" 2>&1 || true
        echo "=== sacct job $jobid ===" >&2
        sacct -j "$jobid" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList%80 -P 2>&1 || true
        echo "=== benchmark log tree: $BENCHMARK_LOGS_DIR ===" >&2
        find "$BENCHMARK_LOGS_DIR" -maxdepth 6 -type f -printf '%p %s bytes\n' 2>/dev/null | sort >&2 || true

        local stdout_path stderr_path
        stdout_path=$(scontrol show job "$jobid" 2>/dev/null | awk -F= '/StdOut=/{print $2; exit}')
        stderr_path=$(scontrol show job "$jobid" 2>/dev/null | awk -F= '/StdErr=/{print $2; exit}')
        for log_path in "$expected_log" "$stdout_path" "$stderr_path"; do
            [[ -n "$log_path" ]] || continue
            echo "=== tail $log_path ===" >&2
            if [[ -s "$log_path" ]]; then
                tail -n 240 "$log_path" >&2 || true
            else
                ls -l "$log_path" >&2 || true
            fi
        done

        shopt -s globstar nullglob
        for server_log in \
            "$BENCHMARK_LOGS_DIR"/logs/slurm_job-${jobid}/*.log \
            "$BENCHMARK_LOGS_DIR"/logs/slurm_job-${jobid}/**/*.log \
            "$BENCHMARK_LOGS_DIR"/server_*.log \
            "$BENCHMARK_LOGS_DIR"/prefill_*.log \
            "$BENCHMARK_LOGS_DIR"/decode_*.log \
            "$BENCHMARK_LOGS_DIR"/lmcache_*.log; do
            [[ -f "$server_log" ]] || continue
            echo "=== server tail $server_log ===" >&2
            tail -n 240 "$server_log" >&2 || true
        done
        shopt -u globstar nullglob
    }

    cleanup_and_save_logs() {
        collect_multinode_debug_logs
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
    BENCHMARK_SCRIPT="benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}"
    if [[ ! -f "$BENCHMARK_SCRIPT" && "$BENCHMARK_SUBDIR" == "multi_node/agentic" ]]; then
        FALLBACK_SCRIPT="benchmarks/multi_node/${SCRIPT_NAME}"
        if [[ -f "$FALLBACK_SCRIPT" ]]; then
            echo "WARNING: $BENCHMARK_SCRIPT not found; falling back to $FALLBACK_SCRIPT" >&2
            BENCHMARK_SCRIPT="$FALLBACK_SCRIPT"
        fi
    fi
    if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
        echo "ERROR: benchmark script not found: $BENCHMARK_SCRIPT" >&2
        exit 1
    fi
    if ! JOB_ID=$(bash "$BENCHMARK_SCRIPT"); then
        echo "ERROR: benchmark script failed before returning a Slurm job id: $BENCHMARK_SCRIPT" >&2
        exit 1
    fi
    if [[ -z "${JOB_ID//[[:space:]]/}" ]]; then
        echo "ERROR: benchmark script returned an empty Slurm job id: $BENCHMARK_SCRIPT" >&2
        exit 1
    fi

    # Wait for job to complete
    LOG_FILE="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID}.out"

    # Give slurm time to start the job and create log file
    sleep 10

    # Wait for log file to appear (also check job is still alive)
    while ! ls "$LOG_FILE" &>/dev/null; do
        if ! squeue -u "$USER" --noheader --format='%i' | grep -q "$JOB_ID"; then
            dump_missing_slurm_log_debug "$JOB_ID" "$LOG_FILE"
            collect_multinode_debug_logs
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
    # written straight to the mounted workspace by process_agentic_result.py.
    if [[ "${IS_AGENTIC:-0}" == "1" ]]; then
        JOB_LOGS_DIR="$BENCHMARK_LOGS_DIR/logs/slurm_job-${JOB_ID}"
        if [ -d "$JOB_LOGS_DIR" ]; then
            shopt -s nullglob
            for result_file in "$JOB_LOGS_DIR"/workspace_artifacts/*.json "$JOB_LOGS_DIR"/*.json; do
                [ -f "$result_file" ] || continue
                echo "Staging agentic result JSON from $result_file"
                cp "$result_file" "$GITHUB_WORKSPACE/"
            done
            shopt -u nullglob

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
                # via sudo (already passwordless here for rm -rf), then force it
                # writable so it always stays cleanable.
                sudo chown -R "$(id -u):$(id -g)" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                chmod -R u+rwX "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                ls -laR "$GITHUB_WORKSPACE/LOGS/agentic"
            elif [ -d "$JOB_LOGS_DIR/LOGS/agentic" ]; then
                echo "Staging vLLM agentic raw artifacts from $JOB_LOGS_DIR/LOGS/agentic"
                mkdir -p "$GITHUB_WORKSPACE/LOGS/agentic"
                cp -r "$JOB_LOGS_DIR/LOGS/agentic/." "$GITHUB_WORKSPACE/LOGS/agentic/"
                mkdir -p "$GITHUB_WORKSPACE/LOGS/agentic/conc_${CONC}"
                cp -r "$JOB_LOGS_DIR/LOGS/agentic/." "$GITHUB_WORKSPACE/LOGS/agentic/conc_${CONC}/"
                sudo chown -R "$(id -u):$(id -g)" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                chmod -R u+rwX "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
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

    set -x
    # Exclude known-bad mi355x compute nodes (KLAUD_DEBUG §5.1 / §5.2):
    #   mia1-p01-g09: pyxis broken (persistently fails to create container filesystem)
    #   mia1-p01-g11: docker.sock permissions denied (cluster-cleanup step fails)
    # Both have been root-caused via #1431/#1432/#1440/#1441/#1443 sweep failures.
    salloc --partition=$PARTITION --exclude=mia1-p01-g09,mia1-p01-g11,mia1-p01-g37 --gres=gpu:$TP --exclusive --cpus-per-task=128 --time=500 --no-shell --job-name="$RUNNER_NAME"
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
    # they are pre-downloaded once to the NFS share instead.
    if [[ "$MODEL" == MiniMaxAI/MiniMax-M3* ]]; then
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
        --container-mounts=/dev/shm:/dev/shm \
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
