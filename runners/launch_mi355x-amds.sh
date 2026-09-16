#!/usr/bin/env bash

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars EVAL_ONLY IS_AGENTIC IS_MULTINODE KEEP_LOGS RUN_EVAL

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

    set -x

    export SLURM_ACCOUNT="$USER"
    export SLURM_PARTITION="compute"
    export SLURM_JOB_NAME="benchmark-sglang-disagg.job"

    export MODEL_NAME=${MODEL##*/}
    export MODEL_PATH="/it-share/data"
    export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"
    export MORI_RDMA_TC=104

    export MODEL_DIR="$MODEL_PATH"  # job.slurm uses MODEL_DIR
    export GPUS_PER_NODE=8          # MI355X has 8 GPUs (set to 4 for MI325X)

    export ISL="$ISL"
    export OSL="$OSL"

    check_env_vars BENCHMARK_LOGS_DIR
    mkdir -p "$BENCHMARK_LOGS_DIR"
    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    # Root-owned container output must go even on early exit, or the next job's
    # checkout hits EACCES; slurm logs are saved as artifacts first. KEEP_LOGS=1
    # disables the trap for local debugging.
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
        sudo rm -rf "$BENCHMARK_LOGS_DIR" 2>/dev/null || true
    }
    if [[ "${KEEP_LOGS}" == "1" ]]; then
        trap '' EXIT
    else
        trap cleanup_and_save_logs EXIT
    fi

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_mi355x_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "sglang-disagg" ]] || [[ "$FRAMEWORK" == "vllm-disagg" ]] || [[ "$FRAMEWORK" == "atom-disagg" ]]; then
        # Agentic recipes under multi_node/agentic/ export the HiCache tunables;
        # fixed-seq-len recipes live at the multi_node/ root.
        if [[ "${SCENARIO_SUBDIR}" == "agentic/" ]]; then
            BENCHMARK_SUBDIR="multi_node/agentic"
        else
            BENCHMARK_SUBDIR="multi_node"
        fi
    else
        BENCHMARK_SUBDIR="single_node/fixed_seq_len"
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

    # -F follows by name and polls; inotify does not work on NFS.
    tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

    wait $POLL_PID

    set -x




    if [[ "${EVAL_ONLY}" != "true" && "${IS_AGENTIC}" != "1" ]]; then
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

        for result_file in $(find $LOGS_DIR -type f); do
            file_name=$(basename $result_file)
            if [ -f $result_file ]; then
                WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
                echo "Found result file ${result_file}. Copying it to ${WORKSPACE_RESULT_FILE}"
                cp $result_file $WORKSPACE_RESULT_FILE
            fi
        done
    fi

    if [[ "${RUN_EVAL}" == "true" ]]; then
        EVAL_DIR=$(find "$BENCHMARK_LOGS_DIR/logs" -type d -name eval_results 2>/dev/null | head -1)
        if [ -n "$EVAL_DIR" ] && [ -d "$EVAL_DIR" ]; then
            echo "Extracting eval results from $EVAL_DIR"
            shopt -s nullglob
            for eval_file in "$EVAL_DIR"/*; do
                [ -f "$eval_file" ] || continue
                eval_dest="$GITHUB_WORKSPACE/$(basename "$eval_file")"
                rm -f "$eval_dest"
                # Eval artifacts are root-owned from the container; sudo overwrites
                # stale root-owned files left by prior runs.
                if sudo cp "$eval_file" "$eval_dest"; then
                    sudo chown "$(id -u):$(id -g)" "$eval_dest" 2>/dev/null || true
                    echo "Copied eval artifact: $(basename "$eval_file")"
                else
                    echo "ERROR: failed to copy eval artifact: $(basename "$eval_file")" >&2
                    exit 1
                fi
            done
            shopt -u nullglob
        else
            echo "WARNING: RUN_EVAL=true but no eval results found under $BENCHMARK_LOGS_DIR/logs"
        fi
    fi

    # benchmark-multinode-tmpl.yml uploads LOGS/agentic/conc_*/... and
    # multinode_server_logs.tar.gz, so preserve trace_replay.sh's conc_<N>/
    # nesting before the logs dir is removed below.
    if [[ "${IS_AGENTIC}" == "1" ]]; then
        JOB_LOGS_DIR="$BENCHMARK_LOGS_DIR/logs/slurm_job-${JOB_ID}"
        if [ -d "$JOB_LOGS_DIR" ]; then
            AGENTIC_SRC="$JOB_LOGS_DIR/agentic"
            if [ -d "$AGENTIC_SRC" ] && find "$AGENTIC_SRC" -mindepth 1 -maxdepth 1 -type d -name 'conc_*' -print -quit 2>/dev/null | grep -q .; then
                echo "Staging agentic raw artifacts from $AGENTIC_SRC"
                mkdir -p "$GITHUB_WORKSPACE/LOGS/agentic"
                cp -r "$AGENTIC_SRC"/. "$GITHUB_WORKSPACE/LOGS/agentic/"
                # Container artifacts arrive root-owned; later jobs, possibly a
                # different runner user, must be able to remove LOGS/.
                sudo chown -R "$(id -u):$(id -g)" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                chmod -R a+rwX "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                ls -laR "$GITHUB_WORKSPACE/LOGS/agentic"
            else
                echo "WARNING: no agentic conc_*/ artifacts found under $JOB_LOGS_DIR/agentic"
            fi
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
    # Synchronous cancel so the NFS file handles are released before cleanup.
    set +x
    scancel_sync $JOB_ID
    set -x
    echo "Canceled the slurm job $JOB_ID"

    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

else

    export HF_HUB_CACHE_MOUNT="/var/lib/hf-hub-cache/"
    export AIPERF_MMAP_CACHE_HOST_PATH="/it-share/aiperf-cache/"
    export PORT_OFFSET=${RUNNER_NAME: -1}
    export PORT=$(( 8888 + ${PORT_OFFSET} ))
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" || "$SPEC_DECODING" == "draft_model" ]] && printf '_mtp' || printf '')

    PARTITION="compute"
    SQUASH_FILE="/var/lib/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    LOCK_FILE="${SQUASH_FILE}.lock"

    check_env_vars GPU_COUNT

    set -x
    salloc --partition=$PARTITION --gres=gpu:$GPU_COUNT --exclusive --cpus-per-task=128 --time=500 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)

    srun --jobid=$JOB_ID bash -c "docker stop \$(docker ps -a -q)"

    # Concurrent jobs import to the same squash file; serialize them.
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

    if [[ "$FRAMEWORK" == "atom" ]] || [[ "$FRAMEWORK" == "sglang" ]]; then
        SLRUM_HOME_MOUNT=""
    else
        SLRUM_HOME_MOUNT=" --container-mount-home "
    fi

    # Avoid a stale saved copy of this checkpoint; read the shared HF cache.
    if [[ ("$FRAMEWORK" == "vllm" || "$FRAMEWORK" == "atom") ]] && [[ "$MODEL" == "deepseek-ai/DeepSeek-V4-Pro" ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    # MiniMax-M3 weights are pre-downloaded to the NFS share, not the node-local
    # /var/lib NVMe cache.
    if [[ "$MODEL" == MiniMaxAI/MiniMax-M3* || "$MODEL" == amd/MiniMax-M3* ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    # GLM-5.2-FP8 is ~756 GB (141 shards). Pull it once to the NFS share rather
    # than once per node-local NVMe cache, so every cell of the sweep (which
    # may land on different nodes) shares a single staged copy.
    if [[ "$MODEL" == "zai-org/GLM-5.2-FP8" ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    # DSv4.1 weights live on the persistent shared cache. Mount this recipe
    # outside /workspace so runtime setup does not create directories there.
    CONTAINER_REPO=/workspace
    if [[ "$MODEL" == "deepseek-ai/DeepSeek-V4.1-Flash" ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
        CONTAINER_REPO=/ix
        export INFMAX_CONTAINER_WORKSPACE="$CONTAINER_REPO"
        case "${RESULT_DIR:-}" in
            /workspace/*) export RESULT_DIR="/ix/${RESULT_DIR#/workspace/}" ;;
        esac
    fi

    SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_mi355x"
    check_env_vars SCENARIO_SUBDIR
    SCRIPT_FW="benchmarks/single_node/${SCENARIO_SUBDIR}${SCRIPT_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    check_env_vars SCENARIO_SUBDIR
    SCRIPT_FALLBACK="benchmarks/single_node/${SCENARIO_SUBDIR}${SCRIPT_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
    if [[ -f "$SCRIPT_FW" ]]; then
        BENCHMARK_SCRIPT="$SCRIPT_FW"
    else
        BENCHMARK_SCRIPT="$SCRIPT_FALLBACK"
    fi

    if [[ "$BENCHMARK_SCRIPT" == "benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_atom_mtp.sh" ]]; then
        export MODEL_PATH="$MODEL"
        export ENABLE_PREFIX_CACHING=true
        export AITER_LOG_LEVEL=WARNING
        export EVAL_TASKS_DIR=infx/evals/gsm8k.yaml
    fi

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:$CONTAINER_REPO/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        $SLRUM_HOME_MOUNT \
        --container-writable \
        --container-workdir=$CONTAINER_REPO/ \
        --container-remap-root \
        --no-container-entrypoint --export=ALL,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash "$BENCHMARK_SCRIPT"
    benchmark_rc=$?

    scancel $JOB_ID

    if ls gpucore.* 1> /dev/null 2>&1; then
        echo "gpucore files exist. not good"
        rm -f gpucore.*
    fi

    exit "$benchmark_rc"
fi
