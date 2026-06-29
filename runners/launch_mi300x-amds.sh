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

# Pin to the known-good mi300x nodes; others are unavailable:
#   chi-mi300x-033,037: down*
#   chi-mi300x-049:     down 
export SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-chi-mi300x-033,chi-mi300x-037,chi-mi300x-049}"

if [[ "$IS_MULTINODE" == "true" ]]; then
    set -x

    export SLURM_ACCOUNT="$USER"
    export SLURM_PARTITION="compute"
    export SLURM_JOB_NAME="benchmark-${FRAMEWORK}.job"

    export MODEL_NAME=${MODEL##*/}
    export MODEL_PATH="/raid/hf-hub-cache"
    export IBDEVICES="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7"
    export MORI_RDMA_TC=104
    export VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly-20260629-e667ebb}"

    export MODEL_DIR="$MODEL_PATH"
    export GPUS_PER_NODE=8

    export ISL="$ISL"
    export OSL="$OSL"

    export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$GITHUB_WORKSPACE/benchmark_logs}"
    mkdir -p "$BENCHMARK_LOGS_DIR"
    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    save_multinode_diagnostics() {
        local art_dir="$GITHUB_WORKSPACE/benchmark_artifacts"
        mkdir -p "$art_dir"

        cp -r "$BENCHMARK_LOGS_DIR"/submit_*.log "$art_dir/" 2>/dev/null || true
        if [[ "${JOB_ID:-}" =~ ^[0-9]+$ ]]; then
            cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$art_dir/" 2>/dev/null || true
            scontrol show job "$JOB_ID" > "$art_dir/scontrol_job_${JOB_ID}.txt" 2>&1 || true
            sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList%80 > "$art_dir/sacct_job_${JOB_ID}.txt" 2>&1 || true
        fi

        squeue -u "$USER" > "$art_dir/squeue_${USER}.txt" 2>&1 || true
        {
            echo "RUNNER_NAME=${RUNNER_NAME:-}"
            echo "RUNNER_TYPE=${RUNNER_TYPE:-}"
            echo "SLURM_ACCOUNT=${SLURM_ACCOUNT:-}"
            echo "SLURM_PARTITION=${SLURM_PARTITION:-}"
            echo "SLURM_EXCLUDE_NODES=${SLURM_EXCLUDE_NODES:-}"
            echo "NODELIST=${NODELIST:-}"
            echo "SCRIPT_NAME=${SCRIPT_NAME:-}"
            echo "BENCHMARK_SUBDIR=${BENCHMARK_SUBDIR:-}"
            echo "BENCHMARK_LOGS_DIR=${BENCHMARK_LOGS_DIR:-}"
            echo "MODEL=${MODEL:-}"
            echo "MODEL_NAME=${MODEL_NAME:-}"
            echo "MODEL_PATH=${MODEL_PATH:-}"
            echo "FRAMEWORK=${FRAMEWORK:-}"
            echo "PRECISION=${PRECISION:-}"
            echo "ISL=${ISL:-}"
            echo "OSL=${OSL:-}"
            echo "CONC_LIST=${CONC_LIST:-}"
            echo "PREFILL_NODES=${PREFILL_NODES:-}"
            echo "PREFILL_NUM_WORKERS=${PREFILL_NUM_WORKERS:-}"
            echo "PREFILL_TP=${PREFILL_TP:-}"
            echo "PREFILL_EP=${PREFILL_EP:-}"
            echo "PREFILL_DP_ATTN=${PREFILL_DP_ATTN:-}"
            echo "DECODE_NODES=${DECODE_NODES:-}"
            echo "DECODE_NUM_WORKERS=${DECODE_NUM_WORKERS:-}"
            echo "DECODE_TP=${DECODE_TP:-}"
            echo "DECODE_EP=${DECODE_EP:-}"
            echo "DECODE_DP_ATTN=${DECODE_DP_ATTN:-}"
            echo "RUN_EVAL=${RUN_EVAL:-}"
            echo "EVAL_ONLY=${EVAL_ONLY:-}"
            echo "EVAL_CONC=${EVAL_CONC:-}"
            echo "RESULT_FILENAME=${RESULT_FILENAME:-}"
        } > "$art_dir/launcher_env.txt" 2>&1 || true

        if compgen -G "$art_dir/*" > /dev/null; then
            tar -czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$art_dir" . 2>/dev/null || true
        fi
    }

    cleanup_and_save_logs() {
        if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
            save_multinode_diagnostics
        fi
        local err_file="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID:-unknown}.err"
        if [[ ! "${JOB_ID:-}" =~ ^[0-9]+$ ]]; then
            err_file="$BENCHMARK_LOGS_DIR/slurm_job-unknown.err"
        fi
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

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_mi300x_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "sglang-disagg" ]] || [[ "$FRAMEWORK" == "vllm-disagg" ]]; then
        BENCHMARK_SUBDIR="multi_node"
    else
        BENCHMARK_SUBDIR="single_node"
    fi

    if [[ -z "${NODELIST:-}" ]]; then
        NUM_NODES_REQUIRED=$((PREFILL_NODES + DECODE_NODES))
        SUBMIT_HOST=$(hostname -s)
        NODELIST_DISCOVERY_TIMEOUT="${NODELIST_DISCOVERY_TIMEOUT:-900}"
        NODELIST_DISCOVERY_INTERVAL="${NODELIST_DISCOVERY_INTERVAL:-30}"
        MI300X_NODE_INVENTORY="${MI300X_NODE_INVENTORY:-chi-mi300x-034 chi-mi300x-035 chi-mi300x-036 chi-mi300x-043 chi-mi300x-049 chi-mi300x-054 chi-mi300x-057 chi-mi300x-058 chi-mi300x-121}"
        discovery_start=$(date +%s)

        while true; do
            SELECTED_NODES=("$SUBMIT_HOST")
            echo "Building NODELIST with submit host first: ${SUBMIT_HOST}"

            for candidate in $MI300X_NODE_INVENTORY; do
                [[ -n "$candidate" ]] || continue
                [[ "$candidate" == "$SUBMIT_HOST" ]] && continue
                if [[ ",${SLURM_EXCLUDE_NODES}," == *",${candidate},"* ]]; then
                    echo "Skipping excluded NODELIST candidate: $candidate"
                    continue
                fi

                if timeout 20s srun --nodes=1 --ntasks=1 --time=00:02:00 --partition="$SLURM_PARTITION" --nodelist="$candidate" \
                    bash -lc "test -d /tmp && test -w /tmp" >/dev/null 2>&1; then
                    SELECTED_NODES+=("$candidate")
                    echo "Added NODELIST candidate with writable /tmp: $candidate"
                else
                    echo "Skipping NODELIST candidate without writable /tmp: $candidate"
                fi

                if [[ "${#SELECTED_NODES[@]}" -ge "$NUM_NODES_REQUIRED" ]]; then
                    break
                fi
            done

            if [[ "${#SELECTED_NODES[@]}" -eq "$NUM_NODES_REQUIRED" ]]; then
                break
            fi

            now=$(date +%s)
            elapsed=$((now - discovery_start))
            if (( elapsed >= NODELIST_DISCOVERY_TIMEOUT )); then
                echo "ERROR: Need ${NUM_NODES_REQUIRED} nodes for multinode job but found ${#SELECTED_NODES[@]} usable nodes with writable /tmp for staging after ${elapsed}s." >&2
                echo "Selected nodes so far: ${SELECTED_NODES[*]}" >&2
                echo "MI300X node inventory checked: ${MI300X_NODE_INVENTORY}" >&2
                sinfo -N -p "$SLURM_PARTITION" -o "%N %T" >&2 || true
                exit 1
            fi

            echo "Only found ${#SELECTED_NODES[@]}/${NUM_NODES_REQUIRED} usable nodes; retrying in ${NODELIST_DISCOVERY_INTERVAL}s..."
            sleep "$NODELIST_DISCOVERY_INTERVAL"
        done

        NODELIST=$(IFS=,; echo "${SELECTED_NODES[*]}")
        export NODELIST
        echo "Using generated NODELIST=${NODELIST}"
    else
        echo "Using caller-provided NODELIST=${NODELIST}"
        IFS=',' read -r -a SELECTED_NODES <<< "$NODELIST"
    fi

    SANITIZED_RUNNER=$(printf '%s' "${RUNNER_NAME:-runner}" | tr -c 'a-zA-Z0-9_.-' '_')
    STAGED_WORKSPACE="/tmp/inferencex-${USER}-${GITHUB_RUN_ID:-manual}-${SANITIZED_RUNNER}"
    JOB_BENCHMARK_LOGS_DIR="${STAGED_WORKSPACE}/benchmark_logs"

    for node in "${SELECTED_NODES[@]}"; do
        echo "Staging workspace to ${node}:${STAGED_WORKSPACE}"
        tar \
            --exclude='./benchmark_logs' \
            --exclude='./benchmark_artifacts' \
            --exclude='./multinode_server_logs.tar.gz' \
            --exclude='./.git' \
            -C "$GITHUB_WORKSPACE" -cf - . | \
            timeout 120s srun --nodes=1 --ntasks=1 --time=00:05:00 --partition="$SLURM_PARTITION" --nodelist="$node" \
                bash -lc "rm -rf '$STAGED_WORKSPACE' && mkdir -p '$STAGED_WORKSPACE' '$JOB_BENCHMARK_LOGS_DIR' && tar -C '$STAGED_WORKSPACE' -xf - && test -f '$STAGED_WORKSPACE/benchmarks/multi_node/amd_utils/job.slurm' && test -d '$JOB_BENCHMARK_LOGS_DIR'"
        stage_status=("${PIPESTATUS[@]}")
        if [[ "${stage_status[0]}" -ne 0 || "${stage_status[1]}" -ne 0 ]]; then
            echo "ERROR: Failed to stage workspace on ${node}" >&2
            exit 1
        fi
    done

    BENCHMARK_LOGS_DIR="$JOB_BENCHMARK_LOGS_DIR"
    export BENCHMARK_LOGS_DIR

    SUBMIT_LOG="$BENCHMARK_LOGS_DIR/submit_${SCRIPT_NAME%.sh}.log"
    GITHUB_WORKSPACE="$STAGED_WORKSPACE" bash "$STAGED_WORKSPACE/benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}" > "$SUBMIT_LOG" 2>&1
    SUBMIT_RC=$?
    cat "$SUBMIT_LOG"
    JOB_ID=$(grep -E '^[0-9]+$' "$SUBMIT_LOG" | tail -n 1 || true)
    if [[ "$SUBMIT_RC" -ne 0 ]]; then
        echo "ERROR: Failed to submit multi-node job via benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}"
        echo "=== Submit log ==="
        cat "$SUBMIT_LOG" || true
        echo "=================="
        exit 1
    fi

    if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Expected numeric Slurm job id, got '$JOB_ID'"
        echo "=== Submit log ==="
        cat "$SUBMIT_LOG" || true
        echo "=================="
        exit 1
    fi

    LOG_FILE="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID}.out"

    sleep 10

    while ! ls "$LOG_FILE" &>/dev/null; do
        if ! squeue -u "$USER" --noheader --format='%i' | grep -q "$JOB_ID"; then
            echo "ERROR: Job $JOB_ID failed before creating log file"
            scontrol show job "$JOB_ID" || true
            save_multinode_diagnostics
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

    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

else

    export HF_HUB_CACHE_MOUNT="/raid/hf-hub-cache/"
    export PORT=8888

    PARTITION="compute"
    SQUASH_FILE="/home/gharunner/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    LOCK_FILE="${SQUASH_FILE}.lock"

    set -x

    EXCLUDE_OPT=()
    if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
        EXCLUDE_OPT=(--exclude "$SLURM_EXCLUDE_NODES")
    fi

    JOB_ID=$(salloc --partition=$PARTITION "${EXCLUDE_OPT[@]}" --gres=gpu:$TP --cpus-per-task=256 --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

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
    bash benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi300x.sh

    scancel $JOB_ID
fi
