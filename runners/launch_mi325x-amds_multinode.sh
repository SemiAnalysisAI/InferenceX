#!/usr/bin/env bash
set -euo pipefail

scancel_sync() {
    local job_id="$1"
    local deadline=$((SECONDS + 600))
    scancel "$job_id" 2>/dev/null || true
    while squeue -h -j "$job_id" 2>/dev/null | grep -q .; do
        if (( SECONDS >= deadline )); then
            echo "WARNING: Slurm job $job_id still exists after 600 seconds" >&2
            return 1
        fi
        sleep 10
    done
}

stage_server_logs() {
    [[ -d "${BENCHMARK_LOGS_DIR:-}" ]] || return 0
    local archive
    archive=$(mktemp /tmp/inferencex-mi325x-server-logs.XXXXXX.tar.gz)
    sudo tar -czf "$archive" \
        --exclude='*/agentic' \
        --exclude='*/agentic/*' \
        --exclude='*/agentic-output' \
        --exclude='*/agentic-output/*' \
        -C "$BENCHMARK_LOGS_DIR" . 2>/dev/null || true
    sudo chown "$USER" "$archive" 2>/dev/null || true
    if [[ -s "$archive" ]]; then
        mv -f "$archive" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"
    else
        rm -f "$archive"
    fi
}

cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n "${JOB_ID:-}" ]] && squeue -h -j "$JOB_ID" 2>/dev/null | grep -q .; then
        scancel_sync "$JOB_ID" || true
    fi
    stage_server_logs || true
    if [[ -n "${BENCHMARK_LOGS_DIR:-}" && -d "$BENCHMARK_LOGS_DIR" ]]; then
        sudo rm -rf -- "$BENCHMARK_LOGS_DIR" 2>/dev/null || true
    fi
    if [[ -n "${BENCHMARK_LOGS_PARENT:-}" && -d "$BENCHMARK_LOGS_PARENT" ]]; then
        rmdir "$BENCHMARK_LOGS_PARENT" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup EXIT

export SLURM_ACCOUNT="$USER"
export SLURM_PARTITION="compute"
export SLURM_JOB_NAME="benchmark-sglang-disagg.job"
export HF_HUB_CACHE_MOUNT="/nfsdata/sa/gharunner/gharunners/hf-hub-cache"
export MODEL_PATH="$HF_HUB_CACHE_MOUNT"
export MODEL_YAML_KEY="${MODEL_YAML_KEY:-${MODEL##*/}}"
export GPUS_PER_NODE=8
export IBDEVICES="${IBDEVICES:-bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8}"
export MORI_RDMA_TC="${MORI_RDMA_TC:-104}"

if [[ -z "${MODEL_NAME:-}" ]]; then
    hf_dir="models--${MODEL//\//--}"
    snapshot_root="$MODEL_PATH/$hf_dir/snapshots"
    if [[ -d "$snapshot_root" ]]; then
        snapshot=$(find "$snapshot_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -1)
    else
        snapshot=""
    fi
    if [[ -n "$snapshot" ]]; then
        export MODEL_NAME="$hf_dir/snapshots/$snapshot"
    elif [[ -d "$MODEL_PATH/${MODEL##*/}" ]]; then
        export MODEL_NAME="${MODEL##*/}"
    else
        echo "ERROR: model '$MODEL' is not staged under $MODEL_PATH" >&2
        exit 1
    fi
fi

runner_tag=${RUNNER_NAME//[^a-zA-Z0-9_.-]/_}
BENCHMARK_LOGS_PARENT="$(dirname "$GITHUB_WORKSPACE")/.inferencex-benchmark-logs"
mkdir -p "$BENCHMARK_LOGS_PARENT"
BENCHMARK_LOGS_DIR=$(mktemp -d "$BENCHMARK_LOGS_PARENT/mi325x-${runner_tag}.XXXXXX")
export BENCHMARK_LOGS_PARENT
export BENCHMARK_LOGS_DIR

script_name="${EXP_NAME%%_*}_${PRECISION}_mi325x_${FRAMEWORK}.sh"
benchmark_script="benchmarks/multi_node/$script_name"
if [[ ! -f "$benchmark_script" ]]; then
    echo "ERROR: $benchmark_script does not exist" >&2
    exit 1
fi

echo "Submitting $benchmark_script with model path $MODEL_NAME"
JOB_ID=$(bash "$benchmark_script" | tail -1)
if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: benchmark submission returned invalid job ID '$JOB_ID'" >&2
    exit 1
fi

log_file="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID}.out"
while [[ ! -f "$log_file" ]]; do
    if ! squeue -h -j "$JOB_ID" 2>/dev/null | grep -q .; then
        echo "ERROR: Slurm job $JOB_ID ended before creating its output log" >&2
        scontrol show job "$JOB_ID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

(
    while squeue -h -j "$JOB_ID" 2>/dev/null | grep -q .; do
        sleep 10
    done
) &
poll_pid=$!
tail -F -s 2 -n+1 "$log_file" --pid="$poll_pid" 2>/dev/null || true
wait "$poll_pid"

job_state=$(sacct -n -X -j "$JOB_ID" --format=State --parsable2 2>/dev/null | head -1 | cut -d'|' -f1)
job_exit=$(sacct -n -X -j "$JOB_ID" --format=ExitCode --parsable2 2>/dev/null | head -1 | cut -d'|' -f1)
echo "Slurm job $JOB_ID finished: state=${job_state:-unknown}, exit=${job_exit:-unknown}"

if [[ "${IS_AGENTIC:-0}" == "1" ]]; then
    mapfile -d '' aggregate_files < <(
        find "$BENCHMARK_LOGS_DIR/logs" -type f \
            -path '*/agentic-output/*.json' -print0 2>/dev/null
    )
    for result_file in "${aggregate_files[@]}"; do
        staged_result=$(mktemp /tmp/inferencex-mi325x-result.XXXXXX.json)
        sudo cp -f "$result_file" "$staged_result"
        sudo chown "$USER" "$staged_result"
        mv -f "$staged_result" "$GITHUB_WORKSPACE/$(basename "$result_file")"
    done

    raw_dir=$(find "$BENCHMARK_LOGS_DIR/logs" -type d -path '*/agentic' -print -quit 2>/dev/null || true)
    if [[ -n "$raw_dir" ]]; then
        staged_raw=$(mktemp -d /tmp/inferencex-mi325x-agentic.XXXXXX)
        sudo cp -a "$raw_dir/." "$staged_raw/"
        sudo chown -R "$USER" "$staged_raw"
        mkdir -p "$GITHUB_WORKSPACE/LOGS/agentic"
        cp -a "$staged_raw/." "$GITHUB_WORKSPACE/LOGS/agentic/"
        rm -rf -- "$staged_raw"
    fi
else
    while IFS= read -r -d '' result_file; do
        output="$GITHUB_WORKSPACE/${RESULT_FILENAME}_$(basename "$result_file")"
        staged_result=$(mktemp /tmp/inferencex-mi325x-result.XXXXXX.json)
        sudo cp -f "$result_file" "$staged_result"
        sudo chown "$USER" "$staged_result"
        mv -f "$staged_result" "$output"
    done < <(find "$BENCHMARK_LOGS_DIR/logs" -type f -name '*.json' -print0 2>/dev/null)
fi

stage_server_logs
scancel_sync "$JOB_ID" || true
JOB_ID=""

if [[ "${job_state:-}" != COMPLETED* || "${job_exit:-1:0}" != "0:0" ]]; then
    echo "ERROR: Slurm benchmark failed: state=${job_state:-unknown}, exit=${job_exit:-unknown}" >&2
    exit 1
fi
