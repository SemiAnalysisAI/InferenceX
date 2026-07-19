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
runner_shared_root="$GITHUB_WORKSPACE"
for _ in 1 2 3 4 5; do
    runner_shared_root=$(dirname "$runner_shared_root")
done
if [[ "$(basename "$runner_shared_root")" != "gharunners" ]]; then
    echo "ERROR: cannot derive shared runner root from $GITHUB_WORKSPACE" >&2
    exit 1
fi
export HF_HUB_CACHE_MOUNT="${MI325X_SHARED_HF_CACHE:-$runner_shared_root/.inferencex-hf-cache}"
export MODEL_PATH="$HF_HUB_CACHE_MOUNT"
export MODEL_YAML_KEY="${MODEL_YAML_KEY:-${MODEL##*/}}"
export GPUS_PER_NODE=8
export IBDEVICES="${IBDEVICES:-bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re7,bnxt_re8}"
export MORI_RDMA_TC="${MORI_RDMA_TC:-104}"

resolve_model_name() {
    local hf_dir="models--${MODEL//\//--}"
    local snapshot_root="$MODEL_PATH/$hf_dir/snapshots"
    local snapshot=""

    if [[ -d "$snapshot_root" ]]; then
        snapshot=$(find "$snapshot_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -1)
    fi
    if [[ -n "$snapshot" ]]; then
        printf '%s\n' "$hf_dir/snapshots/$snapshot"
        return 0
    fi
    if [[ -d "$MODEL_PATH/${MODEL##*/}" ]]; then
        printf '%s\n' "${MODEL##*/}"
        return 0
    fi
    return 1
}

stage_model_to_shared_cache() {
    local hf_dir="models--${MODEL//\//--}"
    local lock_file="$MODEL_PATH/.${hf_dir}.download.lock"
    local python_path=""
    local tool_dir=""

    mkdir -p "$MODEL_PATH"
    if [[ ! -w "$MODEL_PATH" ]]; then
        echo "ERROR: shared model cache '$MODEL_PATH' is not writable by $USER" >&2
        return 1
    fi
    exec 9>"$lock_file"
    if ! flock -w 18000 9; then
        echo "ERROR: timed out waiting to stage '$MODEL' under $MODEL_PATH" >&2
        return 1
    fi

    # Another topology may have completed the download while this launcher
    # waited for the shared-cache lock.
    if resolve_model_name >/dev/null; then
        flock -u 9
        exec 9>&-
        return 0
    fi

    echo "Staging '$MODEL' into shared cache $MODEL_PATH"
    if ! python3 -c 'import huggingface_hub' >/dev/null 2>&1; then
        tool_dir=$(mktemp -d /tmp/inferencex-hf-client.XXXXXX)
        python3 -m pip install --quiet --disable-pip-version-check \
            --target "$tool_dir" 'huggingface_hub>=0.30'
        python_path="$tool_dir"
    fi

    PYTHONPATH="${python_path}${PYTHONPATH:+:$PYTHONPATH}" \
        python3 - "$MODEL" "$MODEL_PATH" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    cache_dir=sys.argv[2],
    token=os.environ.get("HF_TOKEN"),
)
PY

    if [[ -n "$tool_dir" ]]; then
        rm -rf -- "$tool_dir"
    fi
    if ! resolve_model_name >/dev/null; then
        echo "ERROR: download completed but '$MODEL' is unresolved under $MODEL_PATH" >&2
        flock -u 9
        exec 9>&-
        return 1
    fi
    flock -u 9
    exec 9>&-
}

if [[ -z "${MODEL_NAME:-}" ]]; then
    if ! MODEL_NAME=$(resolve_model_name); then
        stage_model_to_shared_cache
        MODEL_NAME=$(resolve_model_name)
    fi
    export MODEL_NAME
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
