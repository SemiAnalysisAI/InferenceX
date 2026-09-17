#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL MODEL_PATH TP AGENTIC_CONC_LIST ENGRAM_SSD_DIR RESULT_DIR \
    RESULT_FILENAME DURATION ENGRAM_GDS_PAGE_CAPACITY ENGRAM_GDS_MAX_ROWS \
    DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE VLLM_ENGINE_READY_TIMEOUT_S
require_agentic_kv_offload_none
export GPU_COUNT="$TP"
export VLLM_USE_RUST_FRONTEND=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768 VLLM_RPC_TIMEOUT=600000
export PYTHONUNBUFFERED=1
mkdir -p "$RESULT_DIR" "$ENGRAM_SSD_DIR"
GDS_FILESYSTEM=$(df -PT "$ENGRAM_SSD_DIR" | awk 'NR==2 {print $2}')
case "$GDS_FILESYSTEM" in
    xfs|ext4) ;;
    *) echo "GDS experiment requires local XFS/ext4, found $GDS_FILESYSTEM" >&2; exit 1 ;;
esac
nvidia-smi
findmnt -T "$ENGRAM_SSD_DIR"
export CUFILE_ALLOW_COMPAT_MODE=false CUFILE_FORCE_COMPAT_MODE=false
python3 "$INFERENCEX_REPO_ROOT/benchmarks/patches/check_engram_gds.py" \
    --directory "$ENGRAM_SSD_DIR" --result-dir "$RESULT_DIR" \
    2>&1 | tee "$RESULT_DIR/gds_probe.log"
export CUFILE_ENV_PATH_JSON="$RESULT_DIR/cufile-gds.json"

VLLM_DIR=$(python3 -c 'import os,vllm; print(os.path.dirname(vllm.__file__))')
GDS_PATCH="$INFERENCEX_REPO_ROOT/benchmarks/patches/vllm-dsv41flash-engram-gds.patch"
patch -p1 --dry-run -d "$VLLM_DIR" < "$GDS_PATCH"
patch -p1 -d "$VLLM_DIR" < "$GDS_PATCH"
cp "$INFERENCEX_REPO_ROOT/benchmarks/patches/engram_gds_reader.py" \
    "$INFERENCEX_REPO_ROOT/benchmarks/patches/engram_gds_reader.cpp" \
    "$VLLM_DIR/models/deepseek_v4_1/common/"
python3 "$INFERENCEX_REPO_ROOT/benchmarks/patches/check_dsv41flash_gds_replay.py" \
    --result-dir "$RESULT_DIR" --disk-dir "$ENGRAM_SSD_DIR" \
    2>&1 | tee "$RESULT_DIR/gds_replay_check.log"

hf download "$MODEL"
resolve_trace_source
install_agentic_deps
BASE_RESULT_FILENAME="$RESULT_FILENAME"
BASE_RESULT_DIR="$RESULT_DIR"
SERVER_PID=""
stop_engine() {
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        for ((attempt=0; attempt<120; attempt++)); do
            if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then
                SERVER_PID=""
                return 0
            fi
            sleep 1
        done
        echo "Engine process group $SERVER_PID did not exit" >&2
        return 1
    fi
}
trap 'stop_engine' EXIT
for CONC in $AGENTIC_CONC_LIST; do
    export CONC
    export RESULT_FILENAME="${BASE_RESULT_FILENAME}_conc${CONC}"
    export RESULT_DIR="${BASE_RESULT_DIR}/conc${CONC}"
    mkdir -p "$RESULT_DIR"
    CAPTURE_SIZE="$DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE"
    while (( CAPTURE_SIZE < CONC * 6 && CAPTURE_SIZE < 2048 )); do
        CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
    done
    select_available_server_port
    export AIPERF_SERVER_URL="http://localhost:$PORT"
    export AIPERF_SERVER_METRICS_URLS="$AIPERF_SERVER_URL/metrics"
    export AIPERF_REQUIRED_SERVER_METRIC_PREFIX=vllm:
    ENGRAM_CONFIG=$(python3 -c 'import json,os; print(json.dumps({"cpu_offload":True,"disk_offload_dir":os.environ["ENGRAM_SSD_DIR"]}))')
    VLLM_CMD=(vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
        --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP"
        --language-model-only --tokenizer-mode deepseek_v41
        --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41
        --engram-config "$ENGRAM_CONFIG" --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
        --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
        --max-model-len 1048576 --max-num-batched-tokens 8192
        --max-cudagraph-capture-size "$CAPTURE_SIZE" --disable-uvicorn-access-log)
    printf '%q ' "${VLLM_CMD[@]}" > "$RESULT_DIR/vllm_command.txt"
    printf '\n' >> "$RESULT_DIR/vllm_command.txt"
    SERVER_LOG="$BASE_RESULT_DIR/server_conc${CONC}.log"
    setsid "${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
    stop_engine
    echo "GDS sweep completed concurrency $CONC"
done
