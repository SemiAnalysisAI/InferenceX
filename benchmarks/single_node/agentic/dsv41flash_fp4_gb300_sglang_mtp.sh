#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash on GB300, using the image's native DSpark.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY SPEC_DECODING IMAGE SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE
require_agentic_kv_offload_none

if [[ "$MODEL" != deepseek-ai/DeepSeek-V4.1-Flash || "$TP" != "$EP_SIZE" || "$SPEC_DECODING" != mtp ]]; then
    echo "This recipe requires DeepSeek-V4.1-Flash, EP_SIZE=TP, and SPEC_DECODING=mtp" >&2
    exit 1
fi
case "$TP:$CONC" in
    4:1|4:2|4:4|4:8|4:32|4:64|4:80|2:16|2:32) ;;
    *) echo "Unsupported GB300 SGLang point: TP=$TP CONC=$CONC" >&2; exit 1 ;;
esac
case "$EVAL_ONLY" in
    true|false) ;;
    *) echo "EVAL_ONLY must be true or false" >&2; exit 1 ;;
esac
case "$SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE" in
    0) ;;
    1) check_env_vars SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT ;;
    *) echo "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE must be 0 or 1" >&2; exit 1 ;;
esac
export GPU_COUNT="$TP"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV41_REASONING_EFFORT=high
export SGLANG_RAGGED_VERIFY_MODE=static

# SGLang graph sizes count requests, not the six target-verify token rows.
# Reserve two requests per trajectory for fan-out; this is a scheduling cap,
# not a guarantee about the number of subagents a trajectory may spawn.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=64
while (( CUDA_GRAPH_MAX_BS < MAX_RUNNING_REQUESTS )); do
    CUDA_GRAPH_MAX_BS=$((2 * CUDA_GRAPH_MAX_BS))
done

# Initial tuning follows the newer B200 TP4 recipe; qualify it on GB300.
SWA_PREFIX_TAILS=$((64 * CONC))
if (( SWA_PREFIX_TAILS < 128 )); then
    SWA_PREFIX_TAILS=128
elif (( SWA_PREFIX_TAILS > 4096 )); then
    SWA_PREFIX_TAILS=4096
fi
if (( CONC >= 32 )); then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# Match the published vLLM arm's thinking-on golden AL, not the earlier
# diagnostic AL=5.4. Accuracy evals must never inherit forced acceptance.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
if [[ "$EVAL_ONLY" == false ]]; then
    export SGLANG_SIMULATE_ACC_LEN=3.51
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    --mem-fraction-static 0.80
    --chunked-prefill-size 4096
    --prefill-decode-interval 16
    --swa-prefix-tails "$SWA_PREFIX_TAILS"
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size 5
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --reasoning-parser auto
    --tool-call-parser auto
    --watchdog-timeout 3600
    --enable-metrics
    --enable-cache-report
)
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{
    printf 'IMAGE=%s\n' "$IMAGE"
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$EVAL_ONLY" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
