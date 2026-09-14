#!/usr/bin/env bash
set -eo pipefail
set -x

# AgentX trace replay for Qwen3.5-397B-A17B FP8 on MI325X with SGLang
# native EAGLE/NEXTN MTP. Throughput uses the committed golden synthetic
# acceptance length; evaluation retains real target-model verification.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

rocm-smi || true
amd-smi || true

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

# This is a single aggregate SGLang engine, so one logical backend metrics
# endpoint is authoritative. build_replay_cmd also discovers the public
# endpoint; AIPerf deduplicates the explicit copy.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

PARALLEL_ARGS=(
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
)

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

# AgentX concurrency counts live session trees rather than HTTP requests.
# Keep capacity for subagent fan-out while bounding graph-capture memory.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

# With EAGLE enabled, SGLang reserves 15% of this budget for the draft model.
# TP2 needs an effective 0.8075 target-model budget to fit Qwen3.5's weights,
# hybrid state, and KV pools; TP4/TP8 retain the established 0.68 budget.
MEM_FRACTION_STATIC=0.80
if [ "$TP" -eq 2 ]; then
    MEM_FRACTION_STATIC=0.95
fi

export PYTHONNOUSERSITE=1
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

# Synthetic rejection sampling is only for performance replay. The AL is the
# committed Qwen3.5 thinking-on value for three speculative tokens. Evals use
# real target-model verification.
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.39
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend aiter
    --quantization fp8
    --kv-cache-dtype fp8_e4m3
    --mamba-ssm-dtype bfloat16
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --watchdog-timeout 1200
    --enable-aiter-allreduce-fusion
    --page-size 16
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 32768
    --chunked-prefill-size 32768
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --stream-interval 50
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-metrics
    --enable-cache-report
)

write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
