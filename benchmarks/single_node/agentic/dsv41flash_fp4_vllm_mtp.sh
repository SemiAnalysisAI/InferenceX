#!/usr/bin/env bash
set -eo pipefail

# Native DeepSeek-V4.1-Flash DSpark and Engram UVA weight offload.
# https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

# Parallelism arms. The matrix always sets EP_SIZE and DP_ATTENTION; stand-alone
# runs default to the pure-TP recipe. TEP keeps TP attention and shards the
# routed experts (--enable-expert-parallel). DEP runs one attention rank per
# GPU (--data-parallel-size TP) behind vllm-router with session affinity.
EP_SIZE="${EP_SIZE:-1}"
DP_ATTENTION="${DP_ATTENTION:-false}"

# Complete/resume partial downloads instead of trusting nonempty directories.
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
ROUTER_LOG="$RESULT_DIR/router.log"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

# Preserve the upstream scheduler defaults; size graph capture for the sweep.
# Each DEP rank sees about CONC/TP sequences; capture for twice that so
# consistent-hash imbalance across ranks still lands on captured graphs.
NUM_SPEC_TOKENS=5
if [[ "$DP_ATTENTION" == "true" ]]; then
    CAPTURE_TARGET=$(( 2 * ((CONC + TP - 1) / TP) * (1 + NUM_SPEC_TOKENS) ))
else
    CAPTURE_TARGET=$(( CONC * (1 + NUM_SPEC_TOKENS) ))
fi
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < CAPTURE_TARGET && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
VLLM_BACKEND_PORT="$PORT"
if [[ "$DP_ATTENTION" == "true" ]]; then
    # vllm-router fronts the DP ranks on PORT and expands the one HTTP backend
    # into one logical worker per rank. Bind every turn of a conversation to
    # the same rank by mapping AIPerf's correlation ID to X-Session-ID.
    VLLM_BACKEND_PORT=$((PORT + 1))
    VLLM_ROUTER_VERSION=0.1.14
    VLLM_ROUTER_METRICS_PORT=$((PORT + 10000))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet "vllm-router==$VLLM_ROUTER_VERSION"
fi
export AIPERF_SERVER_URL="http://localhost:${PORT}"
# AIPerf scrapes the public endpoint's /metrics on its own; under DEP that is the
# router, so name the engine endpoint explicitly (deduplicated for pure TP).
export AIPERF_SERVER_METRICS_URLS="http://localhost:${VLLM_BACKEND_PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# Golden AL: golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Accuracy evals keep real block rejection; throughput fixes acceptance to AL 3.51.
if [[ "${EVAL_ONLY:-false}" == true ]]; then
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
else
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
fi
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [[ "$DP_ATTENTION" == "true" ]]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
    # Under DEP every rank's expert layer sees the tokens dispatched from all
    # DP ranks, so the TRT-LLM FP4 MoE workspace is larger than under TP. With
    # the image default of 0.92 the autotuner warmup died 4.45 GiB short after
    # the KV cache was sized (run 34655656558, DEP4 c64); reserve headroom and
    # let the allocator grow segments instead of fragmenting.
    PARALLEL_ARGS+=(--gpu-memory-utilization 0.85)
    export PYTORCH_ALLOC_CONF=expandable_segments:True
fi
if [[ "$EP_SIZE" -gt 1 ]]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$VLLM_BACKEND_PORT" "${PARALLEL_ARGS[@]}"
    --language-model-only
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    --engram-config '{"cpu_offload":true}'
    --speculative-config "$SPEC_CONFIG"
    --max-model-len 1048576
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$DP_ATTENTION" == "true" ]]; then
    echo "Starting vllm-router on port $PORT for $TP DP ranks..."
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy consistent_hash \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 --port "$PORT" \
        --prometheus-host 127.0.0.1 --prometheus-port "$VLLM_ROUTER_METRICS_PORT" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [[ "${EVAL_ONLY:-false}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
