#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 NVFP4 on B300 using SGLang.
#
# Server flags follow the SGLang cookbook B300 NVFP4 single-node recipes
# (https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2), STP only:
# the cookbook's EAGLE MTP variants are intentionally not wired up yet.
#   DP_ATTENTION=false -> low-latency arm (TP8, fp8 KV, cutedsl bf16 GEMM)
#   DP_ATTENTION=true  -> high-throughput arm (TP8 + DP8 attention-DP)
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ "$KV_OFFLOADING" != "none" ]]; then
    echo "Error: KV_OFFLOADING=$KV_OFFLOADING is not supported by this recipe" >&2
    exit 1
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE.
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
nvidia-smi

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# With attention-DP, front the DP ranks with sglang-router using consistent
# hashing on the AIPerf correlation id so multi-turn sessions stay on the DP
# rank that holds their radix-cache prefix.
USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
ROUTER_LOG="$RESULT_DIR/router.log"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
fi

PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
CHUNKED_PREFILL_SIZE=8192
if [ "$DP_ATTENTION" = "true" ]; then
    # chunked-prefill-size is a whole-engine budget split across DP ranks:
    # the cookbook HT cell's 8192 becomes 1,024 tokens/rank/step under dp8,
    # which starves prefill on the 1M-context agentic corpus (observed: a
    # conc-256 warmup could not drain within AIPerf's 1800s grace period
    # while KV usage sat at ~0.01). Use the cookbook's own dp8 lever from
    # the B200 cells (32768 = ~4096/rank).
    CHUNKED_PREFILL_SIZE=32768
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --tokenizer-worker-num "$TP"
        --dist-init-addr "127.0.0.1:$((PORT + 2000))"
    )
else
    # Cookbook low-latency levers; the DP-attention cell omits them.
    PARALLEL_ARGS+=(
        --kv-cache-dtype fp8_e4m3
        --bf16-gemm-backend cutedsl
        --max-prefill-tokens 8192
    )
fi

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts.
MAX_RUNNING_REQUESTS=$((2 * CONC))
GRAPH_ARGS=()
if [ "$DP_ATTENTION" != "true" ]; then
    # Cookbook low-latency captures graphs up to its request cap; the
    # DP-attention cell leaves the CUDA-graph batch list at SGLang defaults.
    CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
    [ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
    GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
# Agentic warmup dispatches hundreds of large prompts at once; allow up to
# 15 minutes of TCP progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf pins one pooled keep-alive connection per session (client-side
# keep-alive 300s) while uvicorn's default SGLANG_TIMEOUT_KEEP_ALIVE is 5s;
# inter-turn idle gaps (capped at 10s) can reuse a socket exactly as the
# server closes it -> ECONNRESET -> terminal warmup failure. Outlast the
# client pool so the race cannot occur.
export SGLANG_TIMEOUT_KEEP_ALIVE=900

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --quantization modelopt_fp4
    # GLM-5.2 emits the GLM-4.7-style <tool_call>/<arg_key>/<arg_value> format;
    # the glm47 parser is required for structured message.tool_calls (glm45
    # leaves calls as raw text). Without it the SWE-bench mini-swe-agent eval
    # dies with RepeatedFormatError ("No tool calls found in the response") on
    # every instance and scores 0. Reasoning parser keeps hybrid-thinking
    # output in reasoning_content instead of polluting content. Neither flag
    # affects trace-replay throughput (pre-canned replay discards live
    # responses).
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static 0.85
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    "${GRAPH_ARGS[@]}"
    --watchdog-timeout 1800
    --enable-metrics
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

echo "Starting SGLang server for B300..."
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
    echo "Starting SGLang router on port $PORT for $TP DP ranks..."
    python3 -m sglang_router.launch_router \
        --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
        --policy consistent_hashing \
        --request-id-headers x-correlation-id \
        --dp-aware \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$SGLANG_ROUTER_METRICS_PORT" \
        --connect-timeout-secs 900 \
        --request-timeout-secs 14400 \
        --disable-health-check \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
