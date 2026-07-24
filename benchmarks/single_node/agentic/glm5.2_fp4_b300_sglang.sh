#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 NVFP4 on B300 using SGLang.
#
# Server flags follow the SGLang cookbook B300 NVFP4 single-node recipes
# (https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2), STP only:
# the cookbook's EAGLE MTP variants are intentionally not wired up yet.
#   DP_ATTENTION=false -> low-latency arm (TP8, fp8 KV, cutedsl bf16 GEMM)
#   DP_ATTENTION=true  -> high-throughput DEP arm (TP8 + DP8 attention-DP +
#                         EP_SIZE expert-parallel MoE via --ep-size)
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

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

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # HiCache extends RadixAttention: prefixes evicted from the HBM KV pool
    # spill to a pinned host pool instead of being recomputed. On the
    # 1M-context agentic corpus the live working set outgrows HBM past
    # conc 8 (TP8) / 64 (DP8) and the radix hit rate collapses to <0.1
    # against a ~0.97 theoretical ceiling, so every turn re-prefills its
    # whole history; the host tier restores those hits at C2C bandwidth.
    # GLM-5.2 is DSA/MLA-family (attention_backend=dsa): every rank holds
    # complete per-token KV (169.98 GB device pool per rank, replicated on
    # all 8 ranks), so host capacity is controlled through the host/device
    # token-capacity ratio like the DSv4 recipe, NOT a per-rank
    # --hicache-size. A GB-based size of TOTAL_CPU_DRAM_GB/TP pinned the
    # whole 0.80-DRAM budget (8 x 299 GB) at init on top of 465 GB of
    # weights and OOM-killed the node (run 29678598595); DSv4's own
    # ratio=2 default pins 2 x 170 GB x 8 = 2.7 TB here and OOMs too
    # (GLM-5.2's device pool is far larger than DSv4's). Fractional 0.75
    # = ~128 GB/rank = ~1.0 TB total, matching the cluster's proven ~1 TB
    # host-pool envelope; validated on-node 2026-07-19 (boot + 4.2M-token
    # overflow bench forcing eviction through the DSA KV+INDEXER pools).
    DEFAULT_HICACHE_RATIO=0.75
    HICACHE_RATIO="${HICACHE_RATIO:-$DEFAULT_HICACHE_RATIO}"
    if awk -v r="$HICACHE_RATIO" -v cap="$DEFAULT_HICACHE_RATIO" 'BEGIN { exit !(r > cap) }'; then
        echo "Error: HICACHE_RATIO=$HICACHE_RATIO exceeds configured limit $DEFAULT_HICACHE_RATIO" >&2
        exit 1
    fi
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_back}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    echo "HiCache CPU tier: ratio=$HICACHE_RATIO, capacity=${TOTAL_CPU_DRAM_GB} GB, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
fi

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
    "${CACHE_ARGS[@]}"
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
    # GLM-5.2's chat template defaults to reasoning_effort=Max when the
    # client passes no chat_template_kwargs (mini-swe-agent doesn't), and the
    # heavy thinking burns the default 75-step budget: on the 23-instance
    # slice, 12/23 trajectories exited LimitsExceeded unsubmitted while 10 of
    # the 11 that submitted resolved. Double the step budget for this recipe;
    # other recipes keep the shared 75 default.
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
