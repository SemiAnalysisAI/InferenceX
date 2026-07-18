#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.2 FP8 on MI355X using SGLang.
#
# Server flags follow the SGLang cookbook MI355X FP8 single-node recipes
# (https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2): TP8 with the
# DSA tilelang prefill/decode backends, no MTP (spec decoding is not yet
# validated on ROCm gfx950). The cookbook's low-latency / balanced /
# high-throughput strategies differ only in batch-shaping levers
# (chunked-prefill / mem-fraction / graph bs / max-running), which this
# script derives from CONC so one search-space arm traces the full frontier.
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
if [[ "$DP_ATTENTION" = "true" ]]; then
    echo "Error: DP-attention is not part of the GLM-5.2 MI355X cookbook recipe" >&2
    exit 1
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# Weights are pre-staged in the NFS HF hub cache (launch_mi355x-amds.sh mounts
# /it-share/hf-hub-cache for this model); a warm cache makes this a no-op.
if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# Cookbook batch-shaping by concurrency band:
#   low-latency  (conc <= 16): chunked-prefill 131072, mem-fraction 0.80
#   balanced/high-throughput (conc >= 32): chunked-prefill 32768, mem 0.85
# AgentX concurrency counts live session trees, not individual requests, so
# max-running-requests is 2*CONC (subagent fan-out headroom); the CUDA-graph
# range covers it up to the cookbook high-throughput cap of 256.
if [ "$CONC" -le 16 ]; then
    CHUNKED_PREFILL_SIZE=131072
    MEM_FRACTION_STATIC=0.80
else
    CHUNKED_PREFILL_SIZE=32768
    MEM_FRACTION_STATIC=0.85
    # MI355X prefill is slow relative to the 1M-context agentic corpus; give
    # the warmup drain the same extended grace as the B300 saturation arm.
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 256 ] && CUDA_GRAPH_MAX_BS=256

export PYTHONNOUSERSITE=1
# Agentic warmup dispatches hundreds of large prompts at once; allow up to
# 15 minutes of TCP progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf pins one pooled keep-alive connection per session (client-side
# keep-alive 300s) while uvicorn's default SGLANG_TIMEOUT_KEEP_ALIVE is 5s;
# inter-turn idle gaps can reuse a socket exactly as the server closes it.
# Outlast the client pool so the race cannot occur.
export SGLANG_TIMEOUT_KEEP_ALIVE=900

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --ep-size "$EP_SIZE"
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    # GLM-5.2 emits the GLM-4.7-style tool-call format; glm47 is required for
    # structured message.tool_calls (SWE-bench agentic evals die without it).
    # The glm45 reasoning parser keeps hybrid thinking in reasoning_content.
    --tool-call-parser glm47
    --reasoning-parser glm45
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --watchdog-timeout 1800
    --enable-metrics
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

echo "Starting SGLang server for MI355X..."
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
