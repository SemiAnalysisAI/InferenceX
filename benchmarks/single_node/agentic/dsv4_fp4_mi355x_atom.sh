#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using ATOM.
#
# Serving flags follow the validated MI355X recipe from
# https://github.com/ROCm/ATOM/blob/main/recipes/DeepSeek-V4-Agentic-Benchmark.md
# Image is configured in amd-master.yaml.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram is not enabled. 

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# Nightly ROCm image may be missing runtime deps; ensure they are present.
agentic_pip_install --quiet Pillow fastapi uvicorn

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# DeepSeek-V4-Pro weights are large; engine startup can exceed default 600s.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# vllm-project/vllm#43447 keeps local SWA prefix-cache tails sparsely, while
# vllm-project/vllm#44774 applies the same reachability policy to Mooncake's
# store mask. 32k matches the trace-replay tuning validated for this workload.
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768

# VLLM_PREFIX_CACHE_RETENTION_INTERVAL only applies to sliding-window/Mamba
# models; this vLLM build raises ValueError if it is set for DSv4.

# ---- LLM server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
VLLM_BACKEND_PORT="$PORT"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
ROUTER_PID=""
MOONCAKE_MASTER_PID=""

PARALLEL_ARGS=(-tp "$TP")
SPEC_ARGS=()
OFFLOAD_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(-tp "$TP" --enable-dp-attention)
fi
if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--enable-expert-parallel)
fi

# AgentX concurrency counts live session trees, not individual requests.
# Subagent fan-out can push instantaneous request concurrency above CONC, so
# leave 2x headroom rather than clipping those bursts at the scheduler.
MAX_NUM_SEQS=$((2 * CONC))

echo "Starting atom server..."
set -x
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export AITER_LOG_LEVEL=WARNING
export ATOM_DISABLE_MMAP=true
export MEM_FRAC_STATIC=0.9

{ set +x; } 2>/dev/null
ATOM_CMD=(
    python3 -m atom.entrypoints.openai_server
    --model "$MODEL_PATH" 
    --server-port "$VLLM_BACKEND_PORT"
    "${PARALLEL_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    --kv_cache_dtype fp8
    --trust-remote-code 
    --enable_prefix_caching
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$MEM_FRAC_STATIC"
    "${OFFLOAD_ARGS[@]}"
)

printf '%q ' "${ATOM_CMD[@]}" | tee "$RESULT_DIR/atom_command.txt"
printf '\n' | tee -a "$RESULT_DIR/atom_command.txt"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
