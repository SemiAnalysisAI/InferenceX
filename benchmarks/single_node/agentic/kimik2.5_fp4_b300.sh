#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 NVFP4 on B300 using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - vLLM GPU KV only.
#   cpu     - vLLM native simple CPU offload.
#   lmcache - in-process LMCacheConnectorV1 via vLLM's lmcache backend.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION


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
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
PREFIX_CACHE_ARGS=()

case "$OFFLOADING" in
    none) ;;
    cpu)
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    lmcache)
        { set +x; } 2>/dev/null
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        agentic_pip_install --quiet --no-cache-dir lmcache
        python3 -c "import lmcache.integration.vllm.vllm_v1_adapter" >/dev/null

        # vLLM divides the proportional node budget across TP ranks for
        # --kv-offloading-backend=lmcache.
        export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
        # Avoid pinning the full proportional pool during startup. LMCache grows
        # the CPU allocator as agentic prefixes accumulate in the replay.
        export LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR="${LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR:-true}"
        export LMCACHE_LAZY_MEMORY_INITIAL_RATIO="${LMCACHE_LAZY_MEMORY_INITIAL_RATIO:-0.01}"
        export LMCACHE_LAZY_MEMORY_STEP_RATIO="${LMCACHE_LAZY_MEMORY_STEP_RATIO:-0.02}"

        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        OFFLOAD_ARGS=(
            --kv-offloading-backend lmcache
            --kv-offloading-size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    *) echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu, lmcache)" >&2; exit 1 ;;
esac

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size="$TP"
    --gpu-memory-utilization 0.90
    --max-num-seqs "$CONC"
    --reasoning-parser kimi_k2
    --tool-call-parser kimi_k2
    --compilation_config.pass_config.fuse_allreduce_rms true
    --kv-cache-dtype fp8
    --max-cudagraph-capture-size 2048
    --stream-interval 20
    --trust-remote-code
    "${PREFIX_CACHE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
