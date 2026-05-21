#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 NVFP4 on B200 using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - vLLM GPU KV only.
#   cpu     - vLLM native simple CPU offload.
#   lmcache - in-process LMCacheConnectorV1 via vLLM's lmcache backend.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
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
    none)
        ;;
    cpu)
        # B200 DGXC nodes have ~2.7 TiB host DRAM; reserve 2.5 TB for the
        # simple offload connector and leave ~200 GB headroom for worker
        # RSS + page cache. Eager mode (the shortcut form default) is
        # intentional here per user request — Kimi FP4 on B200 has cleared
        # the full eager sweep before.
        TOTAL_CPU_DRAM_GB=2500
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

        # B200 DGXC nodes have ~2.7 TiB host DRAM. Keep the TP=8 LMCache
        # path at the same 2.5 TB envelope as native offload while leaving room
        # for vLLM worker RSS and page cache.
        #
        # vLLM splits --kv-offloading-size across TP ranks for LMCache. In the
        # current vLLM 0.21.0 + LMCache 0.4.5 integrated connector path, Kimi's
        # MLA/HND layout cannot use LazyMixedMemoryAllocator and falls back to a
        # full pinned MixedMemoryAllocator allocation. That means TP=4 with a
        # 2.5 TB total tries to cudaHostAlloc ~625 GB per rank and fails during
        # engine startup, while TP=8 at ~312.5 GB per rank starts successfully.
        # Cap lower-TP LMCache runs to the same proven per-rank envelope.
        TOTAL_CPU_DRAM_GB=2500
        LMCACHE_MAX_LOCAL_CPU_GB_PER_RANK="${LMCACHE_MAX_LOCAL_CPU_GB_PER_RANK:-313}"
        LMCACHE_TOTAL_CPU_DRAM_GB="$TOTAL_CPU_DRAM_GB"
        if (( LMCACHE_TOTAL_CPU_DRAM_GB > TP * LMCACHE_MAX_LOCAL_CPU_GB_PER_RANK )); then
            LMCACHE_TOTAL_CPU_DRAM_GB=$((TP * LMCACHE_MAX_LOCAL_CPU_GB_PER_RANK))
        fi
        echo "LMCache CPU offload pool: ${LMCACHE_TOTAL_CPU_DRAM_GB} GB total across TP=${TP}"
        export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
        # Avoid a noisy failed lazy-allocator fallback; the per-rank cap above is
        # the actual startup guard for this Kimi/vLLM/LMCache combination.
        export LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR="${LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR:-false}"

        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        OFFLOAD_ARGS=(
            --kv-offloading-backend lmcache
            --kv-offloading-size "$LMCACHE_TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu, lmcache)" >&2
        exit 1
        ;;
esac

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
# Disable vLLM v0.21+ CUDA-graph memory estimator. Its pre-reservation
# eats ~32% of HBM upfront which, combined with FP4 weights at TP=4
# (~62 GB/GPU), leaves no room for KV blocks -- _check_enough_kv_cache_memory
# trips before the engine starts. Our --gpu-memory-utilization=0.90 already
# leaves ~18 GB/GPU slack outside vLLM's budget, which is the same safety
# net the estimator provides, so disabling it is redundant rather than
# unsafe.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL"
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

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/aiperf_artifacts" -o "$RESULT_DIR" 2>&1 || true
