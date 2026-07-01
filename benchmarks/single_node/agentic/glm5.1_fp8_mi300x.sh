#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for GLM-5.1 FP8 on MI300X using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
#
# OFFLOADING values:
#   none    - SGLang GPU KV with the default RadixAttention prefix cache.
#   hicache - SGLang HiCache with a local CPU hierarchical cache on top of radix.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ROCm / SGLang performance tuning for MI300X (gfx942)
export SAFETENSORS_FAST_GPU=1

# ---- Cache / offload config -------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
CUDA_GRAPH_MAX_BS="$CONC"
case "$OFFLOADING" in
    none)
        # Leave SGLang's default RadixAttention prefix cache on — agentic
        # replay needs it; --disable-radix-cache would zero the hit rate.
        ;;
    hicache)
        # GLM-5.1 is a dense-KV (non-hybrid) model, so it allocates a single
        # HiCache host pool per TP rank. --hicache-size is per rank per host
        # pool while the workflow input is a node-total DRAM budget, so divide
        # by TP and the host-pool count. Overridable for one-off tuning.
        HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
        HICACHE_MAX_SIZE_GB_PER_RANK_POOL="${HICACHE_MAX_SIZE_GB_PER_RANK_POOL:-180}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
        HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
        HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-layer_first}"
        HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))}"
        if [ "$HICACHE_SIZE_GB" -gt "$HICACHE_MAX_SIZE_GB_PER_RANK_POOL" ]; then
            HICACHE_SIZE_GB="$HICACHE_MAX_SIZE_GB_PER_RANK_POOL"
        fi
        if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
            echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB from TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB, TP=$TP, HICACHE_HOST_POOL_COUNT=$HICACHE_HOST_POOL_COUNT" >&2
            exit 1
        fi
        echo "HiCache CPU pool: ${HICACHE_SIZE_GB} GB per rank per host pool across TP=${TP}, host_pool_count=${HICACHE_HOST_POOL_COUNT}"
        CACHE_ARGS=(
            --enable-hierarchical-cache
            --hicache-size "$HICACHE_SIZE_GB"
            --hicache-io-backend "$HICACHE_IO_BACKEND"
            --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
            --hicache-write-policy "$HICACHE_WRITE_POLICY"
        )
        # Capture ROCm graphs up to full concurrency so the hicache arm is a
        # fair A/B against the none arm (which captures to $CONC). The MI355X
        # recipe caps this at 16 due to a high-conc capture crash on that HW;
        # on MI300X we follow $CONC. Override via env if MI300X hits the same
        # startup crash at high conc.
        HICACHE_CUDA_GRAPH_MAX_BS="${HICACHE_CUDA_GRAPH_MAX_BS:-$CONC}"
        if [ "$HICACHE_CUDA_GRAPH_MAX_BS" -lt "$CUDA_GRAPH_MAX_BS" ]; then
            CUDA_GRAPH_MAX_BS="$HICACHE_CUDA_GRAPH_MAX_BS"
        fi
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

pip install -U transformers

echo "Starting SGLang server..."
export PYTHONNOUSERSITE=1

python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --cuda-graph-max-bs $CUDA_GRAPH_MAX_BS \
    --max-running-requests $CONC \
    --mem-fraction-static 0.85 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 \
    --tokenizer-worker-num $((TP*2)) \
    "${CACHE_ARGS[@]}" \
    --enable-metrics > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"