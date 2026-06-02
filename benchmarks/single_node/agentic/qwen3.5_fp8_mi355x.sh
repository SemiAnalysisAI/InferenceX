#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5 FP8 on MI300X using SGLang.
#
# Base server recipe follows the upstream MI300X reference
# (benchmarks/single_node/qwen3.5_fp8_mi300x.sh, the "AMD Andy" recipe):
# aiter attention backend, aiter allreduce fusion, mem-fraction 0.75.
# The agentic harness (resolve_trace_source / build_replay_cmd /
# run_agentic_replay_and_write_outputs) replaces run_benchmark_serving, and
# --disable-radix-cache is dropped because agentic replay needs prefix reuse.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE
#
# OFFLOADING values:
#   none    - SGLang GPU KV with the default RadixAttention prefix cache.
#   hicache - SGLang HiCache with a local CPU hierarchical cache on top of radix.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR EP_SIZE DP_ATTENTION

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
EP_SIZE=${EP_SIZE:-1}

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
# Cap the replay corpus at 256k (470 traces, max in+out <= 256k) instead of the
# unfiltered 052726 corpus whose ~1M-token traces get rejected and add no perf
# signal at high concurrency.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Cache / offload config -------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
WARMUP_ARGS=()
CUDA_GRAPH_MAX_BS="$CONC"
case "$OFFLOADING" in
    none)
        # Leave SGLang's default RadixAttention prefix cache on — agentic
        # replay needs it; --disable-radix-cache would zero the hit rate.
        ;;
    hicache)
        # Qwen3.5's hybrid GDN/Mamba path allocates two HiCache host pools per
        # TP rank (one hierarchical KV, one hierarchical Mamba), so the
        # node-total DRAM budget divides by TP and the host-pool count.
        TOTAL_CPU_DRAM_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-2500}"
        HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
        HICACHE_MAX_SIZE_GB_PER_RANK_POOL="${HICACHE_MAX_SIZE_GB_PER_RANK_POOL:-${HICACHE_MAX_SIZE_GB_PER_RANK:-300}}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
        # Qwen3.5's hybrid Mamba path runs SGLang's no_buffer scheduler, which
        # requires page_size=1. Keep the safer direct/layer_first copy path;
        # kernel/page_first faults on first prefill in this mode on ROCm.
        HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-1}"
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
            --page-size "$HICACHE_PAGE_SIZE"
            --enable-hierarchical-cache
            --hicache-size "$HICACHE_SIZE_GB"
            --hicache-io-backend "$HICACHE_IO_BACKEND"
            --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
            --hicache-write-policy "$HICACHE_WRITE_POLICY"
        )
        # HiCache startup reaches API readiness but SGLang's internal warmup
        # request can time out on this path; let aiperf own benchmark traffic.
        WARMUP_ARGS=(--skip-server-warmup)
        # Don't force ROCm graph capture at every high concurrency point; conc=16
        # is the highest known-good capture size for this model/server path.
        HICACHE_CUDA_GRAPH_MAX_BS="${HICACHE_CUDA_GRAPH_MAX_BS:-256}"
        if [ "$HICACHE_CUDA_GRAPH_MAX_BS" -lt "$CUDA_GRAPH_MAX_BS" ]; then
            CUDA_GRAPH_MAX_BS="$HICACHE_CUDA_GRAPH_MAX_BS"
        fi
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

echo "Starting SGLang server..."
export PYTHONNOUSERSITE=1

python3 -m sglang.launch_server \
    --attention-backend triton \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --ep-size $EP_SIZE \
    --trust-remote-code \
    --tokenizer-worker-num 6 \
    --enable-aiter-allreduce-fusion \
    --cuda-graph-max-bs $CONC \
    --max-running-requests $CONC \
    --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
    --mem-fraction-static 0.8 \
    "${CACHE_ARGS[@]}" \
    "${WARMUP_ARGS[@]}" \
    --enable-metrics > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"