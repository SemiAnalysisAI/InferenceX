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

rocm-smi || true
amd-smi || true
# ---- Resolve traces and install deps ----------------------------------------
# Cap the replay corpus at 256k (470 traces, max in+out <= 256k) instead of the
# unfiltered 052726 corpus whose ~1M-token traces get rejected and add no perf
# signal at high concurrency.
#export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k
#060226
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_060226_256k

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ROCm / SGLang performance tuning for MI355X
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SAFETENSORS_FAST_GPU=1

# ---- Start SGLang server ----------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

pip install -U transformers

CACHE_ARGS=()
WARMUP_ARGS=()
CUDA_GRAPH_MAX_BS="$CONC"
case "$OFFLOADING" in
    none)
        # Leave SGLang's default RadixAttention prefix cache on — agentic
        # replay needs it; --disable-radix-cache would zero the hit rate.
        ;;
    hicache)
        # GLM-5.1 FP4 uses a standard transformer (no hybrid Mamba path),
        # so one HiCache host pool per TP rank is sufficient.
        # The node-total DRAM budget divides by TP and host-pool count.
        TOTAL_CPU_DRAM_GB=3000
        HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
        HICACHE_MAX_SIZE_GB_PER_RANK_POOL="${HICACHE_MAX_SIZE_GB_PER_RANK_POOL:-${HICACHE_MAX_SIZE_GB_PER_RANK:-500}}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
        # GLM-5.1 uses standard paged attention (no no_buffer scheduler constraint),
        # so page_size can be left at the default. Keep the safer direct/layer_first
        # copy path on ROCm.
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
        HICACHE_CUDA_GRAPH_MAX_BS="${HICACHE_CUDA_GRAPH_MAX_BS:-16}"
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

pip install -U transformers
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL" \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --cuda-graph-max-bs $CONC \
    --max-running-requests $CONC \
    --mem-fraction-static 0.85 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --watchdog-timeout 1200 \
    --kv-cache-dtype fp8_e4m3 \
    --tokenizer-worker-num $((TP*2)) \
    "${CACHE_ARGS[@]}" \
    "${WARMUP_ARGS[@]}" \
    --enable-metrics > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"