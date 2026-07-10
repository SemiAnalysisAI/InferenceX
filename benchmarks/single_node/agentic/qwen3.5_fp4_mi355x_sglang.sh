#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5-397B-A17B MXFP4 on MI355X using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE
#
# KV_OFFLOADING=none                              -> GPU KV only; radix/prefix cache ON
# KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache   -> SGLang hierarchical CPU-DRAM KV tier

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
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
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# Install amd-quark for MXFP4 (manual install due to ROCm packaging gap)
pip install amd-quark || agentic_pip_install amd-quark || true

export SGLANG_USE_AITER=1

# Workaround for MEC FW <177 RCCL memory reclaim issue
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || ${version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
WARMUP_ARGS=()
CUDA_GRAPH_MAX_BS="$CONC"
if require_agentic_kv_offload_backend hicache; then
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    TOTAL_CPU_DRAM_GB="$REQUESTED_HICACHE_TOTAL_GB"
    HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
    HICACHE_MAX_SIZE_GB_PER_RANK_POOL="${HICACHE_MAX_SIZE_GB_PER_RANK_POOL:-${HICACHE_MAX_SIZE_GB_PER_RANK:-180}}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
    # Keep page_size=1 and the direct IO backend on ROCm (the kernel/page_first
    # HiCache path relies on a CUDA-only JIT kernel). Qwen3.5's hybrid Mamba
    # host pool (MambaPoolHost) only supports the page_first_direct layout, and
    # io_backend=direct requires page_first_direct anyway, so pair them.
    HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-1}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    # SGLang --hicache-size is per rank per host pool, while the workflow input
    # is a node-total DRAM budget. Divide by TP and the number of host pools
    # unless HICACHE_SIZE_GB is set directly for one-off tuning.
    MAX_HICACHE_SIZE_GB=$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB exceeds configured per-pool limit $MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
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
    # HiCache startup reaches API readiness, but SGLang's internal warmup request
    # can time out on this Qwen MI355X path. Let aiperf own benchmark traffic
    # instead of blocking server readiness on warmup.
    WARMUP_ARGS=(--skip-server-warmup)
    # Do not force HiCache runs to capture ROCm graphs at every high concurrency
    # point; keep request concurrency as the swept variable.
    HICACHE_CUDA_GRAPH_MAX_BS="${HICACHE_CUDA_GRAPH_MAX_BS:-16}"
    if [ "$HICACHE_CUDA_GRAPH_MAX_BS" -lt "$CUDA_GRAPH_MAX_BS" ]; then
        CUDA_GRAPH_MAX_BS="$HICACHE_CUDA_GRAPH_MAX_BS"
    fi
fi

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then EP_ARGS=(--ep-size "$EP_SIZE"); fi

# Qwen3.5 is a hybrid Mamba model. With radix cache on, SGLang's
# mamba-radix-cache-strategy defaults to "auto", which selects "extra_buffer"
# whenever overlap schedule (the default) is on. extra_buffer needs the
# CUDA-only FLA backend and aborts on ROCm at arg-validation:
#   AssertionError: extra_buffer needs CUDA/MUSA/NPU (FLA).
# Force the ROCm-compatible "no_buffer" strategy, which requires page_size=1
# and overlap schedule off. Applies to both KV modes (both keep radix on).
MAMBA_ARGS=(--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule)
# no_buffer requires page_size=1. The hicache path already sets --page-size in
# CACHE_ARGS; add it here only for the GPU-only (none) path to avoid a dup flag.
if [ "${#CACHE_ARGS[@]}" -eq 0 ]; then
    MAMBA_ARGS+=(--page-size 1)
fi

echo "Starting sglang server..."
export PYTHONNOUSERSITE=1

{ set +x; } 2>/dev/null
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host=0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    "${EP_ARGS[@]}"
    --trust-remote-code
    --attention-backend aiter
    --enable-aiter-allreduce-fusion
    --mem-fraction-static "${MEM_FRAC_STATIC:-0.8}"
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --max-prefill-tokens "${MAX_PREFILL_TOKENS:-32768}"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$CONC"
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --watchdog-timeout 1200
    --enable-metrics
    "${MAMBA_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    "${WARMUP_ARGS[@]}"
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --sleep-interval 60

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
