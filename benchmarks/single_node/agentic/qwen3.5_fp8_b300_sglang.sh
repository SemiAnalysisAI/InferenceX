#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5 FP8 on B300 using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-10}

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

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # HiCache extends RadixAttention, so do not pass --disable-radix-cache.
    # Qwen3.5's hybrid GDN/Mamba path allocates two HiCache host pools per TP
    # rank: one for hierarchical KV cache and one for hierarchical Mamba cache.
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    TOTAL_CPU_DRAM_GB="$REQUESTED_HICACHE_TOTAL_GB"
    HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
    # SGLang --hicache-size is per rank per host pool, while the workflow
    # input is a node-total DRAM budget. Divide by TP and the number of
    # host pools unless HICACHE_SIZE_GB is set directly for one-off tuning.
    MAX_HICACHE_SIZE_GB=$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB exceeds configured per-pool limit $MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
    if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
        echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB from TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB, TP=$TP, HICACHE_HOST_POOL_COUNT=$HICACHE_HOST_POOL_COUNT" >&2
        exit 1
    fi
    echo "HiCache CPU pool: ${HICACHE_SIZE_GB} GB per rank per host pool across TP=${TP}, host_pool_count=${HICACHE_HOST_POOL_COUNT}"
    CACHE_ARGS=(
        --page-size 64
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend kernel
        --hicache-mem-layout page_first
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
    )
fi

echo "Starting SGLang server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true

{ set +x; } 2>/dev/null
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path="$MODEL_PATH" --served-model-name="$MODEL"
    --host=0.0.0.0
    --port="$PORT"
    --served-model-name "Qwen/Qwen3.5-397B-A17B-FP8"
    --trust-remote-code
    --tensor-parallel-size="$TP"
    --data-parallel-size=1
    --expert-parallel-size="$EP_SIZE"
    --enable-symm-mem
    --quantization fp8
    --kv-cache-dtype fp8_e4m3
    --mamba-ssm-dtype bfloat16
    --attention-backend trtllm_mha
    --moe-runner-backend flashinfer_trtllm
    --cuda-graph-max-bs "$CONC"
    --max-running-requests "$CONC"
    --max-prefill-tokens 16384
    --chunked-prefill-size 16384
    --mem-fraction-static 0.80
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --tokenizer-worker-num 6
    --tokenizer-path "$MODEL"
    --enable-metrics
    "${CACHE_ARGS[@]}"
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
