#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5 FP8 on B300 using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - SGLang GPU KV only with radix cache disabled.
#   hicache - SGLang HiCache with local CPU hierarchical cache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-10}
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

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

CACHE_ARGS=()
case "$OFFLOADING" in
    none)
        # Leave SGLang's default RadixAttention prefix cache on — agentic
        # replay needs it; --disable-radix-cache would zero the hit rate.
        ;;
    hicache)
        # HiCache extends RadixAttention, so do not pass --disable-radix-cache.
        # B300 nodes have about 2 TB of usable CPU DRAM. Qwen3.5's hybrid
        # GDN/Mamba path allocates two HiCache host pools per TP rank: one for
        # hierarchical KV cache and one for hierarchical Mamba cache. Keep this
        # local to the script because the workflow currently passes a generic
        # default for TOTAL_CPU_DRAM_GB, not a platform-specific value.
        TOTAL_CPU_DRAM_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-2000}"
        HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-2}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
        # SGLang --hicache-size is per rank per host pool, while the workflow
        # input is a node-total DRAM budget. Divide by TP and the number of
        # host pools unless HICACHE_SIZE_GB is set directly for one-off tuning.
        HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$((TOTAL_CPU_DRAM_GB / TP / HICACHE_HOST_POOL_COUNT))}"
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
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

echo "Starting SGLang server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true

{ set +x; } 2>/dev/null
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path="$MODEL"
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
    --context-length "$MAX_MODEL_LEN"
    --enable-metrics
    "${CACHE_ARGS[@]}"
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
