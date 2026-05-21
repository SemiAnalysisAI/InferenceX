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

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
EP_SIZE=${EP_SIZE:-1}
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
        CACHE_ARGS=(--disable-radix-cache)
        ;;
    hicache)
        # HiCache extends RadixAttention, so do not pass --disable-radix-cache.
        # TOTAL_CPU_DRAM_GB comes from the workflow input and defaults to 600.
        HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$TOTAL_CPU_DRAM_GB}"
        CACHE_ARGS=(
            --page-size 64
            --enable-hierarchical-cache
            --hicache-size "$HICACHE_SIZE_GB"
            --hicache-io-backend kernel
            --hicache-mem-layout page_first
            --hicache-write-policy write_through
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

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/aiperf_artifacts" -o "$RESULT_DIR" 2>&1 || true
