#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on B300 using vLLM.
# Mirrors the fixed-seq-len dsv4_fp4_b300_vllm.sh recipe (TP-only path,
# no DP-attn, ep=1 unless overridden) with --no-enable-prefix-caching
# removed (the agentic trace replay is a prefix-caching benchmark) and a
# 1M max-model-len to exercise DSv4's long-context capability.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
EP_SIZE=${EP_SIZE:-1}
DP_ATTENTION=${DP_ATTENTION:-false}
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=1000000
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# DeepSeek-V4-Pro weights are large; engine startup can exceed default 600s.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=""
case "$OFFLOADING" in
    none) ;;
    cpu)
        # B300 nodes have substantial DRAM; override workflow default
        # (600 GB) so we can offload up to 2.2 TB of KV cache.
        TOTAL_CPU_DRAM_GB=2200
        export VLLM_USE_SIMPLE_KV_OFFLOAD=1
        OFFLOAD_ARGS="--kv_offloading_backend native --kv_offloading_size $TOTAL_CPU_DRAM_GB --disable-hybrid-kv-cache-manager"
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, cpu)" >&2
        exit 1
        ;;
esac

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

MOE_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    MOE_ARGS=(--moe-backend deep_gemm_mega_moe)
fi

echo "Starting vllm server..."
export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_FLOAT32_MATMUL_PRECISION=high

vllm serve "$MODEL" \
--host 0.0.0.0 \
--port "$PORT" \
"${PARALLEL_ARGS[@]}" \
--pipeline-parallel-size 1 \
--kv-cache-dtype fp8 \
--trust-remote-code \
--block-size 256 \
"${EP_ARGS[@]}" \
"${MOE_ARGS[@]}" \
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
--attention_config.use_fp4_indexer_cache True \
--tokenizer-mode deepseek_v4 \
--tool-call-parser deepseek_v4 \
--enable-auto-tool-choice \
--reasoning-parser deepseek_v4 \
--max-cudagraph-capture-size 2048 \
--max-model-len "$MAX_MODEL_LEN" \
--max-num-batched-tokens 2048 \
--max-num-seqs "$CONC" \
$OFFLOAD_ARGS > "$SERVER_LOG" 2>&1 &
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
    "$RESULT_DIR/trace_replay" -o "$RESULT_DIR" 2>&1 || true
