#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5-397B-A17B FP8 on MI355X using
# SGLang with EAGLE MTP (speculative decoding).
#
# MTP variant of qwen3.5_fp8_mi355x_sglang.sh. Enables EAGLE speculative
# decoding (3 steps, 4 draft tokens, topk 1). Throughput uses the committed
# golden synthetic acceptance length; evals retain real target-model
# verification.
#
# Configuration aligned with the NVIDIA B200/B300 Qwen3.5 FP8 AgentX MTP
# scripts, adapted for ROCm/MI355X (attention-backend, allreduce fusion, etc).
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-10}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

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

resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export PYTHONNOUSERSITE=1
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

# ---- MTP settings (EAGLE: num-steps 3, draft-tokens 4, topk 1) ----
SPEC_ALGORITHM="${SPEC_ALGORITHM:-EAGLE}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"

# Throughput runs use synthetic acceptance; eval runs use real verification.
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.39
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

# ---- Concurrency headroom (matches NVIDIA: 2x CONC for sub-agent fan-out) ----
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

# ---- HiCache (host DRAM KV offloading) ----
CACHE_ARGS=()
WARMUP_ARGS=()
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
    HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-1}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-layer_first}"
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
    WARMUP_ARGS=(--skip-server-warmup)
    HICACHE_CUDA_GRAPH_MAX_BS="${HICACHE_CUDA_GRAPH_MAX_BS:-16}"
    if [ "$HICACHE_CUDA_GRAPH_MAX_BS" -lt "$CUDA_GRAPH_MAX_BS" ]; then
        CUDA_GRAPH_MAX_BS="$HICACHE_CUDA_GRAPH_MAX_BS"
    fi
fi

echo "Starting SGLang MTP server for Qwen3.5 FP8 on MI355X..."

{ set +x; } 2>/dev/null
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --attention-backend triton
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host=0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --ep-size "$EP_SIZE"
    --trust-remote-code
    --tokenizer-worker-num 6
    --enable-aiter-allreduce-fusion
    --kv-cache-dtype fp8_e4m3
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 16384
    --chunked-prefill-size 16384
    --mem-fraction-static 0.80
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --tokenizer-path "$MODEL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --speculative-algorithm "$SPEC_ALGORITHM"
    --speculative-num-steps "$SPEC_NUM_STEPS"
    --speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS"
    --speculative-eagle-topk "$SPEC_EAGLE_TOPK"
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
    "${WARMUP_ARGS[@]}"
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

capture_cache_metrics() {
    {
        echo "=== SGLang cache metrics snapshot $(date --iso-8601=seconds) ==="
        curl -fsS "http://localhost:$PORT/metrics" 2>/dev/null \
            | grep -E '^(sglang:(cache_hit_rate|cached_tokens_total|prompt_tokens_total|hicache_host_used_tokens|hicache_host_total_tokens|token_usage|num_requests_running|num_requests_waiting))' \
            || true
        echo "============================================================"
    } >> "$SERVER_LOG"
}

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

capture_cache_metrics
trap capture_cache_metrics EXIT

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
