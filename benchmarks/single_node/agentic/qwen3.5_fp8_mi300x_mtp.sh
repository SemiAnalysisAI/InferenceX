#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.5-397B-A17B FP8 on MI300X (gfx942) with
# SGLang native EAGLE/NEXTN MTP. Throughput uses the committed golden
# synthetic acceptance length; evals retain real target-model verification.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
WARMUP_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    REQUESTED_HICACHE_TOTAL_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-$TOTAL_CPU_DRAM_GB}"
    if [ "$REQUESTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: requested HiCache pool ${REQUESTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    TOTAL_CPU_DRAM_GB="$REQUESTED_HICACHE_TOTAL_GB"

    # Qwen3.5 allocates target KV and Mamba host pools per rank. Native MTP
    # adds a one-attention-layer draft KV pool (1/15 of target KV), so enforce
    # H * 31/15 per rank against the workflow's node-total DRAM budget.
    HICACHE_ALIGNMENT_RESERVE_GB=$TP
    HICACHE_USABLE_TOTAL_GB=$((TOTAL_CPU_DRAM_GB - HICACHE_ALIGNMENT_RESERVE_GB))
    if [ "$HICACHE_USABLE_TOTAL_GB" -lt 1 ]; then
        echo "Error: insufficient DRAM after HiCache alignment reserve" >&2
        exit 1
    fi
    MAX_HICACHE_SIZE_GB=$((HICACHE_USABLE_TOTAL_GB * 15 / TP / 31))
    HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$MAX_HICACHE_SIZE_GB}"
    if [ "$HICACHE_SIZE_GB" -lt 1 ] || [ "$HICACHE_SIZE_GB" -gt "$MAX_HICACHE_SIZE_GB" ]; then
        echo "Error: HICACHE_SIZE_GB=$HICACHE_SIZE_GB outside 1..$MAX_HICACHE_SIZE_GB" >&2
        exit 1
    fi
    PROJECTED_HICACHE_TOTAL_GB=$(((HICACHE_SIZE_GB * TP * 31 + 14) / 15 + HICACHE_ALIGNMENT_RESERVE_GB))
    if [ "$PROJECTED_HICACHE_TOTAL_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: projected HiCache use ${PROJECTED_HICACHE_TOTAL_GB} GB exceeds configured capacity ${TOTAL_CPU_DRAM_GB} GB" >&2
        exit 1
    fi
    echo "HiCache CPU pools: ${HICACHE_SIZE_GB} GB target + Mamba + 1/15 draft per rank across TP=${TP}; projected node total ${PROJECTED_HICACHE_TOTAL_GB} GB <= ${TOTAL_CPU_DRAM_GB} GB"

    # Qwen3.5's hybrid Mamba path uses SGLang's no_buffer scheduler. On ROCm,
    # page_size=1 with direct/layer_first is the exercised HiCache copy path.
    CACHE_ARGS=(
        --page-size 1
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE_GB"
        --hicache-io-backend direct
        --hicache-mem-layout layer_first
        --hicache-write-policy write_through_selective
    )
    WARMUP_ARGS=(--skip-server-warmup)
fi

PARALLEL_ARGS=(
    --tensor-parallel-size "$TP"
    --data-parallel-size 1
    --ep-size "$EP_SIZE"
)

# AgentX concurrency counts live session trees, not individual HTTP requests.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64
if agentic_kv_offload_enabled && [ "$CUDA_GRAPH_MAX_BS" -gt 16 ]; then
    CUDA_GRAPH_MAX_BS=16
fi

export PYTHONNOUSERSITE=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.39
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend aiter
    --enable-aiter-allreduce-fusion
    --mamba-ssm-dtype bfloat16
    --tokenizer-worker-num 6
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 32768
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --mem-fraction-static 0.75
    --tokenizer-path "$MODEL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
    "${WARMUP_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

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
    # Aggregate deployment: the client-facing SGLang engine is the single
    # logical Prometheus target. Make it explicit and require engine metrics.
    export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
    export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
