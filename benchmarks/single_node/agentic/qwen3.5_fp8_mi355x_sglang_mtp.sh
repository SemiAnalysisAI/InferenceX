#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.5-397B-A17B FP8 on MI355X with SGLang native
# EAGLE MTP. Throughput uses the committed golden synthetic acceptance length;
# evaluation retains real target-model verification.
#
# First Qwen3.5 FP8 SGLang AgentX recipe on MI355X: the FP8 precision sibling
# of agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh (amd/Qwen3.5-397B-A17B-MXFP4).
# SKU-level serve shape (AITER unified attention, INT8 quick all-reduce, fp8
# KV, 16k prefill budget, min(2*CONC, 128) graph capture, HiCache tier) is the
# MXFP4 MI355X script unchanged so the two precision curves on this SKU stay
# comparable. The FP8 deltas, marked "FP8:" below, come from the ROCm Qwen3.5
# FP8 sibling on MI325X (agentic/qwen3.5_fp8_mi325x_mtp.sh): the upstream
# Qwen/Qwen3.5-397B-A17B-FP8 checkpoint (~406 GB), --quantization fp8, and
# --mamba-ssm-dtype bfloat16 passed as a server flag. AITER_FLYDSL_FORCE is
# not carried: it forces AITER's FlyDSL MXFP4 GEMM path and has no FP8 role.

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

# Single aggregate SGLang engine: one logical backend metrics endpoint is
# authoritative. build_replay_cmd also discovers the public endpoint; AIPerf
# deduplicates the explicit copy.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# HiCache host-DRAM KV tier, same sizing as the MXFP4 sibling. FP8: the device
# KV pool it is sized against is smaller here (TP4: ~101 GB/rank of weights
# inside 0.80 of 288 GB leaves ~129 GB/rank; MXFP4 TP4: ~53 GB/rank of weights
# leaves ~177 GB/rank), so ratio 1.5 pins proportionally less host DRAM
# (~0.8 TB across TP4) and sits well inside the node budget. Overridable.
CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    echo "HiCache CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT, dram_budget=${TOTAL_CPU_DRAM_GB} GB, tp=$TP"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
fi

PARALLEL_ARGS=(
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
)

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

# AgentX concurrency counts live session trees rather than HTTP requests. Keep
# 2*CONC in flight for subagent fan-out and capture the decode graph to the
# same bound (cap 128) so batches above CONC do not fall to the eager path.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 128 ] && CUDA_GRAPH_MAX_BS=128

# FP8: 0.80 is the MI355X SKU default (MXFP4 sibling) and the MI325X FP8
# sibling's TP4/TP8 value. With EAGLE enabled SGLang reserves 15% of the
# budget for the draft model. The ~406 GB FP8 checkpoint is ~101 GB/rank at
# TP4 and ~51 GB/rank at TP8, leaving ~129 GB and ~179 GB/rank for the KV
# pool and hybrid state; TP2 (~203 GB/rank) would leave ~27 GB and is not
# attempted on day zero (the MI325X FP8 sibling needs 0.95 for TP2 on 256 GB).
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
# Multi-GPU collectives through INT8-quantized ROCm quick all-reduce, per the
# SGLang cookbook MI355X recipe and the MXFP4 sibling (#2737). The MI325X FP8
# sibling's --enable-aiter-allreduce-fusion is not carried: the MI355X MXFP4
# recipe disabled it for EAGLE rank consistency (#2562).
export ROCM_QUICK_REDUCE_QUANTIZATION=INT8
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

# Synthetic rejection sampling is only for performance replay. The AL is the
# committed Qwen3.5 thinking-on value for three speculative tokens
# (golden_al_distribution). Evals use real target-model verification.
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
    # FP8: explicit method, as the MI325X FP8 sibling passes; the checkpoint's
    # quantization_config is fp8 so this only pins what would be auto-detected.
    --quantization fp8
    --kv-cache-dtype fp8_e4m3
    # FP8: keep the Mamba SSM state in bf16 (MI325X FP8 sibling); the MXFP4
    # sibling sets the same via SGLANG_MAMBA_SSM_DTYPE.
    --mamba-ssm-dtype bfloat16
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --watchdog-timeout 1200
    --page-size 16
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 16384
    --chunked-prefill-size 16384
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --stream-interval 50
    "${TOKENIZER_ARGS[@]}"
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
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
