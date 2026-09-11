#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for GLM5.2 FP4 on MI355X using ATOM MTP
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

if [[ -v SLURM_JOB_ID ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [[ -v ROCR_VISIBLE_DEVICES ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "$MODEL_PATH" ]]; then
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

# Require the vLLM Prometheus stream in every official result. AIPerf
# deduplicates this endpoint against its automatic localhost discovery.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

# VRAM space check
wait_for_amd_gpu_clean

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
LMCACHE_PIDS=()
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "ATOM server" 60
    local i
    for i in "${!LMCACHE_PIDS[@]}"; do
        stop_background_process_tree "${LMCACHE_PIDS[$i]}" "LMCache server $i"
    done
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OFFLOAD_ARGS=()

case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        # LMCache settings
        export PYTHONHASHSEED=0
        export LMCACHE_LOCAL_CPU=True
        # Cap at 512 GiB/rank (recipe value for large clusters); use dram-utilization budget on smaller nodes
        export LMCACHE_MAX_LOCAL_CPU_SIZE=$(( TOTAL_CPU_DRAM_GB < 512 ? TOTAL_CPU_DRAM_GB : 512 ))
        export LMCACHE_NUMA_MODE=auto
        export LMCACHE_CHUNK_SIZE=256
        export OFFLOAD_MIN_LOAD_TOKENS=8192

        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}"
        )
        ;;
    *)
        echo "Unsupported KV_OFFLOAD_BACKEND: $KV_OFFLOAD_BACKEND (expected empty or lmcache)" >&2
        exit 1
        ;;
esac

# ---- LLM server config ----------------------------------------------------------

echo "Starting atom server..."
export PYTHONNOUSERSITE=1

# ---- ATOM env ----
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export ATOM_MLA_PAGE_SIZE=1
export ATOM_DCP_REPLICATE_INDEX_CACHE="${ATOM_DCP_REPLICATE_INDEX_CACHE:-0}"

# CUDA/HIPGRAPH settings
case "$CONC" in
  1)  CUDAGRAPH_CAPTURE_SIZES='[1,2]' ;;
  2)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4]' ;;
  4)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8]' ;;
  8)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16]' ;;
  10) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20]' ;;
  12) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24]' ;;
  16) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24,28,32]' ;;
  *)
    echo "Unsupported CONC=$CONC" >&2
    exit 2
    ;;
esac

# PARALLEL settings
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "${DCP_SIZE:-1}" -gt 1 ]; then
    # DCP mode (large concurrency, C16+): adds decode-context parallelism, no spec decoding
    PARALLEL_ARGS+=(--decode-context-parallel-size "$DCP_SIZE")
elif [ "$DP_ATTENTION" = "true" ]; then
    # DPA+EP
    if [ "$EP_SIZE" -gt 1 ]; then
        PARALLEL_ARGS=(--tensor-parallel-size "$TP" --enable-dp-attention --enable-expert-parallel)
    # DPA+TP
    else
        PARALLEL_ARGS=(--tensor-parallel-size "$TP" --enable-dp-attention)
    fi
fi

# SPEC settings
# SIMULATE_ACC_LEN and NUM_SPEC_TOKENS reference:
# https://github.com/ROCm/ATOM/pull/2117
# DCP mode disables speculative decoding entirely.
SIMULATE_ACC_LEN=3.33
NUM_SPEC_TOKENS=4
if [ "${DCP_SIZE:-1}" -gt 1 ]; then
    SPEC_ARGS=()
elif [ "${EVAL_ONLY}" = "true" ]; then
    SPEC_ARGS=(
        --method mtp
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
    )
else
    SPEC_ARGS=(
        --method mtp
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
        --spec-decode-acceptance-length "$SIMULATE_ACC_LEN"
    )
fi
echo "DCP_SIZE=${DCP_SIZE:-1} SIMULATE_ACC_LEN=$SIMULATE_ACC_LEN NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS"

ATOM_CMD=(
    python -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    "${PARALLEL_ARGS[@]}"
    --gpu-memory-utilization 0.95
    --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}'
    --max-num-seqs "$((2 * CONC))"
    --cudagraph-capture-sizes "$CUDAGRAPH_CAPTURE_SIZES"
    --max-num-batched-tokens 16384
    --kv_cache_dtype fp8
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
