#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for MiniMax-M3 MXFP4 on MI355X using ATOM
# with EAGLE3 speculative decoding. Throughput runs pin acceptance to the
# committed golden curve; eval-only runs use the drafter's real acceptance.
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# The AITER page-16 sparse-attention path wants exactly one KV head per
# tensor-parallel rank. MiniMax-M3 has four KV heads, so this recipe is TP4.
if [ "$TP" -ne 4 ] || [ "$EP_SIZE" -ne 1 ] || [ "$DP_ATTENTION" != "false" ]; then
    echo "This recipe requires TP=4, EP_SIZE=1, and DP_ATTENTION=false" >&2
    exit 1
fi
require_agentic_kv_offload_none

# ROCR/HIP visibility
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"
NUM_SPEC_TOKENS=3
# golden_al_distribution/minimaxm3_eagle3.yaml:
# minimax-m3.thinking_on[3]
SPEC_DECODE_AL=2.83

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

hf download "$DRAFT_MODEL"

rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps

# Require the ATOM Prometheus stream in every official result. AIPerf
# deduplicates this endpoint against its automatic localhost discovery.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

# VRAM space check
wait_for_amd_gpu_clean

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_atom_server() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "ATOM server" 60
    exit "$exit_code"
}
trap cleanup_atom_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- ATOM env ---------------------------------------------------------------
echo "Starting atom server..."
export PYTHONNOUSERSITE=1

# Without this the aiter kernel logs flood the server log for the whole replay.
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
# MiniMax-M3 sparse attention runs on the Triton path under ATOM, matching the
# TRITON_ATTN backend the vLLM MI355X recipe pins.
export ATOM_FORCE_ATTN_TRITON=1

# CUDA/HIPGRAPH settings
case "$CONC" in
  1)  CUDAGRAPH_CAPTURE_SIZES='[1,2]' ;;
  4)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8]' ;;
  8)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16]' ;;
  12) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24]' ;;
  16) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24,28,32]' ;;
  *)
    echo "Unsupported CONC=$CONC" >&2
    exit 2
    ;;
esac

# ---- Speculative ------------------------------------------------------------
# Synthetic acceptance standardizes throughput against the committed golden
# EAGLE3 curve. Accuracy evals must use real target verification.
SPEC_ARGS=(
    --method eagle3
    --draft-model "$DRAFT_MODEL"
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
)
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
fi
echo "DRAFT_MODEL=$DRAFT_MODEL NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS SPEC_DECODE_AL=$SPEC_DECODE_AL"

# ---- LLM server -------------------------------------------------------------
# AgentX concurrency counts session trees. Keep 2x scheduler headroom for the
# request bursts produced by subagent fan-out.
ATOM_CMD=(
    python3 -u -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    --tensor-parallel-size "$TP"
    --trust-remote-code
    --block-size 128
    --kv_cache_dtype fp8
    --enable_prefix_caching
    --gpu-memory-utilization 0.8
    --max-num-batched-tokens 16384
    --max-num-seqs "$((2 * CONC))"
    --cudagraph-capture-sizes "$CUDAGRAPH_CAPTURE_SIZES"
    --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","vision_tower","multi_modal_projector","patch_merge_mlp","*block_sparse_moe"]}'
    "${SPEC_ARGS[@]}"
)
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
