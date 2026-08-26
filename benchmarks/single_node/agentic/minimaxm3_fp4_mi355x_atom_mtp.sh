#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for MiniMax-M3 MXFP4 on MI355X using ATOM
# with native MTP speculative decoding.
#
# Serve shape follows the upstream ATOM recipe:
#   https://github.com/ROCm/ATOM/blob/main/recipes/Qwen3.5.md
#     python -m atom.entrypoints.openai_server --model <model> \
#       --kv_cache_dtype fp8 -tp 4 --method mtp --num-speculative-tokens 3
# Everything ATOM does not require is left at its default on purpose, so this
# recipe stays the upstream-faithful arm. MiniMax-M3 ships native MTP modules
# (config.json text_config.num_mtp_modules = 7), so no external drafter is
# loaded.
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

# The upstream recipe is TP4, which is also what MiniMax-M3's four KV heads
# want: one KV head per rank keeps the AITER sparse-attention fast path.
if [ "$TP" -ne 4 ] || [ "$EP_SIZE" -ne 1 ] || [ "$DP_ATTENTION" != "false" ]; then
    echo "This recipe requires TP=4, EP_SIZE=1, and DP_ATTENTION=false" >&2
    exit 1
fi
require_agentic_kv_offload_none

# ROCR/HIP visibility
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

NUM_SPEC_TOKENS=3
# golden_al_distribution/minimaxm3_eagle3.yaml: minimax-m3.thinking_on[3].
# AgentX pins every submission for a model to one golden acceptance curve, so
# the native-MTP arm targets the same acceptance length as the EAGLE3 arm.
SPEC_DECODE_AL=2.83

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

echo "Starting atom server..."
export PYTHONNOUSERSITE=1
# Without this the aiter kernel logs flood the server log for the whole replay.
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"

# ---- Speculative ------------------------------------------------------------
# Synthetic acceptance standardizes throughput against the committed golden
# curve. Accuracy evals must use real target verification.
SPEC_ARGS=(
    --method mtp
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
)
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
fi
echo "NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS SPEC_DECODE_AL=$SPEC_DECODE_AL"

# ---- LLM server -------------------------------------------------------------
# AgentX concurrency counts session trees. Keep 2x scheduler headroom for the
# request bursts produced by subagent fan-out. Every other server knob is left
# at the ATOM default, as the upstream recipe leaves it.
ATOM_CMD=(
    python3 -u -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    --tensor-parallel-size "$TP"
    --trust-remote-code
    --kv_cache_dtype fp8
    --max-num-seqs "$((2 * CONC))"
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
