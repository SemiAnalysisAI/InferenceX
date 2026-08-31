#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using
# ATOM MTP. Throughput runs use the committed golden synthetic acceptance;
# eval-only runs use the model's real MTP acceptance.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

require_agentic_kv_offload_none

echo "Attention mode: $([ "$DP_ATTENTION" = "true" ] && echo dp || echo tp) (DP_ATTENTION=$DP_ATTENTION, CONC=$CONC)"

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

resolve_trace_source
install_agentic_deps

# ATOM runtime settings validated with the DeepSeek-V4-Pro AgentX baseline.
export AITER_BF16_FP8_MOE_BOUND=0
export AITER_LOG_LEVEL=WARNING
export ATOM_MOE_GU_ITLV=1
export ATOM_DISABLE_MMAP=true
export ATOM_DEBUG_PREFIX_HITS=1
export ATOM_PROFILER_MORE=0
export ATOM_PROFILER_TIMEOUT=1200

# DP-attention runs layer ATOM's DPA routing and two-batch-overlap knobs on top of
# the TP settings above (recipe section "Server - DP attention"); exported only for
# the DP band. ATOM_DP_SESSION_AFFINITY is not optional: without it a session's
# turns scatter across DP ranks, the prefix KV written by one turn is unreachable
# by the next, and the multi-turn agentic workload collapses to cold prefill.
# GPU_MAX_HW_QUEUES and ATOM_NUMA_BIND are prerequisites of --enable-tbo.
DP_ATTN_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    export GPU_MAX_HW_QUEUES=5
    export ATOM_NUMA_BIND=1
    export ATOM_DP_SESSION_AFFINITY=1
    export ATOM_DP_LB_REQ_EQUIV=512
    export ATOM_ENABLE_PREFILL_DELAYER=1
    export ATOM_PREFILL_DECODE_INTERVAL=10
    # Client-side counterpart to session affinity: make AIPerf emit a stable
    # session id (x-dynamo-session-id, falling back to the always-sent
    # x-correlation-id) so the DPA router pins each conversation to one rank.
    export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
    DP_ATTN_ARGS=(--enable-dp-attention --enable-tbo)
fi

# Raise the AIPerf HTTP TCP user timeout to 900000 ms (15 min), well above the
# aiperf default of 30000 ms (30 s), so long-stalling AgentX request
# connections are not torn down as dead during extended server-side pauses.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true

# Require ATOM Prometheus metrics in every official result.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

wait_for_amd_gpu_clean

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

# AgentX concurrency counts session trees. Keep 2x scheduler headroom for the
# request bursts produced by subagent fan-out.
MAX_NUM_SEQS=$((2 * CONC))

# golden_al_distribution/dsv4_mtp.yaml: thinking_on, 3 draft tokens -> AL 2.49
# https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/dsv4_mtp.yaml
NUM_SPEC_TOKENS=3
SPEC_DECODE_AL=2.49
SPEC_ARGS=(
    --method mtp
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
)
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
fi

echo "Starting ATOM server with MAX_NUM_SEQS=$MAX_NUM_SEQS NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS SPEC_DECODE_AL=$SPEC_DECODE_AL EVAL_ONLY=${EVAL_ONLY:-false}"
ATOM_CMD=(
    python3 -u -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    --tensor-parallel-size "$TP"
    --kv-cache-dtype fp8
    --index-cache-dtype fp4
    --enable-prefix-caching
    --gpu-memory-utilization 0.9
    --max-num-batched-tokens 16384
    --attn-prefill-chunk-size 16384
    --state-checkpoint-interval-tokens 8192
    --level 3
    --cudagraph-mode FULL
    "${SPEC_ARGS[@]}"
    "${DP_ATTN_ARGS[@]}"
    --max-num-seqs "$MAX_NUM_SEQS"
)
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    # AgentX DSv4 traces already carry fully formed chat payloads; do not apply
    # AIPerf's generic chat template on top of them.
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
