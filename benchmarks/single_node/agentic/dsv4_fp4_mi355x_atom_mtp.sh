#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using
# ATOM MTP. TP throughput runs use the committed golden synthetic acceptance;
# DEP and eval-only runs use the model's real MTP acceptance.

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

# EP is config-driven so the TP band remains TP-only while DEP uses one expert
# shard per GPU.
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# The high-concurrency band uses ATOM's native RCCL DEP transport. Session
# affinity is required: otherwise consecutive turns can land on another DPA
# rank and lose access to the prefix KV produced by the previous turn.
DEP_ARGS=()
STATE_CHECKPOINT_INTERVAL_TOKENS=8192
if [ "$DP_ATTENTION" = "true" ]; then
    if [ "$EP_SIZE" -ne "$TP" ]; then
        echo "ERROR: native RCCL DEP requires EP_SIZE=$TP for TP=$TP, got EP_SIZE=$EP_SIZE" >&2
        exit 1
    fi
    # Keep only runtime controls that are not already expressed by DEP_ARGS.
    export ATOM_DP_SESSION_AFFINITY=1
    export ATOM_DP_LB_REQ_EQUIV=512
    export ATOM_ENABLE_PREFILL_DELAYER=1
    export ATOM_PREFILL_DECODE_INTERVAL=10
    # Client-side counterpart to session affinity: make AIPerf emit a stable
    # session id from its correlation id so the DPA router pins each
    # conversation to one rank.
    export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=1
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
    DEP_ARGS=(
        --enable-dp-attention
        --all2all-backend rccl
        --dp-load-balance least_tokens
        --moe-backend standard
    )
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

# golden_al_distribution/dsv4_mtp.yaml: thinking_on, 3 draft tokens -> AL 2.49.
# https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/dsv4_mtp.yaml
# Native RCCL DEP was validated with the model's real acceptance, so only the
# TP throughput band applies the synthetic golden value.
NUM_SPEC_TOKENS=3
SPEC_DECODE_AL=2.49
SPEC_ARGS=(
    --method mtp
    --num-speculative-tokens "$NUM_SPEC_TOKENS"
)
if [ "${EVAL_ONLY:-false}" != "true" ] && [ "$DP_ATTENTION" != "true" ]; then
    SPEC_ARGS+=(--spec-decode-acceptance-length "$SPEC_DECODE_AL")
fi

echo "Starting ATOM server with MAX_NUM_SEQS=$MAX_NUM_SEQS NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS STATE_CHECKPOINT_INTERVAL_TOKENS=$STATE_CHECKPOINT_INTERVAL_TOKENS DP_ATTENTION=$DP_ATTENTION EP_SIZE=$EP_SIZE EVAL_ONLY=${EVAL_ONLY:-false}"
ATOM_CMD=(
    python3 -u -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    # uvicorn defaults to a 5s idle keep-alive; AIPerf pools sockets for far
    # longer (aiohttp ~15s) and warmup inter-turn gaps under backlog exceed 5s,
    # so the server closes an idle pooled socket and the reused write hits
    # 'Connection reset by peer' (errno 104). One such reset on a root AgentX
    # warmup request aborts the whole run. Outlast the client idle window.
    --timeout-keep-alive 900
    --tensor-parallel-size "$TP"
    --data-parallel-size 1
    --kv-cache-dtype fp8
    --index-cache-dtype fp4
    --enable-prefix-caching
    --gpu-memory-utilization 0.9
    --max-num-batched-tokens 16384
    --attn-prefill-chunk-size 16384
    --state-checkpoint-interval-tokens "$STATE_CHECKPOINT_INTERVAL_TOKENS"
    --level 3
    --cudagraph-mode FULL
    "${SPEC_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${DEP_ARGS[@]}"
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
