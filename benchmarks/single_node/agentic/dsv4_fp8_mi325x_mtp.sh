#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Pro FP8 AgentX replay on one 8xMI325X node. The checkpoint is
# dequantized to FP8 because gfx942 has no native MXFP4 support. The published
# path is pure TP8: current stable and nightly vLLM builds both return invalid
# empty-content responses with expert parallelism on this model/SKU.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL IMAGE TP CONC KV_OFFLOADING RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ "$KV_OFFLOADING" != "none" ]]; then
    echo "ERROR: DeepSeek-V4 MTP on MI325X currently supports GPU-resident KV only" >&2
    exit 1
fi

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

resolve_trace_source
install_agentic_deps
agentic_pip_install --quiet Pillow fastapi uvicorn

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
export VLLM_ENGINE_READY_TIMEOUT_S=10800
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export PYTHONNOUSERSITE=1

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
mkdir -p "$RESULT_DIR"

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [[ "$DP_ATTENTION" == "true" ]]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if (( EP_SIZE > 1 )); then
    EP_ARGS=(--enable-expert-parallel)
fi

USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
ROUTER_PID=""
if [[ "$DP_ATTENTION" == "true" ]]; then
    if (( EP_SIZE != TP )); then
        echo "ERROR: MI325X DP-attention requires EP_SIZE == TP so FP8 experts remain sharded" >&2
        exit 1
    fi
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    export AIPERF_SERVER_METRICS_URLS="http://localhost:${VLLM_BACKEND_PORT}/metrics"
    agentic_pip_install --quiet 'vllm-router==0.1.14'
fi

MAX_NUM_SEQS=$((2 * CONC))
# The existing MI300X/MI325X fixed-sequence recipes use K=2. Keep that MTP
# depth and its measured golden acceptance length for every AgentX point.
NUM_SPEC_TOKENS=2
SYNTHETIC_ACCEPT_LEN=2.27
if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"
else
    SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":${SYNTHETIC_ACCEPT_LEN}}"
fi

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$ROUTER_PID" "vLLM router"
    stop_background_process_tree "${SERVER_PID:-}" "vLLM server" 60
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    --trust-remote-code
    --async-scheduling
    --distributed-executor-backend mp
    --quantization deepseek_v4_fp8
    --kv-cache-dtype fp8
    "${PARALLEL_ARGS[@]}"
    "${EP_ARGS[@]}"
    --gpu-memory-utilization 0.9
    --block-size 256
    --max-num-batched-tokens 16384
    --max-num-seqs "$MAX_NUM_SEQS"
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --speculative-config "$SPEC_CONFIG"
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --reasoning-parser deepseek_v4
    --enable-auto-tool-choice
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-disable-hybrid-kv-cache-manager
)

printf '%q ' "${VLLM_CMD[@]}" > "$RESULT_DIR/vllm_command.txt"
printf '\n' >> "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$USE_VLLM_ROUTER" == "true" ]]; then
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy consistent_hash \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$((PORT + 10000))" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    # Full-context AgentX responses can remain healthy for several minutes
    # after the admission window closes. Let already-admitted requests drain
    # so their observed TTFT/ITL enters the strict coverage calculation. This
    # does not extend admissions, change the workload, or lower the 98% gate.
    REPLAY_CMD+=" --benchmark-grace-period 1800"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
