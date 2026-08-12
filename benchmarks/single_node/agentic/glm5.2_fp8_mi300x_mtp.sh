#!/usr/bin/env bash
set -eo pipefail
set -x

# GLM-5.2 FP8 on one 8xMI300X node with native EAGLE MTP.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING PORT EVAL_ONLY \
    RESULT_DIR DURATION

require_agentic_kv_offload_none

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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

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

export PYTHONNOUSERSITE=1
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_DSA_FUSE_TOPK=false
export SGLANG_OPT_USE_TOPK_V2=false

# Throughput replays use the committed GLM-5.2 thinking-on K=3 golden AL;
# evals retain real target-model verification.
if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
    export SGLANG_SIMULATE_ACC_LEN=2.99
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

MAX_RUNNING_REQUESTS=$((2 * CONC))

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --ep-size "$EP_SIZE"
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    --dsa-topk-backend torch
    --kv-cache-dtype bfloat16
    --tool-call-parser glm47
    --reasoning-parser glm45
    --context-length 1048576
    --max-total-tokens 1048576
    --chunked-prefill-size 131072
    # Full 131072-token DSA prefills need transient workspace beyond the KV
    # pool; 0.80 leaves enough headroom on 192 GB MI300X ranks.
    --mem-fraction-static 0.80
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$MAX_RUNNING_REQUESTS"
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --watchdog-timeout 1800
    --enable-metrics
    --enable-cache-report
)

write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    # Aggregate serving exposes one logical SGLang Prometheus target.
    export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
    export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --benchmark-grace-period 1800"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
