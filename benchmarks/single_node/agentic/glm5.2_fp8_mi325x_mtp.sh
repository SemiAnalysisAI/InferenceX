#!/usr/bin/env bash
set -euo pipefail
set -x

# Full-context AgentX refresh for GLM-5.2 FP8 on one 8xMI325X node.
# The serving shape matches Actions run 29657732517 and adds native EAGLE
# MTP with the committed thinking-on golden acceptance length.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

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

# GLM-5.2 natively supports 1M context, so use the complete AgentX corpus.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
SERVER_PID=""
SERVER_PGID=""

cleanup() {
    if [[ -n "$SERVER_PGID" ]]; then
        kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    elif [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi
    [[ -z "$SERVER_PID" ]] || wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

if agentic_kv_offload_enabled; then
    echo "Error: this refreshed baseline supports GPU-resident KV only" >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
export SGLANG_TIMEOUT_KEEP_ALIVE=900
# The MI30X image's DSA top-k v2 JIT includes CUDA-only headers when compiled
# for gfx942. Use the portable Torch path and disable the fused top-k path.
export SGLANG_DSA_FUSE_TOPK=false
export SGLANG_OPT_USE_TOPK_V2=false

# Synthetic rejection sampling is only for performance replay. Evals retain
# real target-model verification. The AL is GLM-5.2 thinking-on, K=3, from
# golden_al_distribution/glm5.2_mtp.yaml.
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
    --mem-fraction-static 0.85
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

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

echo "Starting SGLang server for MI325X..."
setsid "${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
SERVER_PGID=$SERVER_PID
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    # Long AgentX responses admitted near the end of the measurement window can
    # take several minutes to finish. Bound their post-window drain without
    # extending the measured request-admission window.
    REPLAY_CMD+=" --benchmark-grace-period 1800"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
