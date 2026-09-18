#!/usr/bin/env bash
set -euo pipefail
set -x

# Qwen3.8-Flash-Next Quark MXFP4 (BF16->FP8 PLE, "PLEFP8") AgentX on MI355X
# with SGLang native NEXTN MTP. TP1: the ~126 GiB MXFP4 checkpoint fits on a
# single MI355X (288 GB), so no tensor parallel is needed. Serve flags mirror
# the verified single-node bring-up command; the NEXTN speculative card
# (--speculative-algorithm NEXTN, 3 steps, eagle-topk 1, 4 draft tokens) is the
# same shape the B200/B300/H200 qwen3.8next SGLang AgentX arms run:
# https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
# Checkpoint: https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8
# Quantization is read from the checkpoint (Quark MXFP4 MoE; FP8 PLE). The model
# ships its own multi-step-trained MTP head, so NEXTN needs no external drafter.
# Per the AgentX policy (MODELS.md) agentic recipes run with speculative
# decoding only: throughput pins acceptance to the committed golden AL, evals
# retain real target-model verification.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"
check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
require_agentic_kv_offload_none

if [[ "$EP_SIZE" != 1 ]]; then
    echo "Error: this recipe supports EP_SIZE=1 only" >&2
    exit 1
fi

# Weights are pre-staged on the shared filesystem; read them directly instead
# of downloading from the Hub (avoids gated/cache-permission issues).
export MODEL_PATH=/it-share/data/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8

rocm-smi || true
amd-smi || true

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
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

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
# Honor the explicit --mem-fraction-static instead of AITER's autotuned value.
export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

if [ "${EVAL_ONLY:-false}" != "true" ]; then
    # golden_al_distribution/qwen3.8next_mtp.yaml:
    # qwen3.8-flash-next-fp8.thinking_on[3] = 2.32. --speculative-num-steps 3
    # with 4 draft tokens is 3 speculative tokens per verification step, i.e.
    # the MTP=3 cell; AgentX replays run with thinking on. Same value as the
    # B200/B300/H200 qwen3.8next SGLang arms. EVAL_ONLY leaves simulated
    # acceptance off so evals score real verification.
    export SGLANG_SIMULATE_ACC_LEN=2.32
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path /it-share/data/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp-size "$TP"
    --ep-size "$EP_SIZE"
    --attention-backend aiter
    --moe-runner-backend aiter
    --mamba-ssm-dtype bfloat16
    --page-size 32
    --kv-cache-dtype auto
    --chunked-prefill-size 16384
    --watchdog-timeout 1200
    # TP1 keeps the whole ~126 GiB checkpoint on one 288 GB MI355X, so 0.9 still
    # leaves ample static headroom for weights, KV, and the NEXTN draft head's
    # verification batch. Matches the verified single-node bring-up command.
    --mem-fraction-static 0.9
    --model-loader-extra-config '{"enable_multithread_load": true}'
    # NEXTN silently resets --max-running-requests to 48 when it is unset, so
    # this must stay explicit and sized to the AgentX concurrency.
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --reasoning-parser qwen3-thinking
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path /it-share/data/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8
    --enable-metrics
    --enable-cache-report
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
    # build_replay_cmd points aiperf's client-side --tokenizer at $MODEL (a
    # gated HF id); redirect it to the pre-staged local checkpoint so no Hub
    # access is needed. The server loads its tokenizer from the same path.
    REPLAY_CMD="${REPLAY_CMD/--tokenizer $MODEL/--tokenizer $MODEL_PATH}"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
