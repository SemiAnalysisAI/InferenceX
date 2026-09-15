#!/usr/bin/env bash
set -euo pipefail
set -x

# Qwen3.8-Flash-Next Quark MXFP4 AgentX on MI355X with SGLang native NEXTN
# MTP. Based on the upstream MI355X balanced FP8 recipe plus the cookbook's
# "NEXTN / MTP" speculative card (--speculative-algorithm NEXTN, 3 steps,
# eagle-topk 1, 4 draft tokens), the same shape the B200/B300/H200
# qwen3.8next SGLang AgentX arms run:
# https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
# Checkpoint: https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4
# Quantization is read from the checkpoint (Quark MXFP4 MoE; BF16 PLE). The
# model ships its own multi-step-trained MTP head, so NEXTN needs no external
# drafter. Per the AgentX policy (MODELS.md) agentic recipes run with
# speculative decoding only: throughput pins acceptance to the committed
# golden AL, evals retain real target-model verification.

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

# Let HF validate/resume downloads even when a local directory is nonempty.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

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
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
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
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp-size "$TP"
    --ep-size "$EP_SIZE"
    --attention-backend aiter
    --page-size 32
    --kv-cache-dtype auto
    --chunked-prefill-size 16384
    --watchdog-timeout 1200
    # MTP: leave non-static headroom for the NEXTN draft head's verification
    # batch and AITER spec-decode workspaces. The cookbook STP cell runs 0.9;
    # the B300 NVFP4 MTP sibling runs 0.80 and the H200 FP8 one 0.85. The
    # ~126 GiB checkpoint is ~16 GB/GPU across TP8 on 288 GB parts, so 0.85
    # still leaves a ~229 GB/GPU static share for weights plus KV.
    --mem-fraction-static 0.85
    --model-loader-extra-config '{"enable_multithread_load": true}'
    # NEXTN silently resets --max-running-requests to 48 when it is unset, so
    # this must stay explicit and sized to the AgentX concurrency.
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    # Decode-specific spelling as the MI355X Qwen3.5/DeepSeek-V4/GLM-5.2 SGLang
    # MTP arms use; recent SGLang splits --cuda-graph-max-bs into
    # decode/prefill variants and rejects the old prefix as ambiguous.
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
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
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
