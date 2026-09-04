#!/usr/bin/env bash
set -euo pipefail
set -x

# Controlled PowerX AgentX arm shared by B200 and B300: TP4, FP8, no MTP,
# with GPU prefix reuse and no CPU KV offload.
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars TP CONC EP_SIZE KV_OFFLOADING RESULT_DIR DURATION
if [[ "$TP" != 4 || "$EP_SIZE" != 1 || "${PP_SIZE:-1}" != 1 ||
      "${DCP_SIZE:-1}" != 1 || "${PCP_SIZE:-1}" != 1 ||
      "${DP_ATTENTION:-false}" != false || "${SPEC_DECODING:-none}" != none ]]; then
    echo "Error: the PowerX arm requires TP4/EP1, no context/pipeline/attention parallelism, and no MTP" >&2
    exit 1
fi
require_agentic_kv_offload_none

export MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
export MODEL_REVISION="ea5b4f81096f3901c91dea97f81324302495781d"
export EVAL_FRAMEWORK="lm-eval"
export EVAL_TASKS_DIR="$INFERENCEX_REPO_ROOT/utils/evals/gsm8k.yaml"
export ENABLE_AGENTX_POWER=1
export REQUIRE_POWER=1
export AIPERF_FAILED_REQUEST_THRESHOLD=0
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
for simulation_var in ${!SGLANG_SIMULATE_ACC_@}; do
    unset "$simulation_var"
done

mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
install_agentic_deps
# Staged MODEL_PATH directories do not prove a checkpoint revision. Reuse a
# pinned HF cache snapshot when available, otherwise download that revision.
"$AIPERF_HF_CLI" download "$MODEL" --revision "$MODEL_REVISION" --dry-run \
    | tee "$RESULT_DIR/powerx_model_cache.txt"
MODEL_PATH=$("$AIPERF_HF_CLI" download "$MODEL" --revision "$MODEL_REVISION")
export MODEL_PATH
export AIPERF_TOKENIZER="$MODEL_PATH"
resolve_trace_source

verify_trace_revision() {
    "$AIPERF_PYTHON" - <<'PY' | tee -a "$RESULT_DIR/powerx_dataset_revision.txt"
from huggingface_hub import HfApi

revision = HfApi().dataset_info("semianalysisai/cc-traces-weka-062126-256k").sha
print(revision)
if revision != "8fecd2fc56694469f758f0afbbb6335ad3043740":
    raise SystemExit("PowerX AgentX dataset revision changed")
PY
}
verify_trace_revision

{
    printf 'model=%s\nmodel_revision=%s\nmodel_path=%s\nimage=%s\n' \
        "$MODEL" "$MODEL_REVISION" "$MODEL_PATH" "${IMAGE:-unknown}"
    printf 'inferencex_commit=%s\naiperf_commit=%s\n' \
        "$(git -c safe.directory="$INFERENCEX_REPO_ROOT" -C "$INFERENCEX_REPO_ROOT" rev-parse HEAD)" \
        "$(git -c safe.directory="$AIPERF_DIR" -C "$AIPERF_DIR" rev-parse HEAD)"
    printf 'tp=%s\nep=%s\nconcurrency=%s\nduration=%s\nagentx_fast=%s\ncuda_visible_devices=%s\n' \
        "$TP" "$EP_SIZE" "$CONC" "$DURATION" "${AIPERF_EXPERIMENTAL_FAST:-0}" "${CUDA_VISIBLE_DEVICES:-unset}"
    sha256sum "$MODEL_PATH/config.json" "$MODEL_PATH/tokenizer_config.json"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,power.limit --format=csv
} > "$RESULT_DIR/powerx_runtime.txt"
python3 -m pip freeze > "$RESULT_DIR/powerx_server_packages.txt"
"$AIPERF_UV_BIN" pip freeze --python "$AIPERF_PYTHON" > "$RESULT_DIR/powerx_client_packages.txt"

# Concurrency counts session trees; retain room for subagent requests.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export NCCL_NVLS_ENABLE=1
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
export SGLANG_TIMEOUT_KEEP_ALIVE=1800

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
    --enable-symm-mem
    --quantization fp8
    --kv-cache-dtype fp8_e4m3
    --mamba-ssm-dtype bfloat16
    --attention-backend trtllm_mha
    --moe-runner-backend flashinfer_trtllm
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 16384
    --chunked-prefill-size 16384
    --mem-fraction-static 0.80
    --stream-interval 50
    --scheduler-recv-interval 10
    --tokenizer-worker-num 6
    --tokenizer-path "$MODEL_PATH"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --enable-metrics
    --enable-cache-report
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    capture_cache_metrics
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

capture_cache_metrics() {
    {
        echo "=== SGLang cache metrics snapshot $(date --iso-8601=seconds) ==="
        curl -fsS "http://localhost:$PORT/metrics" 2>/dev/null \
            | grep -E '^(sglang:(cache_hit_rate|cached_tokens_total|prompt_tokens_total|token_usage|num_requests_running|num_requests_waiting))' \
            || true
    } >> "$SERVER_LOG"
}

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
capture_cache_metrics

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
    verify_trace_revision
fi
