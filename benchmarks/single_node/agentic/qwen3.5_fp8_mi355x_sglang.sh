#!/usr/bin/env bash
set -euo pipefail
set -x

# Controlled PowerX AgentX replay: FP8 TP4, native decoding, HBM prefix reuse.
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC EP_SIZE KV_OFFLOADING RESULT_DIR DURATION
require_agentic_kv_offload_none
if [[ "$TP" != 4 || "$EP_SIZE" != 1 || "${SPEC_DECODING:-none}" != none ]]; then
    echo "Error: PowerX requires TP4/EP1 without speculative decoding" >&2
    exit 1
fi

export EVAL_FRAMEWORK=lm-eval
export EVAL_TASKS_DIR="$INFERENCEX_REPO_ROOT/utils/evals/gsm8k.yaml"
export ENABLE_AGENTX_POWER=1
export REQUIRE_POWER=1
export AIPERF_FAILED_REQUEST_THRESHOLD=0
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE

mkdir -p "$RESULT_DIR"
install_agentic_deps
guard_powerx_dataset_revision() {
    "$AIPERF_PYTHON" - <<'PY'
from huggingface_hub import HfApi

dataset = "semianalysisai/cc-traces-weka-062126-256k"
expected = "8fecd2fc56694469f758f0afbbb6335ad3043740"
actual = HfApi().dataset_info(dataset).sha
if actual != expected:
    raise SystemExit(f"PowerX dataset revision changed: {actual} != {expected}")
print(f"{dataset}@{actual}")
PY
}
guard_powerx_dataset_revision > "$RESULT_DIR/powerx_dataset_revision.txt"
resolve_trace_source
export MODEL_REVISION=ea5b4f81096f3901c91dea97f81324302495781d
"$AIPERF_HF_CLI" download "$MODEL" --revision "$MODEL_REVISION" --dry-run \
    | tee "$RESULT_DIR/powerx_model_cache.txt"
MODEL_PATH=$("$AIPERF_PYTHON" - <<'PYMODEL'
import os
from huggingface_hub import snapshot_download

print(snapshot_download(os.environ["MODEL"], revision=os.environ["MODEL_REVISION"]))
PYMODEL
)
export MODEL_PATH
export INFERENCEX_TOKENIZER_PATH="$MODEL_PATH"

SERVER_LOG="$RESULT_DIR/server.log"
{
    printf 'model=%s\nmodel_revision=%s\nmodel_path=%s\nimage=%s\n' \
        "$MODEL" "$MODEL_REVISION" "$MODEL_PATH" "${IMAGE:-unknown}"
    printf 'ROCR_VISIBLE_DEVICES=%s\nHIP_VISIBLE_DEVICES=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
        "${ROCR_VISIBLE_DEVICES:-}" "${HIP_VISIBLE_DEVICES:-}" "${CUDA_VISIBLE_DEVICES:-}"
    git -c safe.directory="$INFERENCEX_REPO_ROOT" -C "$INFERENCEX_REPO_ROOT" rev-parse HEAD
    git -c safe.directory="$AIPERF_DIR" -C "$AIPERF_DIR" rev-parse HEAD
    sha256sum "$MODEL_PATH/config.json" "$MODEL_PATH/tokenizer_config.json"
} > "$RESULT_DIR/powerx_runtime.txt"
amd-smi static --json > "$RESULT_DIR/powerx_gpu_identity.json"
python3 -m pip freeze > "$RESULT_DIR/powerx_server_packages.txt"
"$AIPERF_UV_BIN" pip freeze --python "$AIPERF_PYTHON" > "$RESULT_DIR/powerx_client_packages.txt"

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

MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
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
    --attention-backend aiter
    --enable-aiter-allreduce-fusion
    --quantization fp8
    --kv-cache-dtype fp8_e4m3
    --mamba-ssm-dtype bfloat16
    --mem-fraction-static 0.80
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --watchdog-timeout 1200
    --page-size 16
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 16384
    --chunked-prefill-size 16384
    --scheduler-recv-interval 10
    --stream-interval 50
    --tokenizer-worker-num 6
    --tokenizer-path "$MODEL_PATH"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --enable-metrics
    --enable-cache-report
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = true ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
    guard_powerx_dataset_revision >> "$RESULT_DIR/powerx_dataset_revision.txt"
fi
