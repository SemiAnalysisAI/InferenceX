#!/usr/bin/env bash
set -euo pipefail
set -x

# Small AgentX integration lane for vLLM cache-source Prometheus metrics.
# This is intentionally a plumbing/observability smoke, not a publishable
# performance submission.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION PORT EVAL_ONLY

if [[ "$TP" != "1" ]]; then
    echo "Error: qwen3small cache-source smoke supports TP=1 only" >&2
    exit 1
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    MODEL_PATH="$MODEL"
fi

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OFFLOAD_ARGS=()
GPU_MEMORY_UTILIZATION=0.80
EXPECTED_CACHE_SOURCE=device
if [[ "$KV_OFFLOADING" == "none" ]]; then
    require_agentic_kv_offload_none
elif require_agentic_kv_offload_backend vllm-simple; then
    # Keep only enough GPU KV for one native-length request. Multiple AgentX
    # lanes then evict one another's prefixes into the CPU tier and exercise
    # physical reload attribution, rather than reporting device hits only.
    GPU_MEMORY_UTILIZATION=0.10
    EXPECTED_CACHE_SOURCE=cpu
    CPU_OFFLOAD_BYTES=$((TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000))
    OFFLOAD_CONFIG=$(printf \
        '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"kv_offload_backend":"cpu","cpu_bytes_to_use":%d,"lazy_offload":false}}' \
        "$CPU_OFFLOAD_BYTES")
    OFFLOAD_ARGS=(--kv-transfer-config "$OFFLOAD_CONFIG")
else
    echo "Error: unsupported KV offload backend: ${KV_OFFLOAD_BACKEND:-unset}" >&2
    exit 1
fi

export AIPERF_SERVER_METRICS_URLS="http://127.0.0.1:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
export PYTHONNOUSERSITE=1
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
# Qwen3-0.6B's native context is 40,960 tokens. Use the same limit for vLLM
# and AgentX trace selection so an oversized warmup request cannot reach CUDA.
export MAX_MODEL_LEN=40960

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size 1
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs 8
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-prefix-caching
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$EVAL_ONLY" == "true" ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"

    python3 - "$RESULT_DIR/aiperf_artifacts/server_metrics_export.json" \
        "$EXPECTED_CACHE_SOURCE" <<'PY'
import json
import sys

path = sys.argv[1]
expected_source = sys.argv[2]
with open(path) as file:
    metrics = json.load(file).get("metrics", {})

entry = metrics.get("vllm:prompt_tokens_cached_by_source")
if not isinstance(entry, dict):
    raise SystemExit("missing vllm:prompt_tokens_cached_by_source export")

totals = {}
for series in entry.get("series", []):
    source = series.get("labels", {}).get("source")
    value = series.get("stats", {}).get("total")
    if source is not None and value is not None:
        totals[source] = totals.get(source, 0.0) + float(value)

unexpected = set(totals) - {"device", "cpu"}
if unexpected:
    raise SystemExit(f"unexpected cache-source labels: {sorted(unexpected)}")
if sum(totals.values()) <= 0:
    raise SystemExit(f"no positive cached-token source samples: {totals}")
if totals.get(expected_source, 0) <= 0:
    raise SystemExit(
        f"expected positive {expected_source!r} cached-token samples: {totals}"
    )
print(f"validated AgentX cached-token sources: {totals}")
PY
fi
