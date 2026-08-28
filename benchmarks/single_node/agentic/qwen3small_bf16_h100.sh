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

install_agentic_deps
# The canonical AgentX corpus deliberately contains only long-context traces;
# its 256k-capped variant has no trajectories that fit Qwen3-0.6B's native
# 40,960-token window. This plumbing smoke therefore uses a checked-in Weka
# trajectory with the same growing-prefix shape. AgentX marks the result as an
# unsafe/non-submission run because the fixture is local and intentionally tiny.
export TRACE_SOURCE_FLAG="--input-file /workspace/utils/agentic/fixtures/vllm_cache_source_weka --custom-dataset-type weka_trace"
export AIPERF_UNSAFE_OVERRIDE=true

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

# Match the benchmark request URL's hostname so AIPerf de-duplicates its
# auto-discovered endpoint and this explicit metrics endpoint.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
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

# Scrapers establish a counter baseline from their first observation. Ensure
# every built-in physical tier exists at zero before warmup traffic, otherwise
# tokens served before a newly labelled series first appears are lost from the
# exported delta.
python3 - "http://127.0.0.1:${PORT}" <<'PY'
import math
import sys

from utils.validate_vllm_cache_source_metrics import snapshot

builtins = {"device", "cpu", "disk", "mixed", "external"}
observed = snapshot(sys.argv[1]).cached_by_source
if set(observed) != builtins:
    raise SystemExit(
        f"startup cache-source labels differ: expected {sorted(builtins)}, "
        f"observed {observed}"
    )
nonzero = {source: value for source, value in observed.items() if not math.isclose(value, 0)}
if nonzero:
    raise SystemExit(f"startup cache-source counters are not zero: {nonzero}")
print(f"validated startup cache-source series: {observed}")
PY

if [[ "$EVAL_ONLY" == "true" ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"

    python3 - "$RESULT_DIR/aiperf_artifacts/server_metrics_export.json" \
        "$EXPECTED_CACHE_SOURCE" "$AIPERF_SERVER_METRICS_URLS" <<'PY'
import json
import math
import sys

path = sys.argv[1]
expected_source = sys.argv[2]
expected_endpoint = sys.argv[3]
with open(path) as file:
    metrics = json.load(file).get("metrics", {})

def series_totals(name, label=None):
    entry = metrics.get(name)
    if not isinstance(entry, dict):
        raise SystemExit(f"missing {name} export")
    totals = {}
    for series in entry.get("series", []):
        value = series.get("stats", {}).get("total")
        if value is None:
            continue
        key = series.get("labels", {}).get(label) if label else "total"
        if key is not None:
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


cached_total = sum(series_totals("vllm:prompt_tokens_cached").values())
physical = series_totals("vllm:prompt_tokens_cached_by_source", "source")
logical = series_totals("vllm:prompt_tokens_by_source", "source")

physical_entry = metrics["vllm:prompt_tokens_cached_by_source"]
endpoints = {
    series.get("endpoint_url")
    for series in physical_entry.get("series", [])
    if series.get("endpoint_url") is not None
}
if endpoints != {expected_endpoint}:
    raise SystemExit(
        f"cache-source export contains duplicate or unexpected endpoints: {endpoints}"
    )

builtins = {"device", "cpu", "disk", "mixed", "external"}
if set(physical) != builtins:
    raise SystemExit(
        f"exported cache-source labels differ: expected {sorted(builtins)}, "
        f"observed {physical}"
    )
unexpected_positive = {
    source: value
    for source, value in physical.items()
    if source not in {"device", "cpu"} and not math.isclose(value, 0)
}
if unexpected_positive:
    raise SystemExit(f"unexpected positive cache-source totals: {unexpected_positive}")
if cached_total <= 0:
    raise SystemExit(f"no cached prompt tokens were exported: {cached_total}")
if physical.get(expected_source, 0) <= 0:
    raise SystemExit(
        f"expected positive {expected_source!r} cached-token samples: {physical}"
    )
physical_total = sum(physical.values())
logical_total = sum(
    logical.get(source, 0)
    for source in ("local_cache_hit", "external_kv_transfer")
)
for description, observed in (
    ("physical cache-source", physical_total),
    ("logical cache-source", logical_total),
):
    if not math.isclose(observed, cached_total, rel_tol=0, abs_tol=0.5):
        raise SystemExit(
            f"{description} total does not conserve cached tokens: "
            f"observed={observed}, cached={cached_total}"
        )
print(
    "validated AgentX cached-token conservation: "
    f"cached={cached_total}, physical={physical}, logical={logical}"
)
PY
fi
