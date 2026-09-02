#!/usr/bin/env bash
# Client-only AgentX adapter for an already-ready llm-d Envoy frontend.
set -eo pipefail

: "${INFMAX_CONTAINER_WORKSPACE:?Set the repository mount path}"
export MODEL="$MODEL_NAME"
export SERVED_MODEL_NAME="$MODEL_NAME"
export PORT="$VLLM_PORT"
export AIPERF_SERVER_URL="http://localhost:$ENVOY_PORT"
export RESULT_DIR="$BENCHMARK_LOGS_DIR/agentic"
export AGENTIC_OUTPUT_DIR="$BENCHMARK_LOGS_DIR"
export CONC_LIST="${BENCH_MAX_CONCURRENCY//x/ }"
export CONC="${CONC_LIST%% *}"

# Discovery already enumerates every DP-serving node and excludes headless TP
# followers. Scrape vLLM directly, not Envoy or the decode pd-sidecar port.
AIPERF_METRIC_URLS=$(python3 - "$LLMD_ENDPOINTS_FILE" "$VLLM_PORT" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as source:
    endpoints = yaml.safe_load(source)["endpoints"]
addresses = dict.fromkeys(endpoint["address"] for endpoint in endpoints)
if not addresses:
    raise SystemExit("No llm-d serving endpoints available for metrics")
print(",".join(f"http://{address}:{int(sys.argv[2])}/metrics" for address in addresses))
PY
)
export AIPERF_METRIC_URLS
# benchmark_lib.sh forwards this name to AIPerf's --server-metrics argument.
export AIPERF_SERVER_METRICS_URLS="$AIPERF_METRIC_URLS"

IFS=',' read -r -a metrics_urls <<< "$AIPERF_METRIC_URLS"
metrics_probe=$(mktemp /tmp/llmd-metrics.XXXXXX)
trap 'rm -f "$metrics_probe"' EXIT
for metrics_url in "${metrics_urls[@]}"; do
    curl --fail --silent --show-error --connect-timeout 5 --max-time 10 \
        --retry 6 --retry-connrefused --retry-delay 2 \
        "$metrics_url" --output "$metrics_probe"
    if ! grep -q '^vllm:' "$metrics_probe"; then
        echo "ERROR: no vLLM metrics exposed at $metrics_url" >&2
        exit 1
    fi
    echo "vLLM metrics ready: $metrics_url"
done
rm -f "$metrics_probe"
trap - EXIT

exec bash "$INFMAX_CONTAINER_WORKSPACE/benchmarks/multi_node/agentic_srt.sh"
