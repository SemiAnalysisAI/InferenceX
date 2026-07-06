#!/usr/bin/env bash
set -euo pipefail
set -x

# Client-only agentic trace replay for srt-slurm multinode jobs.
# srt-slurm owns server startup; this script runs as benchmark.type=custom
# against the already-ready frontend on the head node.

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC RESULT_FILENAME DURATION

BASE_RESULT_DIR="${RESULT_DIR:-/logs/agentic}"
BASE_RESULT_FILENAME="$RESULT_FILENAME"
read -r -a CONCURRENCIES <<< "${CONC_LIST:-$CONC}"

if [ "${#CONCURRENCIES[@]}" -eq 0 ]; then
    echo "ERROR: CONC_LIST must contain at least one concurrency" >&2
    exit 1
fi
for concurrency in "${CONCURRENCIES[@]}"; do
    if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid agentic concurrency: $concurrency" >&2
        exit 1
    fi
done

resolve_trace_source
install_agentic_deps

wait_for_agentic_servers_idle() {
    local timeout_seconds="${AIPERF_DRAIN_TIMEOUT_SECONDS:-1800}"
    local poll_seconds="${AIPERF_DRAIN_POLL_SECONDS:-10}"
    local frontend_metrics_url="http://localhost:${PORT}/metrics"

    "$AIPERF_PYTHON" - \
        "$timeout_seconds" \
        "$poll_seconds" \
        "$frontend_metrics_url" \
        "${AIPERF_SERVER_METRICS_URLS:-}" <<'PY'
import sys
import time
import urllib.request

timeout_seconds = int(sys.argv[1])
poll_seconds = int(sys.argv[2])
frontend_url = sys.argv[3]
worker_urls = [url for url in sys.argv[4].split(",") if url]
deadline = time.monotonic() + timeout_seconds
idle_polls = 0


def fetch_metrics(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")


def metric_sum(metrics: str, name: str) -> float:
    total = 0.0
    for line in metrics.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2 or fields[0].split("{", 1)[0] != name:
            continue
        total += float(fields[1])
    return total


while time.monotonic() < deadline:
    try:
        frontend_metrics = fetch_metrics(frontend_url)
        frontend_active = metric_sum(frontend_metrics, "dynamo_frontend_active_requests")
        worker_active = 0.0
        for worker_url in worker_urls:
            worker_metrics = fetch_metrics(worker_url)
            worker_active += metric_sum(worker_metrics, "vllm:num_requests_running")
            worker_active += metric_sum(worker_metrics, "vllm:num_requests_waiting")
        print(
            f"Agentic drain status: frontend_active={frontend_active:g} "
            f"worker_running_or_waiting={worker_active:g}",
            flush=True,
        )
        if frontend_active == 0 and worker_active == 0:
            idle_polls += 1
            if idle_polls >= 3:
                print("Agentic servers remained idle for three polls", flush=True)
                raise SystemExit(0)
        else:
            idle_polls = 0
    except Exception as error:
        idle_polls = 0
        print(f"Agentic drain metrics query failed: {error}", file=sys.stderr, flush=True)
    time.sleep(poll_seconds)

raise SystemExit(f"Agentic servers did not drain within {timeout_seconds} seconds")
PY
}

# The AgentX scenario's first-turn cache-bust marker includes AIPerf's unique
# per-invocation benchmark ID. Each point therefore gets a disjoint KV keyspace
# while its own warmup and profile phases share markers. This makes sequential
# points comparable without restarting the engines or inheriting warmed trace
# prefixes from an earlier concurrency.
for index in "${!CONCURRENCIES[@]}"; do
    concurrency="${CONCURRENCIES[$index]}"
    export CONC="$concurrency"
    export RESULT_FILENAME="${BASE_RESULT_FILENAME}_conc${concurrency}"
    RESULT_DIR="${BASE_RESULT_DIR}/conc_${concurrency}"

    mkdir -p "$RESULT_DIR"

    echo "Running agentic concurrency $concurrency of: ${CONCURRENCIES[*]}"
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"

    if [ "$index" -lt "$(( ${#CONCURRENCIES[@]} - 1 ))" ]; then
        wait_for_agentic_servers_idle
    fi
done
