#!/usr/bin/env bash
set -eo pipefail
set -x

# Client-only AgentX trace replay for single- and multi-node srt-slurm jobs.
# srt-slurm owns server startup; this script runs as benchmark.type=custom
# against the already-ready frontend. Multi-node batches replay each CONC_LIST
# point in turn; a single-node job replays its one CONC point.

# Jobs inherit the legacy scripts' /workspace, which srt-slurm does not mount;
# fall back to the repo mount this client runs from.
if [[ ! -f "${INFMAX_CONTAINER_WORKSPACE:-}/benchmarks/benchmark_lib.sh" ]]; then
    INFMAX_CONTAINER_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
: "${IS_MULTINODE:=false}" "${PORT:=8000}"
export INFMAX_CONTAINER_WORKSPACE IS_MULTINODE PORT
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh" --validation-only
check_env_vars RESULT_DIR EVAL_ONLY
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

if [[ -n "${SRT_FRONTEND_HOST:-}" ]]; then
    check_env_vars SRT_FRONTEND_PORT
    export AIPERF_SERVER_URL="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}"
fi

# benchmark_lib deliberately clears inherited MAX_MODEL_LEN for AgentX so a
# workflow default cannot silently truncate a model's native context. Native
# srt-slurm topologies may still expose a smaller, explicit service limit (for
# example when both P/D roles are configured identically below model-native
# context). Restore that limit only through this dedicated opt-in.
if [[ -n "${AIPERF_MAX_CONTEXT_LENGTH:-}" ]]; then
    if ! [[ "$AIPERF_MAX_CONTEXT_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: AIPERF_MAX_CONTEXT_LENGTH must be a positive integer" >&2
        exit 1
    fi
    export MAX_MODEL_LEN="$AIPERF_MAX_CONTEXT_LENGTH"
fi

check_env_vars \
    MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC \
    RESULT_FILENAME DURATION

if [[ -z "${AIPERF_SERVER_URL:-}" ]]; then
    if [[ -n "${SRT_FRONTEND_HOST:-}" ]]; then
        export AIPERF_SERVER_URL="http://${SRT_FRONTEND_HOST}:${SRT_FRONTEND_PORT}"
    else
        export AIPERF_SERVER_URL="http://localhost:${PORT}"
    fi
fi
echo "Using srt-slurm frontend endpoint: $AIPERF_SERVER_URL"

# A router frontend does not re-export engine metrics; read them from each worker.
if [[ -z "${AIPERF_SERVER_METRICS_URLS:-}" && "${SRTCTL_FRONTEND_TYPE:-}" != dynamo ]]; then
    endpoints="${SRT_AGG_ENDPOINTS:-${SRT_PREFILL_ENDPOINTS:+$SRT_PREFILL_ENDPOINTS,}${SRT_DECODE_ENDPOINTS:-}}"
    if [[ -n "${endpoints%,}" ]]; then
        AIPERF_SERVER_METRICS_URLS=$(sed -E 's#([^,]+)#http://\1/metrics#g' <<< "${endpoints%,}")
        export AIPERF_SERVER_METRICS_URLS
    fi
fi

BASE_RESULT_DIR="${RESULT_DIR}"
BASE_RESULT_FILENAME="$RESULT_FILENAME"
read -r -a CONCURRENCIES <<< "${CONC_LIST:-$CONC}"
if (( ${#CONCURRENCIES[@]} > 1 )); then
    check_env_vars AIPERF_DRAIN_TIMEOUT_SECONDS AIPERF_DRAIN_POLL_SECONDS
fi

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
if [[ "${EVAL_ONLY}" == "true" ]]; then
    _wait_for_openai_chat_route --port "$PORT"
fi

wait_for_agentic_servers_idle() {
    local timeout_seconds="${AIPERF_DRAIN_TIMEOUT_SECONDS}"
    local poll_seconds="${AIPERF_DRAIN_POLL_SECONDS}"
    local frontend_metrics_url="${AIPERF_SERVER_URL%/}/metrics"

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
            worker_active += metric_sum(worker_metrics, "trtllm_num_requests_running")
            worker_active += metric_sum(worker_metrics, "trtllm_num_requests_waiting")
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
    # Multi-node collection expects per-point names; a single-node job keeps the workflow's.
    if [[ -n "${CONC_LIST:-}" ]]; then
        export RESULT_FILENAME="${BASE_RESULT_FILENAME}_conc${concurrency}"
        RESULT_DIR="${BASE_RESULT_DIR}/conc_${concurrency}"
    fi

    mkdir -p "$RESULT_DIR"

    echo "Running agentic concurrency $concurrency of: ${CONCURRENCIES[*]}"
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"

    if [ "$index" -lt "$(( ${#CONCURRENCIES[@]} - 1 ))" ]; then
        wait_for_agentic_servers_idle
    fi
done
