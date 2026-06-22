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

# The AgentX scenario's first-turn cache-bust marker includes AIPerf's unique
# per-invocation benchmark ID. Each point therefore gets a disjoint KV keyspace
# while its own warmup and profile phases share markers. This makes sequential
# points comparable without restarting the engines or inheriting warmed trace
# prefixes from an earlier concurrency.
for concurrency in "${CONCURRENCIES[@]}"; do
    export CONC="$concurrency"
    export RESULT_FILENAME="${BASE_RESULT_FILENAME}_conc${concurrency}"
    RESULT_DIR="${BASE_RESULT_DIR}/conc_${concurrency}"
    mkdir -p "$RESULT_DIR"

    echo "Running agentic concurrency $concurrency of: ${CONCURRENCIES[*]}"
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
done
