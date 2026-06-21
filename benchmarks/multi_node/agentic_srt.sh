#!/usr/bin/env bash
set -euo pipefail
set -x

# Client-only agentic trace replay for srt-slurm multinode jobs.
# srt-slurm owns server startup; this script runs as benchmark.type=custom
# against the already-ready frontend on the head node.

INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/infmax-workspace}"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

check_env_vars MODEL MODEL_PREFIX FRAMEWORK PRECISION CONC RESULT_FILENAME DURATION

RESULT_DIR="${RESULT_DIR:-/logs/agentic}"

mkdir -p "$RESULT_DIR"

configure_slurm_server_metrics() {
    # srt-slurm allocates DYN_SYSTEM_PORT sequentially across worker
    # processes. Recipes opt in with the base port and describe which nodes
    # are logical worker leaders. This lets the benchmark obtain the dynamic
    # hostnames from Slurm without querying Dynamo discovery.
    if [ -z "${AIPERF_SERVER_METRICS_BASE_PORT:-}" ]; then
        return
    fi

    local nodelist="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
    local node_offset="${AIPERF_SERVER_METRICS_NODE_OFFSET:-0}"
    local node_stride="${AIPERF_SERVER_METRICS_NODE_STRIDE:-1}"
    local -a nodes metrics_urls
    local node node_index process_index metrics_port

    if [ -z "$nodelist" ]; then
        echo "ERROR: SLURM_JOB_NODELIST is required to discover worker metrics endpoints" >&2
        return 1
    fi
    if ! command -v scontrol >/dev/null 2>&1; then
        echo "ERROR: scontrol is required to expand SLURM_JOB_NODELIST" >&2
        return 1
    fi
    if ! [[ "$AIPERF_SERVER_METRICS_BASE_PORT" =~ ^[0-9]+$ ]] ||
       ! [[ "$node_offset" =~ ^[0-9]+$ ]] ||
       ! [[ "$node_stride" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: server metrics base port, node offset, and node stride must be non-negative integers (stride > 0)" >&2
        return 1
    fi

    while IFS= read -r node; do
        nodes+=("$node")
    done < <(scontrol show hostnames "$nodelist")
    if [ "${#nodes[@]}" -le "$node_offset" ]; then
        echo "ERROR: Slurm allocation has no worker nodes after offset $node_offset" >&2
        return 1
    fi

    for ((node_index = node_offset; node_index < ${#nodes[@]}; node_index += node_stride)); do
        process_index=$((node_index - node_offset))
        metrics_port=$((AIPERF_SERVER_METRICS_BASE_PORT + process_index))
        metrics_urls+=("http://${nodes[$node_index]}:${metrics_port}/metrics")
    done

    local IFS=,
    AIPERF_SERVER_METRICS_URLS="${metrics_urls[*]}"
    export AIPERF_SERVER_METRICS_URLS
    echo "AIPerf worker metrics endpoints: $AIPERF_SERVER_METRICS_URLS"
}

resolve_trace_source
install_agentic_deps

configure_slurm_server_metrics
build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
