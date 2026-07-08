#!/usr/bin/env bash
# Cluster-agnostic entrypoint for disaggregated benchmarks.
#
# Benchmark recipes own workload intent. The selected runner injects
# MULTINODE_LAUNCHER and owns scheduling, storage, networking, and containers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../benchmark_lib.sh"

check_env_vars \
    MULTINODE_LAUNCHER \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    MODEL \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    PREFILL_EP \
    PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS \
    DECODE_TP \
    DECODE_EP \
    DECODE_DP_ATTN \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO \
    FRAMEWORK

if [[ ! -f "$MULTINODE_LAUNCHER" ]]; then
    echo "Error: runner-provided MULTINODE_LAUNCHER does not exist: $MULTINODE_LAUNCHER" >&2
    exit 1
fi

is_parallel() {
    [[ "${1:-1}" =~ ^[0-9]+$ ]] && (( 10#${1:-1} > 1 ))
}

export PREFILL_ENABLE_EP=false
is_parallel "$PREFILL_EP" && export PREFILL_ENABLE_EP=true
export DECODE_ENABLE_EP=false
is_parallel "$DECODE_EP" && export DECODE_ENABLE_EP=true
export PREFILL_ENABLE_DP="$PREFILL_DP_ATTN"
export DECODE_ENABLE_DP="$DECODE_DP_ATTN"

# Stable benchmark -> runner contract. Cluster-specific values such as
# MODEL_PATH, Slurm settings, host mounts, and network devices are deliberately
# absent and must be supplied by the runner.
export CONTAINER_IMAGE="$IMAGE"
export MULTINODE_BENCHMARK_DIR="$SCRIPT_DIR"
export NODE_LIST="${NODE_LIST:-${NODELIST:-}}"
export SPEC_DECODING="${SPEC_DECODING:-none}"
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

exec bash "$MULTINODE_LAUNCHER"
