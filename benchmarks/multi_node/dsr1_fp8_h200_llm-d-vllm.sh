#!/usr/bin/env bash
#
# Wrapper for the DSR1-FP8 H200 wide-EP llm-d-vllm benchmark.
# Sets topology env (PREFILL_NODES, DECODE_NODES) and calls
# benchmarks/multi_node/llm-d/submit.sh, which prints JOB_ID on stdout.
# Same shape as benchmarks/multi_node/dsr1_fp4_mi355x_sglang-disagg.sh.

set -euo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    MODEL_PATH \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/llm-d" || exit 1

export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# Concurrency list passes through to bench server. Use 'x'-delimited form
# (matches sglang-disagg wrapper convention).
JOB_ID=$(bash ./submit.sh \
    "$PREFILL_NODES" \
    "$DECODE_NODES" \
    "$ISL" "$OSL" "${CONC_LIST// /x}" inf \
    "$RANDOM_RANGE_RATIO")

if [[ -z "$JOB_ID" ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
