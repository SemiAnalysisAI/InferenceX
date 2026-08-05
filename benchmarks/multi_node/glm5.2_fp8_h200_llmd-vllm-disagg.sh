#!/usr/bin/env bash
#
# Wrapper for the GLM-5.2-FP8 H200 llmd-vllm P/D disagg benchmark.
# The runner resolves this script via
#   SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_h200_llmd-vllm-disagg.sh"
# from launch_h200-dgxc-slurm.sh.

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

# H200 DGX = 8 GPUs per node (submit.sh defaults to 8, explicit for clarity).
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

export PREFILL_WORKERS="${PREFILL_WORKERS:-${PREFILL_NUM_WORKERS:-1}}"
export DECODE_WORKERS="${DECODE_WORKERS:-${DECODE_NUM_WORKERS:-1}}"

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
