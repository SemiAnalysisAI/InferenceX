#!/usr/bin/env bash
#
# Wrapper for the DeepSeek-V4-Pro B200 llmd-vllm P/D disagg benchmark
# (agentX-flavored 1P-DEP8/1D-DEP8). Sibling of
# dsv4_fp4_gb200_llmd-vllm-disagg.sh - same shape, different topology
# (B200 = 8 GPUs/node, so each DEP8 role fits on ONE node; GB200 = 4
# GPUs/node, role spans 2 nodes). The runner resolves this script via
#   SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_b200_llmd-vllm-disagg.sh"
# from launch_b200-dgxc-slurm.sh.

set -eo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    IMAGE \
    MODEL_PATH \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO

if [[ "${IS_AGENTIC}" == "1" ]]; then
    check_env_vars DURATION KV_OFFLOADING
    # Positional submit.sh placeholders; AgentX never uses fixed token lengths.
    ISL=0
    OSL=0
else
    check_env_vars ISL OSL
fi

if [[ -n "${SLURM_JOB_ID}" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/llm-d" || exit 1

# B200 DGX = 8 GPUs per node.
export GPUS_PER_NODE="8"

export TIME_LIMIT="08:00:00"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# Worker counts come from the generated matrix.
export PREFILL_WORKERS="$PREFILL_NUM_WORKERS"
export DECODE_WORKERS="$DECODE_NUM_WORKERS"

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
