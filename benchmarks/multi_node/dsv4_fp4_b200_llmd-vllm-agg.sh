#!/usr/bin/env bash
#
# Wrapper for the DeepSeek-V4-Pro B200 llmd-vllm aggregated benchmark
# (agentX-flavored TP8 / DEP8, one engine does both prefill and decode).
# Sibling of dsv4_fp4_b200_llmd-vllm-disagg.sh - same shape, but always
# submits with DECODE_NODES=0 (no decode role at all; see the aggregated
# recipes' header comments and server.sh's IS_AGGREGATED handling). The
# runner resolves this script via
#   SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_b200_llmd-vllm-agg.sh"
# from launch_b200-dgxc-slurm.sh when DISAGG=false.

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

if [[ "$DECODE_NODES" != "0" ]]; then
    echo "Error: dsv4_fp4_b200_llmd-vllm-agg.sh requires DECODE_NODES=0 (got $DECODE_NODES); aggregated mode has no decode role" >&2
    exit 1
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/llm-d" || exit 1

# B200 DGX = 8 GPUs per node (submit.sh defaults to 8, explicit for clarity).
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# Aggregated is always a single engine (no multi-engine high-tpt split), so
# PREFILL_WORKERS is always 1. DECODE_WORKERS is unused (DECODE_NODES=0) but
# still exported since submit.sh/server.sh read it unconditionally.
export PREFILL_WORKERS="${PREFILL_WORKERS:-1}"
export DECODE_WORKERS="${DECODE_WORKERS:-1}"

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
