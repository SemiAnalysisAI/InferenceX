#!/usr/bin/env bash
#
# Wrapper for the DeepSeek-V4-Pro GB200 llmd-vllm P/D disagg benchmark
# (mid-curve 1P1D and high-tpt 2P1D). Sibling of gptoss_fp4_h200_llmd-vllm.sh -
# same shape, different topology (GB200 = 4 GPUs/node, role spans 2 nodes;
# H200 = 8 GPUs/node, role on a single node). The runner resolves this script via
#   SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_gb200_llmd-vllm-disagg.sh"
# from launch_gb200-nv.sh.

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

# GB200 = 4 GPUs per node (Grace+Blackwell). The shared submit.sh
# defaults GPUS_PER_NODE to 8, which is wrong for this SKU and would
# overshoot DP_SIZE = nodes * 8.
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# Worker count per role (Option B multi-engine). Prefer an explicit
# PREFILL_WORKERS/DECODE_WORKERS from the matrix additional-settings; else fall
# back to the matrix num-worker fields (PREFILL_NUM_WORKERS/DECODE_NUM_WORKERS);
# else 1 (single engine = unchanged 1P+1D / mid-curve). submit.sh reads these.
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
