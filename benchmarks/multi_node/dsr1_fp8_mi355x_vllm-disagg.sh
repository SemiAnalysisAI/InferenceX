#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    DECODE_NUM_WORKERS \
    DECODE_TP \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/vllm_disagg_utils" || exit 1

export TIME_LIMIT="08:00:00"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# PREFILL_NODES and DECODE_NODES come from additional-settings in the YAML config.
# NODELIST (optional) constrains which Slurm nodes are used.

JOB_ID=$(bash ./submit.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $ISL $OSL "${CONC_LIST// /x}" inf "${NODELIST:-}" \
    ${RANDOM_RANGE_RATIO})

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
