#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
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
    SGL_SLURM_JOBS_PATH \
    RANDOM_RANGE_RATIO

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

git clone --branch sa-260203 https://github.com/billishyahao/sglang_disagg.git

cd "$SGL_SLURM_JOBS_PATH" || exit 1

# Set up SGL launch script-specific environment variables
export TIME_LIMIT="08:00:00"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME="DeepSeek-R1"
export CONTAINER_IMAGE=$IMAGE

export PREFILL_ENABLE_EP=true
if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
export PREFILL_ENABLE_DP=true
else
export PREFILL_ENABLE_DP=false
fi

export DECODE_ENABLE_EP=true
if [[ "$DECODE_DP_ATTN" == "true" ]]; then
export DECODE_ENABLE_DP=true
else
export DECODE_ENABLE_DP=false
fi

# Launch jobs based on ISL/OSL
# Replace ' ' in CONC_LIST with 'x' such that the concurrency list is represented
# by a list of numbers delimited by 'x'. This is because of how the underlying launch script
# expects the concurrencies.
JOB_ID=$(bash ./submit_disagg.sh $PREFILL_NODES \
    $PREFILL_NUM_WORKERS \
    $DECODE_NODES \
    $DECODE_NUM_WORKERS \
    $ISL $OSL "${CONC_LIST// /x}" inf \
    ${PREFILL_ENABLE_EP} ${PREFILL_ENABLE_DP} \
    ${DECODE_ENABLE_EP} ${DECODE_ENABLE_DP} \
    ${RANDOM_RANGE_RATIO})

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"