#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
    MODEL_NAME \
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

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils"

export TIME_LIMIT="08:00:00"
export CONTAINER_IMAGE="$IMAGE"
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"
export MAX_MODEL_LEN=1048576
export WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126}"

[[ "$PREFILL_EP" -eq 1 ]] && export PREFILL_ENABLE_EP=false || export PREFILL_ENABLE_EP=true
[[ "$PREFILL_DP_ATTN" == "true" ]] && export PREFILL_ENABLE_DP=true || export PREFILL_ENABLE_DP=false
[[ "$DECODE_EP" -eq 1 ]] && export DECODE_ENABLE_EP=false || export DECODE_ENABLE_EP=true
[[ "$DECODE_DP_ATTN" == "true" ]] && export DECODE_ENABLE_DP=true || export DECODE_ENABLE_DP=false

bash ./submit.sh \
    "$PREFILL_NODES" "$PREFILL_NUM_WORKERS" \
    "$DECODE_NODES" "$DECODE_NUM_WORKERS" \
    "$ISL" "$OSL" "${CONC_LIST// /x}" inf \
    "$PREFILL_ENABLE_EP" "$PREFILL_ENABLE_DP" \
    "$DECODE_ENABLE_EP" "$DECODE_ENABLE_DP" \
    "$PREFILL_TP" "$DECODE_TP" \
    "$RANDOM_RANGE_RATIO" "${NODELIST:-}"
