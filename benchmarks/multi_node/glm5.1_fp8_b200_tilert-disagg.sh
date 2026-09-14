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

export MODEL_NAME=glm5
export TILERT_MODEL_TYPE=glm-5
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-202752}"

export DECODE_KV_DTYPE=fp8
export PREFILL_KV_DTYPE=fp8_ds_mla

export TILERT_PARSER=none

exec bash "$(dirname "$0")/tilert_utils/submit.sh"
