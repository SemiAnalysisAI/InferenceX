#!/usr/bin/env bash

_IX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

export PREFILL_MODEL_HOST_DIR=/nfsdata/sa/models
export DECODE_MODEL_HOST_DIR=/nfsdata/sa/models
export IMAGE=lmsysorg/sglang-rocm:v0.5.10rc0-rocm700-mi30x-20260420

export REBUILD_LIBBNXT_IN_CONTAINER=1
export PATH_TO_BNXT_TAR_PACKAGE=/workspace/driver/libbnxt_re-237.1.137.0.tar.gz

export USE_SUDO_FOR_DOCKER=0

export PREFILL_NODE="45.63.71.103"
export DECODE_NODE_1="137.220.60.12"
export DECODE_NODE_2="144.202.59.123"
export PREFILL_TP=8
export DECODE_TP=16
export DECODE_EP=16
export DECODE_DP_ATTN=true
bash "${_IX_ROOT}/run_1p1d_tp16_sglang_mi300_mi325x.sh"
