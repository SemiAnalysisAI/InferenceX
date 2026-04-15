#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export GITHUB_WORKSPACE=$(pwd)
export RUNNER_NAME=mi325x-amd-manual

export MODEL=deepseek-ai/DeepSeek-R1-0528
export EXP_NAME=dsr1_1k1k
export PRECISION=fp8
export FRAMEWORK=sglang-disagg

export IMAGE=ghcr.io/jordannanos/sgl-mi325x-mori:v0.5.9-bnxt-good

export ISL=1024
export OSL=1024
export CONC_LIST="1024 512 256 128 64 32 16 8 4 2 1"
export SPEC_DECODING=none
export RANDOM_RANGE_RATIO=1

export PREFILL_NODES=1
export PREFILL_NUM_WORKERS=1
export PREFILL_TP=4
export PREFILL_EP=1
export PREFILL_DP_ATTN=false

export DECODE_NODES=1
export DECODE_NUM_WORKERS=1
export DECODE_TP=8
export DECODE_EP=1
export DECODE_DP_ATTN=false

bash runners/launch_mi325x-amd.sh

#model files are here:
#/nfsdata/sa/gharunner/gharunners/hf-hub-cache/models--deepseek-ai--DeepSeek-R1-0528