#!/usr/bin/env bash
# Manual test script for MI300X disaggregated inference
# Run from the repo root: bash scripts/manual-test-mi300x.sh
export GITHUB_WORKSPACE=$(pwd)
export RUNNER_NAME=mi300x-amds-manual

export MODEL=deepseek-ai/DeepSeek-R1-0528
export EXP_NAME=dsr1_1k1k
export PRECISION=fp8
export FRAMEWORK=sglang-disagg

export IMAGE=ghcr.io/jordannanos/sgl-mi300x-mori:v0.5.9-bnxt

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

bash runners/launch_mi300x-amds.sh
