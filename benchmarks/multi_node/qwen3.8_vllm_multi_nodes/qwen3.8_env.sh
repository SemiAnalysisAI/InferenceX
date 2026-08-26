#!/usr/bin/env bash
# Shared defaults for the Qwen3.8 two-node vLLM Ray runtime.
#
# Override the host snapshot root with QWEN38_MODEL_PATH or MODEL_PATH before
# launching benchmarks or running verify_model_staging.sh.

: "${QWEN38_MODEL_PATH:=/models/Qwen/Qwen3.8-2.4T-A95B-FP8}"
export MODEL_PATH="${MODEL_PATH:-$QWEN38_MODEL_PATH}"
