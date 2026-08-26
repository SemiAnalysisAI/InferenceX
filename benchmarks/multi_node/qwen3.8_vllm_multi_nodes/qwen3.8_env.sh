#!/usr/bin/env bash
# Shared defaults for the Qwen3.8 two-node vLLM Ray runtime.
#
# Override the host snapshot root with QWEN38_MODEL_PATH or MODEL_PATH before
# launching benchmarks or running verify_model_staging.sh.

: "${QWEN38_MODEL_PATH:=/models/Qwen/Qwen3.8-2.4T-A95B-FP8}"
# Always apply the Qwen3.8 snapshot root when this file is sourced; the generic
# multinode launcher must not leave MODEL_PATH=/it-share/data in place.
export MODEL_PATH="$QWEN38_MODEL_PATH"
