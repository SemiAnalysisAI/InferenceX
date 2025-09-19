#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-sglang-deepseek-r1-fp8.html#run-the-inference-benchmark

export SGLANG_USE_AITER=1

set -x
python3 -m sglang.launch_server \
--model-path=$MODEL --host=0.0.0.0 --port=$PORT --trust-remote-code \
--tensor-parallel-size=$TP \
--mem-fraction-static=0.8 \
--cuda-graph-max-bs=128 \
--chunked-prefill-size=196608 \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=196608 \
--disable-radix-cache
