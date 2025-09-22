#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# PORT
export SGLANG_USE_AITER=1

set -x
python3 -m sglang.launch_server --model-path=$MODEL --trust-remote-code \
--host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP \
--chunked-prefill-size=196608 \
--mem-fraction-static=0.8 \
--disable-radix-cache \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=196608 \
--cuda-graph-max-bs=128

