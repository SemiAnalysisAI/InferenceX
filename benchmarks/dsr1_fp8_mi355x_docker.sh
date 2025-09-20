#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# PORT

export HF_MODULES_CACHE="/tmp/hf_modules_cache/"

set -x
python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tp $TP --cuda-graph-max-bs $CONC --disable-radix-cache
