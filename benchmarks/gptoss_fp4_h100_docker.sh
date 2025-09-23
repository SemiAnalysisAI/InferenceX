#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# MAX_MODEL_LEN
# TP
# CONC

cat > config.yaml << EOF
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

export PYTHONNOUSERSITE=1

set -x
vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config=config.yaml \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC  \
--disable-log-requests
