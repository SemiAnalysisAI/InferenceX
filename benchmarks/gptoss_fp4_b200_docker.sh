#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# IMAGE
# MODEL
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# RESULT_FILENAME
# PORT_OFFSET

sed -i '102,108d' /usr/local/lib/python3.12/dist-packages/flashinfer/jit/cubin_loader.py

cat > config.yaml << EOF
compilation-config: '{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_attn_fusion":true,"enable_noop":true},"custom_ops":["+rms_norm"],"cudagraph_mode":"FULL_AND_PIECEWISE"}'
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

export TORCH_CUDA_ARCH_LIST="10.0"
export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'
export PYTHONNOUSERSITE=1
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

set -x
vllm serve $MODEL --host 0.0.0.0 --port $PORT --config config.yaml \
--gpu-memory-utilization 0.9 --tensor-parallel-size $TP --max-num-seqs 512 \
--disable-log-requests
