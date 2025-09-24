#!/usr/bin/bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

FUSION_FLAG='{'\
'"pass_config": {"enable_fi_allreduce_fusion": true, "enable_attn_fusion": true, "enable_noop": true},'\
'"custom_ops": ["+quant_fp8", "+rms_norm"],'\
'"cudagraph_mode": "FULL_DECODE_ONLY",'\
'"splitting_ops": []'\
'}'
cat > config.yaml <<-EOF
kv-cache-dtype: fp8
compilation-config: '$FUSION_FLAG'
async-scheduling: true
no-enable-prefix-caching: true
max-num-batched-tokens: 8192
max-model-len: $MAX_MODEL_LEN
EOF

cat config.yaml  # Debugging

export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'
export PYTHONNOUSERSITE=1

set -x
vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=$TP \
--max-num-seqs=512 \
--config config.yaml \
--disable-log-requests
