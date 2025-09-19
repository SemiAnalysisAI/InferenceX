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
# PORT

# Create llama-config.yml inline
# For 1k/1k, use batch_wait_max_tokens_ratio and batch_wait_timeout_iters will improve the performance, by default they are all zeros
if [[ "$ISL" == "1024" && "$OSL" == "1024" && ${TP} -lt 8 ]]; then
cat > llama-config.yml << 'EOF'
batch_wait_max_tokens_ratio: 0.9
batch_wait_timeout_iters: 20
cuda_graph_config: 
  enable_padding: true 
  max_batch_size: 1024 
kv_cache_config: 
  dtype: fp8 
  enable_block_reuse: false 
stream_interval: 10
EOF
else 
cat > llama-config.yml << 'EOF'
cuda_graph_config: 
  enable_padding: true 
  max_batch_size: 1024 
kv_cache_config: 
  dtype: fp8 
  enable_block_reuse: false 
stream_interval: 10
EOF
fi

set -x
# Launch TRT-LLM server
mpirun -n 1 --allow-run-as-root --oversubscribe trtllm-serve $MODEL --tp_size $TP --trust_remote_code \
--max_seq_len $MAX_MODEL_LEN --max_num_tokens 16384 --extra_llm_api_options llama-config.yml --port $PORT
