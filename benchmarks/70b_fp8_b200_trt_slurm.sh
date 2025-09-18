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

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

hf download $MODEL
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))


set -x

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

# Launch TRT-LLM server
mpirun -n 1 --oversubscribe --allow-run-as-root trtllm-serve $MODEL --tp_size $TP --trust_remote_code --max_seq_len $MAX_MODEL_LEN --max_num_tokens 16384 --extra_llm_api_options llama-config.yml --port $PORT > $SERVER_LOG 2>&1 &


set +x
while IFS= read -r line; do
    printf '%s\n' "$line"
    if [[ "$line" =~ [Ee][Rr][Rr][Oo][Rr] ]]; then
        sleep 5
        tail -n100 $SERVER_LOG
        echo "JOB $SLURM_JOB_ID ran on NODE $SLURMD_NODENAME"
        exit 1
    fi
    if [[ "$line" == *"Application startup complete"* ]]; then
        break
    fi
done < <(tail -F -n0 "$SERVER_LOG")

set -x
git clone https://github.com/kimbochen/bench_serving.git
python3 bench_serving/benchmark_serving.py \
--model $MODEL --backend openai \
--base-url http://0.0.0.0:$PORT \
--dataset-name random \
--random-input-len $ISL --random-output-len $OSL --random-range-ratio $RANDOM_RANGE_RATIO \
--num-prompts $(( $CONC * 10 )) --max-concurrency $CONC \
--request-rate inf --ignore-eos \
--save-result --percentile-metrics 'ttft,tpot,itl,e2el' \
--result-dir /workspace/ \
--result-filename $RESULT_FILENAME.json
