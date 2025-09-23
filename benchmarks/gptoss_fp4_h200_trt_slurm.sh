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


set +x

export TRTLLM_ENABLE_PDL=1 

set -x
cat > gptoss-config.yml << EOF
cuda_graph_config:
  enable_padding: true
  max_batch_size: $CONC
enable_attention_dp: false
kv_cache_config:
  dtype: auto
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.85
moe_config:
  backend: TRITON
num_postprocess_workers: 4
print_iter_log: true
stream_interval: 20 
EOF


#mpirun -n 1 --oversubscribe --allow-run-as-root trtllm-serve $MODEL --tp_size $TP --trust_remote_code --max_seq_len $MAX_MODEL_LEN --max_num_tokens $MAX_MODEL_LEN --num_postprocess_workers 2 --extra_llm_api_options llama-config.yml --port $PORT > $SERVER_LOG 2>&1 &
mpirun -n 1 --oversubscribe --allow-run-as-root trtllm-serve $MODEL --max_batch_size $CONC --max_num_tokens 20000 --backend pytorch --extra_llm_api_options gptoss-config.yml  --ep_size=$TP --trust_remote_code --gpus_per_node 8 --host 0.0.0.0 --port $PORT --tp_size=$TP --pp_size=1 > $SERVER_LOG 2>&1 &


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
