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

# GPTOSS TRTLLM Deployment Guide:
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/quick-start-recipe-for-gpt-oss-on-trtllm.md

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"

hf download $MODEL
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))

# ========= Determine DP_ATTENTION, EP_SIZE and MOE_BACKEND based on ISL, OSL, CONC =========
EP_SIZE="1"
MOE_BACKEND="TRTLLM"
DP_ATTENTION=false

# Lower concurrencies: Concurrency < 256
# MoE backend=TRTLLM
# Use TP Attention; Switch to MoE Expert parallel for conurrency >=16 (1k1k and 1k8k)
TEP_REQUIRED=false
if [[ "$TP" == "4" || "$TP" == "8" ]]; then 
    if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
        TEP_REQUIRED=true
    elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
        TEP_REQUIRED=true
    fi
fi
if [[ "$TEP_REQUIRED" == "true" && $CONC -ge 16 ]]; then
    EP_SIZE="$TP"
fi

# Higher concurrencies: Concurrency >= 256
#   MoE Backend = CUTLASS
#   Use DP attention with expert parallel MoE
if [[ $CONC -ge 256 ]]; then
    EP_SIZE="$TP"
    DP_ATTENTION=true
    MOE_BACKEND="CUTLASS"
fi

echo "Final configuration: EP_SIZE='$EP_SIZE', MOE_BACKEND='$MOE_BACKEND', DP_ATTENTION='$DP_ATTENTION'"

EXTRA_CONFIG_FILE="gptoss-fp4.yml"
export TRTLLM_ENABLE_PDL=1

cat > $EXTRA_CONFIG_FILE << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CONC
enable_attention_dp: $DP_ATTENTION
kv_cache_config:
    dtype: auto
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.85
print_iter_log: true
stream_interval: 20
num_postprocess_workers: 4
moe_config:
    backend: $MOE_BACKEND
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
    cat << EOF >> $EXTRA_CONFIG_FILE
attention_dp_config:
    enable_balance: true
EOF
fi

echo "Generated config file contents:"
cat $EXTRA_CONFIG_FILE

set -x

MAX_NUM_TOKENS=20000

# Launch TRT-LLM server
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve $MODEL --port=$PORT \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size 512 \
    --max_seq_len=$MAX_MODEL_LEN \
    --max_num_tokens=$MAX_NUM_TOKENS \
    --tp_size=$TP --ep_size=$EP_SIZE \
    --extra_llm_api_options=$EXTRA_CONFIG_FILE \
    > $SERVER_LOG 2>&1 &


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

git clone https://github.com/kimbochen/bench_serving.git
set -x
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