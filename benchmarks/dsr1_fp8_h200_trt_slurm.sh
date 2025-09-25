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

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"

hf download $MODEL

# ========= Determine DP_ATTENTION, EP_SIZE and MOE_BACKEND based on ISL, OSL, CONC =========
EP_SIZE="$TP"
MOE_BACKEND="CUTLASS"
DP_ATTENTION=false

if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    if [[ $CONC -gt 64 ]]; then
        DP_ATTENTION=true
    fi
elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
    if [[ $CONC -gt 64 ]]; then
        DP_ATTENTION=true
    fi
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
    if [[ $CONC -gt 32 ]]; then
        DP_ATTENTION=true
    fi
fi

echo "Final configuration: EP_SIZE='$EP_SIZE', MOE_BACKEND='$MOE_BACKEND', DP_ATTENTION='$DP_ATTENTION'"

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))
EXTRA_CONFIG_FILE="dsr1-fp8.yml"

cat > $EXTRA_CONFIG_FILE << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 128
enable_attention_dp: $DP_ATTENTION
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.75
    enable_block_reuse: false 
stream_interval: 10
moe_config:
    backend: $MOE_BACKEND
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
    cat << EOF >> $EXTRA_CONFIG_FILE
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
fi

set -x

MAX_NUM_TOKENS=$(( ($CONC+$ISL+64+63)/64*64 ))

# Launch TRT-LLM server
PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve $MODEL --port=$PORT \
    --trust_remote_code \
    --backend=pytorch \
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
