#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# TP
# CONC
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# RESULT_FILENAME
# PORT_OFFSET

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL"

hf download $MODEL

# ========= Determine DP_ATTENTION, EP_SIZE and MOE_BACKEND based on ISL, OSL, CONC, TP =========
EP_SIZE="1"
MOE_BACKEND="TRTLLM"
DP_ATTENTION=false
MTP=3

if [[ "$TP" == "4" ]]; then
    if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
        if [[ $CONC -ge 16 ]]; then
            EP_SIZE="$TP"
        fi
        if [[ $CONC -ge 128 ]]; then
            DP_ATTENTION=true
            MOE_BACKEND="CUTLASS"
            MTP=1
        fi
    elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
        if [[ $CONC -ge 32 ]]; then
            EP_SIZE="$TP"
        fi
        if [[ $CONC -ge 128 ]]; then
            DP_ATTENTION=true
            MOE_BACKEND="CUTLASS"
            MTP=1
        fi
    elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
        if [[ $CONC -ge 32 ]]; then
            EP_SIZE="$TP"
            DP_ATTENTION=true
            MOE_BACKEND="CUTLASS"
            MTP=1
        fi
    fi
elif [[ "$TP" == "8" ]]; then
    if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
        if [[ $CONC -ge 16 ]]; then
            EP_SIZE="$TP"
        fi
        if [[ $CONC -ge 64 ]]; then
            DP_ATTENTION=true
            MOE_BACKEND="CUTLASS"
            MTP=1
        fi
    elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
        if [[ $CONC -ge 8 ]]; then
            EP_SIZE="$TP"
        fi
        if [[ $CONC -ge 128 ]]; then
            DP_ATTENTION=true
            MOE_BACKEND="CUTLASS"
            MTP=1
        fi
    elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
        if [[ $CONC -ge 32 ]]; then
            EP_SIZE="$TP"
            DP_ATTENTION=true
            MOE_BACKEND="CUTLASS"
            MTP=1
        fi
    fi
fi

echo "EP_SIZE='$EP_SIZE', MOE_BACKEND='$MOE_BACKEND', DP_ATTENTION='$DP_ATTENTION', MTP='$MTP'"

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))
EXTRA_CONFIG_FILE="dsr1-fp4-mtp.yml"

cat > $EXTRA_CONFIG_FILE << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 512
enable_attention_dp: $DP_ATTENTION
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
    enable_block_reuse: false 
stream_interval: 10
moe_config:
    backend: $MOE_BACKEND
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: ${MTP}
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
    cat << EOF >> $EXTRA_CONFIG_FILE
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
fi

if [[ "$DP_ATTENTION" == "true" ]]; then
    MAX_BATCH_SIZE=$((CONC/TP))
else
    MAX_BATCH_SIZE=$CONC
fi

MAX_NUM_TOKENS=$(( ((MTP+1)*MAX_BATCH_SIZE+ISL+64+63)/64*64 ))

set -x
# Launch TRT-LLM server
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve $MODEL --port=$PORT \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size=$MAX_BATCH_SIZE \
    --max_seq_len=$MAX_MODEL_LEN \
    --max_num_tokens=$MAX_NUM_TOKENS \
    --tp_size=$TP --ep_size=$EP_SIZE \
    --extra_llm_api_options=$EXTRA_CONFIG_FILE \
    > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend openai \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --use-chat-template
