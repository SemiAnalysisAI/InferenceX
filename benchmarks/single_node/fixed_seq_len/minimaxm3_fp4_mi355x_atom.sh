#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE \
    DP_ATTENTION 

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

SERVER_LOG=/workspace/server.log

PARALLEL_ARGS=(-tp "$TP") #TP
if [ "$DP_ATTENTION" = "true" ]; then
    if [ "$EP_SIZE" -gt 1 ]; then #DP+EP
        PARALLEL_ARGS=(-tp "$TP" --enable-expert-parallel --enable-dp-attention )
    else #DP+TP
        PARALLEL_ARGS=(-tp "$TP" --enable-dp-attention )
    fi
fi 

SPEC_ARGS=()
OPT_ARGS=(--online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}')

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor
MEM_FRAC_STATIC=0.8

set -x
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export MAX_MODEL_LEN=32768
export MAX_NUM_BATCHED_TOKENS=32768
export MAX_NUM_SEQS=256
python3 -m atom.entrypoints.openai_server \
    --model $MODEL \
    --server-port $PORT \
    "${PARALLEL_ARGS[@]}" \
    "${SPEC_ARGS[@]}" \
    "${OPT_ARGS[@]}" \
    --block-size 128 \
    --gpu-memory-utilization $MEM_FRAC_STATIC \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-seqs $MAX_NUM_SEQS \
    --kv_cache_dtype fp8 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

export PYTHONDONTWRITEBYTECODE=1
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code $( [[ ${#SPEC_ARGS[@]} -gt 0 ]] && echo "--use-chat-template" )

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x