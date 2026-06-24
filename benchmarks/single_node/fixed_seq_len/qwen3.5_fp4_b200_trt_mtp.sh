#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    DP_ATTENTION \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

# MTP (multi-token prediction) speculative decode requires the FlashInfer GDN
# prefill path to be disabled.
export TLLM_USE_FLASHINFER_GDN_PREFILL="0"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

nvidia-smi

SERVER_LOG=/workspace/server.log
EXTRA_CONFIG_FILE="qwen3.5-fp4-trt-mtp.yml"
NUM_NEXTN_PREDICT_LAYERS=3

# Attention-DP layouts run CUTEDSL MoE; everything else runs the TRTLLM backend.
# With MTP the served batch is much smaller than raw concurrency: attention-DP
# runs at CONC/8, everything else at CONC. The KV-cache memory fraction is tuned
# per layout (there is no single derivable rule).
if [[ "$DP_ATTENTION" == "true" ]]; then
    MAX_BATCH_SIZE=$(( CONC / 8 ))
    MOE_BACKEND="CUTEDSL"
    # attention-DP: 0.9 up to conc 512, backed off to 0.8 at conc 1024.
    if (( CONC >= 1024 )); then KV_MEMORY_FRACTION=0.8; else KV_MEMORY_FRACTION=0.9; fi
    MODE_CONFIG="enable_attention_dp: true
attention_dp_config:
    enable_balance: true
    batching_wait_iters: 10
    timeout_iters: 500"
else
    MAX_BATCH_SIZE="$CONC"
    MOE_BACKEND="TRTLLM"
    # non-attention-DP fraction, tuned per (ISL, TP, EP) layout.
    case "${ISL}_tp${TP}_ep${EP_SIZE}" in
        1024_tp2_ep1) KV_MEMORY_FRACTION=0.6 ;;
        1024_tp2_ep2) KV_MEMORY_FRACTION=0.75 ;;
        1024_tp8_ep8) KV_MEMORY_FRACTION=0.8 ;;
        8192_tp2_ep1) KV_MEMORY_FRACTION=0.7 ;;
        8192_tp2_ep2) KV_MEMORY_FRACTION=0.6 ;;
        8192_tp4_ep4) KV_MEMORY_FRACTION=0.75 ;;
        8192_tp8_ep8) KV_MEMORY_FRACTION=0.8 ;;
        *)            KV_MEMORY_FRACTION=0.8 ;;
    esac
    # Short-context runs hold less in flight, so they wait on a tighter token
    # ratio before flushing a batch.
    case "$ISL" in
        1024) BATCH_WAIT_MAX_TOKENS_RATIO=0.0625 ;;
        *)    BATCH_WAIT_MAX_TOKENS_RATIO=0.45 ;;
    esac
    MODE_CONFIG="batch_wait_timeout_iters: 50
batch_wait_max_tokens_ratio: $BATCH_WAIT_MAX_TOKENS_RATIO"
fi

cat > "$EXTRA_CONFIG_FILE" << EOF
backend: pytorch
print_iter_log: true
enable_layerwise_nvtx_marker: false
disable_overlap_scheduler: false
enable_iter_perf_stats: true
enable_chunked_prefill: false
stream_interval: 20
num_postprocess_workers: 4
scheduler_config:
    capacity_scheduler_policy: MAX_UTILIZATION
    context_chunking_policy: FIRST_COME_FIRST_SERVED
kv_cache_config:
    free_gpu_memory_fraction: $KV_MEMORY_FRACTION
    enable_block_reuse: false
    dtype: fp8
cuda_graph_config:
    enable_padding: true
    batch_sizes:
    - 1
    - 2
    - 4
    - 8
    - 16
    - 32
    - 64
    - 128
moe_config:
    backend: $MOE_BACKEND
    use_low_precision_moe_combine: true
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: $NUM_NEXTN_PREDICT_LAYERS
$MODE_CONFIG
EOF

echo "Generated config file contents:"
cat "$EXTRA_CONFIG_FILE"

MAX_MODEL_LEN=$(( MAX_MODEL_LEN > 8192 ? MAX_MODEL_LEN : 8192 ))

case "${ISL}_${OSL}" in
    8192_1024) MAX_NUM_TOKENS=32768 ;;
    1024_1024) MAX_NUM_TOKENS=16384 ;;
    *)
        MAX_NUM_TOKENS=$(( ISL + OSL + 256 ))
        MAX_NUM_TOKENS=$(( MAX_NUM_TOKENS > 8192 ? MAX_NUM_TOKENS : 8192 ))
        ;;
esac

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
    MAX_NUM_TOKENS="$EVAL_MAX_MODEL_LEN"
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve "$MODEL" --port="$PORT" \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_seq_len="$MAX_MODEL_LEN" \
    --max_num_tokens="$MAX_NUM_TOKENS" \
    --tp_size="$TP" --ep_size="$EP_SIZE" \
    --extra_llm_api_options="$EXTRA_CONFIG_FILE" \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend openai \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$(( CONC * 10 ))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --use-chat-template

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
