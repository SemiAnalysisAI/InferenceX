#!/usr/bin/env bash

# DeepSeek-V4-Pro single-node TRTLLM recipe for B300. The configured image
# already contains NVIDIA/TensorRT-LLM@feat/deepseek_v4; do not build TRTLLM at
# runtime from this benchmark path.

source "$(dirname "$0")/../benchmark_lib.sh"

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

export TRTLLM_DSV4_USE_MPIRUN="${TRTLLM_DSV4_USE_MPIRUN:-1}"
export TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV="${TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV:-1}"

sanitize_slurm_mpi_env_for_trtllm() {
    if [[ "${TRTLLM_DSV4_SANITIZE_SLURM_MPI_ENV:-0}" != "1" ]]; then
        return 0
    fi

    echo "Sanitizing Slurm/PMI environment for TensorRT-LLM launch"
    while IFS='=' read -r name _; do
        case "$name" in
            SLURM_*|PMIX*|PMI*|OMPI_*|ORTE_*)
                unset "$name"
                ;;
        esac
    done < <(env)
}

sanitize_slurm_mpi_env_for_trtllm

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
echo "NCCL_NVLS_ENABLE: $NCCL_NVLS_ENABLE"

if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

nvidia-smi

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}
EXTRA_CONFIG_FILE="dsv4-fp4-trt.yml"

MOE_BACKEND="TRTLLM"
MAX_BATCH_SIZE=$(( CONC > 16 ? CONC : 16 ))
CUDA_GRAPH_MAX_BATCH_SIZE="$MAX_BATCH_SIZE"
KV_CACHE_FREE_MEM_FRACTION="${KV_CACHE_FREE_MEM_FRACTION:-0.50}"

ATTENTION_DP_CONFIG=""
if [[ "$DP_ATTENTION" == "true" ]]; then
    ATTENTION_DP_CONFIG="
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60"
fi

cat > "$EXTRA_CONFIG_FILE" << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CUDA_GRAPH_MAX_BATCH_SIZE
enable_attention_dp: $DP_ATTENTION$ATTENTION_DP_CONFIG
print_iter_log: true
kv_cache_config:
    tokens_per_block: 128
    dtype: fp8
    free_gpu_memory_fraction: $KV_CACHE_FREE_MEM_FRACTION
    enable_block_reuse: false
stream_interval: 10
num_postprocess_workers: 4
moe_config:
    backend: $MOE_BACKEND
EOF

echo "Generated config file contents:"
cat "$EXTRA_CONFIG_FILE"

MAX_MODEL_LEN=$(( MAX_MODEL_LEN > 8192 ? MAX_MODEL_LEN : 8192 ))
MAX_NUM_TOKENS=$(( ISL + OSL + 256 ))
MAX_NUM_TOKENS=$(( MAX_NUM_TOKENS > 8192 ? MAX_NUM_TOKENS : 8192 ))

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
    MAX_NUM_TOKENS="$EVAL_MAX_MODEL_LEN"
fi

# DeepSeek-V4-Pro has hidden size 7168. The current TRTLLM fused-HC MHC
# path corrupts eval generations for this shape; keep eval servers on the
# unfused path until the fused kernel is guarded or supports 7168.
export TRTLLM_MHC_ENABLE_FUSED_HC=0
echo "TRTLLM_MHC_ENABLE_FUSED_HC: $TRTLLM_MHC_ENABLE_FUSED_HC"

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

set -x
SERVE_CMD=(
    trtllm-serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust_remote_code \
    --backend pytorch \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_seq_len "$MAX_MODEL_LEN" \
    --max_num_tokens "$MAX_NUM_TOKENS" \
    --tp_size "$TP" \
    --ep_size "$EP_SIZE" \
    --custom_tokenizer deepseek_v4 \
    --config "$EXTRA_CONFIG_FILE"
)

if [[ "${TRTLLM_DSV4_USE_MPIRUN:-1}" == "0" ]]; then
    "${SERVE_CMD[@]}" > "$SERVER_LOG" 2>&1 &
else
    mpirun -n 1 --oversubscribe --allow-run-as-root \
        "${SERVE_CMD[@]}" \
        > "$SERVER_LOG" 2>&1 &
fi

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$(( CONC * 10 ))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$PWD/" \
    --trust-remote-code \
    --server-pid "$SERVER_PID"

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
