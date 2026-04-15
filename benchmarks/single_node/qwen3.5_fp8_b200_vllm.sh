#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((ISL + OSL + 200))}
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    CALCULATED_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

apply_yarn_config_if_needed "$MODEL" "$CALCULATED_MAX_MODEL_LEN"

PREFIX_CACHING_CONFIG="no-enable-prefix-caching: true"
if is_isb1_replay_benchmark || is_isb1_kv_stress_benchmark; then
    PREFIX_CACHING_CONFIG=""
fi

cat > config.yaml << EOF
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"fuse_allreduce_rms":true,"eliminate_noops":true}}'
$PREFIX_CACHING_CONFIG
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
max-model-len: $CALCULATED_MAX_MODEL_LEN
EOF

apply_vllm_offload_config

export TORCH_CUDA_ARCH_LIST="10.0"
export PYTHONNOUSERSITE=1
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl

if ! is_isb1_kv_stress_benchmark; then
    start_gpu_monitor
fi

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
--config config.yaml \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size "$TP" \
--max-num-seqs 256 \
--disable-log-requests \
--trust-remote-code $VLLM_OFFLOAD_EXTRA_ARGS $VLLM_EXTRA_ARGS \
> "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_single_node_benchmark \
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
    --server-pid "$SERVER_PID" \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

if ! is_isb1_kv_stress_benchmark; then
    stop_gpu_monitor
fi
set +x
