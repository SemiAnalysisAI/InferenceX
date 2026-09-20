#!/usr/bin/env bash
set -eo pipefail

# Qwen3.8-27B (native BF16 checkpoint of the dense hybrid
# attention model: 48 linear-attention and 16 full attention layers) on one
# H200, served by vLLM with the checkpoint's own MTP head (one MTP layer,
# mtp_num_hidden_layers=1) drafting three tokens per step.
# https://recipes.vllm.ai/Qwen/Qwen3.8-27B
# https://huggingface.co/Qwen/Qwen3.8-27B
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME MAX_MODEL_LEN EVAL_ONLY RUN_EVAL

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  check_env_vars SLURMD_NODENAME
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$TP" -ne 1 ]]; then
  echo "This recipe serves Qwen3.8-27B on a single GPU; got TP=$TP" >&2
  exit 1
fi

# Keep the source FP8 recipe's three-token native MTP workload with BF16 weights.
NUM_SPEC_TOKENS=3

nvidia-smi

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

SERVER_LOG=/workspace/server.log

# Serve the matrix context (isl + osl + slack), not the checkpoint's 262K;
# accuracy evals use the eval context benchmark_lib derives.
MODEL_LEN="$MAX_MODEL_LEN"
if [[ "$EVAL_ONLY" == true ]]; then
    setup_eval_context
    MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

# vLLM's default max-num-seqs (1024) exceeds the GDN/Mamba cache blocks that fit
# next to the weights on the smaller cards (472 next to the bf16 weights on an
# 80 GB H100, run 35357364404) and engine start aborts before graph capture.
# Size the scheduler batch to the sweep point instead; the accuracy eval serves
# up to 256 concurrent requests.
MAX_NUM_SEQS=$(( CONC > 16 ? CONC : 16 ))
if [[ "$EVAL_ONLY" == true ]]; then
    MAX_NUM_SEQS=256
fi

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port

# Native MTP: no draft model, the head ships in the checkpoint (mtp.* tensors).
SPEC_CONFIG=$(printf '{"method":"mtp","num_speculative_tokens":%d}' "$NUM_SPEC_TOKENS")

start_gpu_monitor

VLLM_CMD=(
    vllm serve "$MODEL" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --tensor-parallel-size 1
    # Text-only serving: skip the vision tower of Qwen3_5ForConditionalGeneration.
    --language-model-only
    --trust-remote-code
    # Target and native MTP head retain BF16 weights, compute and KV cache.
    --dtype bfloat16
    --kv-cache-dtype auto
    --max-model-len "$MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    # Every 1k1k request prefills its full random prompt; no prefix-cache hits.
    --no-enable-prefix-caching
    --reasoning-parser qwen3
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
    --speculative-config "$SPEC_CONFIG"
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee /workspace/vllm_command.txt
printf '\n' | tee -a /workspace/vllm_command.txt
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$EVAL_ONLY" == true ]]; then
    run_eval --framework lm-eval --port "$PORT"
    # Non-agentic evals must stage lm-eval's output into the workspace root.
    append_lm_eval_summary
else
    pip install -q datasets pandas
    # Chat-templated prompts preserve realistic MTP acceptance.
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
        --use-chat-template \
        --server-pid "$SERVER_PID"
    if [[ "$RUN_EVAL" == true ]]; then
        run_eval --framework lm-eval --port "$PORT"
        append_lm_eval_summary
    fi
fi

stop_gpu_monitor
