#!/usr/bin/env bash
set -eo pipefail

# Qwen3.8-27B (bf16, dense hybrid attention: 48 linear-attention and 16 full
# attention layers) on one MI300X, served by vLLM with the RadixArk DSpark drafter
# (Doopeworld's vLLM-loadable copy) drafting seven tokens per step.
# https://recipes.vllm.ai/Qwen/Qwen3.8-27B
# https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$TP" -ne 1 ]]; then
  echo "This recipe serves Qwen3.8-27B on a single GPU; got TP=$TP" >&2
  exit 1
fi

DRAFT_MODEL="Doopeworld/Qwen3.8-27B-DSpark-vLLM"
# The drafter's trained block size. Its card measured k=4 and k=6 slower than
# k=7 despite the late positions' low acceptance: per-step overhead dominates
# the per-drafted-token cost.
NUM_SPEC_TOKENS=7

# gfx942: the ROCm vLLM arms run Triton attention and keep the KV cache in bf16
# (no calibrated fp8 attention scales on this card); breakable piecewise graph
# capture is off as in the MiniMax-M3 and DSv4.1 Flash gfx942 arms.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PYTHONNOUSERSITE=1

rocm-smi --showmeminfo vram || true

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
hf download "$DRAFT_MODEL"

SERVER_LOG=/workspace/server.log

# Serve the matrix context (isl + osl + slack), not the checkpoint's 262K;
# accuracy evals use the eval context benchmark_lib derives.
MODEL_LEN="${MAX_MODEL_LEN:-$((ISL + OSL + 256))}"
if [[ "${EVAL_ONLY:-false}" == true ]]; then
    setup_eval_context
    MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

# vLLM's default max-num-seqs (1024) exceeds the GDN/Mamba cache blocks that fit
# next to the bf16 weights (472 on an 80 GB H100, run 35357364404) and engine
# start aborts before graph capture. Size the scheduler batch to the sweep point
# instead; the accuracy eval serves up to 256 concurrent requests.
MAX_NUM_SEQS=$(( CONC > 16 ? CONC : 16 ))
if [[ "${EVAL_ONLY:-false}" == true ]]; then
    MAX_NUM_SEQS=256
fi

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port

# Probabilistic draft sampling measured ~23% faster than greedy on the drafter's
# card. Adaptive verification is rejected by vLLM's GDN attention backend for
# this hybrid architecture, so it stays at its default (off).
SPEC_CONFIG=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":%d,"draft_sample_method":"probabilistic"}' "$DRAFT_MODEL" "$NUM_SPEC_TOKENS")

start_gpu_monitor

VLLM_CMD=(
    vllm serve "$MODEL" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --tensor-parallel-size 1
    # Text-only serving: skip the vision tower of Qwen3_5ForConditionalGeneration.
    --language-model-only
    --trust-remote-code
    --attention-backend TRITON_ATTN
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

if [[ "${EVAL_ONLY:-false}" == true ]]; then
    run_eval --framework lm-eval --port "$PORT"
else
    pip install -q datasets pandas
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
        `# Chat-templated prompts: raw random tokens tank draft acceptance.` \
        --use-chat-template \
        --server-pid "$SERVER_PID"
    if [[ "${RUN_EVAL:-false}" == true ]]; then
        run_eval --framework lm-eval --port "$PORT"
        append_lm_eval_summary
    fi
fi

stop_gpu_monitor
