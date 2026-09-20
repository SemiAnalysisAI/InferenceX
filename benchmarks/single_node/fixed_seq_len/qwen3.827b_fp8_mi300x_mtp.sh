#!/usr/bin/env bash
set -eo pipefail

# Qwen3.8-27B-FP8 (fp8 e4m3 dynamic-activation checkpoint of the dense hybrid
# attention model: 48 linear-attention and 16 full attention layers) on one
# MI300X, served by vLLM with the original BF16 native MTP head (one MTP layer,
# mtp_num_hidden_layers=1) drafting three tokens per step.
# https://recipes.vllm.ai/Qwen/Qwen3.8-27B
# https://huggingface.co/Qwen/Qwen3.8-27B-FP8
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME \
    MAX_MODEL_LEN EVAL_ONLY RUN_EVAL THINKING_MODE

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  check_env_vars SLURMD_NODENAME
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$TP" -ne 1 ]]; then
  echo "This recipe serves Qwen3.8-27B-FP8 on a single GPU; got TP=$TP" >&2
  exit 1
fi

# The three-token recipe uses the measured thinking-on curve from PR #3304.
# Qwen's pinned chat template enables thinking unless explicitly overridden.
if [[ "$THINKING_MODE" != "thinking_on" ]]; then
    echo "This recipe requires thinking_on to match its chat template and golden AL" >&2
    exit 1
fi
NUM_SPEC_TOKENS=3
TARGET_REVISION=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
DRAFT_MODEL=Qwen/Qwen3.8-27B
DRAFT_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

# gfx942: the ROCm vLLM arms run Triton attention and keep the KV cache in bf16
# (no calibrated fp8 attention scales on this card); breakable piecewise graph
# capture is off as in the MiniMax-M3 and DSv4.1 Flash gfx942 arms.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PYTHONNOUSERSITE=1

rocm-smi --showmeminfo vram || true

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ "$MODEL" != /* ]]; then hf download "$MODEL" --revision "$TARGET_REVISION"; fi

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

# Read the measured AL; original BF16 MTP modules must not inherit target FP8.
# Accuracy paths, including combined throughput+eval, use real verification.
MTP_CONFIG_DIR=$(mktemp -d /tmp/inferencex-qwen-mtp.XXXXXX)
python3 -m infx.bench_serving.qwen_mtp \
    --target-model "$MODEL" --target-revision "$TARGET_REVISION" \
    --draft-model "$DRAFT_MODEL" --draft-revision "$DRAFT_REVISION" \
    --tokens "$NUM_SPEC_TOKENS" --thinking-mode "$THINKING_MODE" \
    --eval-only "$EVAL_ONLY" --run-eval "$RUN_EVAL" \
    --golden-file "$INFERENCEX_REPO_ROOT/golden_al_distribution/qwen3.827b_fp8_mtp.yaml" \
    --output-dir "$MTP_CONFIG_DIR"
SPEC_CONFIG=$(cat "$MTP_CONFIG_DIR/speculative-config.json")
HF_OVERRIDES=$(cat "$MTP_CONFIG_DIR/hf-overrides.json")

start_gpu_monitor

VLLM_CMD=(
    vllm serve "$MODEL" --served-model-name "$MODEL"
    --revision "$TARGET_REVISION" --dtype bfloat16
    --hf-overrides "$HF_OVERRIDES"
    --host 0.0.0.0 --port "$PORT"
    --tensor-parallel-size 1
    # Text-only serving: skip the vision tower of Qwen3_5ForConditionalGeneration.
    --language-model-only
    --trust-remote-code
    --attention-backend TRITON_ATTN
    --kv-cache-dtype auto
    --max-model-len "$MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    # Every throughput request prefills its full random prompt; no prefix-cache hits.
    --no-enable-prefix-caching
    --reasoning-parser qwen3
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
    --speculative-config "$SPEC_CONFIG"
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee /workspace/vllm_command.txt
printf '\n' | tee -a /workspace/vllm_command.txt
VLLM_LOG_MODEL_INSPECTION=1 "${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$EVAL_ONLY" == true ]]; then
    run_eval --framework lm-eval --port "$PORT"
    # Non-agentic evals must stage lm-eval's output into the workspace root.
    append_lm_eval_summary
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
        --use-chat-template \
        --server-pid "$SERVER_PID"
    if [[ "$RUN_EVAL" == true ]]; then
        run_eval --framework lm-eval --port "$PORT"
        append_lm_eval_summary
    fi
fi

stop_gpu_monitor
