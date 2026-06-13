#!/usr/bin/env bash

# MiniMax-M3 MXFP8 H200 single-node vLLM recipe with EAGLE3 speculative
# decoding — the repo's spec-decoding=mtp variant of minimaxm3_fp8_h200.sh
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). Adds the
# Inferact/MiniMax-M3-EAGLE3 draft head via --speculative-config with 3
# speculative tokens. Everything else keeps the non-MTP serve shape:
# --block-size 128 is mandatory (MSA sparse_block_size is 128), the benchmark
# is text-only so --language-model-only frees the vision encoder's VRAM, and
# the MXFP8 MoE runs through vLLM's Hopper-compatible backends.
#
# The drafter is pinned to FLASH_ATTN: the EAGLE3 head is MHA, and FlashInfer
# only supports page size 128 through its trtllm-gen kernel, which requires
# GQA/MQA — engine init dies in FlashInferMetadataBuilder otherwise (hit on
# the B300 MTP canary). FLASH_ATTN takes any multiple-of-16 block size, so
# the mandatory 128 is fine for the draft.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

# The shared HF cache lives on a network FS; concurrent day-zero downloads of
# the same ~444 GB checkpoint from sibling nodes hit huggingface_hub's
# WeakFileLock "[Errno 116] Stale file handle" race. Retry the download (it
# resumes), then serve with HF_HUB_OFFLINE=1 so vllm's snapshot_download does
# a lock-free local-cache read instead of re-contending the lock files. The
# EAGLE3 draft is fetched the same way so the offline serve finds it cached.
SERVE_OFFLINE=()
if [[ "$MODEL" != /* ]]; then
  for attempt in 1 2 3 4 5; do
    hf download "$MODEL" && break
    if [ "$attempt" = 5 ]; then echo "hf download failed after $attempt attempts" >&2; exit 1; fi
    echo "hf download attempt $attempt failed; retrying in 60s" >&2
    sleep 60
  done
  for attempt in 1 2 3 4 5; do
    hf download "$DRAFT_MODEL" && break
    if [ "$attempt" = 5 ]; then echo "hf download failed after $attempt attempts" >&2; exit 1; fi
    echo "hf download attempt $attempt failed; retrying in 60s" >&2
    sleep 60
  done
  SERVE_OFFLINE=(env HF_HUB_OFFLINE=1)
fi

SERVER_LOG=/workspace/server.log

export PYTHONNOUSERSITE=1
# ~444 GB of MXFP8 weights off shared FS; engine startup can exceed the
# default 600s readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS="--tensor-parallel-size=1 --data-parallel-size=$TP --enable-expert-parallel"
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS="--tensor-parallel-size=$TP --enable-expert-parallel"
else
  PARALLEL_ARGS="--tensor-parallel-size=$TP"
fi

# use 3 speculative tokens for all configs for now
NUM_SPEC_TOKENS=3

# Fixed-seq-len runs don't need graphs past the decode step's token count:
# with spec decoding every running request contributes 1 + NUM_SPEC_TOKENS
# tokens per step, so capture up to the next power of two >=
# CONC * (1 + NUM_SPEC_TOKENS), capped at vLLM's 2048 ceiling.
CAPTURE_SIZE=4
while (( CAPTURE_SIZE < CONC * (1 + NUM_SPEC_TOKENS) )); do CAPTURE_SIZE=$((CAPTURE_SIZE * 2)); done
(( CAPTURE_SIZE > 2048 )) && CAPTURE_SIZE=2048

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
"${SERVE_OFFLINE[@]}" vllm serve $MODEL --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--language-model-only \
--max-cudagraph-capture-size $CAPTURE_SIZE \
--max-num-batched-tokens "$((ISL * 2 ))" \
--speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"FLASH_ATTN\"}" \
--stream-interval 20 --no-enable-prefix-caching \
--trust-remote-code > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# Spec-decode acceptance rate degrades on raw random tokens; route prompts
# through the chat template as the other MTP recipes do.
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
    --trust-remote-code \
    --use-chat-template

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
