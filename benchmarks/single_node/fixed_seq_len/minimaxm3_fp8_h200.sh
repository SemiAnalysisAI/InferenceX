#!/usr/bin/env bash

# MiniMax-M3 MXFP8 H200 single-node vLLM recipe
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). 427B/26B-active MoE with MSA
# sparse attention. --block-size 128 is mandatory (MSA sparse_block_size is
# 128; the default 16 misaligns sparse indexing). The benchmark is text-only,
# so --language-model-only skips the vision encoder and frees VRAM for KV.
# dp-attn=true maps to DP×EP (DEP) per the recipe's "DP8 + Expert Parallel"
# layout; ep>1 maps to TP+EP (TEP). Hopper has no native MX tensor cores, so
# the MXFP8 MoE runs through vLLM's Hopper-compatible backends (Marlin /
# DeepGEMM) selected by the mxfp8 oracle in the minimax-m3 image.

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

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

# The shared HF cache lives on a network FS; concurrent day-zero downloads of
# the same ~444 GB checkpoint from sibling nodes hit huggingface_hub's
# WeakFileLock "[Errno 116] Stale file handle" race. Retry the download (it
# resumes), then serve with HF_HUB_OFFLINE=1 so vllm's snapshot_download does
# a lock-free local-cache read instead of re-contending the lock files.
SERVE_OFFLINE=()
if [[ "$MODEL" != /* ]]; then
  for attempt in 1 2 3 4 5; do
    hf download "$MODEL" && break
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

# Fixed-seq-len runs don't need graphs past the request concurrency: capture
# up to the next power of two >= CONC (per-DP-rank batch is CONC/DP but ragged
# arrival makes the full CONC bound safer), capped at vLLM's 2048 ceiling.
CAPTURE_SIZE=4
while (( CAPTURE_SIZE < CONC )); do CAPTURE_SIZE=$((CAPTURE_SIZE * 2)); done
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
--stream-interval 20 --no-enable-prefix-caching \
--trust-remote-code > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
