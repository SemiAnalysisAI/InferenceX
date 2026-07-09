#!/usr/bin/env bash

# MiniMax-M3 MXFP8 B200 single-node vLLM recipe with EAGLE3 speculative
# decoding — the repo's spec-decoding=mtp variant of minimaxm3_fp8_b200.sh
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). Adds the
# Inferact/MiniMax-M3-EAGLE3 draft head via --speculative-config with 3
# speculative tokens. Everything else keeps the non-MTP serve shape:
# --block-size 128 is mandatory (MSA sparse_block_size is 128; the default 16
# misaligns sparse indexing), and --language-model-only skips the vision
# encoder for the text-only benchmark. dp-attn=true maps to DP×EP (DEP);
# ep>1 maps to TP+EP (TEP).
#
# The target uses the FlashInfer TRT-LLM attention path. The EAGLE3 drafter is
# pinned separately to TRITON_ATTN.

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

# The 0618 image keeps MiniMax M3 top-k indices in a persistent
# [head_kv, max_tokens, topK] buffer for CUDA graphs. Slicing that buffer to
# the actual prefill length is non-contiguous when TP leaves multiple local KV
# heads, and the MSA CSR builder rejects it. Materialize the slice until the
# image includes this fix.
python3 - <<'PYEOF' || { echo "MiniMax M3 MSA contiguity patch failed" >&2; exit 1; }
import importlib.util
import pathlib

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Could not locate the installed vllm package")

target = (
    pathlib.Path(next(iter(spec.submodule_search_locations)))
    / "models"
    / "minimax_m3"
    / "nvidia"
    / "sparse_attention_msa.py"
)
src = target.read_text()
old = "            prefill_topk = topk[:, nd:num_tokens, :]\n"
new = "            prefill_topk = topk[:, nd:num_tokens, :].contiguous()\n"

if new in src:
    print(f"[minimax-m3-msa-patch] already applied: {target}")
elif src.count(old) == 1:
    target.write_text(src.replace(old, new, 1))
    print(f"[minimax-m3-msa-patch] patched: {target}")
else:
    raise RuntimeError(f"Expected exactly one patch anchor in {target}")
PYEOF

DRAFT_MODEL="Inferact/MiniMax-M3-EAGLE3"

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

# launch_b200-dgxc.sh rewrites MODEL to the pre-downloaded
# /lustre/fsw/gharunners/models/MiniMax-M3-MXFP8 path; only download the target
# when handed a bare HF id (b200-cw / b200-nb runners). The EAGLE3 draft is
# never pre-staged, so fetch it either way: next to the target weights when
# MODEL is a local path (the gharunners tree is writable), into the HF cache
# otherwise.
if [[ "$MODEL" != /* ]]; then
  hf download "$MODEL"
  hf download "$DRAFT_MODEL"
  DRAFT_MODEL_PATH="$DRAFT_MODEL"
else
  DRAFT_MODEL_PATH="$(dirname "$MODEL")/${DRAFT_MODEL##*/}"
  if [[ ! -d "$DRAFT_MODEL_PATH" || -z "$(ls -A "$DRAFT_MODEL_PATH" 2>/dev/null)" ]]; then
    hf download "$DRAFT_MODEL" --local-dir "$DRAFT_MODEL_PATH"
  fi
fi

SERVER_LOG=/workspace/server.log

# 444 GB of MXFP8 weights off shared FS; engine startup can exceed the
# default 600s readiness window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_FLOAT32_MATMUL_PRECISION=high

if [ "${DP_ATTENTION}" = "true" ]; then
  PARALLEL_ARGS="--tensor-parallel-size=1 --data-parallel-size=$TP --enable-expert-parallel"
elif [ "$EP_SIZE" -gt 1 ]; then
  PARALLEL_ARGS="--tensor-parallel-size=$TP --enable-expert-parallel"
else
  PARALLEL_ARGS="--tensor-parallel-size=$TP"
fi

# use 3 speculative tokens for all configs for now
NUM_SPEC_TOKENS=3

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
$PARALLEL_ARGS \
--gpu-memory-utilization 0.90 \
--max-model-len $MAX_MODEL_LEN \
--block-size 128 \
--attention-config '{"backend": "FLASHINFER", "use_trtllm_attention": true}' \
--attention-config.indexer_kv_dtype "fp8" \
--kv-cache-dtype fp8 \
--language-model-only \
--max-cudagraph-capture-size 2048 \
--max-num-batched-tokens "$((ISL * 2 ))" \
--speculative-config "{\"method\": \"eagle3\", \"model\": \"$DRAFT_MODEL_PATH\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"attention_backend\": \"TRITON_ATTN\"}" \
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
