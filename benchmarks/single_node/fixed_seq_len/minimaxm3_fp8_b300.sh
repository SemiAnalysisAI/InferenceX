#!/usr/bin/env bash

# MiniMax-M3 MXFP8 B300 single-node vLLM recipe
# (https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3). Same shape as the B200
# script, but follows the b300 launcher's MODEL/MODEL_PATH split:
# launch_b300-nv.sh keeps MODEL as the HF id and points MODEL_PATH at
# /data/models/<basename> (writable NFS) for models not in the SRE-staged
# /scratch/models list — MiniMax-M3 is not staged. --block-size 128 is
# mandatory (MSA sparse/index cache); the benchmark is text-only, so
# --language-model-only frees the vision encoder's VRAM.

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

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE.
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

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

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --host 0.0.0.0 --port $PORT \
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
