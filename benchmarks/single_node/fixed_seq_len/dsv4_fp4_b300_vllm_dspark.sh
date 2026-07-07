#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    # Run the download whenever MODEL_PATH is writable: `hf download` resumes
    # partial fetches (e.g. after a cancelled run) and no-ops when the
    # snapshot is complete. Only skip read-only pre-staged mounts.
    if [[ ! -d "$MODEL_PATH" || -w "$MODEL_PATH" ]]; then
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

export VLLM_ENGINE_READY_TIMEOUT_S=3600

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

MOE_ARGS=()
if [ "${DP_ATTENTION}" = "true" ]; then
    MOE_ARGS=(--moe-backend deep_gemm_mega_moe)
    MAX_NUM_BATCHED_TOKENS=2048
else
    MAX_NUM_BATCHED_TOKENS=$(( ISL * 2 ))
fi

BENCHMARK_MAX_MODEL_LEN=$MAX_MODEL_LEN

if [ "${EVAL_ONLY}" = "true" ]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$BENCHMARK_MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
    SERVE_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    SERVE_MAX_MODEL_LEN="$BENCHMARK_MAX_MODEL_LEN"
fi

# The DSpark drafter ships inside the DeepSeek-V4-*-DSpark checkpoints, so
# unlike EAGLE3 no separate draft model is passed; vLLM only needs
# method=dspark (vllm-project/vllm#46995, #47093 — requires a build at or
# after the 2026-07-02 merge). 7 draft tokens with greedy draft sampling is
# the checkpoint's reference recipe.
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"

# kv-cache-dtype must be the canonical packed layout, not plain fp8: FlashMLA
# sparse decode only supports fp8_ds_mla (see vLLM
# tests/v1/attention/test_dspark_noncausal_sparse_mla.py), and the fp8 ->
# fp8_ds_mla auto-upgrade does not reach every consumer once the DSpark
# drafter is registered ('kv must have shape ...' at engine init otherwise).
#
# The scheduler reserves draft-token slots for every runnable sequence; with
# the engine-default max_num_seqs and 7 draft tokens that drives
# max_num_scheduled_tokens negative on a 2*ISL token budget. Bound running
# sequences to the benchmark concurrency (floor 32 so eval traffic still
# batches) and grow the token budget by the reserved slots.
MAX_NUM_SEQS=$(( CONC > 32 ? CONC : 32 ))
MAX_NUM_BATCHED_TOKENS=$(( MAX_NUM_BATCHED_TOKENS + MAX_NUM_SEQS * (NUM_SPEC_TOKENS + 1) ))

start_gpu_monitor

set -x
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --pipeline-parallel-size 1 \
    --kv-cache-dtype fp8_ds_mla \
    --trust-remote-code \
    --block-size 256 \
    --no-enable-prefix-caching \
    "${EP_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --attention_config.use_fp4_indexer_cache True \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --max-cudagraph-capture-size 2048 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"draft_sample_method\": \"greedy\"}" \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

# DSpark acceptance rate degrades on raw random tokens; --dsv4 routes prompts
# through chat-formatted encoding as required for speculative decoding benchmarks.
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
    --dsv4

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
