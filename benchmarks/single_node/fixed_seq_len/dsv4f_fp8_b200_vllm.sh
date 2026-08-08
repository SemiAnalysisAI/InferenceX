#!/usr/bin/env bash

# DeepSeek-V4-Flash-0731 B200 single-node vLLM recipe, validated on an
# 8x B200 SXM6 node (driver 580.126.09 / 580.173.02) with
# vllm/vllm-openai:v0.25.0 (digest
# sha256:e1c1ff1af9a15921bfa11d1d95047258c1797392cdbfa296e7639da446b23f97,
# vLLM build commit dd10e03f95f94edbea1975c67ace3a35ec9a8a40).
#
# Two serving arms, both with expert parallel: vLLM v0.25.0's MegaMoE
# backend hard-requires EP and raises NotImplementedError before weight
# load if it is missing.
#   TP arm (DP_ATTENTION=false): tensor-parallel 8 + EP8. This is the
#     single-stream / latency arm; at 1M context the TP8 vs DP8
#     break-even is c=3 (vllm-project/vllm#51454).
#   DP arm (DP_ATTENTION=true):  data-parallel 8 (TP1 per replica) + EP8.
#     This is the concurrency and long-context arm: 7.7x more total KV
#     capacity than TP8 and a same-node 3.44x TTFT p50 advantage at
#     1M c=8 (honest range across metrics and runs: 3.2-3.7x).
# Prefix caching is left at the engine default (enabled): the report
# shows it intact on Flash, with a 22x warm-TTFT speedup on re-sent 1M
# documents, and the random-id benchmark has no shared prefixes either
# way.

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

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

# The Flash checkpoint is ~155 GiB. Download from HF only when handed a
# bare repo id (b200-nb / b200-cw runners); clusters with pre-staged
# weights hand a local path via the launcher instead. The serving runs
# behind this config key were validated against this exact snapshot.
MODEL_REVISION=${MODEL_REVISION:-7872f01b1d1fe23eabc4c98b48bffcef5a386062}
REVISION_ARGS=()
if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL" --revision "$MODEL_REVISION"
    REVISION_ARGS=(--revision "$MODEL_REVISION")
fi

SERVER_LOG=/workspace/server.log

# Engine startup covers MegaMoE init plus fp4 indexer cache allocation;
# allow an hour so cold weight loads do not hit the default readiness
# window.
export VLLM_ENGINE_READY_TIMEOUT_S=3600

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    EVAL_MAX_MODEL_LEN=$(compute_eval_context_length "$MODEL" "$MAX_MODEL_LEN")
    export EVAL_MAX_MODEL_LEN
    SERVE_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    SERVE_MAX_MODEL_LEN="$MAX_MODEL_LEN"
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    "${REVISION_ARGS[@]}" \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --moe-backend deep_gemm_mega_moe \
    --attention_config.use_fp4_indexer_cache=True \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    "${PARALLEL_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

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
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
