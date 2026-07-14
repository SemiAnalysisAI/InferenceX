#!/usr/bin/env bash

# DeepSeek-V4-Pro FP8 single-node on MI300X (gfx942) via SGLang.
#
# BRING-UP RECIPE (unvalidated on-cluster as of 2026-07). gfx942 has no
# native FP4 path, so unlike the MI355X (gfx950) dsv4 recipes this runs the
# model in FP8: FP8 KV cache + AITER MoE, no --attention-backend dsv4 (that
# backend ships only in the mi35x images). Model-level flags (deepseek_v4
# tool/reasoning parsers, page-size 256, SWA ratio, disable-shared-experts-
# fusion) mirror dsv4_fp4_mi355x_sglang.sh; the gfx942 infra flags (AITER,
# MEC-firmware scratch-reclaim guard) mirror dsr1_fp8_mi300x.sh.
#
# Open questions for the debug-runs cluster loop before this leaves bring-up:
#   1. Does lmsysorg/sglang:*-mi30x carry the DeepseekV4 model class + a
#      working gfx942 attention backend? (aiter is the proven gfx942 backend;
#      swap if sglang picks a dsv4-specific one.)
#   2. Does --quantization deepseek_v4_fp8 load on ROCm, or is a pre-quantized
#      FP8 checkpoint required?
#   3. Tune mem-fraction-static / cuda-graph-max-bs for 192GB MI300X.

source "$(dirname "$0")/../../benchmark_lib.sh"

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# If the machine runs a MEC FW older than 177, RCCL cannot reclaim some
# memory; disable that feature to avoid crashes (see dsr1_fp8_mi300x.sh).
version=`rocm-smi --showfw | grep MEC | head -n 1 | awk '{print $NF}'`
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

# Context budget: ISL + OSL plus a small pad, matching the fixed-seq-len
# recipes. DeepSeek-V4-Pro uses SWA, so a bounded context keeps the KV
# footprint predictable on 8x192GB.
CONTEXT_LEN=$(( ISL + OSL + 1024 ))

SERVER_LOG=/workspace/server.log

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
    CONTEXT_LEN=$EVAL_MAX_MODEL_LEN
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
python3 -m sglang.launch_server \
    --model-path $MODEL --served-model-name $MODEL \
    --host=0.0.0.0 --port=$PORT \
    --trust-remote-code \
    --tensor-parallel-size=$TP \
    --quantization deepseek_v4_fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend aiter \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --page-size 256 \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.15 \
    --cuda-graph-max-bs ${CONC} \
    --max-running-requests "$(( CONC * 3 / 2 > 8 ? CONC * 3 / 2 : 8 ))" \
    --context-length $CONTEXT_LEN \
    --chunked-prefill-size $ISL \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$(dirname "$0")/../chat_templates/deepseek_v4_thinking.jinja" \
    --watchdog-timeout 1800 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

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
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
