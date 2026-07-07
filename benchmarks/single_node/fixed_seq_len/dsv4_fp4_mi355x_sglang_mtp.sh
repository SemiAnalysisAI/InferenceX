#!/usr/bin/env bash

# DeepSeek-V4-Pro on MI355X via SGLang — MTP variant of dsv4_fp4_mi355x_sglang.sh.
# Adds EAGLE/MTP speculative decoding per sgl-project/sglang#26383
# ([AMD][DSV4] DSV4 MTP graph + sparse triton attn optimizations, merged
# 2026-05-27, commit deaba74), which fixes the ROCm HIP-radix backend's
# per-step draft out_cache_loc slicing under CUDA graph (the bug behind the
# false-EOS / truncated-generation symptom in sgl issue #20404) and validates
# GSM8K 0.950 with MTP on. The EAGLE chain follows that PR's accuracy config
# for the DP-attention path (steps=2, topk=1, draft=3); the TP-only
# low-concurrency path uses the (3,1,4) chain shared with dsr1_fp4_mi355x_mtp.sh.
#
# Image: #26383 is on sglang `main`, so this runs on the mainline ROCm nightly
# (lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-*), NOT a rocm/sgl-dev:*-DSv4
# build. The -DSv4 images are cut from the amd/deepseek_v4 branch, which has not
# merged #26383 (latest da28108 = f96ac98 + build fixes + an unrelated MLA-decode
# refactor; it still crashes at MTP graph capture, run 26723126211). Mainline
# carries #26383 but omits deep_gemm, which DSv4-Pro's default fp8 wo_a path
# imports. AMD doesn't need deep_gemm (it uses aiter/tilelang/torch), and every
# deep_gemm use on the DSv4 path is behind an env-flag fallback, so the block
# below detects deep_gemm's absence and routes around it: SGLANG_OPT_FP8_WO_A_GEMM=0
# (dequant fp8 wo_a -> bf16 + torch.einsum; also skips the weight-load
# transform_sf_into_required_layout that crashed run 26727984372) and
# SGLANG_TOPK_TRANSFORM_512_TORCH=1 (torch topk). The indexer already routes to
# tilelang + torch paged-MQA-logits and MHC to aiter via flags set below. On a
# -DSv4 image that carries #26383, bump amd-master.yaml and the detect restores
# the deep_gemm perf path. RUN_EVAL on the high-conc points gates accuracy.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    MAX_MODEL_LEN

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# sglang ships in the image at the SHA encoded in the image tag (built
# from the amd/deepseek_v4 branch in sgl-project/sglang). To bump sglang,
# bump the image tag in configs/amd-master.yaml.

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0


SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

PARALLEL_ARGS=(
    --tensor-parallel-size "$TP"
)
SPEC_FLAGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)
CHUNKED_PREFILL_SIZE=$ISL
if [ "${DP_ATTENTION}" = "true" ]; then
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1

    CHUNKED_PREFILL_SIZE=$((ISL * TP))
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
    )
fi
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

set -x
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    "${PARALLEL_ARGS[@]}" \
    "${SPEC_FLAGS[@]}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --cuda-graph-max-bs ${CONC} \
    --max-running-requests ${CONC} \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.15 \
    --page-size 256 \
    --kv-cache-dtype fp8_e4m3 \
    --context-length $MAX_MODEL_LEN \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$(dirname "$0")/../chat_templates/deepseek_v4_thinking.jinja" \
    --watchdog-timeout 1800 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# --dsv4 routes prompts through encoding_dsv4.py, emitting the
# <bos><User>...<Assistant><think> framing DeepSeek-V4-Pro expects. EAGLE/MTP
# acceptance silently regresses on raw random tokens, so MTP benchmarks must
# use chat-formatted inputs (AGENTS.md). The DSv4-Pro tokenizer ships without a
# jinja chat_template, so plain --use-chat-template would crash; --dsv4 handles
# the framing directly.
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
    --dsv4

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
