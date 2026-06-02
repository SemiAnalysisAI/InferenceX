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
# bump the image tag in .github/configs/amd-master.yaml.

# Transformers in the container doesn't recognize the `deepseek_v4` model_type.
# PR #23608's fallback in hf_transformers_utils.get_config tries to handle this
# by writing a patched config to /tmp, but in practice isn't catching the error
# in this image. Patch the cached config.json directly instead: set model_type
# to `deepseek_v3` so AutoConfig.from_pretrained succeeds, and keep
# architectures=['DeepseekV4ForCausalLM'] so SGLang dispatches to its native
# DSv4 model class (python/sglang/srt/models/deepseek_v4.py).
python3 << PYEOF
import json
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="$MODEL", filename="config.json")
with open(path) as f:
    config = json.load(f)
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Patched {path}: model_type deepseek_v4 -> deepseek_v3")
else:
    print(f"No patch needed: model_type is {config.get('model_type')!r}")
PYEOF

# DSv4 FP4-experts path. Tracks the env block in python/run_dsv4.sh on the
# amd/deepseek_v4 branch (HEAD's active block is FP8; we override the two
# FP4-specific flags below):
#   SGLANG_DSV4_FP4_EXPERTS=True   -> route experts through the FP4 kernels
#   SGLANG_FORCE_TRITON_MOE_FP8=0  -> dispatch MoE through aiter and apply
#                                    the swiglu_limit clamp in the triton
#                                    MoE fallback path.
export SGLANG_REASONING_EFFORT=max
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_OPT_USE_OLD_COMPRESSOR=false
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_OPT_USE_AITER_MHC_PRE=true
export SGLANG_OPT_USE_AITER_MHC_POST=true
export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_DPSK_V4_RADIX=1
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=true
export SGLANG_FORCE_TRITON_MOE_FP8=0
export SGLANG_HACK_FLASHMLA_BACKEND=triton
export SGLANG_OPT_USE_TILELANG_INDEXER=true
export SGLANG_OPT_USE_TRITON_SWA_PREPARE=true
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_FUSE_WQA_WKV=true
export SGLANG_OPT_USE_FUSED_PAGED_COMPRESS=true
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0

# MTP-specific knobs landed alongside the graph fix in sgl#26383:
#   SGLANG_OPT_USE_TRITON_FUSED_MHC -> fused Triton mhc_post_pre for low conc
#                                      (defaults True in post-#26383 images;
#                                      set explicitly so the recipe is auditable)
#   SGLANG_OPT_C4_SPARSE_TOPK       -> sparse-attention top-k used in the PR's
#                                      DSv4 MTP accuracy run
export SGLANG_OPT_USE_TRITON_FUSED_MHC=1
export SGLANG_OPT_C4_SPARSE_TOPK=512

# Mainline ROCm nightlies carry #26383 but omit deep_gemm (only rocm/sgl-dev:*-DSv4
# builds bundle it). DSv4-Pro's default fp8 wo_a path imports deep_gemm at weight
# load; detect its absence and route the deep_gemm-touching paths to their torch
# fallbacks. No-op on a deep_gemm-bearing image, so this recipe works on both.
#   SGLANG_OPT_FP8_WO_A_GEMM=0       -> wo_a fp8 weights dequantized to bf16 at load
#                                       (_dequant_fp8_wo_a) + o-proj via torch.einsum;
#                                       also skips the post-load deep_gemm
#                                       transform_sf_into_required_layout that crashed.
#   SGLANG_TOPK_TRANSFORM_512_TORCH=1 -> torch topk-transform instead of the kernel.
#   SGLANG_OPT_USE_TOPK_V2=0          -> skip plan_topk_v2 in the indexer metadata;
#                                       its jit kernel is CUDA-only (topk/ptx.cuh
#                                       #includes <cuda/ptx>) and won't build for
#                                       gfx950. topk_metadata is unused on the torch
#                                       topk path, so empty is fine.
#   SGLANG_ENABLE_JIT_DEEPGEMM=0     -> global off; nothing to JIT without the module.
if python3 -c "import deep_gemm" >/dev/null 2>&1; then
    echo "deep_gemm present -> using fp8 wo_a / deep_gemm perf path"
else
    echo "deep_gemm absent -> routing DSv4 fp8 wo_a / topk around it (mainline nightly)"
    export SGLANG_OPT_FP8_WO_A_GEMM=0
    export SGLANG_TOPK_TRANSFORM_512_TORCH=1
    export SGLANG_OPT_USE_TOPK_V2=0
    export SGLANG_ENABLE_JIT_DEEPGEMM=0
fi

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
# EAGLE chain is selected by DP_ATTENTION. The DP-attention path mirrors the
# sgl#26383 DSv4 ROCm accuracy config (steps=2, topk=1, draft=3); the TP-only
# low-concurrency fallback uses the longer (3,1,4) chain that low batch sizes
# benefit from, matching dsr1_fp4_mi355x_mtp.sh.
SPEC_FLAGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
    )
    SPEC_FLAGS=(
        --speculative-algorithm EAGLE
        --speculative-num-steps 2
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 3
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
    --attention-backend compressed \
    --max-running-requests ${CONC} \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.15 \
    --page-size 256 \
    --context-length $MAX_MODEL_LEN \
    --chunked-prefill-size 8192 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$(dirname "$0")/chat_templates/deepseek_v4_thinking.jinja" \
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
