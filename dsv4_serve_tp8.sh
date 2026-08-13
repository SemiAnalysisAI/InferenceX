#!/usr/bin/env bash
# TP8 arm vLLM serve on the 08-12 nightly + as many campaign PRs as can be
# carried safely. Goal here is RUNNABLE FIRST, maximum PR coverage second.
#
# Carried:  aiter #4417 (in-file guards), vllm #51714 (gluon sparse wiring),
#           vllm #51918 (flydsl_mega_moe kernel config).
# Also out: MegaMoE/DEP8 (needs mori.ir.flydsl, absent from the base) and FSE
#           (#4269 needs aiter fhmoe.py, also absent).
#
# CORRECTION -- do not repeat this mistake. An earlier revision of this header
# claimed the container carried "four aiter@97d0c6e4 gluon kernel FILES" that
# had to be deleted as a cross-version transplant. That was wrong:
# aiter/ops/triton/attention/pa_decode_sparse.py SHIPS IN THE STOCK NIGHTLY.
# Diffing the attention dir between a pristine `dsv4stock` container and the
# working one shows dsv4v is MISSING that base file and has nothing extra --
# the deletion damaged the base rather than removing foreign code. Any container
# built from this script must keep the base file; VLLM_ROCM_DSV4_SPARSE_GLUON=0
# below is enough to keep the gluon path dormant, since #51714's call site is a
# lazy in-function import (rocm_aiter_mla_sparse.py:2211) behind that env.
set -uo pipefail

MODEL_PATH=/it-shared/models/DeepSeek-V4-Pro
SERVED=deepseek-ai/DeepSeek-V4-Pro
PORT=8000
LOG=/home/jiacao/InferenceX/dsv4-serve.log

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
# Default off: #51714's gluon sparse-MLA path is not part of the runnable-first
# baseline. The kernel it dispatches to (pa_decode_sparse.py) is present in the
# stock nightly, so this can be flipped to 1 to A/B the gluon arm once the
# baseline serves.
export VLLM_ROCM_DSV4_SPARSE_GLUON=${VLLM_ROCM_DSV4_SPARSE_GLUON:-0}
# Probe knobs for the profile-run memfault, NOT a fix. Both default to aiter's
# own values so a plain run reproduces the campaign config.
#
# The MoE at inter_dim=384 is NOT the fault, despite being the loudest thing in
# the log. Two dead theories, recorded so nobody re-derives them:
#
#   1. "the tuned row names an illegal opus stage2 tile". Wrong. opus's 256 is a
#      *logical fp4* K step; packed it is K_STEP_PACKED=128, the kernel validates
#      effective_inter_dim % 128, and 384 % 128 == 0. 384 is a first-class opus
#      codegen seed (OPUS_A8W4_CODEGEN_SEED_EFFECTIVE_INTER_DIMS). FlyDSL's
#      tile_n/tile_k must divide inter_dim; opus's step need only divide by 128.
#      The two families collide only in naming.
#   2. "stage1 silently drops to tile_n=128 and opus stage2 still reads a 256
#      layout". Also wrong: stage1's output buffer is (token, topk, inter_dim),
#      independent of tile_n, and .repro/moe384.py drives this exact shape
#      standalone -- preshuffled weights, same kernel pair, same tile_n downgrade
#      warning -- and returns finite results. The kernel is fine.
#
# What the log actually shows: TP6 finishes MoE, loads bf16_tuned_gemm.csv, and
# faults there; the other seven workers are parked on that CSV's baton lock and
# survive. Immediately after come M:65536/M:16384 x N:7168 x K:7168 bf16 GEMMs
# with no tuned entry, running default configs. The fault is in the post-MoE
# bf16 GEMM, not the MoE.
export AITER_BYPASS_TUNE_CONFIG=${AITER_BYPASS_TUNE_CONFIG:-0}
export AITER_FLYDSL_FORCE=${AITER_FLYDSL_FORCE:-1}
export VLLM_ENGINE_READY_TIMEOUT_S=10800

# Overridable so the profile-run peak can be walked down without editing this
# file. The profile run builds a dummy batch of max_num_batched_tokens (16384)
# x the MTP fan-out (num_speculative_tokens=3 -> 4), hence the M:65536 GEMMs in
# the log; that transient is the largest allocation the server ever makes.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}

exec vllm serve "$MODEL_PATH" --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --async-scheduling --distributed-executor-backend mp --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 --data-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --moe-backend aiter \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.49}' \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --max-num-seqs 64 > "$LOG" 2>&1
