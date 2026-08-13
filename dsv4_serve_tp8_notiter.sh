#!/usr/bin/env bash
# TP8 arm vLLM serve -- the config the patch route CAN fully reproduce on a stock
# nightly (moe-backend aiter, so no MegaMoE/flydsl -> no mori.ir.flydsl dep),
# gluon sparse on (VLLM_ROCM_DSV4_SPARSE_GLUON=1, patched), base #4269 FSE on
# (VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1), 384-shard native in the 08-12
# base. Used to validate the patched container actually serves DSv4-Pro FP4 at
# 8k/1k c32. (DEP8/MegaMoE is blocked by a mori version gap -- see notes.)
set -uo pipefail

MODEL_PATH=/it-shared/models/DeepSeek-V4-Pro
SERVED=deepseek-ai/DeepSeek-V4-Pro
PORT=8000
LOG=/home/jiacao/InferenceX/dsv4-serve-triton.log

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=0
# FSE (#4269) intentionally OFF: the 08-12 base aiter has no fhmoe.py, and the
# additive route does not restore it.
# gluon: the additively-injected gluon sparse-MLA kernel (from aiter@97d0c6e4)
# GPU-faults during memory profiling on the 08-12 base aiter runtime
# (hc_head_fuse_tilelang path, VllmWorker GPU coredump). Overridable so we can
# validate a servable baseline with gluon OFF, then flip it back on to reproduce.
export VLLM_ROCM_DSV4_SPARSE_GLUON=${VLLM_ROCM_DSV4_SPARSE_GLUON:-1}
export VLLM_ENGINE_READY_TIMEOUT_S=10800

exec vllm serve "$MODEL_PATH" --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --async-scheduling --distributed-executor-backend mp --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 --data-parallel-size 1 \
    --gpu-memory-utilization 0.8 --moe-backend triton \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.49}' \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --max-num-seqs 64 > "$LOG" 2>&1
