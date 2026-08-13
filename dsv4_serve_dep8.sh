#!/usr/bin/env bash
# DEP8 winner-arm vLLM serve for the patch-route validation on the STOCK nightly
# base (3ee2df30) patched in-place by apply_dsv4_container_patches.sh. This is the
# config the campaign measured as the winner: DP8 + expert-parallel + MegaMoE
# (flydsl_mega_moe, patched) + gluon sparse (VLLM_ROCM_DSV4_SPARSE_GLUON=1,
# patched), MTP num_spec=3 synthetic accept 2.49, kv fp8, gmu 0.8. FSE and the
# 384-shard oracle are TP8-arm only, so they stay OFF here.
set -uo pipefail

# The campaign served /models/DeepSeek-V4-Pro (deepseek_v4_fp8 quant: fp8
# weight-block + fp4 experts). That is /it-shared/models/DeepSeek-V4-Pro here.
# NOT the amd--DeepSeek-V4-Pro-MXFP4 HF snapshot (Quark MXFP4 -- different
# merged-column packing, fails the load_merged_column_weight shape assert).
MODEL_PATH=/it-shared/models/DeepSeek-V4-Pro
SERVED=deepseek-ai/DeepSeek-V4-Pro
PORT=8000
LOG=/home/jiacao/InferenceX/dsv4-serve.log

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_DSV4_SPARSE_GLUON=1
export VLLM_ENGINE_READY_TIMEOUT_S=10800

exec vllm serve "$MODEL_PATH" --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --async-scheduling --distributed-executor-backend mp --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
    --gpu-memory-utilization 0.8 --moe-backend flydsl_mega_moe \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.49}' \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --max-num-seqs 64 > "$LOG" 2>&1
