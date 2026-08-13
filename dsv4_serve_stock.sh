#!/usr/bin/env bash
# CONTROL ARM: serve DSv4-Pro FP4 TP8 on the *pristine* 08-12 nightly, with no
# patches applied at all. Same model, same parallelism, same MTP/compile config
# as dsv4_serve_tp8.sh -- the only difference is the container has an untouched
# aiter/vllm tree.
#
# Purpose: the profile-run memfault has only ever been observed on the PATCHED
# container. Nothing established that the patch is innocent. In particular
# apply/dsv4_patch_additive.sh copies four gluon kernel files out of
# aiter@97d0c6e4 into the nightly's much older aiter tree, and those two aiter
# revisions differ semantically in 418 files -- exactly the setup that imports
# cleanly and then faults at runtime. The earlier VLLM_ROCM_DSV4_SPARSE_GLUON=0
# A/B did NOT clear it: that env only avoids one call path, the foreign files
# were still on sys.path.
#
# If this arm serves, the fault is ours and the patch is the bug.
# If this arm faults too, the fault is in the stock base and the patch is clean.
set -uo pipefail

MODEL_PATH=/it-shared/models/DeepSeek-V4-Pro
SERVED=deepseek-ai/DeepSeek-V4-Pro
PORT=8000
LOG=/home/jiacao/InferenceX/dsv4-serve-stock.log

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
# No gluon knob here: VLLM_ROCM_DSV4_SPARSE_GLUON is introduced by vllm PR #51714,
# which this container does not have. Stock takes its default sparse-MLA path.
export VLLM_ENGINE_READY_TIMEOUT_S=10800

exec vllm serve "$MODEL_PATH" --served-model-name "$SERVED" \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --async-scheduling --distributed-executor-backend mp --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 --data-parallel-size 1 \
    --gpu-memory-utilization 0.8 --moe-backend aiter \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.49}' \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
    --max-num-seqs 64 > "$LOG" 2>&1
