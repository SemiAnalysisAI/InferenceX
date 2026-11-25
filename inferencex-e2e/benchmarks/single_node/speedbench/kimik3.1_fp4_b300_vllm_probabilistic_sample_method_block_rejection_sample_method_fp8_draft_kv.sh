#!/usr/bin/env bash
# Kimi-K3.1-Pro B300 vLLM SPEED-Bench AL matrix collector for the released DSpark draft with an
# fp8 draft KV cache ("kv_cache_dtype": "fp8" in --speculative-config). Everything else (image,
# target, draft revision, client, sampling, probabilistic draft and block verify) is the
# kimik3.1_fp4_b300_vllm_probabilistic_sample_method_block_rejection_sample_method.sh collection,
# whose draft KV cache is vLLM's default (model dtype). It measures the
# kimi-k3.1-pro__dspark__kimi-k3.1-pro-dspark__probabilistic-block__draft-kv-fp8 curve.
#
# Dispatch speedbench-al.yml with -f model=moonshotai/Kimi-K3.1-Pro -f model-prefix=kimik3.1
# -f collector-variant=probabilistic_sample_method_block_rejection_sample_method_fp8_draft_kv
# -f 'thinking-kwargs={"thinking": true}' -f 'thinking-modes=on'.

DRAFT_KV_CACHE_DTYPE=fp8 COLLECTOR="$(basename "$0")" \
    exec bash "$(dirname "$0")/kimik3.1_fp4_b300_vllm_probabilistic_sample_method_block_rejection_sample_method.sh"
