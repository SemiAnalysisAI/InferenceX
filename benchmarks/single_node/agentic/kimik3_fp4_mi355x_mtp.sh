#!/usr/bin/env bash
set -euo pipefail

# DSpark variant of kimik3_fp4_mi355x.sh. The MI355X launcher routes
# spec-decoding=mtp rows to this suffix, while the shared base recipe owns the
# model, KV-offload, AgentX replay, and eval plumbing.
#
# Keep this wrapper aligned with the upstream AMD Kimi-K3 DSpark reproducer.
# The first AgentX validation is deliberately GPU-only at c1 so a server or
# kernel failure cannot be attributed to a KV-offload connector.
export SPEC_DECODE=true
# bf16 KV, against the fp8-everywhere rule for this model, and deliberately so
# for now: mla_gluon's fp8 regime (bh16bn128) asserts batch_size == 1, but the
# DSpark verify step calls it with batch_size = the number of verify tokens (8
# at num_speculative_tokens 7). Stock therefore cannot run DSpark under
# --kv-cache-dtype fp8 at all. The patched b128 kernel this recipe pulls from
# the gist is what lifts that restriction, so fp8 DSpark becomes possible once
# that combination has actually been validated -- until then, auto.
# bf16. fp8 + DSpark is blocked on ROCm at the DRAFTER: it asserts
# supports_quant_query_input, triton_mla sets that False, and ROCM_AITER_MLA
# (which sets it True) is causal-only while DSpark is non-causal. The blog's
# fp8 works because FLASHINFER_MLA is CUDA-only and has both. Documented
# exception to the fp8-everywhere rule; revisit if triton_mla learns to
# dequantize the query. KV usage at our crashes was 31-83%, so the pool is not
# the binding constraint at these concurrencies anyway.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# Keep the DSpark arm off the unmerged vLLM#50578 asm-pad patch. It is inert
# here by construction (use_gluon_decode gates on max_qo_len == 1, and verify
# runs at 8, so decode never reaches the asm path), and leaving it out keeps
# this arm a clean single-variable test of the forward_mqa fix.
# Gluon verify. Padded asm verify does not verify (574/574 accepted,
# per-position all 1.000), and the native MTP path imposes causal masking that
# DSpark cannot take (acceptance 1.41-4.76 vs 6.41-8.00). Both refuted today.
export MLA_ASM_PAD="${MLA_ASM_PAD:-0}"
export DSPARK_ASM_VERIFY="${DSPARK_ASM_VERIFY:-0}"
export DSPARK_MTP_NATIVE="${DSPARK_MTP_NATIVE:-0}"
export DSPARK_MQA_FIX="${DSPARK_MQA_FIX:-1}"
# 0.88: mi355x-amds nodes free only ~272 of 288 GiB and 0.95 fails the
# startup free-memory check.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
# The blog states this outright: "--enable-prefix-caching ... not enabled by
# default". The reproducer emitted no override, so every DSpark cell before
# 2026-08-03 ran at 0.0% hit on a trace whose theoretical hit is 88.4%.
export PREFIX_CACHING="${PREFIX_CACHING:-true}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

exec "$(dirname "$0")/kimik3_fp4_mi355x.sh" "$@"
