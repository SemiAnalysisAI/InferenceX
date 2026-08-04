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
# per-position all 1.000).
export MLA_ASM_PAD="${MLA_ASM_PAD:-0}"
export DSPARK_ASM_VERIFY="${DSPARK_ASM_VERIFY:-0}"
# vllm-project/vllm#50619's causal native MTP verify, cherry-picked onto this
# image. It replaces DSPARK_MQA_FIX, which owned the same branch, so that goes
# to 0 and the launcher refuses to run both.
#
# Everything else here is held at run 30875699986's values (k=7, mns 8, GMU
# 0.88, bf16 KV, prefix caching on) so this is a single-variable test of the
# verify path.
#
# What the old default was hiding: the branch DSPARK_MQA_FIX patches, and the
# stock branch it replaces, both give every verify row the request's entire KV
# range -- and seq_lens already counts this step's tokens. Draft token t
# therefore attends draft tokens t+1..7, so the target grades the draft against
# itself. That is where 6.41-8.00 of 8 came from, and why conc 1 read 8.00
# flat. PR 50619 restores the causal tail upstream has always specified and
# scores GSM8K 1271/1319 against a 1269/1319 no-speculation baseline.
#
# So expect acceptance to FALL here, and do not read that as a regression: on
# the old path acceptance rose as correctness fell. The PR's own figure is
# 54.4% of drafted tokens. Judge this arm on acceptance plus a quality gate,
# never acceptance alone.
#
# The former DSPARK_MTP_NATIVE knob is gone: its patcher was never in this
# branch's lineage, so run 30875699986 exported it to no effect and actually
# ran the image's stock non-causal flatten.
export DSPARK_PR50619="${DSPARK_PR50619:-1}"
export DSPARK_MQA_FIX="${DSPARK_MQA_FIX:-0}"
export SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-7}"
# 0.88: mi355x-amds nodes free only ~272 of 288 GiB and 0.95 fails the
# startup free-memory check.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
# The blog states this outright: "--enable-prefix-caching ... not enabled by
# default". The reproducer emitted no override, so every DSpark cell before
# 2026-08-03 ran at 0.0% hit on a trace whose theoretical hit is 88.4%.
export PREFIX_CACHING="${PREFIX_CACHING:-true}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

exec "$(dirname "$0")/kimik3_fp4_mi355x.sh" "$@"
