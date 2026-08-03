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
# EXPERIMENT BRANCH kimik3-dspark-radixark-asm -- the aggressive arm. Stacks the
# two independent DSpark levers at once, against kimik3-mla-asm-pad as the base:
#   1. RadixArk drafter (2.2B) instead of Inferact (4B). Acceptance is not the
#      motive -- we measure 5.5-8.0 on this trace, above both cards' claims.
#      Draft cost is: the drafter forward is pure per-step overhead and at c8 it
#      competes with real batch work instead of filling idle bandwidth.
#   2. Verify routed to vllm#50578's padded asm decode instead of Gluon. Gluon
#      parallelizes only over heads, so per-token latency is linear in KV length;
#      that handicap is worth 5.67x on ITL in the non-spec arm at this trace
#      depth.
#
# Two unvalidated changes in one cell is deliberate (chasing peak throughput),
# but it is NOT a clean attribution: if this arm beats the Gluon/Inferact c8, we
# will not know the split without the single-variable arms
#   30781568167 (RadixArk, Gluon) and 30778998927 (Inferact, padded asm).
export SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-RadixArk/Kimi-K3-DSpark}"
# bf16 KV, against the fp8-everywhere rule for this model, and deliberately so
# for now: mla_gluon's fp8 regime (bh16bn128) asserts batch_size == 1, but the
# DSpark verify step calls it with batch_size = the number of verify tokens (8
# at num_speculative_tokens 7). Stock therefore cannot run DSpark under
# --kv-cache-dtype fp8 at all. The patched b128 kernel this recipe pulls from
# the gist is what lifts that restriction, so fp8 DSpark becomes possible once
# that combination has actually been validated -- until then, auto.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# Keep the DSpark arm off the unmerged vLLM#50578 asm-pad patch. It is inert
# here by construction (use_gluon_decode gates on max_qo_len == 1, and verify
# runs at 8, so decode never reaches the asm path), and leaving it out keeps
# this arm a clean single-variable test of the forward_mqa fix.
export MLA_ASM_PAD="${MLA_ASM_PAD:-1}"
export DSPARK_ASM_VERIFY="${DSPARK_ASM_VERIFY:-1}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
export PREFIX_CACHING="${PREFIX_CACHING:-auto}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

exec "$(dirname "$0")/kimik3_fp4_mi355x.sh" "$@"
