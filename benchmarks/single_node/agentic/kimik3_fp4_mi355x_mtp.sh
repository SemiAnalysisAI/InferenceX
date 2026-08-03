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
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# EXPERIMENT BRANCH kimik3-dspark-asm-verify -- the ONLY delta from
# kimik3-mla-asm-pad, which defaults both of these to 0. Same config keys on
# both branches, so a dispatch of the same key against each is a single-variable
# A/B. The agentic matrix has no per-cell env channel, which is why this is a
# branch pair rather than a config knob (same pattern as the asm-pad A/B).
#
# Routes DSpark's multi-token verify through vllm#50578's padded asm decode
# instead of the Gluon flatten branch. Gluon parallelizes only over heads, so
# per-token latency is linear in KV length -- at our ~306K p90 trace that is the
# handicap the non-spec arm shed for 5.67x on ITL. The PR's own comment says the
# asm path has no gqa<16 / qseqlen>1 kernel, but that describes STOCK, where 12
# heads cannot be padded at all; after tile-padding the count is gqa=16, which
# is not the excluded case. Testing the claim rather than trusting it.
#
# Correctness is NOT established by a throughput number here. Padding is safe
# for plain decode (MLA is independent per query head, pad heads get sliced
# off), but verify adds rejection sampling on top, so a bad pad interaction
# degrades acceptance length silently instead of crashing. Needs gsm8k on the
# spec path before this goes anywhere near a shipping recipe.
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
