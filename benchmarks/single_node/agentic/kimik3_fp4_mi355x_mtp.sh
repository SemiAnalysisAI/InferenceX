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
# EXPERIMENT BRANCH kimik3-dspark-perf -- DSpark configured for peak throughput
# rather than for fidelity to the upstream AMD reproducer. Four deltas from
# kimik3-mla-asm-pad, all of them independently proven on the NON-DSpark arm of
# this same recipe and SKU:
#
#   PREFIX_CACHING=true   The reproducer emits no override and vLLM resolves the
#                         flag to False for this model, so every DSpark cell so
#                         far ran at "Prefix cache hit rate: 0.0%" while the
#                         non-DSpark asm c8 arm ran at 91.7%. On an agentic trace
#                         every turn resends the whole conversation (ISL mean
#                         335K), so 0% means recomputing a ~300K prefix per turn.
#                         Theoretical hit on this trace is 98.1%. This is the
#                         single largest lever and it invalidates the throughput
#                         numbers from runs 30775983425 / 30776785351 /
#                         30778998927 as perf measurements.
#
#   KV_CACHE_DTYPE=fp8    The pool is 2,156,093 tokens at GMU 0.95 bf16; c8 with
#                         ~300K contexts wants ~2.4M, so we sit on the KV wall.
#                         fp8 roughly doubles it. Proven on this model/SKU at c8
#                         (run 30734222234, the 1707.85 asm-pad result) and
#                         mandatory per policy for every other K3 arm.
#                         SIDE EFFECT THAT MATTERS: the launcher gates the gist
#                         mla_gluon patch on exactly this flag, so fp8 also pulls
#                         in the kernel that lifts the batch_size==1 restriction
#                         -- the leading hypothesis for the hipErrorIllegalAddress
#                         that killed c1/c4/c8 at ~13 turns per slot.
#
#   MLA_ASM_PAD + DSPARK_ASM_VERIFY
#                         Route verify to vllm#50578's padded asm decode instead
#                         of Gluon, whose per-token latency is linear in KV
#                         length -- worth 5.67x on ITL in the non-spec arm at
#                         this trace depth.
#
# Drafter stays Inferact: RadixArk is dead on vLLM (its GQA draft KV page size
# cannot unify with K3's hybrid KDA+MLA cache, run 30782801411).
# bf16 KV, against the fp8-everywhere rule for this model, and deliberately so
# for now: mla_gluon's fp8 regime (bh16bn128) asserts batch_size == 1, but the
# DSpark verify step calls it with batch_size = the number of verify tokens (8
# at num_speculative_tokens 7). Stock therefore cannot run DSpark under
# --kv-cache-dtype fp8 at all. The patched b128 kernel this recipe pulls from
# the gist is what lifts that restriction, so fp8 DSpark becomes possible once
# that combination has actually been validated -- until then, auto.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
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
export PREFIX_CACHING="${PREFIX_CACHING:-true}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

exec "$(dirname "$0")/kimik3_fp4_mi355x.sh" "$@"
