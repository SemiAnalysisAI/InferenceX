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
# ROOT-CAUSE ARM. mla_gluon natively accepts a 4-D MTP query
# [batch, qlen, nhead, dim]; forward_mqa flattens verify tokens into fake batch
# rows instead. That single choice produces batch = num_reqs*qlen (144 and 225
# against the kernel's 128 cap), the per-layer sum(seq_len) index expansion
# behind the 13.99 GiB OOM, and the uniform-qlen mis-mapping. Using the native
# layout makes batch == num_reqs and needs no expansion.
#
# JUDGE THIS ON ACCEPTANCE LENGTH, NOT THROUGHPUT. The MTP kernel is causal
# along qlen (q_pos attends KV [0, seq_len-qlen+q_pos]) while the flatten branch
# is non-causal. Causal-tail is the correct verify semantics, but if acceptance
# collapses from the flatten path's 6.41-8.00 the assumption is wrong.
# REFUTED 2026-08-03, run 30804496473. The native MTP path applied correctly
# and BROKE VERIFICATION:
#   acceptance   2.65-4.76   (flatten: 6.41-8.00)
#   per-position 0.644, 0.534, 0.452, 0.356, 0.342, 0.342, 0.342
#                (flatten: 1.000 x4 then 0.875 x3)
# Position 0 rejected 36% of the time -- the target is scoring drafts against
# the wrong context.
#
# Cause: the MTP kernel masks causally along qlen ("each query position q_pos
# attends KV [0, seq_len-qlen+q_pos]"), but DSpark is NON-CAUSAL by
# construction -- it drafts all k tokens in one parallel block-diffusion pass,
# so verify must score them the same way. Independently confirmed by run
# 30802552602, where ROCM_AITER_MLA was rejected for the DRAFTER with
# "non-causal attention not supported". The flatten branch's non-causality was
# deliberate.
#
# It also did NOT fix the crash: 104 completions, same HSA 0x1016 band.
#
# Defaults back off. dspark_mtp_native.py is kept for the record; re-enabling
# it needs a non-causal MTP kernel, which does not exist today.
export DSPARK_MQA_FIX="${DSPARK_MQA_FIX:-1}"
export DSPARK_MTP_NATIVE="${DSPARK_MTP_NATIVE:-0}"
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
# bf16, not fp8. fp8 + DSpark is closed on this build for THREE independent
# reasons, each hit and confirmed:
#   1. asm verify: asm_mla.cu:903 needs qo_len<=4 or persistent mode, and
#      persistent metadata requires qo_len==1 under spec decode (30784997398 at
#      k=7, 30785599385 at k=3).
#   2. Gluon verify: patched mla_gluon caps batch at 128 while the MQA batch is
#      max_num_seqs*~9 (30786158942 "got 225", 30786908516 "got 144").
#   3. the DRAFTER itself: kimi_k3/nvidia/mla.py:614 asserts "Kimi-K3 fp8 KV
#      cache decode requires a backend that accepts an fp8 (quantized) query
#      input", and TRITON_MLA does not (30787559459).
# Reason 3 fires at first forward, so it blocks fp8 no matter what the target's
# verify path does.
#
# aiter#4474 is dtype-independent -- WITHIN_2GB is about the KV cache exceeding
# 2 GB, and the bf16 pool is 2,156,093 x 576 x 2B = 2.48 GB, over the line. So
# this arm still tests the int64-offset hypothesis, just without fp8's KV
# doubling. Note the gist mla_gluon patch is gated on fp8 and will NOT apply
# here; #4474's anchor is present in the stock image kernel too (verified), and
# the bf16 regime bh16bn64 has no batch_size assert at all.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
# Keep the DSpark arm off the unmerged vLLM#50578 asm-pad patch. It is inert
# here by construction (use_gluon_decode gates on max_qo_len == 1, and verify
# runs at 8, so decode never reaches the asm path), and leaving it out keeps
# this arm a clean single-variable test of the forward_mqa fix.
# asm verify is OFF here, unlike the sibling perf branches. fp8 + asm verify is
# CLOSED on this build: asm_mla.cu:903 says "only support gqa_ratio=16 fp8 mla
# decoding with qo_len <= 4 and qo_len>4 in persistent mode on gfx950". Graph
# capture probes qo_len above 4 even at k=3 (it probed 15 at k=7, i.e. 2k+1),
# and the persistent-mode escape is unreachable because vLLM disables
# persistent metadata for spec decode -- get_mla_metadata_v1 fails at
# qseqlen>1. Runs 30784997398 (k=7) and 30785599385 (k=3) both died there.
#
# So verify goes back to Gluon. That is NOT a fallback to the broken config:
# fp8 gates the gist mla_gluon patch, i.e. the kernel that lifts the
# batch_size==1 restriction, and DSpark verify is the only caller that passes
# batch>1. That patched kernel is the leading hypothesis for the
# hipErrorIllegalAddress at ~13 completions per concurrency slot, and it has
# never been tested against the verify path -- every prior DSpark cell ran
# KV_CACHE_DTYPE=auto, which skips it.
#
# This arm therefore tests the crash fix and both perf levers at once:
# fp8 KV pool (4,240,725 vs 2,156,093 tokens) + prefix caching (0% -> ~92%).
export MLA_ASM_PAD="${MLA_ASM_PAD:-0}"
export DSPARK_ASM_VERIFY="${DSPARK_ASM_VERIFY:-0}"
# 0.88, not the reproducer's 0.95. Runs 30796959716 and 30796973043 both died
# at worker init with "Free memory on device cuda:N (272.0-272.4/287.98 GiB) on
# startup is less than desired GPU memory utilization" -- these mi355x-amds
# nodes free only ~272 of 288 GiB and 0.95 asks for ~273.6. Known ceiling.
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
# mns 8, not the reproducer's 16. The patched gist mla_gluon accepts
# 1 <= batch_size <= 128, and the DSpark MQA batch scales as max_num_seqs * ~9
# (not k+1 = 8 as assumed): run 30786158942 died with "got 225" and 30786908516
# with "got 144" even after cudagraph capture was capped at 128, which shows the
# capture list is in SEQUENCES, not tokens. 16 * 9 = 144 > 128; 8 * 9 = 72 fits
# with margin. mns 8 exactly covers c8 concurrency, since aiperf bounds
# in-flight requests by --concurrency.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
export PREFIX_CACHING="${PREFIX_CACHING:-true}"
# Cap cudagraph capture at the gist mla_gluon patch's batch ceiling. The capture
# list is independent of max_num_seqs and defaults up to 256, while the patched
# kernel supports 1 <= batch_size <= 128; run 30786158942 died at init with
# "got 225". Runtime is already bounded at max_num_seqs * (k+1) = 16 * 8 = 128,
# so only capture overshot. Capping keeps cudagraphs on instead of falling back
# to --enforce-eager.
export CUDAGRAPH_MAX_CAPTURE="${CUDAGRAPH_MAX_CAPTURE:-128}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

exec "$(dirname "$0")/kimik3_fp4_mi355x.sh" "$@"
