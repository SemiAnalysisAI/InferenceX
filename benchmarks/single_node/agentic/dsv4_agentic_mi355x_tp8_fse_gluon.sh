#!/usr/bin/env bash
# DSv4-Pro FP4 agentic-coding on MI355X -- TP8 + fused shared expert + sparse gluon.
#
# This is the fastest validated arm measured on this hardware:
#
#   headline           11,751 tok/s/GPU
#   computed prefill      920.9 tok/s/GPU   (real compute, cache-hit tokens excluded)
#   output tput            83.5 tok/s/GPU
#   ITL p50                25.7 ms
#   TTFT p50                0.849 s
#   gpu prefix hit         93.45%
#   duration                3620 s
#   validation             PASSED -- coverage TTFT=100.0% ITL=100.0% (required 98.0%),
#                          error rate 1/2595 = 0.039% (<= 10%)
#
# Measured 2026-08-07 on smci355-1b112-b23-07 (8x MI355X, gfx950).
# Result JSON: agentic_results/tp8_fse_gluon_validating/
#              dsv4_vllm_fp4_tp8_ep1_conc32_fse_gluon.json
#
# ---------------------------------------------------------------------------
# Why this shape
# ---------------------------------------------------------------------------
# TP8 with expert parallelism OFF. On MI355X this beats the DEP8 shape that
# B200 CI uses by 1.50x -- do not copy the B200 topology here. DEP8 buys a
# 3.5x larger KV pool (22.8M vs 6.6M tokens) by sharding the 384 experts
# across ranks, but at CONC=32 this workload peaks at 27% pool usage, so that
# capacity is never spent while the all2all cost is paid on every layer. See
# dsv4_agentic_mi355x_dep8_megamoe_gluon.sh for the best DEP8 number (7,822).
#
# Fused shared expert (FSE): DSv4-Pro is a mixed checkpoint -- MXFP4 routed
# experts plus a native-FP8 shared expert. FSE folds the FP8 shared expert in
# as slot 385 of the otherwise-MXFP4 grouped GEMM (semantic top-k=6, internal
# E=385 top-k=7) using aiter's heterogeneous fhmoe kernel. No patch or overlay
# is needed on this shape: dsv4_fp4_mi355x_vllm_mtp.sh exports
# VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 unconditionally, and TP8 + non-EP
# is the path upstream already allows. Confirm it engaged by grepping the
# server log for "Fusing the DeepSeek V4 native-FP8 shared expert" (expect 1).
#
# FSE cannot be combined with expert parallelism. aiter/fhmoe.py raises
# NotImplementedError("Heterogeneous MXFP4/FP8 experts do not yet support
# expert masks"), and EP always supplies an expert_map -> expert_mask, since
# each rank holds only 384/8 experts. This is a kernel limitation, not a gate:
# patching the vLLM-side check lets init proceed and then all 8 workers die in
# profile_run. That is why this arm runs EP_SIZE=1.
#
# VLLM_ROCM_DSV4_SPARSE_GLUON=1 enables the gfx950 gluon sparse-MLA decode
# kernel (aiter PR #4382), baked into the pinned image. It is worth 1.44x on
# ITL p50 by itself (36.9 ms -> 25.7 ms against the otherwise identical TP8
# baseline).
#
# ---------------------------------------------------------------------------
# Deliberate omissions
# ---------------------------------------------------------------------------
# No AIPERF_EXPERIMENTAL_FAST. It shortens warmup below aiperf's profile
# metric coverage ratio gate (0.98) and the run fails validation after burning
# the full wall clock. An earlier run of this exact config with FAST=1 reported
# 9,531 tok/s/GPU and was rejected at that gate. Dropping FAST is the only
# difference between that run and this one; ITL p50 reproduced (26.7 -> 25.7 ms)
# and output throughput rose 30%, so the speed was real -- FAST only truncated
# warmup, which depressed prefix hit rate (87.9% -> 93.45%) and with it the
# headline number. Headline tput counts cache-hit tokens, so it moves with hit
# rate; compare arms on computed-prefill and ITL, not on the headline alone.
#
# No DRAM KV offload. Measured to contribute exactly nothing on this workload:
# cpu_cache_hit_rate 0.0 and external_prefix_cache_hits 0 in every arm run,
# because peak KV pool usage is 45% and the pool therefore never evicts.
# Offload only pays when eviction happens. Turning it off also avoids the
# KV_OFFLOAD_BACKEND / _METADATA pairing that process_agentic_result validates
# only after the replay finishes.
#
# No MegaMoE. It requires tp_size==1 plus expert parallelism, and moe_backend
# is a single value, so it is mutually exclusive with FSE.
#
# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------
#   * 8x MI355X (gfx950). The gluon kernel and fhmoe both hard-require gfx950.
#   * ~806 GB for the DeepSeek-V4-Pro weights (64 safetensors shards). Verify
#     the tree is complete -- a partial rsync fails late and obscurely, as a
#     missing tokenizer.json surfaces only as "You need to have sentencepiece
#     or tiktoken installed".
#   * The 1.8 GB trace corpus pre-cached; dsv4_agentic_g05.sh preflights this.
#   * Port 8700 free (--network host).
#
# Override MODEL_DIR / HF_CACHE / SCRATCH / PORT for your machine; everything
# else is the measured configuration and should be left alone to reproduce.
#
# Usage:
#   MODEL_DIR=/path/to/DeepSeek-V4-Pro HF_CACHE=/path/to/hf \
#     ./dsv4_agentic_mi355x_tp8_fse_gluon.sh
set -euo pipefail

# Pinned by digest, not tag: tags are mutable and this measurement is only
# meaningful against these exact kernels. Contents, newest layer first:
#   + megamoe_port/enable_megamoe.sh, MORI_ARCH=gfx950   (inert here -- MegaMoE
#     is only reachable via MOE_BACKEND, which this arm does not set)
#   + aiter PR #4382 -- gfx950 gluon sparse-MLA decode
#   base: DSv4 FP4 + aiter PR #4269 -- fused shared expert
export IMAGE="${IMAGE:-jiahcao/vllm-dsv4@sha256:f2fbeadab5172c919a8f40264a54c6434e3599dcfabc1f32223e5229b8cb2631}"

export MODEL_DIR="${MODEL_DIR:-/mnt/models/DeepSeek-V4-Pro}"
export HF_CACHE="${HF_CACHE:-/var/tmp/$USER-hf}"
export SCRATCH="${SCRATCH:-/var/tmp/$USER-dsv4-agentic}"

export TP=8 EP_SIZE=1 DP_ATTENTION=false
export CONC=32 DURATION=3600
export PORT="${PORT:-8700}"
export VLLM_ROUTER_POLICY=round_robin
export KV_OFFLOADING=none
export GPU_MEM_UTIL=0.86

export VLLM_ROCM_DSV4_SPARSE_GLUON=1

export RUN_TAG="${RUN_TAG:-tp8_fse_gluon}"
export RESULT_FILENAME="${RESULT_FILENAME:-dsv4_vllm_fp4_tp8_ep1_conc32_fse_gluon}"

# Known cosmetic defect: the aggregate JSON records spec_decoding "none" even
# though MTP is active. SPEC_DECODING is not set on this manual path and
# dsv4_agentic_g05.sh does not forward it, so the aggregator cannot see it.
# Performance numbers are unaffected; fix the field before reporting upward.

exec bash "$(dirname "${BASH_SOURCE[0]}")/dsv4_agentic_g05.sh"
