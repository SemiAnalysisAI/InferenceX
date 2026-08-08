#!/usr/bin/env bash
# DSv4-Pro FP4 agentic-coding on MI355X -- DEP8 + MegaMoE + sparse gluon + EPLB.
#
# This is dsv4_agentic_mi355x_dep8_megamoe_gluon.sh with expert-parallel load
# balancing turned on. Read that script first: every constraint it documents
# (MegaMoE preconditions, why DEP8 loses to TP8 here, the deliberate omission
# of DRAM offload) applies unchanged.
#
# Measured against that same arm with --enable-eplb as the only delta, both
# arms on the identical patched model:
#
#                       EPLB off      EPLB on      delta
#   total tok/s         40,452.45    43,976.82     +8.71%
#   output tok/s           210.32       245.95    +16.94%
#   ITL p90               141.9 ms     112.5 ms    -20.73%
#   TTFT p50                2.67 s       2.48 s     -7.04%
#   requests ok                479          522     (0 errors either arm)
#
# Expert balancedness climbed from ~0.45 to 0.55-0.56, with one profile
# rearrangement at 4.34 s.
#
# CAVEAT: those numbers come from a CONC=32 pair at DURATION=1200 with
# AIPERF_EXPERIMENTAL_FAST=1, which is NOT what this script runs. They are a
# single measurement per arm with no variance estimate, and the fast path
# truncates warmup below aiperf's coverage gate so neither arm was validated.
# This script uses the full 3600 s no-FAST settings so its output is
# comparable to dsv4_agentic_mi355x_dep8_megamoe_gluon.sh; the EPLB delta at
# that duration has not been measured yet.
#
# ---------------------------------------------------------------------------
# Why EPLB helps more here than on the aiter backend
# ---------------------------------------------------------------------------
# The same A/B on --moe-backend aiter gives only +1.16% total and *worsens*
# ITL p90 by 38%. Offered as a hypothesis, not a proof: the mega path's
# dispatch/combine chain is synchronous, so an uneven expert distribution
# stalls the critical path directly rather than overlapping with other work,
# and the rebalance pays for itself. On aiter the async rearrangements show up
# as tail latency instead.
#
# ---------------------------------------------------------------------------
# What EPLB requires beyond the base arm
# ---------------------------------------------------------------------------
# A patched DSv4 amd/model.py implementing the MixtureOfExperts protocol, plus
# an amd/mega_moe_experts.py carrying EPLB on the FlyDSL path. Both are baked
# into the image pinned below -- see benchmarks/single_node/agentic/
# dsv4_eplb_image/ for the build context and README.
#
# Without them the runner never registers the model, so --enable-eplb is a
# SILENT NO-OP rather than an error: the arm runs, reports normally, and
# measures nothing. If you repoint IMAGE at the unpatched -megamoe tag you get
# exactly that. The image's build-time verifier is what keeps this honest.
#
# NUM_REDUNDANT_EXPERTS must keep (n_routed_experts + N) % ep_size == 0 -- a
# multiple of 8 for this 384-expert checkpoint on 8 ranks. The patched model
# rejects a bad value at init rather than asserting inside rebalance_execute
# thousands of steps later. 8 gives 392 physical experts, 49 per rank.
#
# Two consequences of 49 experts/rank, neither a bug:
#   * ~2.1 GiB more weight memory (+1 expert x 61 layers), plus EPLB's
#     one-layer expert_buffer at ~1.7 GiB. Ample headroom at GPU_MEM_UTIL 0.86.
#   * --enable-ep-weight-filter becomes a no-op (default_loader.py skips it
#     entirely under enable_eplb, since replica slots need non-local logical
#     weights), so model load is slower. Set NUM_REDUNDANT_EXPERTS=0 for
#     pure-permutation EPLB at 48/rank if that matters.
#
# ---------------------------------------------------------------------------
# MORI_SHMEM_HEAP_SIZE is deliberately NOT set here
# ---------------------------------------------------------------------------
# The base arm hardcodes 12 GiB. The patched mega_moe_experts.py computes it
# instead: _ensure_symmetric_heap_size() derives the requirement from the
# MegaMoEShape (7.73 GiB at max_num_batched_tokens=16384, x1.25 headroom,
# rounded to 10 GiB) and sets the variable before the collective init.
#
# This works because mori reads MORI_SHMEM_HEAP_SIZE *lazily*, at
# shmem_torch_process_group_init -- verified with an 8-rank probe, not assumed.
# The heap not growing on demand is a separate fact and remains true.
#
# Sizing it in the model is the right place: it is the only code that knows
# the shape, and the heap is pre-reserved VRAM, so a blanket launcher value
# taxes every arm that does not need it. EPLB's 49th expert per rank widens
# the buffer by 128 rows -- noise against 7.73 GiB, but it is one more reason
# the number should be derived rather than typed.
#
# The variable is still forwarded by dsv4_agentic_g05.sh and the patch never
# lowers an explicit value, so exporting it on the host still wins.
#
# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------
# Same as the base arm: 8x MI355X (gfx950), ~806 GB of weights, the 1.8 GB
# trace corpus pre-cached, and ports 8700/8701/18700 free. Add ~4 GB of GPU
# memory for the redundant experts and the rearrangement buffer.
#
# Usage:
#   MODEL_DIR=/path/to/DeepSeek-V4-Pro HF_CACHE=/path/to/hf \
#     ./dsv4_agentic_mi355x_dep8_megamoe_gluon_eplb.sh
set -euo pipefail

# Pinned by digest, not tag: tags are mutable -- this one was already re-pushed
# once -- and the measurement is only meaningful against these exact kernels.
# Contents, newest layer first:
#   + amd/model.py, amd/mega_moe_experts.py -- the EPLB patch (this arm needs it)
#   + megamoe_port/enable_megamoe.sh, MORI_ARCH=gfx950
#   + aiter PR #4382 -- gfx950 gluon sparse-MLA decode
#   base: DSv4 FP4 + aiter PR #4269 -- fused shared expert (unused here)
# Built from benchmarks/single_node/agentic/dsv4_eplb_image/ over
# jiahcao/vllm-dsv4@sha256:f2fbeada... , the digest the base arm pins.
export IMAGE="${IMAGE:-jiahcao/vllm-dsv4@sha256:283244483864a2413c2ca5357459799b998361fee4f1d2479282231ff94d7e50}"

export MODEL_DIR="${MODEL_DIR:-/mnt/models/DeepSeek-V4-Pro}"
export HF_CACHE="${HF_CACHE:-/var/tmp/$USER-hf}"
export SCRATCH="${SCRATCH:-/var/tmp/$USER-dsv4-agentic}"

export TP=8 EP_SIZE=8 DP_ATTENTION=true
export CONC=32 DURATION=3600
export PORT="${PORT:-8700}"
export VLLM_ROUTER_POLICY=consistent_hash
export KV_OFFLOADING=none
export GPU_MEM_UTIL=0.86

export MOE_BACKEND=flydsl_mega_moe
export VLLM_ROCM_DSV4_SPARSE_GLUON=1

# step_interval 3000 and window_size 1000 are the values the +8.71% arm ran.
# At CONC=32 over 3600 s that is a handful of rearrangements, each ~4.3 s of
# profile time -- frequent enough to track a drifting trace, rare enough that
# the copies do not dominate.
export ENABLE_EPLB=1
export NUM_REDUNDANT_EXPERTS="${NUM_REDUNDANT_EXPERTS:-8}"
export EPLB_STEP_INTERVAL="${EPLB_STEP_INTERVAL:-3000}"
export EPLB_WINDOW_SIZE="${EPLB_WINDOW_SIZE:-1000}"
export EPLB_LOG_INTERVAL="${EPLB_LOG_INTERVAL:-100}"

export RUN_TAG="${RUN_TAG:-dep8_megamoe_gluon_eplb}"
export RESULT_FILENAME="${RESULT_FILENAME:-dsv4_vllm_fp4_tp8_ep8_conc32_megamoe_gluon_eplb}"

# Three log lines decide whether EPLB actually engaged. Check them before
# trusting any delta -- a silent no-op looks identical to a real run:
#   "MegaMoEV2: building shared instance ... experts=392"  -- once, not twice
#       (twice means MTP and target layers built separate runtimes, which
#        costs a second ~7.7 GiB of symmetric buffers)
#   "EPLB is enabled for model ..."   -- the model registered; absence is the
#                                        silent failure described above
#   "Rearranged experts ... in N s"   -- rebalance_execute actually ran
#
# Known cosmetic defect, same as the base arm: the aggregate JSON records
# spec_decoding "none" even though MTP is active, because SPEC_DECODING is not
# set on this manual path and dsv4_agentic_g05.sh does not forward it.
# Performance numbers are unaffected; fix the field before reporting upward.

exec bash "$(dirname "${BASH_SOURCE[0]}")/dsv4_agentic_g05.sh"
