#!/usr/bin/env bash
# DSv4-Pro FP4 agentic-coding on MI355X -- DEP8 + MegaMoE + sparse gluon.
#
# The best DEP8 number measured on this hardware:
#
#   headline            7,822 tok/s/GPU
#   computed prefill      764.0 tok/s/GPU   (real compute, cache-hit tokens excluded)
#   output tput            54.9 tok/s/GPU
#   ITL p50                47.4 ms
#   TTFT p50                2.215 s
#   gpu prefix hit         92.49%
#   duration                3617 s
#   request errors             0
#
# Measured 2026-08-07 on smci355-1b112-b23-07 (8x MI355X, gfx950).
# Result JSON: agentic_results/megamoe_gluon_3600/
#              dsv4_vllm_fp4_tp8_ep8_conc32_megamoe_gluon.json
#
# NOT run through process_agentic_result validation. The replay itself is
# clean (0 errors, matched duration and hit rate against the other 3600s arms),
# but the validation gate was never exercised on this arm. Treat the number as
# a measurement, not a reportable result, until that is done.
#
# ---------------------------------------------------------------------------
# What this arm is for
# ---------------------------------------------------------------------------
# It is NOT the fastest configuration on MI355X. TP8 + FSE + gluon reaches
# 11,751 tok/s/GPU -- 1.50x this arm -- see dsv4_agentic_mi355x_tp8_fse_gluon.sh.
# Run this one to reproduce the DEP8 topology (which B200 CI uses) on AMD, or
# to measure MegaMoE specifically.
#
# The DEP8-vs-TP8 gap on MI355X is a topology mismatch, not a tuning deficit.
# DEP8 shards the 384 experts across ranks, freeing memory for a 3.5x larger
# KV pool (22.8M vs 6.6M tokens) -- but this workload peaks at 40% pool usage
# at CONC=32, so the capacity is never spent while all2all is paid every layer.
# MegaMoE narrowed the gap from 1.74x to 1.50x; it did not close it.
#
# Against the plain DEP8 baseline (6,745 tok/s/GPU, dram offload, aiter MoE),
# at matched duration and hit rate, MegaMoE is worth: headline +16.0%,
# computed prefill +8.8%, output tput +17.9%, ITL p50 1.43x faster
# (67.6 -> 47.4 ms), TTFT flat.
#
# ---------------------------------------------------------------------------
# MegaMoE preconditions -- all three are required
# ---------------------------------------------------------------------------
#   tp_size == 1 + enable_expert_parallel   (DP_ATTENTION=true EP_SIZE=8; the
#                                            TP=8 below is the launcher's
#                                            world-size knob, the model's own
#                                            tp_size is 1 under DP-attention)
#   MOE_BACKEND=flydsl_mega_moe
#   MORI_SHMEM_HEAP_SIZE large enough       (see below)
#
# MORI_SHMEM_HEAP_SIZE is the one that is not discoverable from the error text
# alone. mori's static symmetric heap defaults to a hardcoded 4 GiB
# (DEFAULT_STATIC_SYMMETRIC_HEAP_SIZE = 4ULL<<30, internal.hpp:43). MegaMoE
# sizes its buffer as num_valid_max * per_row_bytes(10476), which at
# max_num_batched_tokens=16384 is 7.73 GiB -- past the default, so allocation
# fails at init. By max_num_batched_tokens: 16384 -> 7.73 GiB, 8192 -> 3.90,
# 4096 -> 1.98, 2048 -> 1.02. 12 GiB leaves headroom.
#
# Lowering gpu_memory_utilization alone does NOT fix this -- the heap is a
# compile-time constant, not vLLM contention. But GPU_MEM_UTIL=0.86 is still
# required to leave room for the 12 GiB heap once it is allowed to be that big.
#
# MegaMoE does not fuse the shared expert. _init_mega_moe_experts returns early
# and self.shared_experts stays a separate DeepseekV4MLP. What MegaMoE fuses is
# dispatch + both expert GEMMs + combine. It is mutually exclusive with FSE
# because moe_backend is a single value.
#
# VLLM_ROCM_DSV4_SPARSE_GLUON=1 enables the gfx950 gluon sparse-MLA decode
# kernel (aiter PR #4382), baked into the pinned image, and composes with
# MegaMoE.
#
# ---------------------------------------------------------------------------
# Deliberate omissions
# ---------------------------------------------------------------------------
# No DRAM KV offload. Measured to contribute nothing (cpu_cache_hit_rate 0.0,
# 0 external hits across ~20.4M queries) because the pool never fills and so
# never evicts -- and its staging buffer would compete with the 12 GiB
# symmetric heap, which here is a hard init failure rather than a slowdown.
#
# No AIPERF_EXPERIMENTAL_FAST: it truncates warmup below aiperf's coverage
# ratio gate (0.98) and costs the run its validation.
#
# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------
#   * 8x MI355X (gfx950). MegaMoE/mori and gluon both hard-require gfx950.
#   * ~806 GB for the DeepSeek-V4-Pro weights (64 safetensors shards, complete
#     tree including tokenizer.json).
#   * The 1.8 GB trace corpus pre-cached; dsv4_agentic_g05.sh preflights this.
#   * Ports 8700, 8701, 18700 free -- DP-attention needs PORT, PORT+1 and
#     PORT+10000 (--network host).
#
# Expect this in the server log during the run, repeatedly:
#   [aiter] [fused_moe] no tuned FlyDSL config for ('gfx950', 256, 1024, 7168,
#   ...), using heuristic FlyDSL fallback
# The MoE GEMM shapes are untuned on gfx950. It is not an error, and it is a
# concrete open lead for further DEP8 gains.
#
# Usage:
#   MODEL_DIR=/path/to/DeepSeek-V4-Pro HF_CACHE=/path/to/hf \
#     ./dsv4_agentic_mi355x_dep8_megamoe_gluon.sh
set -euo pipefail

# Pinned by digest, not tag: tags are mutable and this measurement is only
# meaningful against these exact kernels. Contents, newest layer first:
#   + megamoe_port/enable_megamoe.sh, MORI_ARCH=gfx950   (required by this arm)
#   + aiter PR #4382 -- gfx950 gluon sparse-MLA decode
#   base: DSv4 FP4 + aiter PR #4269 -- fused shared expert (unused here)
export IMAGE="${IMAGE:-jiahcao/vllm-dsv4@sha256:f2fbeadab5172c919a8f40264a54c6434e3599dcfabc1f32223e5229b8cb2631}"

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
export MORI_SHMEM_HEAP_SIZE=12884901888   # 12 GiB; default 4 GiB is too small

export RUN_TAG="${RUN_TAG:-dep8_megamoe_gluon}"
export RESULT_FILENAME="${RESULT_FILENAME:-dsv4_vllm_fp4_tp8_ep8_conc32_megamoe_gluon}"

# Known cosmetic defect: the aggregate JSON records spec_decoding "none" even
# though MTP is active. SPEC_DECODING is not set on this manual path and
# dsv4_agentic_g05.sh does not forward it, so the aggregator cannot see it.
# Performance numbers are unaffected; fix the field before reporting upward.

exec bash "$(dirname "${BASH_SOURCE[0]}")/dsv4_agentic_g05.sh"
