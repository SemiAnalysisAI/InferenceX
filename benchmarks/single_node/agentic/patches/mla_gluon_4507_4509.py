# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# ---------------------------------------------------------------------------
# VENDORED, NOT AUTHORED HERE. Drop-in replacement for
# aiter/ops/triton/gluon/mla_gluon.py, installed by kimik3_fp4_mi355x.sh.
#
# Base: ROCm/aiter main, blob 64dc4bec418 (the exact base all three PRs below
# diff against -- verified by git hash-object).
#
# Upstream PRs applied, in order, all clean:
#   ROCm/aiter#4474  int64 KV offsets on the >2GB (global_load) path
#   ROCm/aiter#4507  size NUM_KV_SPLITS from the page table, not the unset
#                    min_kv_seq_len hint (bh16bn64)
#   ROCm/aiter#4509  split-major grid + blocked stage-2 reduce + partial waits
#
# Local deltas on top, each marked "LOCAL" at its site:
#   1. fp8 query -> bf16 dequant, and the bh16bn128 batch_size 1..128
#      relaxation. Both forward-ported from the seungrokj gist this file
#      replaces (gist rev 4b0088c5); that gist predates main's nhead<=96
#      head-tiling work, so it could not simply be patched in place.
#   2. #4507's page-table split sizing extended to bh16bn128, since the fp8
#      regime has the same unset-hint bug. AITER_MLA_B128_PT_SPLITS=0 reverts
#      that one knob for A/B.
#   3. get_num_sms import fallback for images whose aiter predates it.
#
# Why this replaced the curl-from-gist: the image's own mla_gluon.py has almost
# no qlen/MTP support (3 mentions vs 31 in the gist), and the gist install was
# gated on KV_CACHE_DTYPE=fp8 -- so the bf16 DSpark arm was silently running the
# stock image kernel with neither the MTP path nor either perf PR.
# ---------------------------------------------------------------------------

# Gluon MLA decode kernel originated from FlashMLA triton kernel(https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py).
# Stage-1 split-KV MLA attention using explicit Gluon layouts. Three regimes:
#
#   REGIME='bh64'      - bf16 Q + bf16 KV, BLOCK_H=64, BLOCK_N=64,
#                        nhead in {64, 128}, batch_size in {64, 128, 256},
#                        NUM_KV_SPLITS auto-picked to fill ~256 WGs (in {1,2,4}).
#                        Fast path: when NUM_KV_SPLITS==1, stage-1 writes the
#                        final output directly to O and stage-2 reduce is skipped.
#   REGIME='bh16bn128' - bf16 Q + fp8 KV, BLOCK_H=16, BLOCK_N=128,
#                        nhead <= 96, batch_size=1, NUM_KV_SPLITS=256.
#                        (batch, split, head_block*qlen) grid. Always splits +
#                        always reduces. A partial last head block
#                        (nhead % BLOCK_H != 0) masks OOB heads on Q load / O store.
#   REGIME='bh16bn64'  - bf16 Q + bf16 KV, BLOCK_H=16, BLOCK_N=64,
#                        nhead <= 96, batch_size >= 1,
#                        (batch, split, head_block*qlen) grid. Full decode
#                        (stage-1 + stage-2 reduce into the final O). A partial
#                        last head block (nhead % BLOCK_H != 0) masks OOB heads.
#
# The bh16 regimes support num_iter in {1, 2, ...} (no gl.assume(num_iter>=3));
# only bh64 assumes >= 3. See epilogue-1 handling below.
#
# Wrapper dispatch: nhead in {64,128} -> bh64; nhead <= 96 routes by KV dtype
# (bf16 -> bh16bn64, fp8 -> bh16bn128), tiling heads into cdiv(nhead,16) blocks.
#
# Full decode for all regimes. For NUM_KV_SPLITS>1 stage-1 writes per-split acc +
# fp32 lse; stage-2 (_mla_softmax_reducev_kernel) reduces into O. RETURN_LSE also
# returns the merged fp32 lse [B, H] (stage-2 for splits>1, else stage-1).
#
# 3-stage software pipeline (double-buffered, BLOCK_N with 2x(BLOCK_N/2) KV slices):
#   AC = async_copy (global->LDS), LL = load (LDS->reg), P = page, K = K-cache, V = V-cache
#
#                      iter i          iter i+1        iter i+2
#   ACP(page):        [i+2]           [i+3]           [i+4]
#   LLP+ACK(K):       [i+1]           [i+2]           [i+3]
#   LLK+MFMA+LLV:     [i]             [i+1]           [i+2]
#
#   Within each loop iteration (operating on buf_idx=current, async_idx=next):
#     ACP                                -- async_copy page numbers [i+2]
#     LLP, ACK                           -- local_load pages [i+1], async_copy K/KPE [i+1]
#     LLK, MFMA0, softmax, LLV, MFMA1   -- compute on [i]: QK dot, softmax, PV dot

# isort and black disagree here: isort wants two blank lines after this block,
# black folds them back to one because the `# fmt: off` below starts a
# formatting-disabled region. black is the one CI enforces, so it wins.
import os

import torch  # noqa: I001
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton import arch_info

try:
    from aiter.ops.triton.utils.device_info import get_num_sms, get_num_xcds
except ImportError:  # LOCAL: older aiter has no get_num_sms; #4507 needs it.
    from aiter.ops.triton.utils.device_info import get_num_xcds

    def get_num_sms():
        return torch.cuda.get_device_properties(0).multi_processor_count


# LOCAL: see the bh16bn128 split-sizing block in mla_gluon(). Set to 0 to restore
# the upstream (min_kv_seq_len-derived) formula on the fp8 KV regime.
_B128_PT_SPLITS = os.environ.get("AITER_MLA_B128_PT_SPLITS", "1") == "1"

# fmt: off
@gluon.jit
def _mla_gluon(
    Q_nope,
    Q_pe,
    Kv_c_cache,
    K_pe_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    Attn_sink,
    sm_scale,
    kv_scale,
    stride_q_nope_bs,
    stride_q_nope_s,  # MTP: q_pos (qlen) stride; 0 when QLEN==1
    stride_q_nope_h,
    stride_q_pe_bs,
    stride_q_pe_s,  # MTP: q_pos (qlen) stride; 0 when QLEN==1
    stride_q_pe_h,
    stride_kv_c_bs,
    stride_k_pe_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_s,  # MTP: q_pos (qlen) stride on O/logits; 0 when QLEN==1
    stride_o_h,
    stride_o_split,
    Mid_lse,  # split>1: per-split fp32 lse [B, QLEN, H, NUM_KV_SPLITS] (else None)
    stride_mid_lse_b,
    stride_mid_lse_s,  # MTP: q_pos stride; 0 when QLEN==1
    stride_mid_lse_h,
    stride_mid_lse_split,
    Final_lse,  # RETURN_LSE only: merged fp32 lse [B, QLEN, H] (else None)
    stride_final_lse_b,
    stride_final_lse_s,  # MTP: q_pos stride; 0 when QLEN==1
    stride_final_lse_h,
    BLOCK_H: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    HEAD_DIM_CKV: gl.constexpr,
    HEAD_DIM_KPE: gl.constexpr,
    KV_PE_OFFSET: gl.constexpr,
    USE_2D_VIEW: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NHEAD: gl.constexpr,
    REGIME: gl.constexpr,
    RETURN_LSE: gl.constexpr,
    QLEN: gl.constexpr,  # MTP query length; 1 for plain decode
    # --- dsv4-prefill knobs ---
    HAS_PE: gl.constexpr,
    HAS_ATTN_SINK: gl.constexpr,
    # bh16 only: swap grid axes 0 and 1 so the split index varies fastest.
    # Host-chosen constexpr; a pure relabeling, see the launcher for why.
    SPLIT_MAJOR: gl.constexpr = False,
):
    # Grid mapping: bh64 uses 3-D XCD-aware multi-batch; bh16bn64 and bh16bn128
    # use 2-D (batch, split) — for batch_size=1 this is (1, NUM_KV_SPLITS).
    # MTP: an extra q_pos axis carries the query position within QLEN. bh64 packs
    # it into grid axis 1 (after the head-block index); bh16 uses grid axis 2.
    # When QLEN==1, q_pos is always 0 and the layout below is identical to before.
    if REGIME == 'bh64':
        NUM_M_BLOCKS: gl.constexpr = (NHEAD + BLOCK_H - 1) // BLOCK_H
        cur_batch = gl.program_id(0) + (gl.program_id(2) // NUM_KV_SPLITS) * NUM_XCDS
        cur_head_id = gl.program_id(1) % NUM_M_BLOCKS
        q_pos = gl.program_id(1) // NUM_M_BLOCKS
        split_kv_id = gl.program_id(2) % NUM_KV_SPLITS
    else:
        # bh16*: grid axis 2 carries (head_block, q_pos). For nhead <= 16 there is
        # a single head block (NUM_M_BLOCKS==1) so cur_head_id==0 and q_pos==pid(2),
        # identical to the original 2-D+qlen mapping. For nhead > 16 (e.g. 96) the
        # head range is tiled into NUM_M_BLOCKS = cdiv(NHEAD, BLOCK_H) blocks of 16.
        NUM_M_BLOCKS: gl.constexpr = (NHEAD + BLOCK_H - 1) // BLOCK_H
        if SPLIT_MAJOR:
            split_kv_id = gl.program_id(0)
            cur_batch = gl.program_id(1)
        else:
            cur_batch = gl.program_id(0)
            split_kv_id = gl.program_id(1)
        cur_head_id = gl.program_id(2) % NUM_M_BLOCKS
        q_pos = gl.program_id(2) // NUM_M_BLOCKS

    # USE_2D_VIEW=True: fixed len or max padded VarLen
    # Req_to_tokens = block_table[batch, max_seqlen], B_seq_len = cache_seqlens[batch]
    # USE_2D_VIEW=False: flattened VarLen
    # Req_to_tokens = kv_indices[total_kv],           B_seq_len = kv_indptr[batch+1]
    if USE_2D_VIEW:
        batch_page_start = stride_req_to_tokens_bs * cur_batch
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch)
    else:
        batch_page_start = gl.load(B_seq_len + cur_batch)
        cur_batch_seq_len = gl.load(B_seq_len + cur_batch + 1) - batch_page_start

    # split-KV: each program covers [split_kv_start, split_kv_end).
    # OLD: ceil-based per_split. The LAST split could be empty (num_iter=0),
    # which breaks the unconditional epilogue-2 consume. Kept here as commented
    # reference; remove in cleanup.
    # kv_len_per_split = gl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    # split_kv_start = kv_len_per_split * split_kv_id
    # split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    #
    # NEW: floor per_split with the last split absorbing the remainder
    # (remainder = seq mod NUM_KV_SPLITS, in [0, NUM_KV_SPLITS)). Combined with
    # the wrapper bound min_kv_seq_len >= NUM_KV_SPLITS this guarantees every
    # split is non-empty (split_len >= floor >= 1, hence num_iter >= 1); bh64
    # additionally bounds min_kv_seq_len so num_iter >= 3 for its gl.assume.
    # Trade-off: at seqs just above the wrapper minimum the last CU does up to
    # ~(floor + NUM_KV_SPLITS - 1)/floor more work than the others.
    kv_len_per_split = cur_batch_seq_len // NUM_KV_SPLITS
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = split_kv_start + kv_len_per_split
    if split_kv_id == NUM_KV_SPLITS - 1:
        split_kv_end = cur_batch_seq_len
    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    start_n = split_kv_start

    # >2GB KV cache (global_load path): widen strides to int64 so kv offsets don't overflow int32.
    if not WITHIN_2GB:
        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)

    # early return with empty kv slice to save compute
    if split_kv_start >= split_kv_end:
        return

    # MTP causal tail mask: query position q_pos may attend KV
    # [0, seq_len-QLEN+q_pos] only, so score_end is its per-program valid-score
    # bound. For QLEN==1 this equals split_kv_end, keeping the original code
    # path untouched.
    if QLEN > 1:
        score_end = gl.minimum(split_kv_end, cur_batch_seq_len - QLEN + q_pos + 1)
    else:
        score_end = split_kv_end

    ######### layout setting begin #########
    # Q-side layouts + mfma_layout: switch by BLOCK_H.
    # bh64 has BLOCK_H=64; bh16bn128 and bh16bn64 share BLOCK_H=16 (identical Q layouts + mfma orientation).
    if BLOCK_H == 64:
        # bh64: Q is [64, 512] / [64, 64]; warps tile M.
        blocked_q_nope: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 64],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]],
            cga_layout=[],
            shape=[64, 512]
        )
        blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (32, 0)),
            lane_bases=((0, 8), (0, 16), (0, 32), (4, 0), (8, 0), (16, 0)),
            warp_bases=((1, 0), (2, 0)),
            block_bases=[],
            shape=[64, 64],
        )
        shared_q_pe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [1, 0], [2, 0], [32, 0]],
            cga_layout=[],
            shape=[64, 64]
        )
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[4, 1],
        )
    else:
        # BLOCK_H == 16: shared by bh16bn128 and bh16bn64. Q is [16, 512] / [16, 64]; warps tile K.
        blocked_q_nope: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[1, 64],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        shared_q_nope: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [1, 0], [2, 0], [4, 0], [8, 0]],
            cga_layout=[],
            shape=[16, 512]
        )
        blocked_q_pe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4)),
            lane_bases=((0, 8), (0, 16), (0, 32), (1, 0), (2, 0), (4, 0)),
            warp_bases=((8, 0), (0, 0)),
            block_bases=[],
            shape=[16, 64],
        )
        shared_q_pe: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=8, order=[1, 0])
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[1, 4],
        )

    # KV-side layouts: switch by BLOCK_N.
    # bh16bn128 (BLOCK_N=128, fp8 KV) needs distinct K layouts; bh64 and bh16bn64 share BLOCK_N=64 bf16 KV.
    if BLOCK_N == 128:
        # bh16bn128: K is [512, 128]fp8, KPE is [64, 128]fp8.
        blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 4), (0, 32), (0, 64)),
            lane_bases=((16, 0), (32, 0), (64, 0), (128, 0), (256, 0), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 128],
        )
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[1024, 32], [8192, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 16], [0, 1], [0, 2], [0, 8], [0, 4], [0, 32], [0, 64]],
            cga_layout=[],
            shape=[512, 128]
        )
        blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 2)),
            lane_bases=((16, 0), (32, 0), (0, 4), (0, 8), (0, 16), (0, 32)),
            warp_bases=((0, 64), (0, 1)),
            block_bases=[],
            shape=[64, 128],
        )
        shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[2048, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2]],
            cga_layout=[],
            shape=[64, 128]
        )
        blocked_page: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0,),),
            lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
            warp_bases=((64,), (0,)),
            block_bases=[],
            shape=[128],
        )
        blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 4), (0, 32)),
            lane_bases=((16, 0), (32, 0), (64, 0), (128, 0), (256, 0), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 64],
        )
    else:
        # BLOCK_N == 64: shared by bh64 and bh16bn64 (both bf16 KV).
        # K is [512, 64]bf16, KPE is [64, 64]bf16.
        blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16), (0, 32)),
            lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 64],
        )
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 1], [0, 2], [0, 8], [0, 4], [0, 16], [0, 32]],
            cga_layout=[],
            shape=[512, 64]
        )
        blocked_kpe: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 32)),
            lane_bases=((8, 0), (16, 0), (32, 0), (0, 4), (0, 8), (0, 16)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[64, 64],
        )
        shared_kpe: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[512, 16]],
            offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 1], [0, 2], [0, 32]],
            cga_layout=[],
            shape=[64, 64]
        )
        blocked_page: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0,),),
            lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
            warp_bases=((0,), (0,)),
            block_bases=[],
            shape=[64],
        )
        blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16)),
            lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[512, 32],
        )

    # linear_v: each regime has unique warp/reg mapping (bh64 has degenerate warp_bases,
    # bh16bn128 has an extra K reg base for the 128-wide K, bh16bn64 has the bh16 warp layout at 64-wide K).
    if REGIME == 'bh64':
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((0, 0), (0, 0)),
            block_bases=[],
            shape=[512, 64],
        )
    elif REGIME == 'bh16bn128':
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (0, 64), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((16, 0), (32, 0)),
            block_bases=[],
            shape=[512, 128],
        )
    else:
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((16, 0), (32, 0)),
            block_bases=[],
            shape=[512, 64],
        )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)
    dtype = Q_nope.type.element_ty
    kvtype = Kv_c_cache.type.element_ty
    ######### layout setting end #########

    buf_q_nope = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_CKV], layout=shared_q_nope)
    if HAS_PE:
        buf_q_pe = gl.allocate_shared_memory(dtype, shape=[BLOCK_H, HEAD_DIM_KPE], layout=shared_q_pe)

    # load q_nope
    offs_d_ckv = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, blocked_q_nope))
    cur_head = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope))
    offs_q_nope = cur_batch * stride_q_nope_bs + q_pos * stride_q_nope_s + cur_head[:, None] * stride_q_nope_h + offs_d_ckv[None, :]
    ### For nhead < BLOCK_H, mask OOB heads to zero on Q load and skip OOB O stores; wasted MFMA lanes are free (memory-bound).
    gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_nope, Q_nope, offs_q_nope, mask = (cur_head < NHEAD)[:, None] if NHEAD % BLOCK_H != 0 else None)
    gl.amd.cdna4.async_copy.commit_group()

    # load q_pe
    if HAS_PE:
        offs_d_kpe = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(0, blocked_q_pe))
        cur_head_qpe = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_pe))
        offs_q_pe = cur_batch * stride_q_pe_bs + q_pos * stride_q_pe_s + cur_head_qpe[:, None] * stride_q_pe_h + offs_d_kpe[None, :]
        gl.amd.cdna4.async_copy.buffer_load_to_shared(buf_q_pe, Q_pe, offs_q_pe, mask = (cur_head_qpe < NHEAD)[:, None] if NHEAD % BLOCK_H != 0 else None)
        gl.amd.cdna4.async_copy.commit_group()

    e_max = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)) - float("inf")
    e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_H, HEAD_DIM_CKV], dtype=gl.float32, layout=mfma_layout)

    # Fold KV dequant scale into the QK temperature. For fp8 KV the real
    # logits are (Q @ K_fp8^T) * kv_scale * sm_scale; softmax is shift- but
    # not scale-invariant, so kv_scale must affect qk (not just acc).
    # For bf16 KV the wrapper passes kv_scale=1.0, so this is a no-op.
    qk_scale = sm_scale * kv_scale

    ### bufs of page_number
    shared_page: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    bufs_page = gl.allocate_shared_memory(gl.int32, shape=[2, BLOCK_N], layout=shared_page)
    gl.static_assert(PAGE_SIZE == 1)

    offs_page_raw = gl.arange(0, BLOCK_N, layout=blocked_page)

    ################ prologue
    #### global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(0), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    start_n += BLOCK_N
    #### global load page number
    offs_n_page = start_n + offs_page_raw
    offs_page = batch_page_start + offs_n_page // PAGE_SIZE
    gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(1), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
    gl.amd.cdna4.async_copy.commit_group()

    #### local load Q
    gl.amd.cdna4.async_copy.wait_group(2)
    q_nope = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_nope, mfma_layout_a)
    if HAS_PE:
        q_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_q_pe, mfma_layout_a)

    #################### move here to work around allocate_shared_memory bug
    bufs_kv = gl.allocate_shared_memory(kvtype, shape=[2, HEAD_DIM_CKV, BLOCK_N], layout=shared_kv)
    if HAS_PE:
        bufs_kpe = gl.allocate_shared_memory(kvtype, shape=[2, HEAD_DIM_KPE, BLOCK_N], layout=shared_kpe)

    #### global load K
    # local load page number
    gl.amd.cdna4.async_copy.wait_group(1)
    if HAS_PE:
        kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(0), gl.SliceLayout(0, blocked_kpe))
        # simplify for page_size 1
        kv_loc_pe = kv_page_number_pe

    # local load page number for slice 0
    bufs_page_0 = bufs_page.index(0).slice(0, BLOCK_N // 2, 0)
    kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
    kv_loc0 = kv_page_number_0

    # global load K_nope slice 0
    offs_n_nope0 = split_kv_start + gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, blocked_kv_slice))
    offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
    offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    bufs_kv0 = bufs_kv.index(0).slice(0, BLOCK_N // 2, 1)
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0)
    gl.amd.cdna4.async_copy.commit_group()

    # global load K_pe
    if HAS_PE:
        offs_n_pe0 = split_kv_start + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
        offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
        offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(0), K_pe_cache, offs_k_pe, mask=offs_n_pe0[None, :] < split_kv_end)
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(0), K_pe_cache + offs_k_pe)
        gl.amd.cdna4.async_copy.commit_group()

    # local load page number for slice 1
    bufs_page_1 = bufs_page.index(0).slice(BLOCK_N // 2, BLOCK_N // 2, 0)
    kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
    kv_loc1 = kv_page_number_1

    # global load K_nope slice 1
    offs_n_nope1 = offs_n_nope0 + BLOCK_N // 2
    bufs_kv1 = bufs_kv.index(0).slice(BLOCK_N // 2, BLOCK_N // 2, 1)
    offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n_nope1[None, :] < split_kv_end)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1)
    gl.amd.cdna4.async_copy.commit_group()

    if REGIME == 'bh64':
        # bh64 guarantees >= 3 iters/split; this constant-folds the
        # `if num_iter >= 2` epilogue-1 guard below so its codegen is unchanged.
        gl.assume(num_iter >= 3)
    buf_idx = 0
    ################ loop
    for i in range(num_iter - 2):
        async_idx = (buf_idx + 1) % 2

        # Only the page numbers are needed here. Outstanding groups at this point,
        # oldest first, are the four (three without PE) committed by the previous
        # trip: page -> bufs_page[async_idx], then kv0, kpe and kv1 -> the buffers
        # this trip reads at the local loads further down. Draining all of them
        # (wait_group(0)) serialized the page prefetch chain behind the whole KV
        # stream and left one KV block in flight; retiring just the oldest group
        # keeps the previous trip's K/K_pe copies in flight across the issue block
        # below, so two KV blocks are resident at no extra LDS. The prologue
        # leaves the same set outstanding, so trip 0 behaves like the rest.
        gl.amd.cdna4.async_copy.wait_group(3 if HAS_PE else 2)
        #### global load page number
        offs_n_page = start_n + BLOCK_N + offs_page_raw
        offs_page = batch_page_start + offs_n_page // PAGE_SIZE
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_page.index(buf_idx), Req_to_tokens, offs_page, offs_n_page < split_kv_end)
        gl.amd.cdna4.async_copy.commit_group()

        #### global load K
        bufs_kv0 = bufs_kv.index(async_idx).slice(0, BLOCK_N // 2, 1)
        bufs_kv1 = bufs_kv.index(async_idx).slice(BLOCK_N // 2, BLOCK_N // 2, 1)
        # local load page number for slice 0
        bufs_page_0 = bufs_page.index(async_idx).slice(0, BLOCK_N // 2, 0)
        kv_page_number_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_0, gl.SliceLayout(0, blocked_kv_slice))
        kv_loc0 = kv_page_number_0
        # global load K_nope slice 0
        offs_n_nope0 = start_n + gl.arange(0, BLOCK_N // 2, layout=gl.SliceLayout(0, blocked_kv_slice))
        offs_d_ckv_10 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv_slice))
        offs_k_c0 = kv_loc0[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv0, Kv_c_cache, offs_k_c0, mask=offs_n_nope0[None, :] < split_kv_end)
        else:
            # No mask needed on global_load path in the loop body: all
            # iterations are guaranteed in-bounds by num_iter arithmetic.
            # Only the epilogue uses mask + other=0 for the last
            # potentially-partial block.
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv0, Kv_c_cache + offs_k_c0)
        gl.amd.cdna4.async_copy.commit_group()

        # local load page_number_pe
        if HAS_PE:
            kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe))
            kv_loc_pe = kv_page_number_pe
            # global load K_pe
            offs_n_pe = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
            offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
            offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
            if WITHIN_2GB:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache, offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end)
            else:
                # No mask needed: loop iterations are in-bounds (see KV slice 0 comment).
                gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe)
            gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part0)
        # Second half of the split wait. The previous trip's K / K_pe copies into
        # bufs_kv[buf_idx] / bufs_kpe[buf_idx] must have landed before these local
        # loads. Outstanding here, oldest first: prev kv0, prev kpe, prev kv1,
        # this page, this kv0, this kpe -- so leaving three pending (two without
        # PE) retires exactly the three needed and keeps this trip's prefetches in
        # flight through the MFMA and softmax block. Same granularity as
        # epilogue 1's wait below.
        gl.amd.cdna4.async_copy.wait_group(3 if HAS_PE else 2)
        k_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)
        if HAS_PE:
            k_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kpe.index(buf_idx), mfma_layout_b)
            qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)

        # local load page number for slice 1
        bufs_page_1 = bufs_page.index(async_idx).slice(BLOCK_N // 2, BLOCK_N // 2, 0)
        kv_page_number_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page_1, gl.SliceLayout(0, blocked_kv_slice))
        kv_loc1 = kv_page_number_1
        # global load K_nope slice 1
        offs_n1 = offs_n_nope0 + BLOCK_N // 2
        offs_k_c1 = kv_loc1[None, :] * stride_kv_c_bs + offs_d_ckv_10[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv1, Kv_c_cache, offs_k_c1, mask=offs_n1[None, :] < split_kv_end)
        else:
            # No mask needed: loop iterations are in-bounds (see KV slice 0 comment).
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv1, Kv_c_cache + offs_k_c1)
        gl.amd.cdna4.async_copy.commit_group()

        #### dot, softmax, dot (part1)
        qk *= qk_scale
        offs_n_qk = split_kv_start + i * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        LOG2E: gl.constexpr = 1.4426950408889634
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
        if QLEN > 1:
            # MTP: a leading/whole fully-masked split keeps e_max=n_e_max=-inf,
            # making re_scale/p NaN. Force them to 0 so the split cleanly yields
            # e_sum=0 -> lse=-inf, which stage-2 drops.
            re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
            p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_kv.index(buf_idx), linear_v)
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    LOG2E: gl.constexpr = 1.4426950408889634

    ################ epilogue 1
    # Skip when num_iter < 2 (possible for bh16bn64 / bh16bn128 in either mode).
    # bh64 has gl.assume(num_iter >= 3) above so the compiler folds this branch
    # out there; for the bh16 regimes it stays a runtime branch.
    if num_iter >= 2:
        async_idx = (buf_idx + 1) % 2

        #### global load K
        # local load page number
        gl.amd.cdna4.async_copy.wait_group(3 if HAS_PE else 2)
        kv_page_number = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kv))
        kv_loc = kv_page_number
        if HAS_PE:
            kv_page_number_pe = gl.amd.cdna4.async_copy.load_shared_relaxed(bufs_page.index(async_idx), gl.SliceLayout(0, blocked_kpe))
            kv_loc_pe = kv_page_number_pe
        # global load K_nope
        offs_n_nope = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kv))
        offs_d_ckv_1 = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(1, blocked_kv))
        offs_k_c = kv_loc[None, :] * stride_kv_c_bs + offs_d_ckv_1[:, None]
        if WITHIN_2GB:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache, offs_k_c, mask=offs_n_nope[None, :] < split_kv_end)
        else:
            # No mask needed: out-of-range positions are discarded by the qk score mask
            gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kv.index(async_idx), Kv_c_cache + offs_k_c)
        gl.amd.cdna4.async_copy.commit_group()
        # global load K_pe
        if HAS_PE:
            offs_n_pe = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kpe))
            offs_d_kpe_1 = gl.arange(0, HEAD_DIM_KPE, layout=gl.SliceLayout(1, blocked_kpe))
            offs_k_pe = kv_loc_pe[None, :] * stride_k_pe_bs + offs_d_kpe_1[:, None] + KV_PE_OFFSET
            if WITHIN_2GB:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache, offs_k_pe, mask=offs_n_pe[None, :] < split_kv_end)
            else:
                gl.amd.cdna4.async_copy.global_load_to_shared(bufs_kpe.index(async_idx), K_pe_cache + offs_k_pe)
            gl.amd.cdna4.async_copy.commit_group()

        # dot, softmax, dot
        gl.amd.cdna4.async_copy.wait_group(2 if HAS_PE else 1)
        k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
        zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)

        if HAS_PE:
            k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
            qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
        qk *= qk_scale
        offs_n_qk = split_kv_start + (num_iter - 2) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
        qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
        if QLEN > 1:
            re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
            p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        e_max = n_e_max
        p = p.to(dtype)
        p = gl.convert_layout(p, mfma_layout_a)
        acc *= re_scale[:, None]
        v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
        v_c = v_c.to(dtype)
        v_c = gl.permute(v_c, [1, 0])
        v_c = gl.convert_layout(v_c, mfma_layout_b)
        acc = gl.amd.cdna4.mfma(p, v_c, acc)

        start_n += BLOCK_N
        buf_idx = (buf_idx + 1) % 2

    ################ epilogue 2
    #### dot, softmax, dot
    gl.amd.cdna4.async_copy.wait_group(0)
    k_c = bufs_kv.index(buf_idx).load(layout=mfma_layout_b)
    zeros = gl.zeros([BLOCK_H, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q_nope, k_c.to(dtype), zeros)

    if HAS_PE:
        k_pe = bufs_kpe.index(buf_idx).load(layout=mfma_layout_b)
        qk = gl.amd.cdna4.mfma(q_pe, k_pe.to(dtype), qk)
    qk *= qk_scale
    offs_n_qk = split_kv_start + (num_iter - 1) * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    qk = gl.where(offs_n_qk[None, :] < score_end, qk, float("-inf"))
    n_e_max = gl.maximum(gl.max(qk, 1), e_max)
    re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
    p = gl.exp2((qk - n_e_max[:, None]) * LOG2E)
    if QLEN > 1:
        re_scale = gl.where(e_max == float("-inf"), 0.0, re_scale)
        p = gl.where(n_e_max[:, None] == float("-inf"), 0.0, p)
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    e_max = n_e_max
    p = p.to(dtype)
    p = gl.convert_layout(p, mfma_layout_a)
    acc *= re_scale[:, None]
    v_c = bufs_kv.index(buf_idx).load(layout=linear_v)
    v_c = v_c.to(dtype)
    v_c = gl.permute(v_c, [1, 0])
    v_c = gl.convert_layout(v_c, mfma_layout_b)
    acc = gl.amd.cdna4.mfma(p, v_c, acc)

    cur_head_o = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout))
    offs_d_ckv_o = gl.arange(0, HEAD_DIM_CKV, layout=gl.SliceLayout(0, mfma_layout))
    offs_o = cur_batch * stride_o_b + q_pos * stride_o_s + cur_head_o[:, None] * stride_o_h + split_kv_id * stride_o_split + offs_d_ckv_o[None, :]

    if HAS_ATTN_SINK:
        # Fold the optional per-head sink into the softmax denom (no V contribution).
        # e_max/e_sum are natural-log units (the *LOG2E is inside exp2), so is sink.
        if NHEAD % BLOCK_H != 0:
            sink = gl.load(Attn_sink + cur_head_o, mask=cur_head_o < NHEAD, other=float("-inf")).to(gl.float32)
        else:
            sink = gl.load(Attn_sink + cur_head_o).to(gl.float32)
        n_e_max = gl.maximum(e_max, sink)
        re_scale = gl.exp2((e_max - n_e_max) * LOG2E)
        acc *= re_scale[:, None]
        e_sum = e_sum * re_scale + gl.exp2((sink - n_e_max) * LOG2E)
        e_max = n_e_max

    acc *= kv_scale
    rcp = 1.0 / e_sum
    stored_value = (acc * rcp[:, None]).to(dtype)
    if NHEAD % BLOCK_H != 0:
        gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o, mask=(cur_head_o < NHEAD)[:, None])
    else:
        gl.amd.cdna4.buffer_store(stored_value, ptr=O, offsets=offs_o)

    ### store lse
    blocked_lse: gl.constexpr = gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[64], warps_per_cta=[4], order=[0])
    cur_head_lse = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=blocked_lse)
    if RETURN_LSE and NUM_KV_SPLITS == 1:
        # split==1: single split is the whole sequence, so its lse is the final lse.
        offs_final_lse = cur_batch * stride_final_lse_b + q_pos * stride_final_lse_s + cur_head_lse * stride_final_lse_h
        lse = e_max + gl.log(e_sum)
        lse = gl.convert_layout(lse, blocked_lse)
        if NHEAD % BLOCK_H != 0:
            gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse, mask=(cur_head_lse < NHEAD))
        else:
            gl.amd.cdna4.buffer_store(lse, ptr=Final_lse, offsets=offs_final_lse)
    elif NUM_KV_SPLITS > 1:
        # per-split lse for stage-2 reduce.
        offs_mid_lse = cur_batch * stride_mid_lse_b + q_pos * stride_mid_lse_s + cur_head_lse * stride_mid_lse_h + split_kv_id * stride_mid_lse_split
        lse = e_max + gl.log(e_sum)
        lse = gl.convert_layout(lse, blocked_lse)
        if NHEAD % BLOCK_H != 0:
            gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse, mask=(cur_head_lse < NHEAD))
        else:
            gl.amd.cdna4.buffer_store(lse, ptr=Mid_lse, offsets=offs_mid_lse)
# fmt: on


# fmt: off
@triton.jit
def _mla_softmax_reducev_kernel(
    Logits,
    Mid_lse,
    O,
    Final_lse,
    B_seq_len,  # same seq_info as the decode kernel to derive empty kv splits
    stride_l_b,
    stride_l_qs,  # MTP: q_pos (qlen) stride; 0 when QLEN==1
    stride_l_h,
    stride_l_s,
    stride_ml_b,
    stride_ml_qs,  # MTP: q_pos stride; 0 when QLEN==1
    stride_ml_h,
    stride_ml_s,
    stride_o_b,
    stride_o_qs,  # MTP: q_pos stride; 0 when QLEN==1
    stride_o_h,
    stride_fl_b,
    stride_fl_qs,  # MTP: q_pos stride; 0 when QLEN==1
    stride_fl_h,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HAS_FINAL_LSE: tl.constexpr,
    USE_2D_VIEW: tl.constexpr,
    BLOCK_S: tl.constexpr,
    D_SPLIT: tl.constexpr,
):
    """Merge the per-split (acc, lse) partials, BLOCK_S splits at a time.

    BLOCK_S=1, D_SPLIT=1 reproduces the original scalar walk exactly. Larger
    values only re-associate the same online-softmax merge: BLOCK_S splits are
    loaded as one [BLOCK_S, BLOCK_D] tile so the serial rescale chain is BLOCK_S
    times shorter and the memory system sees BLOCK_S independent loads, and the
    head dim is tiled across D_SPLIT programs on grid axis 2. Each head-dim slice
    re-derives the same scalar lse statistics from the same Mid_lse vector and
    owns a disjoint slice of `o`, so the slices never communicate.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    # Grid axis 2 packs (q_pos, head-dim slice).
    q_pos = tl.program_id(2) // D_SPLIT
    cur_d = tl.program_id(2) % D_SPLIT

    # Recompute this batch's seq len exactly as the decode kernel did, so we can
    # rederive which splits are empty. Stage-1 early-returns on empty splits
    # (num_iter == 0) and writes nothing, so their logits_buf / mid_lse slots hold
    # raw, uninitialised memory - they cannot be loaded or reduced.
    if USE_2D_VIEW:
        cur_batch_seq_len = tl.load(B_seq_len + cur_batch)
    else:
        batch_page_start = tl.load(B_seq_len + cur_batch)
        cur_batch_seq_len = tl.load(B_seq_len + cur_batch + 1) - batch_page_start
    kv_len_per_split = cur_batch_seq_len // NUM_KV_SPLITS

    BLOCK_D: tl.constexpr = HEAD_DIM_CKV // D_SPLIT
    offs_d_ckv = cur_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_l = cur_batch * stride_l_b + q_pos * stride_l_qs + cur_head * stride_l_h + offs_d_ckv
    offs_ml = cur_batch * stride_ml_b + q_pos * stride_ml_qs + cur_head * stride_ml_h
    offs_s = tl.arange(0, BLOCK_S)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    LOOP_START = NUM_KV_SPLITS - 1 if kv_len_per_split == 0 else 0
    # Same empty-split semantics as the scalar walk: start on a block boundary at
    # or below LOOP_START and mask the uninitialised slots off inside the block.
    BLK_START = (LOOP_START // BLOCK_S) * BLOCK_S
    for split_kv_id in range(BLK_START, NUM_KV_SPLITS, BLOCK_S):
        s = split_kv_id + offs_s
        s_ok = (s >= LOOP_START) & (s < NUM_KV_SPLITS)
        # [BLOCK_S] per-split lse; a masked slot carries no weight.
        logits_1 = tl.load(Mid_lse + offs_ml + s * stride_ml_s, mask=s_ok, other=-float("inf"))
        # [BLOCK_S, BLOCK_D] per-split partial numerators.
        logits = tl.load(Logits + offs_l[None, :] + s[:, None] * stride_l_s, mask=s_ok[:, None], other=0.0)

        n_e_max = tl.maximum(tl.max(logits_1, 0), e_max)
        old_scale = tl.where(e_max == -float("inf"), 0.0, tl.exp(e_max - n_e_max))
        exp_logic = tl.where(logits_1 == -float("inf"), 0.0, tl.exp(logits_1 - n_e_max))
        # MTP: a fully causal-masked split stores NaN logits with lse=-inf; drop
        # the whole row so NaN*0 doesn't poison acc (no-op for plain decode).
        contrib = tl.where(logits_1[:, None] == -float("inf"), 0.0, exp_logic[:, None] * logits)

        acc = acc * old_scale + tl.sum(contrib, 0)
        e_sum = e_sum * old_scale + tl.sum(exp_logic, 0)
        e_max = n_e_max

    out = acc / e_sum if e_sum > 0.0 else tl.zeros([BLOCK_D], dtype=tl.float32)
    tl.store(
        O + cur_batch * stride_o_b + q_pos * stride_o_qs + cur_head * stride_o_h + offs_d_ckv,
        out,
    )
    # Every head-dim slice derived the same lse; slice 0 owns the one store.
    if HAS_FINAL_LSE and cur_d == 0:
        tl.store(
            Final_lse + cur_batch * stride_fl_b + q_pos * stride_fl_qs + cur_head * stride_fl_h,
            e_max + tl.log(e_sum),
        )
# fmt: on


def mla_gluon(
    q_nope,  # [batch, nhead, kv_lora_rank] or MTP [batch, qlen, nhead, kv_lora_rank]
    q_pe,  # [batch, nhead, qk_rope_head_dim] or MTP [batch, qlen, nhead, qk_rope_head_dim]
    # Shared: kv_c=[N, kv_lora_rank+qk_rope_head_dim], k_pe=None,              kv_pe_offset=kv_lora_rank
    # Split:  kv_c=[N, kv_lora_rank],                k_pe=[N,qk_rope_head_dim], kv_pe_offset=0
    kv_c,
    # final output [batch, nhead, kv_lora_rank] or MTP [batch, qlen, nhead, kv_lora_rank].
    o,
    page_table,  # 2D: block_table [batch, max_seqlen] | 1D: kv_indices [total_kv]
    seq_info,  # 2D: cache_seqlens [batch]           | 1D: kv_indptr [batch+1]
    sm_scale,
    k_pe=None,
    kv_pe_offset=512,
    use_2d_view=True,
    kv_scale=1.0,
    min_kv_seq_len=1,
    return_lse=False,
    has_pe=True,
    attn_sink=None,  # [nhead] fp32 per-head sink bias, None means no sink
):
    """Unified Gluon MLA entry (gfx950 / CDNA4) — decode and DeepSeek V4 sparse prefill.

    `mla_gluon` supports the full decode (stage-1 + stage-2 reduce, or the stage-1-only
    fast path when NUM_KV_SPLITS==1) and writes the final attention into the
    caller's `o`.

    MTP (Multi-Token Prediction): pass 4-D q_nope/q_pe/o shaped
    [batch, qlen, nhead, dim]. qlen is a runtime value (one compiled kernel
    serves any qlen). Each query position q_pos attends KV [0, seq_len-qlen+q_pos]
    (causal tail). The plain-decode 3-D path ([batch, nhead, dim], qlen=1) is
    unchanged. Implementation: q_pos is an extra grid axis, so KV is currently
    re-read per q_pos, but those re-reads are mostly served from L2/MALL cache so
    the path is efficient at ctx<=16384.

    return_lse=False (default): returns (o, None).

    return_lse=True: additionally returns the merged log-sum-exp, a separate
        fp32 tensor [batch, qlen, nhead]

    DSv4 Sparse prefill packs NoPE and RoPE in to one contiguous row (448+64).
    To run DSv4 prefill, it requires has_pe=False, prepares valid Q / K in q_nope / kv_c,
    and attn_sink, q_pe / k_pe are unused placeholders.
    """
    if k_pe is None:
        k_pe = kv_c

    # Accept plain decode (3-D, qlen=1) or MTP (4-D, [batch, qlen, nhead, dim]).
    # DSv4 sparse prefill (has_pe=False) always uses the 3-D path.
    assert q_nope.dim() in (3, 4), f"q_nope must be 3-D or 4-D, got {q_nope.dim()}-D"
    IS_MTP = q_nope.dim() == 4
    if IS_MTP:
        batch_size, qlen, nhead, head_dim_ckv = q_nope.shape
    else:
        batch_size, nhead, head_dim_ckv = q_nope.shape
        qlen = 1
    # Decode carries a real q_pe [.., 64];
    # DSV4 prefill (HAS_PE=False) has no PE, so q_pe may be None and RoPE head_dim is the fixed 64.
    head_dim_kpe = q_pe.shape[-1] if has_pe else 64
    if not has_pe:
        q_pe = q_nope
        k_pe = kv_c
        kv_pe_offset = 0
        use_2d_view = False

    # LOCAL (from the seungrokj fp8-KV gist, not upstream): fp8 KV-cache paths can
    # hand us fp8 queries, but the kernel MFMA expects bf16 q_nope/q_pe (only
    # kv_c/k_pe stay fp8). Dequantize the queries to bf16.
    if q_nope.dtype == torch.float8_e4m3fn:
        q_nope = q_nope.to(torch.bfloat16)
    if q_pe is not None and q_pe.dtype == torch.float8_e4m3fn:
        q_pe = q_pe.to(torch.bfloat16)

    assert (
        arch_info.get_arch() == "gfx950"
    ), f"mla_gluon requires gfx950 (CDNA4), got {arch_info.get_arch()}"
    assert (
        head_dim_ckv == 512
    ), f"mla_gluon requires head_dim_ckv=512, got {head_dim_ckv}"
    assert head_dim_kpe == 64, f"mla_gluon requires head_dim_kpe=64, got {head_dim_kpe}"

    # attn sink: decode never sets one; prefill may pass a per-head [H] bias.
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=o.device, dtype=torch.float32)  # dummy ptr

    # Pick regime by (nhead, kv dtype). MTP (qlen>1) uses the grid-axis path:
    # q_pos is grid axis 2, so each query position is a separate program.
    if nhead in (64, 128):
        REGIME = "bh64"
    elif 1 <= nhead <= 96:
        # bh16 path: heads are tiled into cdiv(nhead, 16) blocks of BLOCK_H=16 on
        # grid axis 2 (alongside q_pos). nhead <= 16 is a single block (unchanged);
        # nhead > 16 (e.g. 96) adds head-block programs. A partial last block
        # (nhead % 16 != 0) masks OOB heads on Q load / O store.
        if kv_c.dtype == torch.bfloat16:
            REGIME = "bh16bn64"
        elif kv_c.dtype == torch.float8_e4m3fn:  # gfx950 fp8 (e4m3fn, not e4m3fnuz)
            REGIME = "bh16bn128"
        else:
            raise AssertionError(
                f"mla_gluon[bh16*] requires kv_c.dtype in (bfloat16, float8_e4m3fn), got {kv_c.dtype}"
            )
    else:
        raise AssertionError(
            f"mla_gluon requires nhead <= 96 [bh16bn128/bh16bn64] or nhead in (64,128) [bh64], got {nhead}"
        )

    PAGE_SIZE = 1

    if REGIME == "bh64":
        BLOCK_H, BLOCK_N = 64, 64
        NUM_XCDS = get_num_xcds()
        # Auto-pick NUM_KV_SPLITS so the launch fills ~256 workgroups (one wave on
        # MI350). For the supported (batch, nhead) matrix the result is in {1, 2, 4}.
        base_grid = (
            NUM_XCDS * triton.cdiv(nhead, BLOCK_H) * qlen * (batch_size // NUM_XCDS)
        )
        NUM_KV_SPLITS = max(1, triton.next_power_of_2(triton.cdiv(256, base_grid)))

        assert (
            batch_size % 64 == 0
        ), f"mla_gluon[bh64] requires batch_size divisible by 64, got {batch_size}"
        # gl.assume(num_iter > 3) inside the kernel requires every split to have
        # > 3*BLOCK_N tokens. Smallest split (last) for batch length s is
        # s - (k-1)*ceil(s/k); a sufficient bound is min_kv_seq_len > k*(3*BLOCK_N + k).
        min_kv_seq_len_required = NUM_KV_SPLITS * (3 * BLOCK_N + NUM_KV_SPLITS)
        assert (
            min_kv_seq_len > min_kv_seq_len_required
        ), f"mla_gluon[bh64] requires min_kv_seq_len > {min_kv_seq_len_required} (NUM_KV_SPLITS={NUM_KV_SPLITS}), got {min_kv_seq_len}"
        assert (
            q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
        ), f"q_nope/q_pe must be bf16, got {q_nope.dtype}/{q_pe.dtype}"
        assert (
            kv_c.dtype == torch.bfloat16 and k_pe.dtype == torch.bfloat16
        ), f"kv_c/k_pe must be bf16, got {kv_c.dtype}/{k_pe.dtype}"
    else:  # bh16bn128 (fp8 KV) or bh16bn64 (bf16 KV)
        BLOCK_H = 16
        BLOCK_N = 128 if REGIME == "bh16bn128" else 64
        kv_dtype = torch.float8_e4m3fn if REGIME == "bh16bn128" else torch.bfloat16
        NUM_XCDS = 1  # unused by 2-D split grid mapping
        # nhead > 16 tiles heads into cdiv(nhead, BLOCK_H) blocks on grid axis 2.
        # Total stage-1 WGs = batch * NUM_KV_SPLITS * (NUM_M_BLOCKS * qlen), so the
        # ~256-WG split budget divides by NUM_M_BLOCKS as well (head blocks already
        # fill the machine, mirroring how bh64 folds head blocks into the grid).
        NUM_M_BLOCKS = triton.cdiv(nhead, BLOCK_H)
        # 2-D grid (batch, split). Both bh16 regimes support num_iter in {1, 2, ...}
        # (no gl.assume(num_iter >= 3) in the kernel); the only correctness need is
        # that every split is non-empty (floor split size = min_kv_seq_len //
        # NUM_KV_SPLITS >= 1). Each clamp below keeps NUM_KV_SPLITS <= min_kv_seq_len,
        # Split to fill one wave (total WGs = batch * NUM_KV_SPLITS *
        # NUM_M_BLOCKS * qlen, ~one per CU), while keeping at least one BLOCK_N
        # block of KV per split. Per-request context is the page table's
        # allocated width, host-known and shape-stable, so the constexpr is
        # identical at HIP-graph capture time and on every replay.
        #
        # LOCAL: upstream #4507 computes this inline in the bh16bn64 branch. It is
        # hoisted here because we reuse it for bh16bn128 as well (see below).
        if use_2d_view:
            pt_tokens_per_req = int(page_table.shape[-1])
        else:
            pt_tokens_per_req = max(1, int(page_table.numel()) // max(1, batch_size))
        pt_num_kv_splits = max(
            1,
            min(
                get_num_sms() // max(1, batch_size * qlen * NUM_M_BLOCKS),
                triton.cdiv(pt_tokens_per_req, BLOCK_N),
            ),
        )
        if REGIME == "bh16bn128":
            # LOCAL (from the seungrokj fp8-KV gist, not upstream): upstream still
            # asserts batch_size == 1 here, which is what blocks fp8 KV under
            # DSpark verify (batch = the number of verify tokens). The gist lifted
            # the restriction; that image is the one validated at 8x380K requests.
            assert (
                1 <= batch_size <= 128
            ), f"mla_gluon[bh16bn128] requires 1 <= batch_size <= 128, got {batch_size}"
            # LOCAL: #4507 fixes bh16bn64 only, but bh16bn128 sizes its splits from
            # the same unset min_kv_seq_len hint (default 1), so fp8 KV silently
            # runs at NUM_KV_SPLITS=1 too -- which also makes #4509's split-major
            # grid and blocked stage-2 reduce inert on the fp8 arm. Extend the
            # page-table sizing here. AITER_MLA_B128_PT_SPLITS=0 restores the
            # upstream formula for A/B against the validated fp8 behaviour.
            if _B128_PT_SPLITS:
                NUM_KV_SPLITS = pt_num_kv_splits
            else:
                NUM_KV_SPLITS = max(
                    1, min(256 // (batch_size * qlen * NUM_M_BLOCKS), min_kv_seq_len)
                )
        else:  # bh16bn64
            NUM_KV_SPLITS = pt_num_kv_splits
        assert (
            q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
        ), f"q_nope/q_pe must be bf16, got {q_nope.dtype}/{q_pe.dtype}"
        assert (
            kv_c.dtype == kv_dtype and k_pe.dtype == kv_dtype
        ), f"kv_c/k_pe must be {kv_dtype}, got {kv_c.dtype}/{k_pe.dtype}"

    # buffer_load uses scalar base + 32-bit offsets, limiting addressable range.
    # For KV caches > 2 GB the kernel falls back to global_load (64-bit pointers).
    max_kv_bytes = kv_c.shape[0] * kv_c.stride(0) * kv_c.element_size()
    within_2gb = max_kv_bytes <= 0x80000000  # 2 GB

    # Normalized Q strides: (batch, q_pos, head). For plain decode (3-D) the
    # q_pos stride is 0 and q_pos is always 0, so the kernel address math is
    # identical to before.
    if IS_MTP:
        stride_q_nope_bs, stride_q_nope_s, stride_q_nope_h = q_nope.stride()[:3]
        stride_q_pe_bs, stride_q_pe_s, stride_q_pe_h = q_pe.stride()[:3]
        stride_o_b_final, stride_o_s_final, stride_o_h_final = o.stride()[:3]
    else:
        stride_q_nope_bs, stride_q_nope_s, stride_q_nope_h = (
            q_nope.stride(0),
            0,
            q_nope.stride(1),
        )
        stride_q_pe_bs, stride_q_pe_s, stride_q_pe_h = q_pe.stride(0), 0, q_pe.stride(1)
        stride_o_b_final, stride_o_s_final, stride_o_h_final = (
            o.stride(0),
            0,
            o.stride(1),
        )

    if NUM_KV_SPLITS == 1:
        # Fast path: stage-1 writes the final attention (and lse) directly to o.
        # View o with an explicit (q_pos, split) layout so stage-1 strides are uniform.
        if IS_MTP:
            logits_buf = o.view(batch_size, qlen, nhead, NUM_KV_SPLITS, head_dim_ckv)
        else:
            logits_buf = o.view(batch_size, 1, nhead, NUM_KV_SPLITS, head_dim_ckv)
        mid_lse = None
        stride_mid_lse_b = stride_mid_lse_s = stride_mid_lse_h = (
            stride_mid_lse_split
        ) = 0
    else:
        # stage-1 -> per-split (acc, lse); stage-2 reduces into o.
        logits_buf = torch.empty(
            (batch_size, qlen, nhead, NUM_KV_SPLITS, head_dim_ckv),
            dtype=o.dtype,
            device=o.device,
        )
        mid_lse = torch.empty(
            (batch_size, qlen, nhead, NUM_KV_SPLITS),
            dtype=torch.float32,
            device=o.device,
        )
        stride_mid_lse_b, stride_mid_lse_s, stride_mid_lse_h, stride_mid_lse_split = (
            mid_lse.stride()
        )

    # logits_buf is [batch, qlen, nhead, split, dim] in both paths; reuse its
    # strides for stage-1's O write (fast path aliases o, so this stays correct).
    stride_o_b, stride_o_s, stride_o_h, stride_o_split, _ = logits_buf.stride()

    if return_lse:
        final_lse = torch.empty(
            (batch_size, qlen, nhead), dtype=torch.float32, device=q_nope.device
        )
        stride_final_lse_b, stride_final_lse_s, stride_final_lse_h = final_lse.stride()
    else:
        final_lse = None
        stride_final_lse_b, stride_final_lse_s, stride_final_lse_h = 0, 0, 0

    split_major = False
    if REGIME == "bh64":
        grid = (
            NUM_XCDS,
            triton.cdiv(nhead, BLOCK_H) * qlen,
            (batch_size // NUM_XCDS) * NUM_KV_SPLITS,
        )
    else:
        # Grid axis 2 carries (head_block, q_pos): cdiv(nhead, BLOCK_H) head blocks
        # times qlen query positions. For nhead <= 16 this is just qlen (one head
        # block), i.e. the original grid-axis MTP mapping.
        #
        # Axes 0 and 1 are swapped when the split count is a multiple of the XCD
        # count. Triton linearizes axis 0 fastest, so grid=(split, batch, ...)
        # gives workgroup id `split + NUM_KV_SPLITS * batch + ...`, which the
        # hardware round-robins over the XCDs. Only when NUM_KV_SPLITS is a
        # multiple of NUM_XCD does that id stay congruent mod NUM_XCD for every
        # batch, i.e. only then do all requests' workgroups for the same split
        # land on the same XCD and share its L2. With prefix caching those
        # workgroups read the same KV pages, so they hit in L2 instead of every
        # XCD pulling its own copy from HBM.
        #
        # This only relabels which workgroup computes which (batch, split) item;
        # each program still covers the same KV range, so results are unchanged.
        num_xcds = get_num_xcds()
        split_major = (
            batch_size > 1 and NUM_KV_SPLITS > 1 and NUM_KV_SPLITS % num_xcds == 0
        )
        head_q_axis = triton.cdiv(nhead, BLOCK_H) * qlen
        if split_major:
            grid = (NUM_KV_SPLITS, batch_size, head_q_axis)
        else:
            grid = (batch_size, NUM_KV_SPLITS, head_q_axis)
    stride_page_bs = page_table.stride(0) if use_2d_view else 0

    _mla_gluon[grid](
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        page_table,
        seq_info,
        logits_buf,
        attn_sink,
        sm_scale,
        kv_scale,
        stride_q_nope_bs,
        stride_q_nope_s,
        stride_q_nope_h,
        stride_q_pe_bs,
        stride_q_pe_s,
        stride_q_pe_h,
        kv_c.stride(-2),
        k_pe.stride(-2),
        stride_page_bs,
        stride_o_b,
        stride_o_s,
        stride_o_h,
        stride_o_split,
        mid_lse,
        stride_mid_lse_b,
        stride_mid_lse_s,
        stride_mid_lse_h,
        stride_mid_lse_split,
        final_lse,
        stride_final_lse_b,
        stride_final_lse_s,
        stride_final_lse_h,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=PAGE_SIZE,
        HEAD_DIM_CKV=head_dim_ckv,
        HEAD_DIM_KPE=head_dim_kpe,
        KV_PE_OFFSET=kv_pe_offset,
        USE_2D_VIEW=use_2d_view,
        WITHIN_2GB=within_2gb,
        NUM_XCDS=NUM_XCDS,
        NHEAD=nhead,
        REGIME=REGIME,
        RETURN_LSE=return_lse,
        QLEN=qlen,
        HAS_PE=has_pe,
        HAS_ATTN_SINK=has_attn_sink,
        SPLIT_MAJOR=split_major,
    )

    if NUM_KV_SPLITS == 1:
        # Fast path: stage-1 already wrote o (and lse) directly.
        return o, final_lse

    # Stage-2: reduce per-split (acc, lse) into o (and lse when return_lse).
    #
    # The reduce walks the splits serially, and each step's rescale depends on the
    # previous e_max while moving only head_dim_ckv elements, so a large split
    # count turns it into a deep latency-bound chain. On a (batch, nhead, qlen)
    # grid it is also only batch*nhead*qlen workgroups, which does not fill the
    # device at small batch. Merge BLOCK_S splits per iteration to shorten the
    # chain, and tile the head dim across D_SPLIT programs on grid axis 2 to widen
    # the grid. BLOCK_S=1, D_SPLIT=1 is the original kernel.
    RED_D_SPLIT = 8
    while RED_D_SPLIT > 1 and (
        head_dim_ckv % RED_D_SPLIT or head_dim_ckv // RED_D_SPLIT < 16
    ):
        RED_D_SPLIT //= 2
    RED_BLOCK_S = min(256, triton.next_power_of_2(NUM_KV_SPLITS))
    # grid axis 2 packs (q_pos, head-dim slice). o uses the caller's layout.
    grid_reduce = (batch_size, nhead, qlen * RED_D_SPLIT)
    sl_b, sl_qs, sl_h, sl_split, _ = logits_buf.stride()
    _mla_softmax_reducev_kernel[grid_reduce](
        logits_buf,
        mid_lse,
        o,
        final_lse,
        seq_info,
        sl_b,
        sl_qs,
        sl_h,
        sl_split,
        stride_mid_lse_b,
        stride_mid_lse_s,
        stride_mid_lse_h,
        stride_mid_lse_split,
        stride_o_b_final,
        stride_o_s_final,
        stride_o_h_final,
        stride_final_lse_b,
        stride_final_lse_s,
        stride_final_lse_h,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        HEAD_DIM_CKV=head_dim_ckv,
        HAS_FINAL_LSE=return_lse,
        USE_2D_VIEW=use_2d_view,
        BLOCK_S=RED_BLOCK_S,
        D_SPLIT=RED_D_SPLIT,
        num_warps=4,
    )

    return o, final_lse
