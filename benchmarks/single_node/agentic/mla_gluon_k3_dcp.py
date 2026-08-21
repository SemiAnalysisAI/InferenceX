# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Gluon MLA decode kernel originated from FlashMLA triton kernel(https://github.com/deepseek-ai/FlashMLA/blob/main/benchmark/bench_flash_mla.py).
# Stage-1 split-KV MLA attention using explicit Gluon layouts. Three regimes:
#
#   REGIME='bh64'      - bf16 Q + bf16 KV, BLOCK_H=64, BLOCK_N=64,
#                        nhead in {64, 128}, batch_size in {64, 128, 256},
#                        NUM_KV_SPLITS auto-picked to fill ~256 WGs (in {1,2,4}).
#                        Fast path: when NUM_KV_SPLITS==1, stage-1 writes the
#                        final output directly to O and stage-2 reduce is skipped.
#   REGIME='bh16bn128' - bf16 Q + fp8 KV, BLOCK_H=16, BLOCK_N=128,
#                        nhead <= 96, batch_size >= 1,
#                        (batch, split, head_block*qlen) grid. Full decode
#                        (stage-1 + stage-2 reduce into the final O). A partial
#                        last head block (nhead % BLOCK_H != 0) masks OOB heads.
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
from aiter.ops.triton.utils.device_info import get_num_xcds

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

    # NUM_KV_SPLITS is a launch-time budget only. 
    # the partition is derived here from the runtime per-batch KV length.
    # kv_len_per_split = max(BLOCK_N, floor(seq / NUM_KV_SPLITS)):
    #   - the BLOCK_N floor keeps every split at >= 1 full block, so a short seq
    #     is spread over fewer, whole-block splits instead of many partial ones;
    kv_len_per_split = gl.maximum(BLOCK_N, cur_batch_seq_len // NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    if split_kv_id == NUM_KV_SPLITS - 1:
        split_kv_end = cur_batch_seq_len
    # early return for inactive split
    if split_kv_start >= split_kv_end:
        return
    num_iter = gl.cdiv(split_kv_end - split_kv_start, BLOCK_N)
    start_n = split_kv_start

    # >2GB KV cache (global_load path): widen strides to int64 so kv offsets don't overflow int32.
    if not WITHIN_2GB:
        stride_kv_c_bs = stride_kv_c_bs.to(gl.int64)
        stride_k_pe_bs = stride_k_pe_bs.to(gl.int64)

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

        gl.amd.cdna4.async_copy.wait_group(0)
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
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    q_pos = tl.program_id(2)

    # Recompute this batch's seq len exactly as the decode kernel did, so we can
    # rederive which splits are empty. Stage-1 early-returns on empty splits
    # (num_iter == 0) and writes nothing, so their logits_buf / mid_lse slots hold
    # raw, uninitialised memory - they cannot be loaded or reduced.
    if USE_2D_VIEW:
        cur_batch_seq_len = tl.load(B_seq_len + cur_batch)
    else:
        batch_page_start = tl.load(B_seq_len + cur_batch)
        cur_batch_seq_len = tl.load(B_seq_len + cur_batch + 1) - batch_page_start
    # Mirror stage-1's partition (see _mla_gluon) so both stages agree on how many
    # splits actually hold data. Splits >= active were early-returned by stage-1
    # and never written; they are masked out of the reduce below.
    kv_len_per_split = tl.maximum(BLOCK_N, cur_batch_seq_len // NUM_KV_SPLITS)
    active_kv_splits = tl.minimum(
        tl.cdiv(cur_batch_seq_len, kv_len_per_split), NUM_KV_SPLITS
    )

    offs_d_ckv = tl.arange(0, HEAD_DIM_CKV)
    offs_s = tl.arange(0, BLOCK_S)
    base_l = cur_batch * stride_l_b + q_pos * stride_l_qs + cur_head * stride_l_h
    base_ml = cur_batch * stride_ml_b + q_pos * stride_ml_qs + cur_head * stride_ml_h

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)

    # The split-KV LSE merge is associative, so instead of a serial dependent loop
    # over NUM_KV_SPLITS (latency-bound on ~batch*nhead workgroups), reduce BLOCK_S
    # splits at a time as a vectorized [BLOCK_S, HEAD_DIM] tile. This overlaps the
    # per-split loads and shortens the dependent chain to NUM_KV_SPLITS/BLOCK_S.
    # Upper bound is active_kv_splits, not the constexpr NUM_KV_SPLITS: active can
    # be far below the launched budget, so bounding by it collapses the reduce to
    # ceil(active / BLOCK_S) tiles instead of always sweeping the full budget.
    for start in range(0, active_kv_splits, BLOCK_S):
        s_ids = start + offs_s  # [BLOCK_S]
        s_mask = s_ids < active_kv_splits
        lse = tl.load(
            Mid_lse + base_ml + s_ids * stride_ml_s, mask=s_mask, other=-float("inf")
        )  # [BLOCK_S]
        logits = tl.load(
            Logits + base_l + s_ids[:, None] * stride_l_s + offs_d_ckv[None, :],
            mask=s_mask[:, None],
            other=0.0,
        )  # [BLOCK_S, HEAD_DIM_CKV]

        tile_max = tl.max(lse, axis=0)  # scalar; masked/empty splits are -inf
        n_e_max = tl.maximum(e_max, tile_max)
        old_scale = tl.where(e_max == -float("inf"), 0.0, tl.exp(e_max - n_e_max))
        w = tl.where(lse == -float("inf"), 0.0, tl.exp(lse - n_e_max))  # [BLOCK_S]
        logits = tl.where(lse[:, None] == -float("inf"), 0.0, logits)
        acc = acc * old_scale + tl.sum(w[:, None] * logits, axis=0)
        e_sum = e_sum * old_scale + tl.sum(w, axis=0)
        e_max = n_e_max

    out = acc / e_sum if e_sum > 0.0 else tl.zeros([HEAD_DIM_CKV], dtype=tl.float32)
    tl.store(
        O + cur_batch * stride_o_b + q_pos * stride_o_qs + cur_head * stride_o_h + offs_d_ckv,
        out,
    )
    if HAS_FINAL_LSE:
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
        # Fixed ~256-WG launch budget, independent of sequence length so CUDA
        # Graph capture cannot freeze it; the kernels derive the per-batch
        # partition from the runtime KV length. Head blocks and MTP qlen already
        # consume part of the wave, so the budget divides by them too.
        NUM_M_BLOCKS = triton.cdiv(nhead, BLOCK_H)
        NUM_KV_SPLITS = max(1, 256 // (batch_size * qlen * NUM_M_BLOCKS))
        # Flattened DSpark verify is B=qlen rows with qlen=1, so the 256-WG
        # budget freezes splits at 14. Past ~28k global / DCP8 that is 3
        # BLOCK_N iters/split and the 3-stage pipeline faults. Grow splits so
        # each split stays near 2*BLOCK_N tokens. Snap to a power of two and
        # cap at 32 so CUDA-graph / compile variants stay a small discrete set.
        if min_kv_seq_len > 1:
            need = max(1, (int(min_kv_seq_len) + 2 * BLOCK_N - 1) // (2 * BLOCK_N))
            need = min(32, triton.next_power_of_2(need))
            NUM_KV_SPLITS = max(NUM_KV_SPLITS, need)
        forced_splits = int(os.getenv("VLLM_GLUON_NUM_KV_SPLITS", "0"))
        if forced_splits > 0:
            NUM_KV_SPLITS = forced_splits
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
        grid = (batch_size, NUM_KV_SPLITS, triton.cdiv(nhead, BLOCK_H) * qlen)
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
    )

    if NUM_KV_SPLITS == 1:
        # Fast path: stage-1 already wrote o (and lse) directly.
        return o, final_lse

    # Stage-2: reduce per-split (acc, lse) into o (and lse when return_lse).
    # grid axis 2 is q_pos (qlen). o uses the caller's layout (3-D or 4-D).
    grid_reduce = (batch_size, nhead, qlen)
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
        BLOCK_S=min(64, triton.next_power_of_2(NUM_KV_SPLITS)),
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    return o, final_lse
