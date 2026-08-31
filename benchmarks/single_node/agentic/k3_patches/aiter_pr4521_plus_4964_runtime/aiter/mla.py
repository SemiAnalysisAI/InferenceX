# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import functools
import os

import torch
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.core import is_experimental_enabled
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.attention import get_mla_decode_fwd_max_splits


@triton.jit
def _fwd_kernel_stage2_asm(
    Mid_O,
    Mid_lse,
    O,
    Final_lse,
    qo_indptr,
    kv_indptr,
    kv_last_page_lens,
    num_kv_splits_indptr,
    valid_split_count,
    stride_mid_ob: tl.int64,
    stride_mid_oh: tl.int64,
    stride_mid_os: tl.int64,
    stride_obs: tl.int64,
    stride_oh: tl.int64,
    stride_lse_bs: tl.int64,
    page_size: tl.constexpr,
    KV_INDPTR_IS_PAGE_LEVEL: tl.constexpr,
    MAYBE_FINAL_OUT: tl.constexpr,
    HAS_FINAL_LSE: tl.constexpr,
    USE_VALID_SPLIT_COUNT_REDUCE: tl.constexpr,
    BATCH_NUM: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    mgc: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_qo_start = tl.load(qo_indptr + cur_batch)
    cur_qo_end = tl.load(qo_indptr + cur_batch + 1)
    cur_split_start = tl.load(num_kv_splits_indptr + cur_batch)
    cur_split_end = tl.load(num_kv_splits_indptr + cur_batch + 1)
    num_max_kv_splits = tl.load(num_kv_splits_indptr + BATCH_NUM)
    cur_kv_start = tl.load(kv_indptr + cur_batch)
    cur_kv_end = tl.load(kv_indptr + cur_batch + 1)
    cur_kv_seq_len = cur_kv_end - cur_kv_start
    if KV_INDPTR_IS_PAGE_LEVEL:
        cur_kv_page_count = cur_kv_seq_len
        cur_kv_seq_len = (cur_kv_page_count - 1) * page_size + tl.load(
            kv_last_page_lens + cur_batch
        )
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    offs_logic = cur_qo_start * stride_mid_ob + cur_head * stride_mid_oh
    offs_v = offs_logic * Lv + offs_d
    num_valid_kv_splits = tl.minimum(
        cur_split_end - cur_split_start, tl.cdiv(cur_kv_seq_len, mgc)
    )
    if USE_VALID_SPLIT_COUNT_REDUCE:
        num_valid_kv_splits = tl.minimum(
            num_valid_kv_splits, tl.load(valid_split_count + cur_batch)
        )
    FINAL_OUT = MAYBE_FINAL_OUT and num_max_kv_splits == BATCH_NUM

    for cur_qo in range(cur_qo_start, cur_qo_end):
        if FINAL_OUT:
            input_ptr = Mid_O.to(tl.pointer_type(O.type.element_ty))
            out = tl.load(
                input_ptr
                + Lv * (cur_qo * stride_mid_os + cur_head * stride_mid_oh)
                + offs_d,
                mask=mask_d,
                other=0.0,
            )
            tl.store(
                O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
                out,
                mask=mask_d,
            )
        else:
            e_sum = 0.0
            e_max = -float("inf")
            acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
            for split_kv_id in range(num_valid_kv_splits):
                tv = tl.load(
                    Mid_O + offs_v + split_kv_id * stride_mid_os * Lv,
                    mask=mask_d,
                    other=0.0,
                )
                tlogic = tl.load(Mid_lse + offs_logic + split_kv_id * stride_mid_os)
                n_e_max = tl.maximum(tlogic, e_max)

                old_scale = tl.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc += exp_logic * tv

                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max
            offs_logic += stride_mid_ob
            offs_v += stride_mid_ob * Lv
            tl.store(
                O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
                acc / e_sum,
                mask=mask_d,
            )
            if HAS_FINAL_LSE:
                tl.store(
                    Final_lse + cur_qo * stride_lse_bs + cur_head,
                    e_max + tl.log(e_sum),
                )


@functools.lru_cache
def get_meta_param(
    num_kv_splits,
    bs,
    total_kv,
    nhead,
    max_seqlen_q,
    dtype,
    tg_factor=1,
    ignore_total_kv=0,
):
    # ignore_total_kv: experiment switch (default 0 = legacy behavior). When 1,
    # the split-count search ignores `total_kv` entirely: the auto-search keeps
    # only the CU-occupancy factor (drops the avg_kv / (avg_kv + overhead*i) HBM
    # term) and the fp8 min-block cap is skipped. This yields the pure
    # occupancy-driven pick (fill bs_occ*splits into cu_num, capped at 16),
    # independent of how long each sequence's KV is.
    # tg_factor: number of thread-groups (workgroups) the kernel launches per
    # (seq, kv-split) along the head dim. Default 1. For variants that synthesize
    # a larger logical head count from multiple WGs -- e.g. the v4 nm gqa=128
    # path runs gdx=ceil(128/64)=2 WGs per (seq, split) -- the GPU's CU occupancy
    # is driven by `bs * tg_factor * num_kv_splits`, not `bs * num_kv_splits`.
    # Only the occupancy term scales; avg_kv, the fp8 block cap, and the indptr
    # all stay keyed on the real `bs`. The auto-search keeps V3's cap of 16.
    if num_kv_splits is None:
        cu_num = get_cu_num()
        avg_kv = total_kv / bs
        overhead = 84.1
        # Head-group workgroup multiplier feeding CU occupancy. Two sources:
        #   - tg_factor (caller-supplied): the v4 nm wrapper passes
        #     ceil(num_heads/64) so gqa=128 (2 head-group WGs) is counted as 2x.
        #   - wg_per_split (auto, from main): qh128 decode on gfx1250 launches 2
        #     head-group workgroups per (batch, split) along z (mirrors gdz =
        #     kv_split*2 in asm_mla.cu).
        # Take the max so either path applies; for V3 callers (tg_factor=1) the
        # gfx1250 auto-rule still kicks in, and for v4 callers the explicit
        # tg_factor governs.
        is_gfx1250 = get_gfx() == "gfx1250"
        wg_per_split = 2 if nhead == 128 and is_gfx1250 else 1
        bs_occ = bs * max(tg_factor, wg_per_split)
        if ignore_total_kv:
            # Occupancy-only: drop the avg_kv HBM term so the pick depends solely
            # on how well bs_occ*i fills cu_num (ties -> smallest i).
            tmp = [
                (
                    bs_occ * i / ((bs_occ * i + cu_num - 1) // cu_num * cu_num),
                    i,
                )
                for i in range(1, 17)
            ]
        else:
            tmp = [
                (
                    bs_occ
                    * i
                    / ((bs_occ * i + cu_num - 1) // cu_num * cu_num)
                    * avg_kv
                    / (avg_kv + overhead * i),
                    i,
                )
                for i in range(1, 17)
            ]
        num_kv_splits = max(tmp, key=lambda x: x[0])[1]

    get_block_n_fp8 = {
        8: 64,
        16: 128,
        24: 128,
        32: 128,
        48: 64,
        64: 64,
        128: 32,
        256: 32,
        384: 32,
        512: 32,
    }

    if dtype == dtypes.fp8 and not ignore_total_kv:
        min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]
        # ceil(avg_kv / min_block_n) computed in pure integers (avg_kv = total_kv/bs).
        num_kv_splits = min(
            num_kv_splits,
            (total_kv + bs * min_block_n - 1) // (bs * min_block_n),
        )
        if num_kv_splits > 1:
            num_kv_splits = min(
                num_kv_splits,
                int(abs(total_kv / bs - max_seqlen_q) // min_block_n) + 1,
            )

    num_kv_splits_indptr = torch.arange(
        0, (bs + 1) * num_kv_splits, num_kv_splits, dtype=torch.int, device="cuda"
    )

    return num_kv_splits, num_kv_splits_indptr


# Persistent MLA-decode kernel gate: the persistent kernel
# ("mla_a16w16_qh16..._ps") is slower than the non-persistent split-KV kernel
# ("mla_dec_stage1...") above a concurrency threshold (~batch 16-64 on gfx950 bf16
# 16-head decode). Fall back to non-persistent at/above the batch threshold.
# Env-tunable (0 disables the gate). Provisional default pending microbench
# calibration (AIOSS-5156).
_MLA_DECODE_PERSISTENT_MAX_BATCH_DEFAULT = 32


@functools.lru_cache(maxsize=1)
def _persistent_mla_decode_max_batch():
    """Cached read of AITER_MLA_DECODE_PERSISTENT_MAX_BATCH (read once per process)."""
    try:
        return int(
            os.getenv(
                "AITER_MLA_DECODE_PERSISTENT_MAX_BATCH",
                _MLA_DECODE_PERSISTENT_MAX_BATCH_DEFAULT,
            )
        )
    except (TypeError, ValueError):
        return _MLA_DECODE_PERSISTENT_MAX_BATCH_DEFAULT


def _fold_seqlen_indptr(indptr, fold_factor):
    """Repeat each batch's seqlen ``fold_factor`` times (head-folding pseudo-batches)."""
    lens = indptr[1:] - indptr[:-1]
    folded_lens = lens.repeat_interleave(fold_factor)
    out = torch.empty(
        indptr.shape[0] + (fold_factor - 1) * (indptr.shape[0] - 1),
        dtype=indptr.dtype,
        device=indptr.device,
    )
    out[0] = 0
    out[1:] = torch.cumsum(folded_lens, dim=0).to(indptr.dtype)
    return out


def _use_persistent_mla_decode(bs, nhead, max_seqlen_q, q_dtype, kv_dtype):
    """Whether to keep the persistent MLA decode kernel.

    True keeps the caller's persistent request; False falls back to the
    non-persistent split-KV kernel at/above the concurrency (batch) threshold.
    Scoped to the characterized gfx950 bf16 16-head single-token decode profile.
    """
    is_regression_profile = (
        get_gfx() == "gfx950"
        and q_dtype == dtypes.bf16
        and kv_dtype == dtypes.bf16
        and nhead == 16
        and max_seqlen_q == 1
    )
    if not is_regression_profile:
        return True

    max_batch = _persistent_mla_decode_max_batch()

    # max_batch <= 0 disables the gate (always keep persistent).
    if max_batch <= 0:
        return True
    return bs < max_batch


def mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    page_size=1,
    nhead_kv=1,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
    num_kv_splits_indptr=None,  # for experts only!!!
    work_meta_data=None,
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,
    q_scale=None,
    kv_scale=None,
    intra_batch_mode=False,
    return_logits=False,
    return_lse=False,
    # round-robin context-parallel (CP): when cp_world_size > 1 the local KV is a
    # round-robin shard of a global sequence (global pos p -> rank p % W). The
    # kernel maps a local kv index j back to its global position g(j)=j*W+r to
    # apply the causal mask on GLOBAL positions. g_kv_indptr carries the GLOBAL
    # per-request kv_indptr (global KV length). cp_world_size==1 disables it.
    g_kv_indptr=None,
    cp_world_size=1,
    cp_rank=0,
):
    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    if kv_buffer.dtype != torch.uint8:
        _, _, _, qk_head_dim = kv_buffer.shape
    else:
        _, _, qk_head_dim = q.shape

    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    ori_total_s, ori_nhead, _ori_v_head_dim = o.shape
    total_s, nhead, v_head_dim = o.shape
    bs = qo_indptr.shape[0] - 1
    total_kv = kv_indices.shape[0]

    persistent_mode = work_meta_data is not None

    # Above the concurrency threshold the persistent kernel is slower, so
    # fall back to the non-persistent split-KV path (which rebuilds its own
    # num_kv_splits and ignores work_meta_data). See _use_persistent_mla_decode.
    #
    # Scope: only the dense vLLM-style decode (num_kv_splits left to aiter) is
    # gated. Callers that pass an explicit num_kv_splits -- the sparse/expert
    # op paths -- depend on the persistent kernel's work_meta_data layout; the
    # non-persistent fallback can't honor their top-k/split scheme and would
    # return wrong results, so never downgrade them.
    if persistent_mode and num_kv_splits is None:
        persistent_mode = _use_persistent_mla_decode(
            bs, nhead, max_seqlen_q, q.dtype, kv_buffer.dtype
        )

    io_transformed = False
    qseqlen_folded = False

    if not persistent_mode:
        if num_kv_splits is None or num_kv_splits_indptr is None:
            num_kv_splits, num_kv_splits_indptr = get_meta_param(
                num_kv_splits, bs, total_kv, nhead, max_seqlen_q, q.dtype
            )
        mgc = (
            64
            if nhead in [8, 16]
            and (max_seqlen_q == 1 or (nhead == 8 and max_seqlen_q == 2))
            else 16
        )
        mgc = (
            32
            if (
                nhead == 128 and q.dtype == dtypes.fp8 and kv_buffer.dtype == dtypes.fp8
            )
            or (
                nhead == 64
                and q.dtype == dtypes.bf16
                and kv_buffer.dtype == dtypes.bf16
                and max_seqlen_q == 1
            )
            else mgc
        )

        MAYBE_FINAL_OUT = True

        if nhead in [8, 16] and max_seqlen_q == 1:
            MAYBE_FINAL_OUT = False

        logits = (
            o.view((total_s, num_kv_splits, nhead, v_head_dim))
            if (
                num_kv_splits == 1
                and (
                    q.dtype == dtypes.fp8
                    or (q.dtype == dtypes.bf16 and max_seqlen_q == 4)
                    or (
                        q.dtype == dtypes.bf16
                        and kv_buffer.dtype == dtypes.bf16
                        and nhead == 32
                    )
                    or (
                        q.dtype == dtypes.bf16
                        and kv_buffer.dtype == dtypes.bf16
                        and nhead == 8
                    )
                )
            )
            else torch.empty(
                (total_s, num_kv_splits, nhead, v_head_dim),
                dtype=dtypes.fp32,
                device=device,
            )
        )

        attn_lse = torch.empty(
            (total_s, num_kv_splits, nhead, 1), dtype=dtypes.fp32, device=device
        )
        final_lse = (
            torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)
            if return_lse
            else None
        )

        # Per-batch valid KV split count writeback buffer. Always allocated (and
        # passed to stage1) so the asm kernel has a valid destination; whether
        # stage2 actually uses it is gated by use_valid_split_count_reduce.
        # Initialized to num_kv_splits so a min() against it is a no-op until the
        # kernel overwrites it with the real (smaller) valid count.
        valid_split_count = torch.full(
            (bs,), num_kv_splits, dtype=dtypes.i32, device=device
        )
        use_valid_split_count_reduce = int(num_kv_splits > 1)

        aiter.mla_decode_stage1_asm_fwd(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_kv_splits_indptr,
            None,
            None,
            None,
            max_seqlen_q,
            page_size,
            nhead_kv,
            sm_scale,
            logits,
            attn_lse,
            o,
            final_lse,
            q_scale,
            kv_scale,
            g_kv_indptr,
            cp_world_size,
            cp_rank,
            valid_split_count,
            use_valid_split_count_reduce,
        )

        if num_kv_splits == 1 and (
            q.dtype == dtypes.fp8
            or (q.dtype == dtypes.bf16 and max_seqlen_q == 4)
            or (
                q.dtype == dtypes.bf16
                and kv_buffer.dtype == dtypes.bf16
                and nhead == 32
            )
            or (
                q.dtype == dtypes.bf16 and kv_buffer.dtype == dtypes.bf16 and nhead == 8
            )
        ):
            lse = final_lse if return_lse else attn_lse
            return logits.view(total_s, nhead, v_head_dim), lse

        Lv = v_head_dim
        BLOCK_DV = triton.next_power_of_2(Lv)
        grid = (bs, nhead)
        extra_kargs = {"waves_per_eu": 4}

        has_final_lse = final_lse is not None
        final_lse_buf = (
            final_lse
            if has_final_lse
            else torch.empty((1,), dtype=dtypes.fp32, device=device)
        )

        _fwd_kernel_stage2_asm[grid](
            logits,
            attn_lse,
            o,
            final_lse_buf,
            qo_indptr,
            kv_indptr,
            kv_last_page_lens,
            num_kv_splits_indptr,
            valid_split_count,
            attn_lse.stride(0),
            attn_lse.stride(2),
            attn_lse.stride(1),
            o.stride(0),
            o.stride(1),
            final_lse_buf.stride(0) if has_final_lse else 0,
            page_size=page_size,
            KV_INDPTR_IS_PAGE_LEVEL=page_size > 1,
            MAYBE_FINAL_OUT=MAYBE_FINAL_OUT,
            HAS_FINAL_LSE=has_final_lse,
            USE_VALID_SPLIT_COUNT_REDUCE=use_valid_split_count_reduce,
            BATCH_NUM=bs,
            BLOCK_DV=BLOCK_DV,
            Lv=Lv,
            mgc=mgc,
            # One (token, head) per CTA, no cross-lane reduction: the merge is
            # memory-latency bound, so max occupancy (num_warps=1) beats wider
            # tiles. Measured 1.7-1.9x over num_warps=4 on gfx950 decode shapes.
            num_warps=1,
            num_stages=2,
            **extra_kargs,
        )
    else:
        if num_kv_splits is None:
            num_kv_splits = get_mla_decode_fwd_max_splits(
                ori_nhead, max_seqlen_q, q.dtype, kv_buffer.dtype
            )
        if (
            nhead == 16
            or (
                get_gfx() == "gfx942"
                and nhead == 128
                and q.dtype == dtypes.fp8
                and kv_buffer.dtype == dtypes.fp8
            )
            or (
                get_gfx() == "gfx950"
                and nhead == 128
                and q.dtype == dtypes.fp8
                and kv_buffer.dtype == dtypes.fp8
                and is_experimental_enabled()
            )
            or (
                get_gfx() == "gfx942"
                and nhead in (16, 32, 64)
                and nhead * max_seqlen_q == 128
                and q.dtype == dtypes.fp8
                and kv_buffer.dtype == dtypes.fp8
                and is_experimental_enabled()
            )
            or (
                get_gfx() == "gfx950"
                and q.dtype == dtypes.fp8
                and kv_buffer.dtype == dtypes.fp8
                and ((nhead == 32) or (nhead == 64) or (nhead == 128))
            )
            or (
                get_gfx() == "gfx950"
                and nhead == 8
                and q.dtype == dtypes.fp8
                and kv_buffer.dtype == dtypes.fp8
                and max_seqlen_q == 4
            )
            or (
                get_gfx() == "gfx942"
                and nhead == 8
                and q.dtype == dtypes.bf16
                and kv_buffer.dtype == dtypes.bf16
                and max_seqlen_q == 2
            )
            or (
                get_gfx() == "gfx942"
                and nhead == 64
                and q.dtype == dtypes.fp8
                and kv_buffer.dtype == dtypes.fp8
                and max_seqlen_q == 1
            )
            or (
                get_gfx() == "gfx950"
                and q.dtype == dtypes.bf16
                and kv_buffer.dtype == dtypes.bf16
            )
            or (
                get_gfx() == "gfx950"
                and nhead == 96
                and q.dtype == dtypes.fp8
                and kv_buffer.dtype == dtypes.fp8
                and max_seqlen_q <= 6
            )
        ):
            # Natively support cases
            pass
        elif nhead in range(32, 128 + 1, 16) and persistent_mode:
            fold_factor = ori_nhead // 16
            nhead = 16
            total_s = ori_total_s * fold_factor
            if max_seqlen_q == 1:
                q = q.view(total_s, nhead, -1)
            else:
                q = (
                    q.reshape(
                        ori_total_s // max_seqlen_q,
                        max_seqlen_q,
                        ori_nhead // nhead,
                        nhead,
                        -1,
                    )
                    .permute(0, 2, 1, 3, 4)
                    .reshape(total_s, nhead, -1)
                )
                o_orig = o

            o = o.view(total_s, nhead, -1)

            qo_indptr = _fold_seqlen_indptr(qo_indptr, fold_factor)
            if g_kv_indptr is not None:
                g_kv_indptr = _fold_seqlen_indptr(g_kv_indptr, fold_factor)
                # Each pseudo-batch shares the original batch's local kv begin.
                kv_indptr = torch.cat(
                    [kv_indptr[:-1].repeat_interleave(fold_factor), kv_indptr[-1:]]
                )
            io_transformed = True
        else:
            assert False, f"{nhead=} and {max_seqlen_q=} not supported"

        logits = torch.empty(
            (reduce_partial_map.size(0) * max_seqlen_q, 1, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
        attn_lse = torch.empty(
            (reduce_partial_map.size(0) * max_seqlen_q, 1, nhead, 1),
            dtype=dtypes.fp32,
            device=device,
        )
        final_lse = (
            torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)
            if return_lse
            else None
        )

        use_hk = (
            get_gfx() in ("gfx942", "gfx950")
            and nhead * max_seqlen_q == 128
            and q.dtype == dtypes.fp8
            and kv_buffer.dtype == dtypes.fp8
            and page_size in (1, 64)
            and is_experimental_enabled()
        ) or (
            get_gfx() == "gfx950"
            and nhead * max_seqlen_q == 64
            and q.dtype == dtypes.fp8
            and kv_buffer.dtype == dtypes.fp8
            and page_size in (1, 64)
            and is_experimental_enabled()
        )

        if use_hk:
            aiter.hk_mla_v32_decode_fwd(
                q,
                kv_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                work_indptr,
                work_info_set,
                max_seqlen_q,
                sm_scale,
                logits,
                attn_lse,
                o,
            )
        else:
            aiter.mla_decode_stage1_asm_fwd(
                q,
                kv_buffer,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                num_kv_splits_indptr,
                work_meta_data,
                work_indptr,
                work_info_set,
                max_seqlen_q,
                page_size,
                nhead_kv,
                sm_scale,
                logits,
                attn_lse,
                o,
                final_lse,
                q_scale,
                kv_scale,
                g_kv_indptr,
                cp_world_size,
                cp_rank,
                None,
                0,
            )

        aiter.mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            max_seqlen_q,
            num_kv_splits,
            o,
            final_lse,
        )

    if io_transformed:
        if return_logits:
            logits = logits.view(-1, 1, ori_nhead, v_head_dim)

        if max_seqlen_q == 1 or qseqlen_folded:
            q = q.view(ori_total_s, ori_nhead, -1)
            o = o.view(ori_total_s, ori_nhead, -1)
            if final_lse is not None:
                final_lse = final_lse.view(ori_total_s, ori_nhead)
        else:
            new_o = (
                o.reshape(
                    ori_total_s // max_seqlen_q,
                    ori_nhead // nhead,
                    max_seqlen_q,
                    nhead,
                    -1,
                )
                .permute(0, 2, 1, 3, 4)
                .reshape(ori_total_s, ori_nhead, -1)
                .contiguous()
            )
            o_orig.set_(new_o)
            o = o_orig

            if final_lse is not None:
                final_lse = (
                    final_lse.reshape(
                        ori_total_s // max_seqlen_q,
                        ori_nhead // nhead,
                        max_seqlen_q,
                        nhead,
                    )
                    .permute(0, 2, 1, 3)
                    .reshape(ori_total_s, ori_nhead)
                    .contiguous()
                )

    return logits, final_lse


# V4.0 layout constants (mirrored from op_tests/test_mla_v4_persistent.py).
_V40_DIM_NOPE = 448
_V40_DIM_ROPE = 64
_V40_DIM_QK_PACKED = 512  # NOPE 448 + dup-E8M0 14 + unused trailing pad 50


def mla_v40_decode_fwd(
    q,  # [total_q, nhead, 512]                         FP8 (NOPE+scale+pad)
    q_rope,  # [total_q, nhead, 64]                          BF16
    kv_buffer,  # [num_page, page_size, 1, 512]                 FP8
    kv_buffer_rope,  # [num_page, page_size, 1, 64]                  BF16
    o,  # [total_q, nhead, 512]                         BF16
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    work_indptr,
    work_info_set,
    reduce_indptr,
    reduce_final_map,
    reduce_partial_map,
    sm_scale=None,  # default 1.0/sqrt(512)
    return_lse=False,
    attn_sink=None,  # optional [nhead] fp32: per-head sink logit (virtual K, zero V)
):
    """
    V4.0 MLA decode router. Picks a suitable backend (HK persistent today;
    asm / triton / gluon / flydsl as they land) given the layout, dtype, and
    target chip. Writes attention output into `o` in-place; returns
    `(logits, final_lse)` for the caller (final_lse is None unless
    return_lse=True).

    attn_sink (optional): [nhead] fp32. Per-head bias logit treated as a
    virtual K column with zero V. The kernel folds it into the LAST
    split's running denominator (gated by kv_offset == 0); the reducer's
    standard formula then routes exp(sink) into the global denominator
    exactly once.
    """
    device = q.device
    total_s, nhead, v_head_dim = o.shape

    assert q.shape[-1] == _V40_DIM_QK_PACKED, (
        f"mla_v40_decode_fwd: packed Q last dim must be {_V40_DIM_QK_PACKED}, "
        f"got q.shape={tuple(q.shape)}"
    )
    assert kv_buffer.shape[-1] == _V40_DIM_QK_PACKED, (
        f"mla_v40_decode_fwd: packed KV last dim must be {_V40_DIM_QK_PACKED}, "
        f"got kv_buffer.shape={tuple(kv_buffer.shape)}"
    )
    assert q_rope.shape[-1] == _V40_DIM_ROPE, (
        f"mla_v40_decode_fwd: Q rope last dim must be {_V40_DIM_ROPE}, "
        f"got q_rope.shape={tuple(q_rope.shape)}"
    )
    assert kv_buffer_rope.shape[-1] == _V40_DIM_ROPE, (
        f"mla_v40_decode_fwd: KV rope last dim must be {_V40_DIM_ROPE}, "
        f"got kv_buffer_rope.shape={tuple(kv_buffer_rope.shape)}"
    )

    if sm_scale is None:
        sm_scale = 1.0 / ((_V40_DIM_NOPE + _V40_DIM_ROPE) ** 0.5)  # 1/sqrt(512)

    page_size = kv_buffer.shape[1]

    logits = torch.empty(
        (reduce_partial_map.size(0) * max_seqlen_q, 1, nhead, v_head_dim),
        dtype=dtypes.fp32,
        device=device,
    )
    attn_lse = torch.empty(
        (reduce_partial_map.size(0) * max_seqlen_q, 1, nhead, 1),
        dtype=dtypes.fp32,
        device=device,
    )
    final_lse = (
        torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)
        if return_lse
        else None
    )

    use_hk = (
        get_gfx() == "gfx950"
        and nhead * max_seqlen_q in (64, 128)
        and q.dtype == dtypes.fp8
        and kv_buffer.dtype == dtypes.fp8
        and q_rope.dtype == dtypes.bf16
        and kv_buffer_rope.dtype == dtypes.bf16
        and page_size in (1, 64)
        and is_experimental_enabled()
    )

    if use_hk:
        aiter.hk_mla_v40_decode_fwd(
            q,
            q_rope,
            kv_buffer,
            kv_buffer_rope,
            qo_indptr,
            kv_indices,
            kv_last_page_lens,
            work_indptr,
            work_info_set,
            max_seqlen_q,
            sm_scale,
            logits,
            attn_lse,
            o,
            attn_sink,
        )
    else:
        # TODO: dispatch to asm / triton / gluon / flydsl V4.0 backends once
        # they land. Today HK is the only available implementation.
        raise NotImplementedError(
            f"mla_v40_decode_fwd: no backend available for "
            f"gfx={get_gfx()} nhead={nhead} max_seqlen_q={max_seqlen_q} "
            f"q.dtype={q.dtype} kv_buffer.dtype={kv_buffer.dtype} "
            f"q_rope.dtype={q_rope.dtype} kv_buffer_rope.dtype={kv_buffer_rope.dtype} "
            f"page_size={page_size} experimental={is_experimental_enabled()}"
        )

    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_q,
        0,
        o,
        final_lse,
    )

    return logits, final_lse


def mla_prefill_fwd(
    q,  # [num_seqs, num_heads, head_size]
    kv_buffer,  # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    o,  # [num_seqs, num_heads, v_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
):
    device = q.device
    _num_page, _page_size, _nhead_kv, qk_head_dim = kv_buffer.shape
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)
    bs, nhead, v_head_dim = o.shape

    num_kv_splits = 1

    logits = o.view(bs, num_kv_splits, nhead, v_head_dim)
    # logits = torch.empty(
    #     (bs, num_kv_splits, nhead, v_head_dim), dtype=dtypes.fp32, device=device
    # )
    attn_lse = torch.empty(
        (bs, num_kv_splits, nhead, 1), dtype=dtypes.fp32, device=device
    )

    aiter.mla_prefill_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q,
        sm_scale,
        logits,
        attn_lse,
    )

    # return logits.view(bs, nhead, v_head_dim).to(o.dtype), attn_lse
    return o.view(bs, nhead, v_head_dim), attn_lse


def mla_prefill_ps_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    work_indptr: torch.Tensor | None,
    work_info_set: torch.Tensor | None,
    max_seqlen_q: int,
    is_causal: bool,
    reduce_indptr: torch.Tensor | None = None,
    reduce_final_map: torch.Tensor | None = None,
    reduce_partial_map: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    device = Q.device
    total_s, nhead, v_head_dim = output.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / (v_head_dim**0.5)

    tile_q = 256
    logits = torch.empty(
        (reduce_partial_map.size(0) * tile_q, nhead, v_head_dim),
        dtype=dtypes.fp32,
        device=device,
    )
    attn_lse = torch.empty(
        (reduce_partial_map.size(0) * tile_q, nhead), dtype=dtypes.fp32, device=device
    )
    final_lse = torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)

    aiter.mla_prefill_ps_asm_fwd(
        Q,
        K,
        V,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        work_indptr,
        work_info_set,
        max_seqlen_q,
        softmax_scale,
        is_causal,
        logits,
        attn_lse,
        output,
        q_scale,
        k_scale,
        v_scale,
    )

    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        tile_q,
        0,
        output,
        final_lse,
    )

    return output.view(total_s, nhead, v_head_dim), attn_lse


@triton.jit
def _mla_prefill_reduce_kernel(
    # Input tensors
    partial_output_ptr,  # [padded_num_tokens * available_tgs, num_head_q, v_head_dim]
    partial_lse_ptr,  # [padded_num_tokens * available_tgs, num_head_q]
    # Metadata tensors
    reduce_indptr_ptr,  # [num_reduce_groups + 1]
    reduce_final_map_ptr,  # [num_reduce_groups, 2]: [qo_start, qo_end]
    reduce_partial_map_ptr,  # [num_partial_tiles]: [partial_qo_loc]
    # Output tensor
    output_ptr,  # [total_tokens, num_head_q, v_head_dim]
    # Strides
    stride_po_tok,
    stride_po_head,
    stride_po_dim,
    stride_lse_tok,
    stride_lse_head,
    stride_o_tok,
    stride_o_head,
    stride_o_dim,
    # Constants
    TILE_Q: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    MAX_PARTIALS: tl.constexpr,
):
    """
    Each program processes one (reduce_group, head, token) combination.
    Grid: (num_reduce_groups, num_heads, TILE_Q)

    All heads are uniformly split and reduced together.
    """

    group_id = tl.program_id(0)
    head_id = tl.program_id(1)
    tok_offset = tl.program_id(2)  # q_tile

    # Load reduce group metadata (read once per block)
    start_idx = tl.load(reduce_indptr_ptr + group_id)
    end_idx = tl.load(reduce_indptr_ptr + group_id + 1)
    num_partials = end_idx - start_idx

    if num_partials == 0:
        return

    # Load final map: [qo_start, qo_end]
    final_map_offset = group_id * 2
    qo_start = tl.load(reduce_final_map_ptr + final_map_offset + 0)
    qo_end = tl.load(reduce_final_map_ptr + final_map_offset + 1)

    q_len = qo_end - qo_start
    tok_id = tok_offset

    # Skip if beyond valid range
    if tok_id >= q_len:
        return

    # compute max LSE and collect LSE values
    max_lse = -float("inf")
    lse_values = tl.zeros((MAX_PARTIALS,), dtype=tl.float32) - float("inf")

    for p_idx in range(num_partials):
        if p_idx < num_partials:
            partial_qo_loc = tl.load(reduce_partial_map_ptr + start_idx + p_idx)

            lse_offset = (
                partial_qo_loc + tok_id
            ) * stride_lse_tok + head_id * stride_lse_head
            lse = tl.load(partial_lse_ptr + lse_offset)

            is_valid = lse == lse  # noqa: PLR0124
            lse = tl.where(is_valid, lse, -float("inf"))

            lse_values = tl.where(tl.arange(0, MAX_PARTIALS) == p_idx, lse, lse_values)

            # Update max
            max_lse = tl.maximum(max_lse, lse)

    # compute sum_exp
    sum_exp = 0.0
    for p_idx in tl.static_range(MAX_PARTIALS):
        if p_idx < num_partials:
            # Extract the lse value for this partition
            lse = tl.sum(tl.where(tl.arange(0, MAX_PARTIALS) == p_idx, lse_values, 0.0))
            exp_val = tl.exp(lse - max_lse)
            sum_exp += exp_val

    final_lse = max_lse + tl.log(sum_exp)

    # accumulate weighted outputs in chunks
    # Process V_HEAD_DIM in chunks of BLOCK_DIM
    num_dim_blocks = tl.cdiv(V_HEAD_DIM, BLOCK_DIM)

    for dim_block_id in range(num_dim_blocks):
        dim_offs = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offs < V_HEAD_DIM

        acc = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

        for p_idx in tl.static_range(MAX_PARTIALS):
            if p_idx < num_partials:
                partial_qo_loc = tl.load(reduce_partial_map_ptr + start_idx + p_idx)

                # Extract lse value
                lse = tl.sum(
                    tl.where(tl.arange(0, MAX_PARTIALS) == p_idx, lse_values, 0.0)
                )

                scale = tl.exp(lse - final_lse)

                # load partial output
                out_offset = (
                    (partial_qo_loc + tok_id) * stride_po_tok
                    + head_id * stride_po_head
                    + dim_offs * stride_po_dim
                )
                partial_out = tl.load(
                    partial_output_ptr + out_offset, mask=dim_mask, other=0.0
                )

                # Handle NaN in output (NaN != NaN)
                is_valid_out = partial_out == partial_out  # noqa: PLR0124
                partial_out = tl.where(is_valid_out, partial_out, 0.0)

                acc += scale * partial_out

        output_offset = (
            (qo_start + tok_id) * stride_o_tok
            + head_id * stride_o_head
            + dim_offs * stride_o_dim
        )
        tl.store(
            output_ptr + output_offset,
            acc.to(output_ptr.dtype.element_ty),
            mask=dim_mask,
        )


def mla_prefill_reduce_triton(
    partial_output: torch.Tensor,  # [padded_num_tokens * available_tgs, num_head_q, v_head_dim]
    partial_lse: torch.Tensor,  # [padded_num_tokens * available_tgs, num_head_q]
    reduce_indptr: torch.Tensor,  # [num_reduce_groups + 1], int32
    reduce_final_map: torch.Tensor,  # [num_reduce_groups, 2], int32: [qo_start, qo_end]
    reduce_partial_map: torch.Tensor,  # [num_partial_tiles], int32: [partial_qo_loc]
    output: torch.Tensor,  # [total_tokens, num_head_q, v_head_dim], output buffer
    tile_q: int = 256,  # Q tile size (for padding)
    max_partials_static: (
        int | None
    ) = None,  # Maximum number of partials, defaults to num_cu
) -> None:
    """Triton version of mla_prefill_reduce.
    All heads are uniformly split and reduced together.
    """
    MAX_PARTIALS_STATIC = (
        max_partials_static if max_partials_static is not None else get_cu_num()
    )

    num_reduce_groups = reduce_indptr.shape[0] - 1
    _, num_heads, v_head_dim = partial_output.shape

    if num_reduce_groups == 0:
        return

    # Check max_partials doesn't exceed the fixed constant in kernel
    max_partials = 0
    for i in range(num_reduce_groups):
        num_p = (reduce_indptr[i + 1] - reduce_indptr[i]).item()
        max_partials = max(max_partials, num_p)

    if max_partials > MAX_PARTIALS_STATIC:
        raise ValueError(
            f"max_partials={max_partials} exceeds MAX_PARTIALS_STATIC={MAX_PARTIALS_STATIC}. "
            "Consider increasing MAX_PARTIALS_STATIC."
        )

    # Choose block size for v_head_dim chunks
    BLOCK_DIM = 64
    if v_head_dim <= 64:
        BLOCK_DIM = triton.next_power_of_2(v_head_dim)

    # Grid: (num_reduce_groups, num_heads, TILE_Q)
    grid = (num_reduce_groups, num_heads, tile_q)

    _mla_prefill_reduce_kernel[grid](
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        output,
        # Strides
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # Constants
        TILE_Q=tile_q,
        V_HEAD_DIM=v_head_dim,
        BLOCK_DIM=BLOCK_DIM,
        MAX_PARTIALS=MAX_PARTIALS_STATIC,
        num_warps=4,
    )


def mla_prefill_reduce(
    partial_output: torch.Tensor,  # [padded_num_tokens * available_tgs, num_head_q, v_head_dim]
    partial_lse: torch.Tensor,  # [padded_num_tokens * available_tgs, num_head_q]
    reduce_indptr: torch.Tensor,  # [num_reduce_groups + 1], int32
    reduce_final_map: torch.Tensor,  # [num_reduce_groups, 2], int32: [qo_start, qo_end]
    reduce_partial_map: torch.Tensor,  # [num_partial_tiles], int32: [partial_qo_loc]
    output: torch.Tensor,  # [total_tokens, num_head_q, v_head_dim], output buffer
    tile_q: int = 256,  # Q tile size (for padding)
    use_triton: bool = True,  # Whether to use Triton kernel
) -> None:

    if True:
        try:
            return mla_prefill_reduce_triton(
                partial_output,
                partial_lse,
                reduce_indptr,
                reduce_final_map,
                reduce_partial_map,
                output,
                tile_q,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Triton reduce failed ({e}), falling back to PyTorch")

    # torch implementation, just for reference
    num_reduce_groups = reduce_indptr.shape[0] - 1
    device = partial_output.device
    dtype = partial_output.dtype
    _, num_heads, _v_head_dim = partial_output.shape

    for group_id in range(num_reduce_groups):
        start_idx = reduce_indptr[group_id].item()  # 0
        end_idx = reduce_indptr[group_id + 1].item()  # 2
        num_partials = end_idx - start_idx

        if num_partials == 0:
            continue

        final_map = reduce_final_map[group_id]
        qo_start = final_map[0].item()
        qo_end = final_map[1].item()

        q_len = qo_end - qo_start  # actual length (may be < tile_q for last tile)
        read_len = tile_q

        # Collect partial indices
        partial_indices = []
        for partial_idx in range(start_idx, end_idx):
            partial_qo_loc = reduce_partial_map[partial_idx].item()
            partial_indices.append(partial_qo_loc)

        # Process all heads together
        for head_idx in range(num_heads):
            partial_lses = []
            partial_outputs = []

            for partial_qo_loc in partial_indices:
                lse = partial_lse[partial_qo_loc : partial_qo_loc + read_len, head_idx]
                partial_lses.append(lse)

                out = partial_output[
                    partial_qo_loc : partial_qo_loc + read_len, head_idx, :
                ]
                partial_outputs.append(out)

            if len(partial_lses) == 0:
                continue

            partial_lses = torch.stack(partial_lses, dim=0)  # [K, tile_q]
            partial_outputs = torch.stack(partial_outputs, dim=0)  # [K, tile_q, D]

            nan_mask = torch.isnan(partial_lses)  # [K, tile_q]
            neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
            zero = torch.tensor(0.0, device=device, dtype=dtype)

            partial_lses_clean = torch.where(nan_mask, neg_inf, partial_lses)

            max_lse = torch.max(partial_lses_clean, dim=0)[0]  # [tile_q]

            # Compute sum_exp (NaN values contribute 0 to sum)
            # exp(-inf - max) = 0, so NaN values are automatically excluded
            sum_exp = torch.sum(
                torch.where(
                    nan_mask, zero, torch.exp(partial_lses - max_lse.unsqueeze(0))
                ),
                dim=0,
            )

            final_lse = max_lse + torch.log(sum_exp)

            scales = torch.exp(partial_lses_clean - final_lse.unsqueeze(0)).unsqueeze(
                -1
            )  # [K, tile_q, 1]

            nan_output_mask = torch.isnan(partial_outputs)  # [K, tile_q, D]
            partial_outputs_clean = torch.where(nan_output_mask, zero, partial_outputs)

            final_output = torch.sum(
                partial_outputs_clean * scales, dim=0
            )  # [tile_q, v_head_dim]

            output[qo_start:qo_end, head_idx, :] = final_output[:q_len, :]


# ---------------------------------------------------------------------------
# DSv4 MLA -- additive entry point. Does NOT touch any existing
#   gqa_ratio=64, sub_Q=64, page_size=1, q_dtype=fp8, kv_dtype=fp8
# ---------------------------------------------------------------------------
def mla_decode_fwd_v4_nm(
    q,  # [total_query_len, num_heads, head_size]   FP8 packed Q+e8m0
    qrope,  # [total_query_len, num_heads, kv_rotary]   BF16
    kv_buffer,  # [num_page, page_size, num_kv_heads, dim_qk_packed]
    kvrope,  # [num_page, page_size, num_kv_heads, kv_rotary]
    output,  # [total_query_len, num_heads, v_head_dim]  BF16 (used for out_16_nosplit=1)
    qo_indptr,  # [num_seqs+1]
    kv_indptr,  # [num_seqs+1]
    kv_page_indices,  # [num_page_used]
    max_seqlen_q,
    *,
    sink,  # REQUIRED [num_heads] FP32 -- attention sink logit
    split_indptr=None,  # [num_seqs+1] int32; auto-built by get_meta_param if None
    sm_scale=None,  # ignored on v4 nm; kernel hardcodes 1/sqrt(512)
    out_16_nosplit=0,  # DEPRECATED/IGNORED: derived internally from num_kv_splits (V3-style)
    num_kv_splits=None,  # None -> auto-pick via V3 get_meta_param heuristic
    logits=None,
    attn_lse=None,
    kv_last_page_lens=None,  # [num_seqs] int32; OPTIONAL, defaults None. Unused on
    # the nm path (page_size=1 -> kv_seq_len comes from the token-level kv_indptr).
    # None flows through to a nullptr kernarg; the host guards the deref and the
    # kernel never loads through it, so no buffer is allocated.
):
    """v4 MLA decode forward.

    Routes through the canonical aiter JIT C-ABI module
    `module_mla_v4_asm` (csrc/py_itfs_cu/asm_mla_v4.cu). Returns
    `(logits, attn_lse)` -- both 4D, in **kernel-native layout**:
        logits:   [total_q, num_kv_splits, num_heads, v_head_dim]   FP32
        attn_lse: [total_q, num_kv_splits, num_heads, 1]            FP32
    where `total_q = num_seqs * max_seqlen_q` and
    `num_heads = num_kv_heads * gqa_ratio` (gqa-flattened, mirrors V3
    convention).

    Single-pass mode (`num_kv_splits == 1`):
      V3-style: the kernel ALWAYS writes the final packed-BF16 result straight
      into `output` (zero-copy). `out_16_nosplit` is derived internally (=1
      here) and the caller-facing arg is ignored. The returned `logits` is a
      BF16 view aliased onto `output` (shape [total_q, 1, num_heads,
      v_head_dim]); read the per-token result from `output` (or `logits[:, 0]`,
      same bytes). There is no separate FP32 partial to reduce.

    Split-count selection (`num_kv_splits`):
      `None` (default) auto-picks the split count via V3's `get_meta_param`
      heuristic (CU-occupancy x HBM-efficiency, capped by the fp8 min block
      for the kernel tile) -- identical to `mla_decode_fwd`'s non-persistent
      path. Pass an explicit int to override. Note V4 nm is always
      non-persistent, so only that branch of `get_meta_param` applies.

    Multi-pass mode (`num_kv_splits > 1`):
      1. If `split_indptr` is None, build a uniform one:
         `[0, N, 2N, ..., bs*N]` so every seq uses all N passes. When
         `num_kv_splits` is auto-picked, `get_meta_param` returns this
         indptr directly.
      2. Force `out_16_nosplit = 0` -- the kernel's bf16-direct-write path
         does not support multi-pass.
      3. After the kernel returns FP32 partials, run V3's
         `_fwd_kernel_stage2_asm` triton kernel which performs the
         FlashAttention LSE merge across the `num_kv_splits` axis and
         writes the merged result directly into the BF16 `output` tensor.

      `sink` (REQUIRED, keyword-only):
      Per-Q-head attention sink logit, total `num_heads` FP32 values
      (= `num_kv_heads * gqa_ratio`). Adds a virtual K-column of logit
      `sink[h]` to the softmax denominator for head `h`; the same
      value is shared across ALL q_token positions.

    """
    num_seqs = qo_indptr.shape[0] - 1
    num_heads = q.size(1)
    v_head_dim = output.size(2)
    num_kv_heads = kv_buffer.size(2)
    gqa_ratio = num_heads // num_kv_heads

    # ---- sink shape / dtype validation -----------------------------------
    expected_sink_size = num_heads
    if sink.dtype != dtypes.fp32:
        raise ValueError(
            f"mla_decode_fwd_v4_nm: `sink` must be FP32, got {sink.dtype}."
        )
    if not sink.is_contiguous():
        raise ValueError(
            f"mla_decode_fwd_v4_nm: `sink` must be contiguous (the kernel "
            f"reads it as a flat fp32 buffer). Got strides={sink.stride()}."
        )
    if sink.numel() != expected_sink_size:
        raise ValueError(
            f"mla_decode_fwd_v4_nm: `sink` numel {sink.numel()} != expected "
            f"{expected_sink_size} (= num_heads = "
            f"num_kv_heads({num_kv_heads}) * gqa_ratio({gqa_ratio})). "
            f"Accepted shapes: ({num_heads},) flat or "
            f"({num_kv_heads}, {gqa_ratio}) 2D row-major. Under-sized "
            f"buffers silently OOB into HBM padding. "
        )
    if sink.device != q.device:
        raise ValueError(
            f"mla_decode_fwd_v4_nm: `sink` must live on the same device as "
            f"`q` (got sink={sink.device}, q={q.device})."
        )

    # ---- V3-style split-count + split_indptr resolution -------------------
    # Port of mla_decode_fwd's non-persistent branch (line ~204), adapted to
    # V4 nm's stricter buffer contract. V4 nm is always non-persistent, so we
    # only need that branch (no work_meta_data / persistent path).
    #
    #   num_kv_splits is None  -> auto-pick via get_meta_param's
    #       CU-occupancy x HBM-efficiency heuristic (and take the matching
    #       uniform indptr it returns, an int32/torch.int cuda tensor).
    #   num_kv_splits given     -> RESPECT it verbatim. The caller may have
    #       pre-allocated logits/attn_lse sized to exactly this split count
    #       (see test_v4_nm_multi_split_covers_full_kv), so we must NOT route
    #       an explicit value back through get_meta_param's fp8 cap (which
    #       could shrink it and desync the buffer shapes). Only synthesize a
    #       uniform split_indptr if the caller didn't pass one.
    total_kv = kv_page_indices.shape[0]
    if num_kv_splits is None or split_indptr is None:
        tg_factor = max(1, -(-num_heads // 64))  # ceil(num_heads / 64)
        # v4 nm forces occupancy-only split selection: ignore total_kv so the
        # split count is driven purely by CU occupancy (drops the avg_kv HBM
        # term + fp8 min-block cap in get_meta_param).
        ignore_total_kv = 1
        meta_num_kv_splits, meta_split_indptr = get_meta_param(
            num_kv_splits,
            num_seqs,
            total_kv,
            num_heads,
            max_seqlen_q,
            q.dtype,
            tg_factor,
            ignore_total_kv,
        )
        if num_kv_splits is None:
            num_kv_splits = meta_num_kv_splits
        if split_indptr is None:
            split_indptr = meta_split_indptr.to(device=q.device, dtype=torch.int32)

    out_16_nosplit = 1 if num_kv_splits == 1 else 0

    # ---- Allocate splitData / splitLse in KERNEL-NATIVE layout --------------
    # kernel-native layout:   [num_seqs, max_seqlen_q, num_kv_splits, num_heads, v_head_dim]
    # where num_heads = num_kv_heads * gqa_ratio (the gqa-flattened head dim
    # used by V3's `_fwd_kernel_stage2_asm` triton merge).
    #
    # The reference host harness then transposes `(max_seqlen_q,
    # num_kv_splits)` *before* its host-side reduce so
    # the public-facing dump shape becomes
    #   [num_seqs, num_kv_splits, num_kv_heads, max_seqlen_q*gqa_ratio, v_head_dim]
    # We *don't* do that reorder here. Instead we keep the kernel-native
    # layout and feed it straight into `_fwd_kernel_stage2_asm` which
    # takes explicit per-axis stride args.
    #
    # The 4D contiguous view `[total_q, num_kv_splits, num_heads, dv]` is
    # bit-equivalent to the kernel's native write order because:
    #   total_q (= num_seqs * max_seqlen_q) is contiguous-row-major over
    #   (batch, q_logical), exactly matching the kernel's outer two axes.
    total_q = num_seqs * max_seqlen_q
    num_heads = num_kv_heads * gqa_ratio  # gqa-flattened head dim

    expected_logits_shape = (total_q, num_kv_splits, num_heads, v_head_dim)
    expected_lse_shape = (total_q, num_kv_splits, num_heads, 1)
    if out_16_nosplit != 0:
        # V3-style zero-copy final output. When out_16_nosplit=1 (single-pass),
        # the kernel writes the final DENSELY-PACKED BF16 result into ptr_R
        # (= the `logits`/splitData buffer).
        logits = output.view(expected_logits_shape)
    elif logits is None:
        logits = torch.empty(
            expected_logits_shape,
            dtype=dtypes.fp32,
            device=q.device,
        )
    elif tuple(logits.shape) != expected_logits_shape:
        raise ValueError(
            f"mla_decode_fwd_v4_nm: caller-provided `logits` has shape "
            f"{tuple(logits.shape)}, expected {expected_logits_shape} "
            f"(kernel-native layout: [total_q, num_kv_splits, num_heads, v_head_dim] "
            f"with total_q=num_seqs*max_seqlen_q and num_heads=num_kv_heads*gqa_ratio). "
            f"NOTE: this differs from the older 5D shape "
            f"[num_seqs, num_kv_splits, num_kv_heads, max_seqlen_q*gqa_ratio, v_head_dim]; "
            f"the wrapper now keeps the kernel-native layout to skip a permute+copy "
            f"and feed `_fwd_kernel_stage2_asm` directly. Pass logits=None to let "
            f"the wrapper allocate, or update your shape to the new layout."
        )
    if attn_lse is None:
        attn_lse = torch.empty(
            expected_lse_shape,
            dtype=dtypes.fp32,
            device=q.device,
        )
    elif tuple(attn_lse.shape) != expected_lse_shape:
        raise ValueError(
            f"mla_decode_fwd_v4_nm: caller-provided `attn_lse` has shape "
            f"{tuple(attn_lse.shape)}, expected {expected_lse_shape}. See the "
            f"note in the `logits` shape error above for details."
        )

    # softmax_scale is ignored by the v4 nm kernel (hardcodes 1/sqrt(512));
    # we still pass *something* through to satisfy the C ABI.
    sm_scale_arg = 0.0 if sm_scale is None else float(sm_scale)

    # Per-seq valid split count writeback buffer (mirrors the stage1 path above).
    # Left uninitialized: the valid-split-exporting decode kernel writes the real
    # (valid) count before stage2 reads it, so pre-filling is dead work.
    valid_split_count = torch.empty((num_seqs,), dtype=dtypes.i32, device=q.device)

    use_valid_split_count_reduce = int(num_kv_splits > 1)

    aiter.mla_decode_v4_asm(
        q,
        qrope,
        kv_buffer,
        kvrope,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        split_indptr,
        sink,
        max_seqlen_q,
        sm_scale_arg,
        int(out_16_nosplit),
        int(num_kv_splits),
        logits,
        attn_lse,
        output,
        valid_split_count,
        use_valid_split_count_reduce,
        kv_last_page_lens,  # tail: unused on nm path, None -> nullptr
    )

    # ---- Cross-split FlashAttention merge via _fwd_kernel_stage2_asm ------
    if num_kv_splits > 1:
        device = logits.device
        Lv = v_head_dim
        BLOCK_DV = triton.next_power_of_2(Lv)
        # mgc = reduce's per-split KV granularity. On gfx950 the v4 nm decode
        # tiles at SUB_KV=32, so mgc must be 32 (not 64): otherwise cdiv(kv, mgc)
        # floors below the per-seq valid split count the kernel exports and drops
        # a short seq's splits (that mismatch is what previously required the host
        # ragged rebuild). Other archs keep the original mgc=64 for the gqa16/8
        # single-token tile.
        if max_seqlen_q == 1 and num_heads in (8, 16):
            mgc = 32 if get_gfx() == "gfx950" else 64
        else:
            mgc = 16

        final_lse_buf = torch.empty((1,), dtype=dtypes.fp32, device=device)

        _fwd_kernel_stage2_asm[(num_seqs, num_heads)](
            logits,  # Mid_O   [total_q, num_kv_splits, num_heads, dv]
            attn_lse,  # Mid_lse [total_q, num_kv_splits, num_heads, 1]
            output,  # final O [total_q, num_heads, dv]   BF16
            final_lse_buf,
            qo_indptr,
            kv_indptr,
            kv_last_page_lens,
            split_indptr,  # num_kv_splits_indptr
            valid_split_count,  # [num_seqs] i32; written by the asm kernel
            attn_lse.stride(0),  # stride_mid_ob = num_kv_splits * num_heads
            attn_lse.stride(2),  # stride_mid_oh = 1
            attn_lse.stride(1),  # stride_mid_os = num_heads
            output.stride(0),  # stride_obs    = num_heads * v_head_dim
            output.stride(1),  # stride_oh     = v_head_dim
            0,  # stride_lse_bs (unused, HAS_FINAL_LSE=False)
            page_size=1,  # v4 nm KV cache is page_size=1
            KV_INDPTR_IS_PAGE_LEVEL=False,  # page_size=1 -> token-level indptr
            MAYBE_FINAL_OUT=True,
            HAS_FINAL_LSE=False,
            USE_VALID_SPLIT_COUNT_REDUCE=int(num_kv_splits > 1),
            BATCH_NUM=num_seqs,
            BLOCK_DV=BLOCK_DV,
            Lv=Lv,
            mgc=mgc,
            # One (token, head) per CTA, no cross-lane reduction: the merge is
            # memory-latency bound, so max occupancy (num_warps=1) beats wider
            # tiles. Measured 1.7-1.9x over num_warps=4 on gfx950 decode shapes.
            num_warps=1,
            num_stages=2,
            waves_per_eu=4,
        )

    return logits, attn_lse
