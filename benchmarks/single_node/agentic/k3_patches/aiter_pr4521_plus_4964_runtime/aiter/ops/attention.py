# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
import triton
import triton.language as tl

from aiter import dtypes
from aiter.ops.enum import Enum, MlaVersion, QuantType
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
from csrc.cpp_itfs.pa.pa import paged_attention_rocm as paged_attention_rocm_core
from csrc.cpp_itfs.pa.pa_ragged import (
    paged_attention_ragged as paged_attention_ragged_core,
)
from csrc.cpp_itfs.pa.pa_v1 import paged_attention_v1 as paged_attention_v1_core
from csrc.cpp_itfs.torch_utils import direct_register_custom_op

from ..jit.core import compile_ops, is_experimental_enabled
from ..jit.utils.chip_info import get_cu_num, get_gfx

MD_NAME = "module_attention"

direct_register_custom_op(
    "pa_decode_gluon",
    pa_decode_gluon,
    ["output", "exp_sums", "max_logits", "temporary_output"],
)


def gen_pa_fwd_native_fake(
    # [num_seqs, num_heads, head_size]
    query: torch.Tensor,
    # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    key_cache: torch.Tensor,
    # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache: torch.Tensor,
    # [num_seqs, max_num_blocks_per_seq]
    block_tables: torch.Tensor,
    # [num_seqs]
    context_lens: torch.Tensor,
    k_dequant_scales: torch.Tensor,
    v_dequant_scales: torch.Tensor,
    max_seq_len: int,
    num_kv_heads: int,
    scale_s: float,
    scale_k: float,
    scale_v: float,
    block_size: int,
    quant_algo: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    else:
        return torch.empty_like(query)


def gen_pa_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    max_qlen: int = 1,
    K_QScale: torch.Tensor | None = None,
    V_QScale: torch.Tensor | None = None,
    out_: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    high_precision: (
        int | None
    ) = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: str | None = None,
):
    if out_ is not None:
        return out_
    else:
        return torch.empty_like(Q)


@compile_ops("module_attention", gen_fake=gen_pa_fwd_native_fake)
def pa_fwd_naive(
    # [num_seqs, num_heads, head_size]
    query: torch.Tensor,
    # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    key_cache: torch.Tensor,
    # [num_blocks, num_kv_heads, head_size, block_size]
    value_cache: torch.Tensor,
    # [num_seqs, max_num_blocks_per_seq]
    block_tables: torch.Tensor,
    # [num_seqs]
    context_lens: torch.Tensor,
    k_dequant_scales: torch.Tensor,
    v_dequant_scales: torch.Tensor,
    max_seq_len: int,
    num_kv_heads: int,
    scale_s: float,
    scale_k: float,
    scale_v: float,
    block_size: int,
    quant_algo: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor: ...


@compile_ops(
    "module_attention_asm", fc_name="pa_fwd", ffi_type="ctypes", gen_fake=gen_pa_fwd_asm
)
def _pa_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    max_qlen: int = 1,
    K_QScale: torch.Tensor | None = None,
    V_QScale: torch.Tensor | None = None,
    out_: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    high_precision: int | None = 1,
    kernelName: str | None = None,
) -> None: ...


def pa_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    max_qlen: int = 1,
    K_QScale: torch.Tensor | None = None,
    V_QScale: torch.Tensor | None = None,
    out_: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    high_precision: (
        int | None
    ) = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: str | None = None,
) -> torch.Tensor:
    output = out_ if out_ is not None else torch.empty_like(Q)
    _pa_fwd_asm(
        Q,
        K,
        V,
        block_tables,
        context_lens,
        block_tables_stride0,
        max_qlen,
        K_QScale,
        V_QScale,
        output,
        qo_indptr,
        high_precision,
        kernelName,
    )
    return output


def _should_use_asm_kernel(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    kv_cache_tensor_dtype: torch.dtype,
    high_precision: int,
) -> bool:
    # ASM kernel only supports head_size == 128; all other head sizes use HIP.
    if head_size != 128:
        return False

    # high_precision == 2 forces ASM for maximum precision (fp8 kvcache only)
    if high_precision == 2:
        return True

    # int8 kv cache always uses ASM
    if kv_cache_tensor_dtype == torch.int8:
        return True

    # Get GPU compute units (CUs)
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    # ASM kernel becomes relevant, once the total_heads is sufficiently large compared to CUs
    total_heads = num_seqs * num_heads
    return total_heads > 2 * cu_num


def paged_attention_common(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    scale: float,
    max_qlen: int = 1,
    max_seq_len: int = 1,
    K_QScale_hip: torch.Tensor | None = None,  # [num_seqs, num_heads]
    V_QScale_hip: torch.Tensor | None = None,
    K_QScale_asm: torch.Tensor | None = None,  # [num_blocks, num_kv_heads, block_size]
    V_QScale_asm: torch.Tensor | None = None,
    out_: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    high_precision: (
        int | None
    ) = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: str | None = None,
    kv_cache_dtype: str = "auto",
    kv_cache_tensor_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Paged attention forward pass with automatic kernel selection.
    ASM is favored for int8 kv caches, for short ctx_len, or when the workload exceeds
    the heuristic thresholds for larger ctx_len values.
    PA is normally using per tensor quant and this is what has been tested, however,
    per head quant can be supported as well in principle, but not tested.
    """
    kv_cache_tensor_dtype = (
        kv_cache_tensor_dtype if kv_cache_tensor_dtype is not None else K.dtype
    )
    num_seqs, num_heads, head_size = Q.shape

    use_asm_kernel = _should_use_asm_kernel(
        num_seqs, num_heads, head_size, kv_cache_tensor_dtype, high_precision
    )

    if use_asm_kernel:
        output = pa_fwd_asm(
            Q,
            K,
            V,
            block_tables,
            context_lens,
            block_tables_stride0,
            max_qlen,
            K_QScale_asm,
            V_QScale_asm,
            out_,
            qo_indptr,
            high_precision,
            kernelName,
        )
        return output

    # Use ROCm paged attention kernel for smaller workloads / common path.
    output = out_ if out_ is not None else torch.empty_like(Q)

    paged_attention_rocm(
        out=output,
        exp_sums=exp_sums,
        max_logits=max_logits,
        tmp_out=tmp_out,
        query=Q,
        key_cache=K,
        value_cache=V,
        num_kv_heads=int(K.size(1)),
        scale=scale,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=int(K.size(3)),
        max_context_len=max_seq_len,
        alibi_slopes=None,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=K_QScale_hip,
        v_scale=V_QScale_hip,
        fp8_out_scale=None,
        partition_size=256,
        mtp=1,
        q_scale=None,
    )
    return output


def gen_pa_ps_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,  # better have ?
    max_qlen: int = 1,
    K_QScale: torch.Tensor | None = None,
    V_QScale: torch.Tensor | None = None,
    out_: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    # work_meta_data: Optional[torch.Tensor] = None,
    work_indptr: torch.Tensor | None = None,
    work_info: torch.Tensor | None = None,
    splitData: torch.Tensor | None = None,
    splitLse: torch.Tensor | None = None,
    high_precision: (
        int | None
    ) = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: str | None = None,
    quant_type: Enum | None = QuantType.per_Token.value,
) -> torch.Tensor:
    if out_ is not None:
        return out_
    else:
        return torch.empty_like(Q)


@compile_ops(
    "module_attention_asm",
    fc_name="pa_ps_fwd",
    ffi_type="ctypes",
    gen_fake=gen_pa_ps_fwd_asm,
)
def _pa_ps_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    max_qlen: int = 1,
    K_QScale: torch.Tensor | None = None,
    V_QScale: torch.Tensor | None = None,
    out_: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    work_indptr: torch.Tensor | None = None,
    work_info: torch.Tensor | None = None,
    splitData: torch.Tensor | None = None,
    splitLse: torch.Tensor | None = None,
    mask: int = 0,
    high_precision: int | None = 1,
    kernelName: str | None = None,
    quant_type: Enum | None = QuantType.per_Token.value,
) -> None: ...


def pa_ps_fwd_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    max_qlen: int = 1,
    K_QScale: torch.Tensor | None = None,
    V_QScale: torch.Tensor | None = None,
    out_: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    work_indptr: torch.Tensor | None = None,
    work_info: torch.Tensor | None = None,
    splitData: torch.Tensor | None = None,
    splitLse: torch.Tensor | None = None,
    mask: int = 0,
    high_precision: (
        int | None
    ) = 1,  # [0, 1, 2] 2 is the highest precision, this is only for fp8 kvcache
    kernelName: str | None = None,
    quant_type: Enum | None = QuantType.per_Token.value,
) -> torch.Tensor:
    output = out_ if out_ is not None else torch.empty_like(Q)
    _pa_ps_fwd_asm(
        Q,
        K,
        V,
        kv_indptr,
        kv_page_indices,
        context_lens,
        softmax_scale,
        max_qlen,
        K_QScale,
        V_QScale,
        output,
        qo_indptr,
        work_indptr,
        work_info,
        splitData,
        splitLse,
        mask,
        high_precision,
        kernelName,
        quant_type,
    )
    return output


# ---------------------------------------------------------------------------
# pa_decode_bf16_asm (gfx1250) -- persistent / split-KV paged-attention decode.
#
# Wraps the SP3 kernel PA_DECODE_D64_1TG_4W_PS (head_dim=64, page_size=256,
# gqa=8).  FP8 Q **and** FP8 paged KV cache, bf16 output, **per-tensor** scalar
# dequant scales for Q/K/V (distinct from the per-token/per-block scale tensors
# used by pa_ps_fwd_asm).  GPT-OSS style attention sink (per-Q-head fp32 logits
# in the SCALED-logit domain, exp(sink); kernel divides by s_eff internally) is
# always read by the kernel.
#
# Memory-allocation policy: all GPU tensors are allocated on the Python side;
# the C++ entry point performs only pointer + stride bookkeeping and the kernel
# launch (no torch dependency).  The public wrapper `pa_decode_bf16_asm` below
# handles output/scale/sink allocation and folds the attention softmax scale
# into key_scale (matching the reference host file sched2/pa_ps.cpp).
# ---------------------------------------------------------------------------
@compile_ops(
    "module_pa_decode_bf16_asm",
    fc_name="pa_decode_bf16_asm",
    ffi_type="ctypes",
)
def _pa_decode_bf16_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_indices: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    out: torch.Tensor,
    qo_indptr: torch.Tensor | None,
    kv_indptr: torch.Tensor,
    work_indptr: torch.Tensor | None,
    work_info: torch.Tensor | None,
    split_o: torch.Tensor | None,
    split_lse: torch.Tensor | None,
    sink: torch.Tensor,
    gqa: int,
    mtp: int,
    kernelName: str | None,
) -> None: ...


def pa_decode_bf16_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    kv_indices: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    kv_indptr: torch.Tensor,
    gqa: int = 8,
    mtp: int = 0,
    query_scale: torch.Tensor | None = None,
    key_scale: torch.Tensor | None = None,
    value_scale: torch.Tensor | None = None,
    qo_indptr: torch.Tensor | None = None,
    work_indptr: torch.Tensor | None = None,
    work_info: torch.Tensor | None = None,
    split_o: torch.Tensor | None = None,
    split_lse: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    kernelName: str | None = None,
) -> torch.Tensor:
    """Public wrapper for the gfx1250 PA decode kernel.

    Contract details:
      * `Q`/`K`/`V` are FP8; `out` is bf16 with Q's logical shape.
      * `query_scale`/`key_scale`/`value_scale` are the per-tensor FP8 dequant
        scales as 1-element fp32 tensors (None means 1.0); the attention
        `softmax_scale` (typically 1/sqrt(head_dim)) is
        passed BY VALUE (kernarg 0x60) and the kernel forms
        scl_log2e = query_scale * key_scale * softmax_scale * log2e.
      * `sink` (optional) holds per-Q-head fp32 logits in the SCALED-logit
        domain (exp(sink), Triton/GPT-OSS convention; the kernel divides by
        s_eff internally), shape [kv_head_num * gqa].  The kernel always reads
        this slot, so when `sink` is None a -inf buffer is allocated, making the
        sink a numerical no-op.
    """
    device = Q.device
    kv_head_num = K.shape[1]
    q_head_num = kv_head_num * gqa

    if out is None:
        out = torch.empty(Q.shape, dtype=torch.bfloat16, device=device)

    # query/key/value_scale are 1-element fp32 dequant scales, passed straight to
    # the kernel. softmax_scale is passed BY VALUE (kernarg 0x60); the kernel
    # applies it, so do NOT pre-fold it into key_scale.

    if sink is None:
        # The kernel is compiled sink-enabled (always reads + merges the sink
        # slot), so default to a FINITE large-negative buffer (numerical no-op:
        # exp2((sink-max)*scl) underflows to 0) rather than -inf, which can
        # produce inf/NaN in the in-kernel sink merge.
        sink = torch.full((q_head_num,), -1.0e30, dtype=torch.float32, device=device)
    else:
        assert sink.dtype == torch.float32, "sink must be in fp32 for pa ASM"

    _pa_decode_bf16_asm(
        Q,
        K,
        V,
        kv_indices,
        context_lens,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        out,
        qo_indptr,
        kv_indptr,
        work_indptr,
        work_info,
        split_o,
        split_lse,
        sink,
        gqa,
        mtp,
        kernelName,
    )
    return out


def pa_reduce_v1(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor | None,
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    final_lse: torch.Tensor | None = None,
    # num_kv_splits is trailing+optional so the ATOM call site (which passes 8
    # positional args, no split count) stays aligned. The kernel uses
    # max(SM_count, num_kv_splits), so the default 0 means "auto" (SM_count).
    num_kv_splits: int = 0,
) -> None:
    mla_reduce_v1(
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_q,
        num_kv_splits,
        final_output,
        final_lse,
    )


def pa_persistent_fwd(
    Q: torch.Tensor,  # [sum_qlen, kv_heads * gqa + kv_heads * 2, head_dim]
    K: torch.Tensor,  # [num_blocks, kv_heads, head_dim / x, block_size, x]
    V: torch.Tensor,  # [num_blocks, kv_heads, block_size / x, head_dim, x]
    output: torch.Tensor,
    max_qlen: int,  # default = 1
    qo_indptr: torch.Tensor,  # [batch+1], qolen prefix sum
    kv_indptr: torch.Tensor,  # [batch+1], kv_used_pages prefix sum
    kv_indices: torch.Tensor,  # [sum_kv_used_pages], packed kv ids
    context_lens: torch.Tensor,  # [batch]
    # work_meta_data: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    K_QScale: torch.Tensor | None = None,  # [num_blocks, kv_heads, block_size]
    V_QScale: torch.Tensor | None = None,  # [num_blocks, kv_heads, block_size]
    softmax_scale: float | None = None,
    mask: int = 0,
    quant_type: QuantType = QuantType.per_Token,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = Q.device
    total_s, nhead, v_head_dim = output.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / (v_head_dim**0.5)
    logits = torch.empty(
        (reduce_partial_map.size(0) * max_qlen, 1, nhead, v_head_dim),
        dtype=dtypes.fp32,
        device=device,
    )
    splitLse = torch.empty(
        (reduce_partial_map.size(0) * max_qlen, 1, nhead, 1),
        dtype=dtypes.fp32,
        device=device,
    )
    final_lse = torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)

    pa_ps_fwd_asm(
        Q,
        K,
        V,
        kv_indptr,
        kv_indices,
        context_lens,
        softmax_scale,
        max_qlen,
        K_QScale,
        V_QScale,
        output,
        qo_indptr,
        work_indptr,
        work_info,
        logits,
        splitLse,
        mask,
        quant_type=quant_type,
    )
    pa_reduce_v1(
        logits,
        splitLse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_qlen,
        output,
        final_lse,
    )

    return logits, final_lse


def paged_attention_rocm(
    out: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: torch.Tensor | None = None,
    partition_size: int = 256,
    mtp: int = 1,
    q_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    paged_attention_rocm_core(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        fp8_out_scale,
        partition_size,
        mtp,
        q_scale,
    )
    return out


direct_register_custom_op(
    "paged_attention_rocm",
    paged_attention_rocm,
    ["out", "exp_sums", "max_logits", "tmp_out"],
)


def paged_attention_v1(
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    cu_query_lens: torch.Tensor | None,
    context_lens: torch.Tensor,
    max_context_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: torch.Tensor | None = None,
    partition_size: int = 256,
    mtp: int = 1,
    sliding_window: int = 0,
) -> torch.Tensor:
    paged_attention_v1_core(
        out,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        cu_query_lens,
        context_lens,
        max_context_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale,
        partition_size,
        mtp,
        sliding_window=sliding_window,
    )
    return out


direct_register_custom_op(
    "paged_attention_v1",
    paged_attention_v1,
    ["out", "workspace_buffer"],
)


def paged_attention_ragged(
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    block_size: int,
    max_num_partitions: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: torch.Tensor | None = None,
    partition_size: int = 256,
    mtp: int = 1,
) -> torch.Tensor:
    paged_attention_ragged_core(
        out,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        block_size,
        max_num_partitions,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale,
        partition_size,
        mtp,
    )
    return out


direct_register_custom_op(
    "paged_attention_ragged",
    paged_attention_ragged,
    ["out", "workspace_buffer"],
)


MD_NAME = "module_mla_asm"


@compile_ops(MD_NAME, ffi_type="ctypes")
def mla_decode_stage1_asm_fwd(
    # [num_seqs, num_heads, head_size]
    Q: torch.Tensor,
    # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    KV: torch.Tensor,
    # [batch_size+1]
    qo_indptr: torch.Tensor,
    # [batch_size+1]
    kv_indptr: torch.Tensor,
    # [num_page_used]
    kv_page_indices: torch.Tensor,
    # [batch_size]
    kv_last_page_lens: torch.Tensor,
    num_kv_splits_indptr: torch.Tensor | None,
    work_meta_data: torch.Tensor | None,
    work_indptr: torch.Tensor | None,
    work_info_set: torch.Tensor | None,
    max_seqlen_q: int,
    page_size: int,
    nhead_kv: int,
    softmax_scale: float,
    # [batch_size, num_kv_splits, num_heads, v_head_dim]
    splitData: torch.Tensor,
    # [batch_size, num_kv_splits, num_heads,  1]
    splitLse: torch.Tensor,
    output: torch.Tensor,
    # [batch_size, num_heads, v_head_dim]
    lse: torch.Tensor | None = None,
    # [1] per-tensor
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    # round-robin context-parallel (CP) extension:
    #   g_kv_indptr   : [batch_size+1] GLOBAL kv_indptr (per-request global KV length)
    #   cp_world_size : number of CP ranks (W); 1 == disabled
    #   cp_rank       : this rank id (r); local kv idx j -> global pos j*W + r
    g_kv_indptr: torch.Tensor | None = None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    # [batch_size] scratch for gfx1250 packed MLA kernels
    valid_split_count: torch.Tensor | None = None,
    use_valid_split_count_reduce: int = 0,
) -> None: ...


MD_NAME_V4 = "module_mla_v4_asm"


@compile_ops(MD_NAME_V4, ffi_type="ctypes")
def mla_decode_v4_asm(
    # [total_query_len, num_heads, head_size]   FP8 packed Q + e8m0 scale region
    Q: torch.Tensor,
    # [total_query_len, num_heads, kv_rotary]   BF16
    qrope: torch.Tensor,
    # [num_page, page_size, num_kv_heads, head_size]  FP8
    KV: torch.Tensor,
    # [num_page, page_size, num_kv_heads, kv_rotary]  BF16
    kvrope: torch.Tensor,
    # [num_seqs+1]
    qo_indptr: torch.Tensor,
    # [num_seqs+1]
    kv_indptr: torch.Tensor,
    # [num_page_used]
    kv_page_indices: torch.Tensor,
    # [num_seqs+1]
    split_indptr: torch.Tensor,
    # [num_heads] FP32 -- attention sink logit. Loaded by the kernel via
    # kernarg slot 18 (byte offset 0x120). Caller must ALWAYS pass a real
    # tensor; there is no nullable-sink convention on the C ABI. Pass
    # torch.full((num_heads,), float("-inf")) for "no sink" math.
    sink: torch.Tensor,
    max_seqlen_q: int,
    # ignored on v4 nm; kernel hardcodes 1/sqrt(kV4DimNope+kV4DimRope)=1/sqrt(512)
    softmax_scale: float,
    # 0 = fp32 split-out path; 1 = bf16 nosplit reduce path
    out_16_nosplit: int,
    num_kv_splits: int,
    # outputs
    # [num_seqs, num_kv_splits, num_kv_heads, gqa*max_seqlen_q, v_head_dim] FP32
    splitData: torch.Tensor,
    # [num_seqs, num_kv_splits, num_kv_heads, gqa*max_seqlen_q, 1]          FP32
    splitLse: torch.Tensor,
    # [total_query_len, num_heads, v_head_dim] BF16 (used when out_16_nosplit==1)
    output: torch.Tensor,
    # [num_seqs] int32 scratch for gfx1250 packed MLA kernels. Holds the
    # per-request valid kv-split count the kernel writes (slot 19). Pass a
    # real tensor when use_valid_split_count_reduce != 0; otherwise the
    # kernel skips the write and nullptr is fine.
    valid_split_count: torch.Tensor | None = None,
    use_valid_split_count_reduce: int = 0,
    # [num_seqs] int32. Unused on the v4 nm path (page_size=1 -> the kernel derives
    # kv_seq_len from the token-level kv_indptr). Optional/nullable: None sends a
    # nullptr; the host guards the deref (asm_mla_v4.cu) and the kernel never loads
    # through it. Placed at the tail because it carries no data on this path.
    kv_last_page_lens: torch.Tensor | None = None,
) -> None: ...


@compile_ops(MD_NAME, ffi_type="ctypes")
def mla_prefill_asm_fwd(
    # [num_seqs, num_heads, head_size]
    Q: torch.Tensor,
    # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    KV: torch.Tensor,
    # [batch_size+1]
    qo_indptr: torch.Tensor,
    # [batch_size+1]
    kv_indptr: torch.Tensor,
    # [num_page_used]
    kv_page_indices: torch.Tensor,
    # [batch_size]
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    # [batch_size, num_kv_splits, num_heads, v_head_dim]
    splitData: torch.Tensor,
    # [batch_size, num_kv_splits, num_heads,  1]
    splitLse: torch.Tensor,
) -> None: ...


def get_pa_metadata_info_v1(
    batch_size: int,
    num_head_k: int = 1,
):
    """
    Returns:
        1. Shape of work_metadata_ptrs followed by its scalar type.
        2. Shape of work_indptr followed by its scalar type.
        3. Shape of work_info_set followed by its scalar type.
        4. Shape of reduce_indptr followed by its scalar type.
        5. Shape of reduce_final_map followed by its scalar type.
        6. Shape of reduce_partial_map followed by its scalar type.
    """

    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count

    tile_cnt = batch_size
    max_work = (tile_cnt + cu_num - 1) * num_head_k
    max_split_tiles = min(batch_size + cu_num - 1, (cu_num - 1) * 2)

    return (
        ((2), torch.uint64),  # work_metadata_ptrs
        ((cu_num + 1), torch.int32),  # work_indptr
        ((max_work, 8), torch.int32),  # work_info_set
        ((tile_cnt + 1), torch.int32),  # reduce_indptr
        ((tile_cnt, 2), torch.int32),  # reduce_final_map
        (max_split_tiles, torch.int32),  # reduce_partial_map
    )


@compile_ops("module_pa_metadata")
def get_pa_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    pages_kv_indptr: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    kv_granularity: int = 16,
    block_size: int = 16,
    max_seqlen_qo: int = -1,
    uni_seqlen_qo: int = -1,
    fast_mode: bool = True,
    topk: int = -1,
    max_split_per_batch: int = -1,
) -> None:
    """
    Inputs:
        cumulated seqlens of q/o: (batch_size + 1), dtype torch.int32.
        cumulated used pages of k/v: (batch_size + 1), dtype torch.int32.
        context_lens: seqlens of k/v, dtype torch.int32.
        num_heads_per_head_k: Equals to num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
        is_causal: Whether causal mask is enabled.
        Options: Detailed settings for spliting. All of them are optional.
            kv_granularity: default=16. The granularity on kv sequence length when cutting batch.
            max_seqlen_qo: default=-1. Used to check lds usage and save time. value less than 1 means unknown.
            uni_seqlen_qo: default=-1. Sequence length of qo is uniform across batches. value less than 1 means the
                           length is not fixed.
            fast_mode: default=True. Whether user wants metadata become as fast as possible. Note that fast
                       mode may lead to bad overall performance.
            topk: default=-1. Top-k tokens selected for sparse attention. -1 means non-sparse attention.
    Outputs:
        [0] work_metadata_ptrs  (2)                 Two 64-bits pointers point to the 1st element of work_indptr and
                                                    work_info.
        [1] work_indptr:        (#cu_part + 1),     The IDs of work handled by each cu_part.
        [2] work_info           (#work, 8)
        [2.0] bs_index:         (#work),            The index of batch handled by each work.
        [2.1] partial_index:    (#work),            The index of tile in output buffer when splits. -1 means no split.
        [2.2] q_start:          (#work),            The global index in seq where q/o starts. Use global index here can
                                                    reduce memory access count in kernel.
        [2.3] q_end:            (#work),            The global index in seq where q/o ends (not included).
        [2.4] kv_start:         (#work),            The global index in kv_indices where k/v starts.
        [2.5] kv_end:           (#work),            The global index in kv_indices where k/v ends (not included). Note
                                                    that this value indicates the end of last qo sequence if there are
                                                    multiple qo sequences included in the current work and causal mask
                                                    is enabled.
        [2.6] kv_offset:        (#work),            Not used.
        [2.7] pad               (#work, 1),         The start index(low 16bits) and end index(high 16bits) of q heads.
        [3] reduce_indptr:      (sum(qo_seqlen_blk_count) + 1),
                                                    The IDs in reduce_partial_map indicates the tiles should be merged
                                                    together.
        [4] reduce_final_map:   (sum(qo_seqlen_blk_count)),
                                                    The final output location of each group of tiles.
        [5] reduce_partial_map: (#partial_tiles),   The locations in partial buffer of partial tiles waiting for being
                                                    reduced.
    """


def get_ps_metadata_info_v1(
    batch_size: int,
    num_head_k: int,
    max_qlen: int,
    qlen_granularity: int = 256,
):
    """
    Returns:
        1. Shape of work_metadata_ptrs followed by its scalar type.
        2. Shape of work_indptr followed by its scalar type.
        3. Shape of work_info followed by its scalar type.
        4. Shape of reduce_indptr followed by its scalar type.
        5. Shape of reduce_final_map followed by its scalar type.
        6. Shape of reduce_partial_map followed by its scalar type.
    """

    device = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device)
    cu_num = device_properties.multi_processor_count

    num_clusters = math.gcd(num_head_k, cu_num)
    cus_per_cluster = cu_num // num_clusters

    max_qo_split_per_batch = math.ceil(max_qlen / qlen_granularity)

    qo_tile_cnt = batch_size * max_qo_split_per_batch
    # TODO: consider split q to reduce max_works & max_partials
    max_works = (batch_size + cus_per_cluster - 1) * max_qo_split_per_batch * num_head_k
    max_partials = (
        min(batch_size + cus_per_cluster - 1, (cus_per_cluster - 1) * 2)
        * max_qo_split_per_batch
    )

    return (
        (2, torch.uint64),  # work_metadata_ptrs
        (cu_num + 1, torch.int32),  # work_indptr
        ((max_works, 8), torch.int32),  # work_info
        (qo_tile_cnt + 1, torch.int32),  # reduce_indptr
        ((qo_tile_cnt, 2), torch.int32),  # reduce_final_map
        (max_partials, torch.int32),  # reduce_partial_map
    )


@compile_ops("module_ps_metadata")
def get_ps_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    pages_kv_indptr: torch.Tensor,
    context_lens: torch.Tensor,
    gqa_ratio: int,
    num_heads_k: int,
    work_metadata_ptrs: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    qhead_granularity: int = 1,
    qlen_granularity: int = 256,
    kvlen_granularity: int = 16,
    block_size: int = 16,
    is_causal: bool = True,
) -> None: ...


@compile_ops(MD_NAME, ffi_type="ctypes")
def mla_prefill_ps_asm_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    work_indptr: torch.Tensor | None,
    work_info_set: torch.Tensor | None,
    max_seqlen_q: int,
    softmax_scale: float,
    is_causal: bool,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
    output: torch.Tensor,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None: ...


def get_mla_decode_fwd_occupancy(
    num_head_qo: int,
    max_seqlen_qo: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> int:
    """Occupancy of the HK MLA decode fwd kernel that will be dispatched for
    these (num_head_qo, max_seqlen_qo, dtypes). The m16x4 kernel (gfx950 +
    fp8/fp8, 64 q-tokens per tile, gated on AITER_ENABLE_EXPERIMENTAL) runs at
    occupancy=2; all other kernels run at occupancy=1.

    Used wherever code must agree with the metadata kernel's cluster count
    (which is `multiProcessorCount * occupancy / num_heads_k`):
      - get_mla_metadata_info_v1 (buffer sizing)
      - mla_decode_fwd (per-tile num_kv_splits upper bound for the reduce)
      - C++ metadata at csrc/kernels/mla/metadata/v1_2_device.cuh
    """
    is_hk_m16x4 = (
        get_gfx() == "gfx950"
        and q_dtype == dtypes.fp8
        and kv_dtype == dtypes.fp8
        and (num_head_qo * max_seqlen_qo == 64)
        and is_experimental_enabled()
    )
    return 2 if is_hk_m16x4 else 1


def get_mla_decode_fwd_max_splits(
    num_head_qo: int,
    max_seqlen_qo: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> int:
    """Upper bound on per-tile num_splits produced by the metadata kernel for
    the HK MLA decode fwd dispatch. Equals `cu_num * occupancy` (num_heads_k=1
    is assumed, matching the only configuration the HK kernels support). This
    is the value `mla_reduce_v1` needs for its LDS layout so
    `p_lds_reduce_partial_map` is sized to fit every split the fwd kernel can
    emit.
    """
    occupancy = get_mla_decode_fwd_occupancy(
        num_head_qo, max_seqlen_qo, q_dtype, kv_dtype
    )
    return get_cu_num() * occupancy


def get_mla_metadata_info_v1(
    batch_size: int,
    max_seqlen_qo: int,
    num_head_qo: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    is_sparse: bool,
    fast_mode: bool = True,
    num_kv_splits: int = 32,
    intra_batch_mode: bool = False,
    max_split_per_batch: int = -1,
):
    """
    Returns:
        1. Shape of work_metadata_ptrs followed by its scalar type.
        2. Shape of work_indptr followed by its scalar type.
        3. Shape of work_info_set followed by its scalar type.
        4. Shape of reduce_indptr followed by its scalar type.
        5. Shape of reduce_final_map followed by its scalar type.
        6. Shape of reduce_partial_map followed by its scalar type.
    """

    assert num_head_qo % 8 == 0
    max_splits = get_mla_decode_fwd_max_splits(
        num_head_qo, max_seqlen_qo, q_dtype, kv_dtype
    )

    effective_seqlen_qo = 1 if is_sparse else max_seqlen_qo
    packed_qo_len = effective_seqlen_qo * num_head_qo
    max_qo_tiles_per_batch = math.ceil(packed_qo_len / 16)

    if (
        get_gfx() == "gfx950"
        and q_dtype == dtypes.bf16
        and kv_dtype == dtypes.bf16
        and packed_qo_len >= 64
        and num_head_qo <= 64
        and (packed_qo_len < 128 or num_head_qo == 48)
    ) or (
        get_gfx() == "gfx950"
        and q_dtype == dtypes.fp8
        and kv_dtype == dtypes.fp8
        and (num_head_qo == 32)
        and (effective_seqlen_qo == 3)
    ):
        if num_head_qo * 2 > 64:
            # e.g. nhead=48: C++ does  `return seqlen_qo`  (not ceil)
            max_qo_tiles_per_batch = effective_seqlen_qo
        else:
            max_qo_tiles_per_batch = math.ceil(packed_qo_len / 64)
    elif (
        num_head_qo == 16
        or (
            get_gfx() == "gfx942"
            and num_head_qo == 128
            and kv_dtype == dtypes.fp8
            and q_dtype == dtypes.fp8
        )
        or (
            get_gfx() == "gfx942"
            and num_head_qo in (16, 32, 64)
            and num_head_qo * effective_seqlen_qo == 128
            and kv_dtype == dtypes.fp8
            and q_dtype == dtypes.fp8
            and is_experimental_enabled()
        )
        or (
            get_gfx() == "gfx950"
            and kv_dtype == dtypes.fp8
            and q_dtype == dtypes.fp8
            and (
                (num_head_qo == 32 and effective_seqlen_qo == 4)
                or (num_head_qo == 64)
                or (num_head_qo == 128)
            )
        )
        or (
            get_gfx() in ("gfx942", "gfx950")
            and num_head_qo == 64
            and q_dtype == dtypes.fp8
            and kv_dtype == dtypes.fp8
            and effective_seqlen_qo == 1
        )
    ):
        max_qo_tiles_per_batch = math.ceil(packed_qo_len / 128)
    elif (
        get_gfx() == "gfx950"
        and (packed_qo_len >= 128 or num_head_qo > 64)
        and kv_dtype == dtypes.bf16
        and q_dtype == dtypes.bf16
        and num_head_qo != 48
    ) or (
        get_gfx() == "gfx950"
        and q_dtype == dtypes.fp8
        and kv_dtype == dtypes.fp8
        and num_head_qo == 96
        and effective_seqlen_qo <= 6
    ):
        if num_head_qo * 2 > 128:
            max_qo_tiles_per_batch = effective_seqlen_qo
        else:
            max_qo_tiles_per_batch = math.ceil(packed_qo_len / 128)

    batch_size = batch_size * max_seqlen_qo if is_sparse else batch_size
    tile_cnt = batch_size * max_qo_tiles_per_batch

    if fast_mode:
        max_work = (batch_size + max_splits - 1) * max_qo_tiles_per_batch
        max_split_tiles = (
            min(batch_size + max_splits - 1, (max_splits - 1) * 2)
            * max_qo_tiles_per_batch
        )
    else:
        max_work = tile_cnt * max_splits
        max_split_tiles = tile_cnt * max_splits

    # Metadata's global split cap is `min(cu_num, max_split_per_batch * batch_size)`
    # (see csrc/kernels/mla/metadata/v1_2_device.cuh:560-562). This is a GLOBAL
    # budget shared across all tiles, so the total number of partial reduce
    # entries is bounded by the base tiles (one per tile) plus at most the global
    # split budget of EXTRA splits distributed across them:
    #     reduce_partial_map <= tile_cnt + per_tile_cap
    # The previous `tile_cnt * per_tile_cap` assumed every tile could individually
    # absorb the whole global budget simultaneously, which the shared budget
    # forbids. With cudagraph batch_size >> cu_num that product collapsed to
    # tile_cnt * cu_num (e.g. 512 * 256 = 131072), and aiter mla_decode_fwd sizes
    # its fp32 `logits` from reduce_partial_map.size(0) -> ~32 GiB OOM at capture.
    if max_split_per_batch > 0:
        per_tile_cap = min(max_splits, max_split_per_batch * batch_size)
        max_split_tiles = max(max_split_tiles, tile_cnt + per_tile_cap)

    if not intra_batch_mode:
        return (
            ((2), torch.uint64),  # work_metadata_ptrs
            ((max_splits + 1), torch.int32),  # work_indptr
            ((max_work, 8), torch.int32),  # work_info_set
            ((tile_cnt + 1), torch.int32),  # reduce_indptr
            ((tile_cnt, 2), torch.int32),  # reduce_final_map
            (max_split_tiles, torch.int32),  # reduce_partial_map
        )
    else:
        return (
            ((2), torch.uint64),  # work_metadata_ptrs
            (max_splits + 1, torch.int32),  # work_indptr
            ((tile_cnt * num_kv_splits, 8), torch.int32),  # work_info_set
            ((tile_cnt + 1), torch.int32),  # reduce_indptr
            ((tile_cnt, 2), torch.int32),  # reduce_final_map
            (tile_cnt * num_kv_splits, torch.int32),  # reduce_partial_map
        )


@compile_ops("module_mla_metadata", fc_name="get_mla_metadata_v1")
def _get_mla_metadata_v1_impl(
    seqlens_qo_indptr: torch.Tensor,
    seqlens_kv_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_info_set: torch.Tensor,
    work_indptr: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    page_size: int = 1,
    kv_granularity: int = 16,
    max_seqlen_qo: int = -1,
    uni_seqlen_qo: int = -1,
    fast_mode: bool = True,
    topk: int = -1,
    max_split_per_batch: int = -1,
    intra_batch_mode: bool = False,
    is_cp_round_robin: bool = False,
    mla_version: int = MlaVersion.V32.value,
    dtype_q_nope: torch.dtype | None = None,
    dtype_q_rope: torch.dtype | None = None,
    dtype_kv_nope: torch.dtype | None = None,
    dtype_kv_rope: torch.dtype | None = None,
) -> None:
    """Compiled binding for ``get_mla_metadata_v1`` (bound via ``fc_name``).

    Internal: call the ``get_mla_metadata_v1`` wrapper below instead, which also
    accepts the legacy ``dtype_q`` / ``dtype_kv`` aliases. This entry takes the
    per-component dtypes directly.

    Args:
        seqlens_qo_indptr: (batch_size + 1) int32. Cumulative q/o seqlens.
        seqlens_kv_indptr: (batch_size + 1) int32. Cumulative k/v page indices.
        kv_last_page_lens: (batch_size) int32. Length of the last k/v page.
        num_heads_per_head_k: num_heads_q // num_heads_k.
        num_heads_k: Number of k heads.
        is_causal: Whether the causal mask is enabled.
        work_metadata_ptrs / work_info_set / work_indptr / reduce_indptr /
            reduce_final_map / reduce_partial_map: Pre-allocated output buffers
            (see get_mla_metadata_info_v1 for shapes/dtypes) filled in place.
        page_size: Page size (default 1).
        kv_granularity: Granularity of kv page counts when splitting a batch (default 16).
        max_seqlen_qo: Max qo seqlen; used to bound LDS usage. < 1 means unknown (default -1).
        uni_seqlen_qo: Uniform qo seqlen across batches. < 1 means not fixed (default -1).
        fast_mode: Prefer fastest metadata generation; may hurt overall perf (default True).
        topk: Top-k tokens for sparse attention. -1 means non-sparse (default -1).
        max_split_per_batch: Cap on KV splits per batch. -1 means unbounded (default -1).
        intra_batch_mode: Fake non-persistent mode; same split count per batch (default False).
        is_cp_round_robin: Context-parallel round-robin work assignment (default False).
        mla_version: MLA kernel version (default V32).
        dtype_q_nope / dtype_kv_nope: Q/KV NoPE dtypes. None -> bf16 (C++ default).
        dtype_q_rope / dtype_kv_rope: Q/KV RoPE dtypes. None -> the matching NoPE dtype.

    Outputs (written into the buffers above):
        [0] work_metadata_ptrs  (2)             Two 64-bit pointers to the 1st elements of
                                                work_indptr and work_info.
        [1] work_indptr         (#cu_part + 1)  IDs of the work handled by each cu_part.
        [2] work_info           (#work, 8)      Per-work descriptor, 8 DWs each:
              [.0] bs_index        Batch index handled by this work.
              [.1] partial_index   Output-buffer tile index when split; -1 means no split.
              [.2] q_start         Global seq index where q/o starts.
              [.3] q_end           Global seq index where q/o ends (exclusive).
              [.4] kv_start        Global page index where k/v starts.
              [.5] kv_end          Global page index where k/v ends (exclusive). With causal
                                   mask and page_size==1, this is the end of the last qo seq
                                   when the work covers multiple qo sequences.
              [.6] kv_offset       Remaining seq length from kv_end to the end of the batch.
              [.7] pad             Padding to 8 DWs.
        [3] reduce_indptr       (sum(qo_seqlen_blk_count) + 1)  Groups reduce_partial_map
                                                entries that must be merged together.
        [4] reduce_final_map    (sum(qo_seqlen_blk_count))      Final output location per group.
        [5] reduce_partial_map  (#partial_tiles)                Partial-buffer locations of the
                                                tiles awaiting reduction.
    """


def get_mla_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    seqlens_kv_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_info_set: torch.Tensor,
    work_indptr: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    page_size: int = 1,
    kv_granularity: int = 16,
    max_seqlen_qo: int = -1,
    uni_seqlen_qo: int = -1,
    fast_mode: bool = True,
    topk: int = -1,
    max_split_per_batch: int = -1,
    intra_batch_mode: bool = False,
    is_cp_round_robin: bool = False,
    mla_version: int = MlaVersion.V32.value,
    dtype_q_nope: torch.dtype | None = None,
    dtype_q_rope: torch.dtype | None = None,
    dtype_kv_nope: torch.dtype | None = None,
    dtype_kv_rope: torch.dtype | None = None,
    dtype_q: torch.dtype | None = None,
    dtype_kv: torch.dtype | None = None,
) -> None:
    """Forward-compatible wrapper around the compiled ``get_mla_metadata_v1``.

    ``dtype_q`` / ``dtype_kv`` are legacy aliases kept for callers written
    before the NoPE/RoPE dtype split. When provided, each fills in the
    corresponding NoPE and RoPE dtype only if that specific one was not already
    given (an explicit ``dtype_q_nope`` / ``dtype_q_rope`` etc. always wins).
    See the compiled op's docstring for the full input/output description.
    """
    if dtype_q is not None:
        if dtype_q_nope is None:
            dtype_q_nope = dtype_q
        if dtype_q_rope is None:
            dtype_q_rope = dtype_q
    if dtype_kv is not None:
        if dtype_kv_nope is None:
            dtype_kv_nope = dtype_kv
        if dtype_kv_rope is None:
            dtype_kv_rope = dtype_kv

    return _get_mla_metadata_v1_impl(
        seqlens_qo_indptr,
        seqlens_kv_indptr,
        kv_last_page_lens,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=page_size,
        kv_granularity=kv_granularity,
        max_seqlen_qo=max_seqlen_qo,
        uni_seqlen_qo=uni_seqlen_qo,
        fast_mode=fast_mode,
        topk=topk,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=intra_batch_mode,
        is_cp_round_robin=is_cp_round_robin,
        mla_version=mla_version,
        dtype_q_nope=dtype_q_nope,
        dtype_q_rope=dtype_q_rope,
        dtype_kv_nope=dtype_kv_nope,
        dtype_kv_rope=dtype_kv_rope,
    )


@compile_ops("module_mla_metadata")
def get_mla_metadata_v1_no_redundant(
    seqlens_qo_indptr: torch.Tensor,
    seqlens_kv_indptr: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    kv_granularity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cumulated seqlens of q/o: (batch_size + 1), dtype torch.int32.
        cumulated seqlens of k/v: (batch_size + 1), dtype torch.int32.
        num_heads_per_head_k: Equals to num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.
        is_causal: whether causal mask is enabled.
        kv_granularity: the granularity on kv sequence length when cutting batch.
    Returns:
        [0] work_metadata_ptrs  (2)                  Two 64-bits pointers point to the 1st element of work_indptr and
                                                     work_info.
        [1] work_indptr:        (#work_cu + 1),      The IDs of work handled by each cu_part.
        [2] work_info           (#work, 8)
        [2.0] bs_index:         (#work),             The index of batch handled by each work.
        [2.1] partial_index:    (#work),             The index of tile in output buffer when splits. -1 means no split.
        [2.2] q_start:          (#work),             The global index in seq where q/o starts. Use global index here can
                                                     reduce memory access count in kernel.
        [2.3] q_end:            (#work),             The global index in seq where q/o ends (not included).
        [2.4] kv_start:         (#work),             The global index in seq where k/v starts.
        [2.5] kv_end:           (#work),             The global index in seq where k/v ends (not included).
        [2.6] pad               (#work, 2),          Pad to 8 DWs.
        [3] reduce_indptr:      (#reduce_tiles + 1), The IDs in reduce_partial_map indicates the tiles should be merged
                                                     together.
        [4] reduce_final_map:   (#reduce_tiles),     The final output location of each group of tiles.
        [5] reduce_partial_map: (#partial_tiles),    The locations in partial buffer of partial tiles waiting for being
                                                     reduced.
    """


@compile_ops("module_mla_reduce", develop=True)
def mla_reduce_v1(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor | None,
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    num_kv_splits: int,
    final_output: torch.Tensor,
    final_lse: torch.Tensor | None = None,
) -> None:
    """
    Cross-split (flash-style) reduction for split-KV MLA decode.

    The decode kernel splits each (batch, head) sequence into KV tiles and
    writes one *partial* attention output + log-sum-exp per split. This op
    combines the partials belonging to the same query tile into the final
    output using the standard online-softmax combine
    (``out = sum_k exp(lse_k - lse*) * out_k``), then normalizes.

    Args:
        partial_output: [max(reduce_partial_map)+s, h, dv] fp32. Per-split
            partial attention outputs (unnormalized numerators) to merge.
        partial_lse: [max(reduce_partial_map)+s, h] fp32. Per-split
            log-sum-exp denominators paired with ``partial_output``.
        reduce_indptr: [#reduce_tiles + 1] int32. Group boundaries into
            ``reduce_partial_map``: the partials for reduce tile ``i`` are
            ``reduce_partial_map[reduce_indptr[i] : reduce_indptr[i+1]]``.
        reduce_final_map: optional [#reduce_tiles, 2] int32. The final-output
            location of each merged group. ``None`` means the reduce tiles
            map to final rows in order (no indirection).
        reduce_partial_map: [reduce_indptr[-1]] int32. Locations in the
            partial buffer of the tiles waiting to be reduced.
        max_seqlen_q: max query length (tokens) per decode step.
        num_kv_splits: sizing hint for the reducer's per-split LDS scratch
            (``max_splits = max(device_cu_count, num_kv_splits)``).
            **``0`` means auto** -- size to the device CU count. Pass a value
            larger than the CU count only to force a bigger split budget;
            values <= CU count (incl. 0) are clamped up to it.
        final_output: [bs, h, dv]. Combined, normalized output (written
            in-place).
        final_lse: optional [bs, h] fp32. Combined LSE; written only when
            provided.
    """


@triton.jit(do_not_specialize=["tile_reduce_cnt"])
def decode_update_mla_metadata_v1_kernel(
    seqlens_qo_indptr,
    seqlens_kv_indptr,
    kv_last_page_lens,
    num_heads_per_head_k: tl.constexpr,
    num_heads_k: tl.constexpr,
    is_causal: tl.constexpr,
    work_info,
    work_indptr,
    reduce_indptr,
    reduce_final_map,
    reduce_partial_map,
    page_size: tl.constexpr,
    kv_granularity: tl.constexpr,
    cu_num: tl.constexpr,
    qk_batch_ratio: tl.constexpr,
    tile_reduce_cnt,
    num_reject_tokens,
    has_num_reject_tokens: tl.constexpr,
):
    work_id = tl.program_id(0)
    num_workers = tl.load(work_indptr + cu_num)
    if work_id >= num_workers:
        return
    batch_id = tl.load(work_info + work_id * 8 + 0)
    real_batch_id = batch_id // qk_batch_ratio

    # seq_kv_start = tl.load(seqlens_kv_indptr + real_batch_id).to(tl.int32)
    seq_kv_end = tl.load(seqlens_kv_indptr + real_batch_id + 1).to(tl.int32)
    # seq_kv_last = tl.load(kv_last_page_lens + real_batch_id).to(tl.int32)
    # seq_kv_len = (seq_kv_end - seq_kv_start - 1) + seq_kv_last

    seq_kv_delta = 1
    if has_num_reject_tokens:
        seq_kv_delta -= tl.load(num_reject_tokens + real_batch_id).to(tl.int32)

    q_len = 1
    partial_index = tl.load(work_info + work_id * 8 + 1)
    q_start = tl.load(work_info + work_id * 8 + 2)
    q_end = tl.load(work_info + work_id * 8 + 3)
    kv_start = tl.load(work_info + work_id * 8 + 4)
    kv_end = tl.load(work_info + work_id * 8 + 5)
    kv_offset = tl.load(work_info + work_id * 8 + 6)
    ori_partial_index = partial_index
    work_kv_len = kv_end - kv_start
    if kv_offset == 0:
        if work_kv_len > 0:
            kv_end = seq_kv_end
            if work_kv_len + seq_kv_delta > 0:
                kv_start = kv_end - work_kv_len - seq_kv_delta
            else:
                kv_start = kv_end - 1
    else:
        kv_offset += seq_kv_delta
        if kv_offset <= 0:
            work_kv_len += kv_offset - 1
            work_kv_len = max(work_kv_len, 1)
            kv_offset = 1
        kv_end = seq_kv_end - kv_offset
        kv_start = kv_end - work_kv_len

    q_len = q_end - q_start
    if q_len > 1:
        q_start = batch_id
        q_end = batch_id + 1
        if partial_index >= 0:
            partial_index = partial_index // q_len  # qlen must be same for all batches
            # partial_index = work_id

    tl.store(work_info + work_id * 8 + 1, partial_index)
    tl.store(work_info + work_id * 8 + 2, q_start)
    tl.store(work_info + work_id * 8 + 3, q_end)
    tl.store(work_info + work_id * 8 + 4, kv_start)
    tl.store(work_info + work_id * 8 + 5, kv_end)
    tl.store(work_info + work_id * 8 + 6, kv_offset)
    tl.store(work_info + work_id * 8 + 7, 0)

    if q_len > 1 and ori_partial_index >= 0:
        tile_idx = batch_id
        partial_start = tl.load(reduce_indptr + tile_idx)
        partial_end = tl.load(reduce_indptr + tile_idx + 1)
        if kv_offset == 0:
            tl.store(reduce_final_map + tile_idx * 2, q_start)
            tl.store(reduce_final_map + tile_idx * 2 + 1, q_end)
        found_partial_index = False
        for i in range(partial_start, partial_end):
            if not found_partial_index:
                partial_index_i = tl.load(reduce_partial_map + i)
                if partial_index_i == ori_partial_index:
                    tl.store(reduce_partial_map + i, partial_index)
                    found_partial_index = True


def decode_update_mla_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    seqlens_kv_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_info_set: torch.Tensor,
    work_indptr: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    page_size: int = 1,
    kv_granularity: int = 16,
    max_seqlen_qo: int = 1,
    dtype_q: torch.dtype = dtypes.bf16,
    dtype_kv: torch.dtype = dtypes.bf16,
    num_reject_tokens: torch.Tensor | None = None,
) -> None:
    """
    Update MLA metadata incrementally for decode steps where the batch
    composition has not changed. It will also convert qlen > 1 to qlen = 1.
    """
    assert kv_granularity % page_size == 0
    assert num_heads_k == 1
    assert kv_granularity >= 16
    assert page_size == 1
    # assert not (dtype_q == dtypes.bf16 and dtype_kv == dtypes.bf16 and num_heads_per_head_k == 128), "In this case, use get_mla_metadata_v1 instead"
    q_is_fp8 = dtype_q == dtypes.fp8
    kv_is_fp8 = dtype_kv == dtypes.fp8
    arch_id = get_gfx()
    natively_supported = (
        (num_heads_per_head_k == 16)
        or (
            arch_id == "gfx950"
            and num_heads_per_head_k == 32
            and q_is_fp8
            and kv_is_fp8
            and max_seqlen_qo == 4
        )
        or (
            arch_id in ("gfx942", "gfx950")
            and num_heads_per_head_k == 128
            and q_is_fp8
            and kv_is_fp8
        )
        or (
            arch_id == "gfx950"
            and num_heads_per_head_k == 96
            and q_is_fp8
            and kv_is_fp8
            and max_seqlen_qo <= 6
        )
    )
    cu_num = work_indptr.shape[0] - 1
    tile_reduce_cnt = reduce_indptr.shape[0] - 1
    max_work = work_info_set.shape[0]
    batch_size = seqlens_qo_indptr.shape[0] - 1
    qk_batch_ratio = 1
    if not natively_supported and num_heads_per_head_k % 16 == 0:
        qk_batch_ratio = num_heads_per_head_k // 16
        num_heads_per_head_k = 16
        batch_size *= qk_batch_ratio
    grid = (max_work,)
    decode_update_mla_metadata_v1_kernel[grid](
        seqlens_qo_indptr,
        seqlens_kv_indptr,
        kv_last_page_lens,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size,
        kv_granularity,
        cu_num,
        qk_batch_ratio,
        tile_reduce_cnt,
        num_reject_tokens,
        num_reject_tokens is not None,
    )


@compile_ops(
    "module_hk_mla_v32_fwd_mi3xx", fc_name="hk_mla_v32_decode_fwd", develop=True
)
def hk_mla_v32_decode_fwd_mi3xx(
    # [num_seqs, num_heads, head_size]
    query: torch.Tensor,
    # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    kv_buffer: torch.Tensor,
    # [batch_size+1]
    qo_indptr: torch.Tensor,
    # [batch_size+1]
    kv_indptr: torch.Tensor,
    # [num_page_used]
    kv_page_indices: torch.Tensor,
    # [batch_size]
    kv_last_page_lens: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    # [batch_size, num_kv_splits, num_heads, v_head_dim]
    split_output: torch.Tensor,
    # [batch_size, num_kv_splits, num_heads,  1]
    split_lse: torch.Tensor,
    final_output: torch.Tensor,
) -> None: ...


def hk_mla_v32_decode_fwd(
    query: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    split_output: torch.Tensor,
    split_lse: torch.Tensor,
    final_output: torch.Tensor,
) -> None:
    """Arch-dispatching entry point for the HK V3.2 MLA decode kernel."""
    arch_id = get_gfx()
    if arch_id in ("gfx942", "gfx950"):
        hk_mla_v32_decode_fwd_mi3xx(
            query,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_page_indices,
            kv_last_page_lens,
            work_indptr,
            work_info_set,
            max_seqlen_q,
            softmax_scale,
            split_output,
            split_lse,
            final_output,
        )
    else:
        raise NotImplementedError(
            f"hk_mla_v32_decode_fwd has no implementation for arch {arch_id}"
        )


@compile_ops(
    "module_hk_mla_v40_fwd_mi3xx", fc_name="hk_mla_v40_decode_fwd", develop=True
)
def hk_mla_v40_decode_fwd_mi3xx(
    # [total_q, num_heads, V4_DIM_QK_PACKED=512]  FP8
    #   per-token bytes: NOPE 448 + dup-E8M0 14 + pad 50
    query: torch.Tensor,
    # [total_q, num_heads, V4_DIM_ROPE=64]        BF16
    query_rope: torch.Tensor,
    # [num_page, page_size, num_kv_heads=1, 512]  FP8 (same packing as Q)
    kv_buffer: torch.Tensor,
    # [num_page, page_size, num_kv_heads=1, 64]   BF16
    kv_buffer_rope: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    split_output: torch.Tensor,
    split_lse: torch.Tensor,
    final_output: torch.Tensor,
    attn_sink: torch.Tensor | None = None,
) -> None: ...


def hk_mla_v40_decode_fwd(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_buffer_rope: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    split_output: torch.Tensor,
    split_lse: torch.Tensor,
    final_output: torch.Tensor,
    attn_sink: torch.Tensor | None = None,
) -> None:
    """Arch-dispatching entry point for the HK V4.0 MLA decode kernel."""
    arch_id = get_gfx()
    if arch_id in ("gfx942", "gfx950"):
        hk_mla_v40_decode_fwd_mi3xx(
            query,
            query_rope,
            kv_buffer,
            kv_buffer_rope,
            qo_indptr,
            kv_page_indices,
            kv_last_page_lens,
            work_indptr,
            work_info_set,
            max_seqlen_q,
            softmax_scale,
            split_output,
            split_lse,
            final_output,
            attn_sink,
        )
    else:
        raise NotImplementedError(
            f"hk_mla_v40_decode_fwd has no implementation for arch {arch_id}"
        )
