# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import random
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# current supported case in decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16, nhead128
# qdtype fp8, kdtype fp8: nhead16, nhead128


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    return True


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
) -> torch.Tensor:
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias
    lse = attn_weights.logsumexp(dim=-1)
    attn_weights = torch.softmax(attn_weights, dim=-1)

    out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
    return out.to(dtype), lse


def torch_mha_extend(
    q,  # [total_q, nheads, headdim_q]
    k,  # [num_page * page_size, nhead_kv, qk_head_dim]
    v,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    dtype,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    ks = torch.tensor_split(k, kv_indptr.tolist()[1:])
    vs = torch.tensor_split(v, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    for i in range(bs):
        q = qs[i]
        k = ks[i]
        v = vs[i]
        o, _ = ref_masked_attention(q, k, v, sm_scale, dtype)
        os.append(o)
    o = torch.concat(os)
    return o


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=is_causal)
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses, dim=1).transpose(0, 1)
    return o, lse


@benchmark()
def test_mla(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    varlen,
    decode_qlen,
    split_per_batch=None,
    return_lse=False,
    is_causal=True,
    sequential_page_indices=False,
):
    ret = {}

    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
    kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
    if sequential_page_indices:
        # page_id == logical token index; needs pool >= ctx and byte offset can exceed 2^32
        num_page = max(num_page, kv_indptr[-1].item() + 10000)
    n_kv_idx = kv_indptr[-1].item() + 10000
    if sequential_page_indices:
        kv_indices = torch.arange(n_kv_idx, dtype=torch.int)
    else:
        kv_indices = torch.randint(0, num_page, (n_kv_idx,), dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    max_seqlen_kv = seq_lens_kv.max().item()
    total_qo = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()
    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    # for none absorb (mha)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # ############################## normal: prefill
    def test_normal_prefill():
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)
        k = torch.randn((total_kv, nhead, qk_head_dim), dtype=torch.bfloat16)
        v = torch.randn((total_kv, nhead, v_head_dim), dtype=torch.bfloat16)

        out_ref = torch_mha_extend(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            dtype=dtype,
        )

        out_aiter, us_aiter = run_perftest(
            aiter.flash_attn_varlen_func,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            max_seqlen_qo,
            max_seqlen_kv,
            softmax_scale=sm_scale,
            causal=True,
        )

        flop = (
            batch_size
            * nhead
            * 2
            * (ctx_lens * qk_head_dim * ctx_lens + ctx_lens * ctx_lens * v_head_dim)
        )
        checkAllclose(
            out_ref.to(torch.float),
            out_aiter.to(torch.float),
            msg=f"mla_prefill-normal    [torch vs  aiter_ck]: {us_aiter:>8.2f} us...... {flop/us_aiter/1000/1000:>8.2f} TFlops",
        )
        return us_aiter

    out_dtype = torch.bfloat16

    us_aiter = None
    # Prefill ref builds [nhead, (batch*ctx)^2] fp32 attn weights; bound both
    # the lazy "tile area" gate and the per-call ctx so decode-scale ctx_lens
    # (1M+) never trigger the O(N^2) ref.
    if (
        (dtype == torch.bfloat16 and kvtype == torch.bfloat16)
        and batch_size * ctx_lens * nhead < 256 * 8192 * 16
        and ctx_lens <= 16384
    ):
        us_aiter = test_normal_prefill()
        ret["prefill:ck_192"] = us_aiter

    torch.cuda.empty_cache()
    # absorb init
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # test prefill
    # ############################## absorb: prefill
    def test_absorb_prefill():
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)

        out_ref, _ = torch_mla_extend(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
        )

        # #triton version
        # prefix_indptr = kv_indptr - qo_indptr
        # tmp = kv_indptr[1:] - seq_lens_qo
        # tmp_inpptr, _ = torch.concat([kv_indptr[1:], tmp]).sort()
        # prefix_kv_indices = kv_indices.tensor_split(tmp_inpptr.tolist())
        # extend_kv_indices = torch.concat(
        #     [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 1]
        # )
        # prefix_kv_indices = torch.concat(
        #     [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 0]
        # )
        # extend_kvc = torch.index_select(kv_buffer, 0, extend_kv_indices)
        # out_triton = torch.empty((total_qo, nhead, v_head_dim), dtype=dtype).fill_(-1)
        # _, us_triton = run_perftest(
        #     mla_extend_ref.extend_attention_fwd,
        #     q,
        #     extend_kvc,
        #     extend_kvc[..., :kv_lora_rank],
        #     out_triton,
        #     kv_buffer,
        #     kv_buffer[..., :kv_lora_rank],
        #     qo_indptr,
        #     prefix_indptr,
        #     prefix_kv_indices,
        #     None,
        #     None,
        #     max_seqlen_qo,
        #     sm_scale,
        #     num_iters=5,
        # )
        # checkAllclose(
        #     out_ref,
        #     out_triton,
        #     msg=f"mla_prefill-absorb    [torch vs    triton]:{us_torch:>8.2f} us vs {us_triton:>8.2f} us......",
        # )

        out_asm = torch.empty((total_qo, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm = run_perftest(
            aiter.mla.mla_prefill_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
        )

        checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_prefill-absorb    [torch vs aiter_asm]: {us_asm:>8.2f} us......",
        )
        return us_asm

    us_asm = None
    # Absorb-prefill ref (mla_torch) builds [nhead, (batch*ctx_kv)^2] fp32 attn
    # weights -- O(N^2) memory. Tile-area gate alone is not enough: bh16 CI
    # sweeps run with decode-scale ctx_lens (-c 49152, -c 98304, -c 10000000)
    # and would OOM the host. Mirror the normal-prefill gate's explicit
    # ctx_lens <= 16384 cap to skip the ref for those configs.
    if (
        (dtype == torch.bfloat16 and kvtype == torch.bfloat16 and nhead in [16, 128])
        and batch_size * ctx_lens * nhead < 32 * 8192 * 16
        and ctx_lens <= 16384
    ):
        us_asm = test_absorb_prefill()
        ret["prefill:asm_576"] = us_asm

    torch.cuda.empty_cache()

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and decode_qlen != 1:
    #     return
    seq_lens_qo.fill_(decode_qlen)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    # troch implementation
    out_ref, lse_ref = torch_mla_extend(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=is_causal,
        dtype=out_dtype,
    )

    # Triton implementation
    # if decode_qlen == 1:
    #     if qk_head_dim != v_head_dim:
    #         out_triton = q.new_empty((total_q, nhead, v_head_dim)).fill_(-1)
    #     else:
    #         out_triton = torch.empty_like(q)

    #     num_kv_splits = 16
    #     attn_logits = torch.empty(
    #         (total_q, nhead, num_kv_splits, v_head_dim + 1),
    #         dtype=dtypes.fp32,
    #     )
    #     _, us_ref = run_perftest(
    #         mla_decode_ref.decode_attention_fwd,
    #         q,
    #         kv_buffer,
    #         kv_buffer[..., :kv_lora_rank],
    #         out_triton,
    #         kv_indptr,
    #         kv_indices,
    #         attn_logits,
    #         num_kv_splits,
    #         sm_scale,
    #         num_iters=5,
    #     )
    #     # logits_ref, lse_ref = attn_logits.split([v_head_dim, 1], dim=-1)
    #     # logits_ref = rearrange(logits_ref, "bs h sp d -> bs sp h d")
    #     # lse_ref = rearrange(lse_ref, "bs h sp d -> bs sp h d")
    #     checkAllclose(
    #         out_ref,
    #         out_triton,
    #         msg=f"mla_decode-absorb    [golden vs    triton]:{us_torch_decode:>8.2f} us vs {us_ref:>8.2f} us......",
    #     )

    def test_absorb_decode_bf16():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=split_per_batch,
            return_lse=return_lse,
        )

        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        if return_lse and attn_lse is not None:
            checkAllclose(
                lse_ref,
                attn_lse.reshape(total_q, nhead),
                msg=f"mla_decode-absorb    [lse_ref vs attn_lse]: {us_asm_decode:>8.2f} us......",
            )
        return err, us_asm_decode

    def test_absorb_decode_fp8():
        if dtype != dtypes.fp8 and nhead == 128:
            aiter.logger.info("don't support this case:\n")
            return None, 1e12
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_fp8 = q.to(dtype)
        q_scale = None
        if dtype == dtypes.fp8:
            q_scale = torch.ones([1], dtype=torch.float, device="cuda")
        else:
            aiter.logger.info("don't support this case.")
            return None, 1e12

        kv_buffer_fp8 = kv_buffer.to(kvtype)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            num_kv_splits=split_per_batch,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )

        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    def test_absorb_decode_gluon():
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon

        out_gluon = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_nope = q[:, :, :v_head_dim].view(batch_size, nhead, v_head_dim)
        q_pe = q[:, :, v_head_dim:].view(batch_size, nhead, qk_head_dim - v_head_dim)

        # KV: flat [N, 576] buffer; the kernel uses KV_PE_OFFSET (default 512)
        # to reach k_pe columns and picks buffer_load vs global_load internally.
        kv_c = kv_buffer.view(-1, qk_head_dim)

        # Varlen=False: reshape kv_indices as block_table [batch, ctx_lens]
        # Varlen=True : pass kv_indices + kv_indptr
        if not varlen:
            page_table = kv_indices[:total_kv].view(batch_size, ctx_lens)
            seq_info = seq_lens_kv
            use_2d_view = True
        else:
            page_table = kv_indices
            seq_info = kv_indptr
            use_2d_view = False

        (attn_logits, attn_lse), us_gluon_decode = run_perftest(
            mla_gluon,
            q_nope,
            q_pe,
            kv_c,
            out_gluon.view(batch_size, nhead, v_head_dim),
            page_table,
            seq_info,
            sm_scale,
            use_2d_view=use_2d_view,
            min_kv_seq_len=ctx_lens,
            return_lse=return_lse,
        )

        err = checkAllclose(
            out_ref,
            out_gluon,
            msg=f"mla_decode-absorb    [golden vs gluon_mla]: {us_gluon_decode:>8.2f} us......",
        )
        if return_lse and attn_lse is not None:
            checkAllclose(
                lse_ref,
                attn_lse.reshape(total_q, nhead),
                msg=f"mla_decode-absorb    [lse_ref vs gluon_mla_lse]: {us_gluon_decode:>8.2f} us......",
            )
        return err, us_gluon_decode

    def test_absorb_decode_gluon_bh16(name):
        # Shared bh16bn{64,128} runner. The wrapper dispatches on
        # (nhead, kv dtype): name='bh16bn128' -> cast kv to fp8;
        # name='bh16bn64' -> keep bf16. -lse also validates the returned lse.
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon

        out_gluon = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        q_nope = q[:, :, :v_head_dim].view(batch_size, nhead, v_head_dim)
        q_pe = q[:, :, v_head_dim:].view(batch_size, nhead, qk_head_dim - v_head_dim)

        kv_c = kv_buffer.view(-1, qk_head_dim)
        if name == "bh16bn128":
            kv_c = kv_c.to(dtypes.fp8)

        if not varlen:
            page_table = kv_indices[:total_kv].view(batch_size, ctx_lens)
            seq_info = seq_lens_kv
            use_2d_view = True
        else:
            page_table = kv_indices
            seq_info = kv_indptr
            use_2d_view = False

        (_, lse), us_decode = run_perftest(
            mla_gluon,
            q_nope,
            q_pe,
            kv_c,
            out_gluon.view(batch_size, nhead, v_head_dim),
            page_table,
            seq_info,
            sm_scale,
            use_2d_view=use_2d_view,
            kv_scale=1.0,
            min_kv_seq_len=ctx_lens,
            return_lse=return_lse,
        )

        err = checkAllclose(
            out_ref,
            out_gluon,
            msg=f"mla_decode-absorb    [golden vs gluon_{name}]: {us_decode:>8.2f} us......",
        )
        cal_diff(out_ref, out_gluon, f"out_gluon_{name}", use_fp8=(name == "bh16bn128"))
        if return_lse and lse is not None:
            checkAllclose(
                lse_ref,
                lse.reshape(total_q, nhead),
                msg=f"mla_decode-absorb    [lse_ref vs gluon_{name}_lse]: {us_decode:>8.2f} us......",
            )
        return err, us_decode

    err = None
    us_asm_decode = 1e12
    # The ASM decode baseline aborts for these MLA configs when lse is requested
    if return_lse:
        pass
    elif (dtype == torch.bfloat16 and kvtype == torch.bfloat16) and nhead in [
        8,
        16,
        32,
        64,
        128,
    ]:
        err, us_asm_decode = test_absorb_decode_bf16()
    elif kvtype == dtypes.fp8 and nhead in [8, 16, 32, 128]:
        err, us_asm_decode = test_absorb_decode_fp8()

    ret["decode:err"] = err
    ret["decode:asm_576"] = us_asm_decode

    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim * (torch.finfo(kvtype).bits // 8)
        + total_q * nhead * qk_head_dim * (torch.finfo(dtype).bits // 8)
        + total_q * nhead * v_head_dim * (torch.finfo(out_dtype).bits // 8)
    )

    ret["decode:flops"] = flops
    ret["decode:bytes"] = bytes
    ret["decode:TFLOPS"] = flops / us_asm_decode / 1e6
    ret["decode:TB/s"] = bytes / us_asm_decode / 1e6

    # Gluon MLA decode test
    # Example: -c 16384 -b 64 128 -n 64,1 128,1 -d bf16 -kvd bf16
    NUM_XCDS_GFX950 = 8
    BLOCK_H_GLUON = 64
    if (
        get_gfx() == "gfx950"
        and dtype == torch.bfloat16
        and kvtype == torch.bfloat16
        and nhead in (64, 128)
        and decode_qlen == 1
        and v_head_dim == 512
        and (qk_head_dim - v_head_dim) == 64
        and batch_size in (64, 128, 256)
        and page_size == 1
    ):
        base_grid = (
            NUM_XCDS_GFX950
            * ((nhead + BLOCK_H_GLUON - 1) // BLOCK_H_GLUON)
            * (batch_size // NUM_XCDS_GFX950)
        )
        splits_needed = max(1, (256 + base_grid - 1) // base_grid)
        # Round up to a power of two: 1 << (n - 1).bit_length() for n >= 1.
        num_kv_splits = 1 << (splits_needed - 1).bit_length()
        # PIPELINE_STAGES=3, BLOCK_N=64 → 192; mirror wrapper's bound.
        min_ctx_required = num_kv_splits * (192 + num_kv_splits)
        if ctx_lens > min_ctx_required:
            err_gluon, us_gluon_decode = test_absorb_decode_gluon()
            ret["decode:gluon_err"] = err_gluon
            ret["decode:gluon_576"] = us_gluon_decode
            ret["decode:gluon_TFLOPS"] = flops / us_gluon_decode / 1e6
            ret["decode:gluon_TB/s"] = bytes / us_gluon_decode / 1e6

    # Gluon MLA bh16bn128 decode test
    # Example: -c 10000000 -b 1 -n 16,1 -d bf16 -kvd fp8
    if (
        get_gfx() == "gfx950"
        and dtype == torch.bfloat16
        and kvtype == dtypes.fp8
        and nhead <= 16
        and decode_qlen == 1
        and batch_size == 1
        and v_head_dim == 512
        and (qk_head_dim - v_head_dim) == 64
        and page_size == 1
        and ctx_lens >= 1
    ):
        err_gluon, us_gluon_decode = test_absorb_decode_gluon_bh16("bh16bn128")
        ret["decode:gluon_err"] = err_gluon
        ret["decode:gluon_576"] = us_gluon_decode
        ret["decode:gluon_TFLOPS"] = flops / us_gluon_decode / 1e6
        ret["decode:gluon_TB/s"] = bytes / us_gluon_decode / 1e6

    # Gluon MLA bh16bn64 decode test
    # Example: -c 10000 -b 1 3 4 -n 16,1 -d bf16 -kvd bf16 [-lse]
    if (
        get_gfx() == "gfx950"
        and dtype == torch.bfloat16
        and kvtype == torch.bfloat16
        and nhead <= 16
        and decode_qlen == 1
        and v_head_dim == 512
        and (qk_head_dim - v_head_dim) == 64
        and page_size == 1
        and 1 <= batch_size <= 256
        and ctx_lens >= 1
    ):
        err_gluon, us_gluon_decode = test_absorb_decode_gluon_bh16("bh16bn64")
        ret["decode:gluon_err"] = err_gluon
        ret["decode:gluon_576"] = us_gluon_decode
        ret["decode:gluon_TFLOPS"] = flops / us_gluon_decode / 1e6
        ret["decode:gluon_TB/s"] = bytes / us_gluon_decode / 1e6

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qn",
    "--qk_nope_head_dim",
    type=int,
    default=128,
    help="""qk nope head dim.
    e.g.: -qn 128""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=128,
    help="""v head dim.
    e.g.: -vh 128""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default="bf16,",
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    metavar="{bf16, fp8}",
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    nargs="*",
    type=dtypes.str2Dtype,
    default="bf16,",
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    metavar="{bf16, fp8}",
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[21, 64, 256, 512, 1200, 3200, 5200, 8192],
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[1, 3, 5, 16, 32, 64, 128, 256],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    choices=[
        (4, 1),
        (8, 1),
        (8, 2),
        (12, 1),
        (16, 1),
        (16, 2),
        (16, 4),
        (32, 1),
        (32, 2),
        (32, 4),
        (64, 1),
        (128, 1),
        (128, 2),
        (128, 4),
    ],
    nargs="*",
    const=None,
    default=[(16, 1), (16, 2), (16, 4), (128, 1), (128, 2)],
    help="""Number of nhead and decode_qlen.
    e.g.: -n 16,1""",
)
parser.add_argument(
    "-splits",
    "--split_per_batch",
    type=int,
    nargs="*",
    default=[None],
    help="""kv seqlens split num for per batch.
    e.g.: -ms 32""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "-lse",
    "--return_lse",
    action="store_true",
    help="""return lse. Default: False.
    --lse # True""",
)
parser.add_argument(
    "--sequential-page-indices",
    action="store_true",
    help="""Use kv_indices[i]=i (sequential physical page id) instead of random pages.
    Expands KV pool to cover ctx length (tests 64-bit page_idx * stride).""",
)
parser.add_argument(
    "--causal",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="""Enable/disable causal masking. Default: True.
    --causal / --no-causal""",
)


args = parser.parse_args()

for nhead, decode_qlen in args.nhead:
    df = []
    for dtype, kvtype, ctx_len, batch_size, split_per_batch in itertools.product(
        args.dtype, args.kv_dtype, args.ctxLen, args.batchSize, args.split_per_batch
    ):
        if check_support(dtype, kvtype, nhead):
            ret = test_mla(
                ctx_len,
                batch_size,
                nhead,
                args.kv_lora_rank,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
                args.v_head_dim,
                dtype,
                kvtype,
                args.block_size,
                varlen=args.varlen,
                decode_qlen=decode_qlen,
                split_per_batch=split_per_batch,
                return_lse=args.return_lse,
                is_causal=args.causal,
                sequential_page_indices=args.sequential_page_indices,
            )
            df.append(ret)
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}decode_qlen{decode_qlen}.csv")
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla summary (markdown):\n%s", df_md)
