"""Shared contiguous attention inputs and bottom-right causal SDPA semantics.

MLA measures materialized Q/K/V attention only; cache projection and RoPE are
outside this operator. MHA expands grouped KV heads outside the torch timer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from operatorx.core import Op, UnsupportedOpError


def prepare(op: Op, *, layout: str = "bhsd") -> dict:
    a = op.args
    if a.get("kv_layout", "contig") != "contig" or a.get("sliding_window") is not None:
        raise UnsupportedOpError(
            "attention requires contiguous KV without a sliding window"
        )
    mla = op.type == "attention_mla"
    names = (
        [a["dtype_q"], a["dtype_kv"]]
        if mla
        else [a[k] for k in ("dtype_q", "dtype_k", "dtype_v")]
    )
    names.append(a.get("dtype_o", "bf16"))
    if len(set(names)) != 1 or names[0] not in {"bf16", "fp16", "fp32"}:
        raise UnsupportedOpError(
            "attention requires uniform bf16/fp16/fp32 inputs and output"
        )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        names[0]
    ]
    b, sq, sk, h = (
        a[k] for k in ("batch_size", "seq_len_q", "seq_len_kv", "num_heads")
    )
    hkv = h if mla else a["num_heads_kv"]
    dq = a["head_dim_qk_nope"] + a["head_dim_qk_rope"] if mla else a["head_dim"]
    dv = a["head_dim_v"] if mla else dq
    if (
        any(type(n) is not int or n <= 0 for n in (b, sq, sk, h, hkv, dq, dv))
        or h % hkv
    ):
        raise ValueError(
            "attention dimensions must be positive; query heads must divide by KV heads"
        )
    dims = ((b, h, sq, dq), (b, hkv, sk, dq), (b, hkv, sk, dv))
    if layout == "bshd":
        dims = tuple((batch, seq, heads, dim) for batch, heads, seq, dim in dims)
    elif layout != "bhsd":
        raise ValueError(f"invalid attention layout: {layout}")
    q, k, v = (torch.randn(*shape, dtype=dtype, device="cuda") for shape in dims)
    causal = bool(a.get("causal", True))
    ctx = {"q": q, "k": k, "v": v, "causal": causal, "mask": None}
    if layout == "bhsd":
        if hkv != h:
            ctx["k"] = k.repeat_interleave(h // hkv, dim=1)
            ctx["v"] = v.repeat_interleave(h // hkv, dim=1)
        # PyTorch is_causal aligns upper-left. Decode queries refer to the last
        # positions of the KV sequence, so rectangular masks align bottom-right.
        if causal and sq != sk:
            ctx["causal"] = False
            if sq != 1 or sq > sk:
                ctx["mask"] = torch.ones(sq, sk, dtype=torch.bool, device="cuda").tril(
                    sk - sq
                )
    return ctx


def kernel(ctx: dict) -> None:
    ctx["out"] = F.scaled_dot_product_attention(
        ctx["q"], ctx["k"], ctx["v"], attn_mask=ctx["mask"], is_causal=ctx["causal"]
    )
