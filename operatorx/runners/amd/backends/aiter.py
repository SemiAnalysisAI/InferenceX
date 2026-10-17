"""AITER fused MHA/GQA and materialized MLA attention (no torch fallback)."""

from __future__ import annotations

import aiter

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions
from operatorx.runners import attention


def versions() -> dict[str, str]:
    return lookup_versions("amd-aiter", "torch")


def prepare(op: Op) -> dict:
    a = op.args
    if a["dtype_q"] not in {"bf16", "fp16"}:
        raise UnsupportedOpError("AITER attention currently supports bf16/fp16")
    dq = (
        a["head_dim"]
        if op.type == "attention_mha"
        else a["head_dim_qk_nope"] + a["head_dim_qk_rope"]
    )
    dv = a["head_dim"] if op.type == "attention_mha" else a["head_dim_v"]
    if dq > 256 or dv > 256 or dq % 8 or dv % 8:
        raise UnsupportedOpError(
            "AITER attention requires head dimensions divisible by 8 and <=256"
        )
    return attention.prepare(op, layout="bshd")


def kernel(ctx: dict) -> None:
    ctx["out"] = aiter.flash_attn_func(
        ctx["q"], ctx["k"], ctx["v"], dropout_p=0.0, causal=ctx["causal"]
    )


IMPLS = [
    BackendImpl(op_type=name, prepare=prepare, kernel=kernel)
    for name in ("attention_mha", "attention_mla")
]
