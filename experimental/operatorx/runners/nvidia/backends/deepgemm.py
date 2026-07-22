from __future__ import annotations

import deep_gemm
import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("deep_gemm", "torch")


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a["dtype_a"] != "fp8" or a["dtype_b"] != "fp8":
        raise UnsupportedOpError(f"deepgemm requires fp8 inputs (got {a['dtype_a']}/{a['dtype_b']})")
    if a.get("bias"):
        raise UnsupportedOpError("deepgemm fp8_gemm_nt does not expose a bias parameter")
    if a.get("activation") is not None:
        raise UnsupportedOpError(
            f"deepgemm fp8_gemm_nt has no fused activation; got activation={a['activation']!r}"
        )
    if a["m"] <= 0 or a["n"] <= 0 or a["k"] <= 0:
        raise UnsupportedOpError(f"degenerate gemm shape m={a['m']} n={a['n']} k={a['k']}")
    # Pad K up to a multiple of 128 (block size).
    K_pad = ((a["k"] + 127) // 128) * 128
    M, N, K = a["m"], a["n"], K_pad
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    A_fp8, A_scales = deep_gemm.per_token_cast_to_fp8(A_bf16, use_ue8m0=True)
    B_fp8, B_scales = deep_gemm.per_block_cast_to_fp8(B_bf16, use_ue8m0=True)

    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    return {"lhs": (A_fp8, A_scales), "rhs": (B_fp8, B_scales), "out": out}


def _kernel_gemm(ctx: dict) -> None:
    # FIXME: hardcoded layout
    deep_gemm.fp8_gemm_nt(ctx["lhs"], ctx["rhs"], ctx["out"])


IMPLS = [
    BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm),
]
