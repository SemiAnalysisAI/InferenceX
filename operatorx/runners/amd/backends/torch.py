"""Dense GEMM through ROCm PyTorch, including architecture-correct FP8."""

from __future__ import annotations

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("torch")


def resolve_dtype(name: str) -> torch.dtype:
    ordinary = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name in ordinary:
        return ordinary[name]
    if name == "e4m3":
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
        arch = arch.split(":")[0]
        if arch == "gfx942":
            return torch.float8_e4m3fnuz
        if arch == "gfx950":
            return torch.float8_e4m3fn
        raise UnsupportedOpError(f"ROCm FP8 GEMM is not qualified for {arch}")
    raise UnsupportedOpError(f"ROCm torch GEMM does not support dtype={name!r}")


def prepare(op: Op) -> dict:
    a = op.args
    if a.get("activation") is not None:
        raise UnsupportedOpError("ROCm torch GEMM has no native activation fusion")
    qa, qb = a["a"], a["b"]
    if qa["dtype"] != qb["dtype"]:
        raise UnsupportedOpError("ROCm torch GEMM requires matching input dtypes")
    fp8 = qa["dtype"] == "e4m3"
    # _scaled_mm here takes one static fp32 scale per operand; unscaled GEMMs take none
    per_tensor = {"dtype": "fp32", "static": True, "group": [-1, -1]}
    # the FP8 path multiplies pre-quantized operands: activation quantization is not in the op
    want = (({"dtype": "e4m3", "scale": per_tensor, "input": "e4m3"}, {"dtype": "e4m3", "scale": per_tensor})
            if fp8 else ({"dtype": qa["dtype"]},) * 2)
    if (qa, qb) != want:
        raise UnsupportedOpError(f"ROCm torch GEMM supports unscaled or static per-tensor fp8; got a={qa} b={qb}")
    dtype = resolve_dtype(qa["dtype"])
    out_name = a.get("out") or "bf16"
    if out_name not in {"bf16", "fp16", "fp32"}:
        raise UnsupportedOpError(f"unsupported GEMM output dtype={out_name!r}")
    out_dtype = resolve_dtype(out_name)
    m, n, k = a["m"], a["n"], a["k"]
    if fp8:
        left = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(dtype)
        right = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(dtype).t()
        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")
    else:
        if dtype != out_dtype:
            raise UnsupportedOpError("unscaled GEMM output must match the input dtype")
        left = torch.randn(m, k, device="cuda", dtype=dtype)
        right = torch.randn(k, n, device="cuda", dtype=dtype)
        scale_a = scale_b = None
    bias = torch.randn(n, device="cuda", dtype=out_dtype) if a.get("bias") else None
    return {
        "A": left,
        "B": right,
        "bias": bias,
        "out_dtype": out_dtype,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "fp8": fp8,
    }


def kernel(ctx: dict) -> None:
    if ctx["fp8"]:
        ctx["C"] = torch._scaled_mm(
            ctx["A"],
            ctx["B"],
            ctx["scale_a"],
            ctx["scale_b"],
            out_dtype=ctx["out_dtype"],
            bias=ctx["bias"],
        )
    elif ctx["bias"] is not None:
        ctx["C"] = torch.addmm(ctx["bias"], ctx["A"], ctx["B"])
    else:
        ctx["C"] = torch.matmul(ctx["A"], ctx["B"])


IMPLS = [
    BackendImpl(op_type="gemm", prepare=prepare, kernel=kernel),
]
