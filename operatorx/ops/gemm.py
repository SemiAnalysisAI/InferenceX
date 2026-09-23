from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register

ELEMENT_DTYPES = {"bf16", "fp16", "fp32", "e4m3", "e5m2", "e2m1", "int4", "int8"}
SCALE_DTYPES = {"fp32", "bf16", "fp16", "e4m3", "ue8m0"}


def quant(dtype: str, scale: dict | None = None, scale2: dict | None = None,
          symmetric: bool = True) -> dict[str, Any]:
    """Canonical operand descriptor; defaults are omitted so equal operands hash equal."""
    d: dict[str, Any] = {"dtype": dtype}
    if scale is not None:
        d["scale"] = scale
    if scale2 is not None:
        d["scale2"] = scale2
    if not symmetric:
        d["symmetric"] = False
    return d


def scale(dtype: str, static: bool, group: tuple[int, int] | list[int]) -> dict[str, Any]:
    return {"dtype": dtype, "static": static, "group": [int(group[0]), int(group[1])]}


def _check_scale(s: Any, where: str) -> None:
    if not isinstance(s, dict) or set(s) != {"dtype", "static", "group"}:
        raise ValueError(f"{where} must be {{dtype, static, group}}, got {s!r}")
    if s["dtype"] not in SCALE_DTYPES:
        raise ValueError(f"{where}.dtype {s['dtype']!r} not in {sorted(SCALE_DTYPES)}")
    if not isinstance(s["static"], bool):
        raise ValueError(f"{where}.static must be a bool")
    g = s["group"]
    if not (isinstance(g, list) and len(g) == 2 and all(isinstance(x, int) and (x == -1 or x >= 1) for x in g)):
        raise ValueError(f"{where}.group must be [rows, cols] with each -1 or >= 1, got {g!r}")


def _check_operand(d: Any, where: str) -> None:
    if not isinstance(d, dict) or "dtype" not in d or set(d) - {"dtype", "scale", "scale2", "symmetric"}:
        raise ValueError(f"{where} must be {{dtype[, scale, scale2, symmetric]}}, got {d!r}")
    if d["dtype"] not in ELEMENT_DTYPES:
        raise ValueError(f"{where}.dtype {d['dtype']!r} not in {sorted(ELEMENT_DTYPES)}")
    if "scale" in d:
        _check_scale(d["scale"], f"{where}.scale")
    elif "scale2" in d:
        raise ValueError(f"{where}.scale2 needs a scale")
    if "scale2" in d:
        _check_scale(d["scale2"], f"{where}.scale2")
    if d.get("symmetric", True) is not True and d["symmetric"] is not False:
        raise ValueError(f"{where}.symmetric must be a bool")


@dataclass(frozen=True)
class GemmArgs:
    """C[M,N] = activation(A[M,K] @ B[N,K]^T + bias); A = activation, B = weight.

    a, b: operand descriptors, {"dtype", "scale"?, "scale2"?, "symmetric"?}
      dtype: storage element type (bf16, e4m3, e2m1, int4, ...)
      scale: {"dtype", "static", "group": [rows, cols]}, the checkpoint's scale
        format. group is the block of the operand, in its stored layout
        (A [M, K], B [N, K]), that shares one scale; -1 spans the dimension:
        [-1, -1] per-tensor, [1, -1] per-token, [-1, 1] per-channel,
        [1, g] per-row groups of g along K, [r, c] 2-D blocks.
        static=False means computed at runtime (activation quantization inside the op).
      scale2: optional second-level scale (e.g. NVFP4's per-tensor fp32 global scale).
      symmetric: False when the format carries zero points.
    An unquantized operand is just {"dtype": "bf16"}. The fields mirror vLLM's
    QuantKey / ScaleDesc / GroupShape.
    """
    m: int
    n: int
    k: int
    a: dict
    b: dict
    out: str = "bf16"
    bias: bool = False
    activation: str | None = None

    def __post_init__(self):
        _check_operand(self.a, "a")
        _check_operand(self.b, "b")
        if self.out not in ELEMENT_DTYPES:
            raise ValueError(f"out {self.out!r} not in {sorted(ELEMENT_DTYPES)}")


GEMM = OpSpec(
    type="gemm",
    arg_schema=GemmArgs,
    description="C = activation(A[M,K] @ B[N,K]^T + bias)",
)

register(GEMM)
