from __future__ import annotations

from dataclasses import dataclass

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register


@dataclass(frozen=True)
class GemmArgs:
    m: int
    n: int
    k: int
    dtype_a: str = "bf16"
    dtype_b: str = "bf16"
    dtype_out: str = "bf16"
    bias: bool = False
    activation: str | None = None


GEMM = OpSpec(
    type="gemm",
    arg_schema=GemmArgs,
    description="C = activation(A[M,K] @ B[K,N] + bias)",
)

register(GEMM)
