from __future__ import annotations

from dataclasses import dataclass

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register


@dataclass(frozen=True)
class GemmArgs:
    """C[M,N] = activation(A[M,K] @ B[K,N] + bias), A = activation, B = weight.

    dtype_*: storage element type (bf16, e4m3, e2m1, int4).
    scale_a: how A is quantized inside the op (serving feeds bf16 activations):
      none | per_tensor_static | per_tensor_dynamic | per_token_dynamic |
      group_1x128_dynamic | group_16_dynamic | group_32_dynamic
    scale_b: weight scale granularity:
      none | per_tensor | per_channel | block_128x128 | group_16 | group_32
    scale_dtype_*: fp32 | ue8m0 | e4m3 (group_16 e4m3 scales also carry an fp32
      global scale).
    """
    m: int
    n: int
    k: int
    dtype_a: str = "bf16"
    dtype_b: str = "bf16"
    dtype_out: str = "bf16"
    scale_a: str = "none"
    scale_b: str = "none"
    scale_dtype_a: str = "none"
    scale_dtype_b: str = "none"
    bias: bool = False
    activation: str | None = None


GEMM = OpSpec(
    type="gemm",
    arg_schema=GemmArgs,
    description="C = activation(A[M,K] @ B[K,N] + bias)",
)

register(GEMM)
