"""Dense GEMM and MoE layers through vLLM's own layers and kernel selection."""
from operatorx.runners.common.vllm import linear, moe
from operatorx.runners.common.vllm.linear import versions

IMPLS = [*linear.IMPLS, *moe.IMPLS]

__all__ = ["IMPLS", "versions"]
