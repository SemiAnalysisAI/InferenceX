"""Dense GEMM, MoE and attention modules through vLLM's own layers and kernel selection."""
from operatorx.runners.common.vllm import attention, linear, moe
from operatorx.runners.common.vllm.linear import versions

IMPLS = [*linear.IMPLS, *moe.IMPLS, *attention.IMPLS]

__all__ = ["IMPLS", "versions"]
