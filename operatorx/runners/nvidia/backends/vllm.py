"""GEMM, MoE and attention through vLLM's layers and kernel selection."""
from operatorx.runners.common.vllm import attention, linear, moe
from operatorx.runners.common.vllm.linear import versions

IMPLS = [*linear.IMPLS, *moe.IMPLS, *attention.IMPLS]

__all__ = ["IMPLS", "resident", "versions"]


def resident() -> bool:
    """Whether a built layer or engine is cached (the harness checks for leaks only without one)."""
    return linear.resident() or attention.resident()
