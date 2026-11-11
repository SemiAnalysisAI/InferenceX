"""GEMM, MoE and attention through vLLM's layers (ROCm), AITER on as in InferenceX's ROCm launches."""
import os

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

from operatorx.runners.common.vllm import attention, linear, moe  # noqa: E402
from operatorx.runners.common.vllm.linear import versions  # noqa: E402

IMPLS = [*linear.IMPLS, *moe.IMPLS, *attention.IMPLS]

__all__ = ["IMPLS", "resident", "versions"]


def resident() -> bool:
    """Whether a built layer or engine is cached (the harness checks for leaks only without one)."""
    return linear.resident() or attention.resident()
