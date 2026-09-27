"""Dense GEMM, MoE and attention modules through vLLM's own layers and kernel selection (ROCm).

AITER is enabled as in InferenceX's ROCm vLLM launches.
"""
import os

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

from operatorx.runners.common.vllm import attention, linear, moe  # noqa: E402
from operatorx.runners.common.vllm.linear import versions  # noqa: E402

IMPLS = [*linear.IMPLS, *moe.IMPLS, *attention.IMPLS]

__all__ = ["IMPLS", "versions"]
