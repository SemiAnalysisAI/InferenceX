"""Dense GEMM and MoE layers through vLLM's own layers and kernel selection (ROCm).

AITER is enabled as in InferenceX's ROCm vLLM launches.
"""
import os

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

from operatorx.runners.common.vllm import linear, moe  # noqa: E402
from operatorx.runners.common.vllm.linear import versions  # noqa: E402

IMPLS = [*linear.IMPLS, *moe.IMPLS]

__all__ = ["IMPLS", "versions"]
