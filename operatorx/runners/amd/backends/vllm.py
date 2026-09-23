"""Dense GEMM through vLLM's own linear layers and kernel selection (ROCm).

AITER is enabled as in InferenceX's ROCm vLLM launches.
"""
import os

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

from operatorx.runners.vllm_linear import IMPLS, versions  # noqa: E402

__all__ = ["IMPLS", "versions"]
