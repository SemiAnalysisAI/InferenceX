"""Dense GEMM through vLLM's own linear layers and kernel selection."""
from operatorx.runners.vllm_linear import IMPLS, versions

__all__ = ["IMPLS", "versions"]
