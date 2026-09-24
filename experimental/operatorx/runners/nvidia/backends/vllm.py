"""Dense GEMM and MoE layers through vLLM's own layers and kernel selection."""
from operatorx.runners.common import vllm_linear, vllm_moe
from operatorx.runners.common.vllm_linear import versions

IMPLS = [*vllm_linear.IMPLS, *vllm_moe.IMPLS]

__all__ = ["IMPLS", "versions"]
