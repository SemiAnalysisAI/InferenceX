"""deepep-v25: DeepEP V2.5's EPBuffer under vLLM's `deepep_v2` call configuration.

A library row: vLLM v0.32.0 still builds V2's ElasticBuffer. V2.5 keeps the constructor and the
dispatch/combine arguments the V2 adapter passes, so this subclass changes only what V2.5 changed.
"""

from __future__ import annotations

from ep_deepep_v2 import DeepEPV2Backend


class DeepEPV25Backend(DeepEPV2Backend):
    name = "deepep-v25"
    # V2.5 removed V1, including the legacy low-latency Buffer and NVSHMEM; EPBuffer's no-sync
    # dispatch, which `normal` already runs, is its decode path.
    SUPPORTED_MODES = ("normal",)
    CUDA_GRAPH_MODES = ("normal",)
    kernel_generation = "v25-ep-buffer-vllm-nosync"
    _buffer_class = "EPBuffer"
    _handle_copy = False  # V2.5 asserts it off: handle copying was removed
