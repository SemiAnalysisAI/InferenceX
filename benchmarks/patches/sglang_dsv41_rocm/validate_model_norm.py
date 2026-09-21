"""Verify the installed V4.1 RMSNorm-to-linear path on an allocated MI355X."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv41_rocm.fp8_linear import (
    rocm_v41_block_fp8_linear,
)
from sglang.srt.models.deepseek_v4 import _fused_rmsnorm_fp8_quant

torch.manual_seed(41)
config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v41"))
with patch("sglang.srt.runtime_context.process_model_config", return_value=config):
    for rows, width in ((1, 1280), (8, 1280), (8, 4096)):
        # WQ_A is sliced from fused QKV output, so its rows are non-contiguous.
        x = torch.randn(rows, width + 512, device="cuda", dtype=torch.bfloat16)
        x = x[:, :width]
        norm_weight = torch.ones(width, device="cuda", dtype=torch.bfloat16)
        first, normalized = _fused_rmsnorm_fp8_quant(x, norm_weight, 1e-6)
        assert first.dtype == torch.bfloat16
        assert first.data_ptr() == normalized.data_ptr()
        reference = (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        torch.testing.assert_close(normalized, reference, rtol=0.01, atol=0.015)
        weight = torch.randn(128, width, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(4, width // 32, device="cuda")
        result = rocm_v41_block_fp8_linear(first, weight, [32, 32], scales)
        assert result.shape == (rows, 128) and torch.isfinite(result).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_first, _ = _fused_rmsnorm_fp8_quant(x, norm_weight, 1e-6)
            graph_result = rocm_v41_block_fp8_linear(
                graph_first, weight, [32, 32], scales
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_result, result, rtol=0, atol=0)
        print(rows, width, "installed V4.1 norm/linear eager+graph PASS", flush=True)
print("All V4.1 norm/linear checks PASS", flush=True)
