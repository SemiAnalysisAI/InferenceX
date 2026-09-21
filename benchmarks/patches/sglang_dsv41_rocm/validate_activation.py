"""GPU numerical check for native AITER activation at V4.1 partition widths."""

import torch
from sglang.srt.layers.attention.dsv41_rocm.activation import (
    rocm_v41_silu_and_mul_clamp,
)

torch.manual_seed(41)
for rows, width in ((0, 576), (1, 576), (12, 576), (12, 1152)):
    x = (torch.randn(rows, 2 * width, device="cuda") * 12).bfloat16()
    out = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
    gate, up = x.float().chunk(2, -1)
    expected = (
        torch.nn.functional.silu(gate.clamp(max=7)) * up.clamp(-7, 7)
    ).bfloat16()
    rocm_v41_silu_and_mul_clamp(x, out, 7.0)
    torch.testing.assert_close(out, expected, rtol=0.008, atol=0.015)
    if rows:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rocm_v41_silu_and_mul_clamp(x, out, 7.0)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0.008, atol=0.015)
    print(rows, width, "AITER masked activation eager/graph PASS", flush=True)
