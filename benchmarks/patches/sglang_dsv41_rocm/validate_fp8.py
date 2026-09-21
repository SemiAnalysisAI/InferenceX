import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "dsv41_rocm"))
from fp8_linear import quantize_ue8m0, rocm_v41_block_fp8_linear

assert torch.cuda.device_count() >= 1
print("DEVICE", torch.cuda.get_device_name(), flush=True)
torch.manual_seed(42)
for rows, width in [(0, 128), (1, 128), (8, 4096), (64, 1024)]:
    for magnitude in [0.0, 1e-12, 1.0, 448.0, 65536.0]:
        x = (torch.randn(rows, width, device="cuda") * magnitude).bfloat16()
        q, s = quantize_ue8m0(x, 32)
        v = x.float().view(rows, width // 32, 32)
        a = v.abs().amax(-1).clamp_min(1e-10)
        expected_s = torch.exp2(torch.ceil(torch.log2(a / 448.0)))
        expected_q = (
            (v / expected_s.unsqueeze(-1))
            .clamp(-448, 448)
            .view(rows, width)
            .to(torch.float8_e4m3fn)
        )
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
        torch.testing.assert_close(
            q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
        )
        if rows:
            for _ in range(3):
                quantize_ue8m0(x, 32)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                cq, cs = quantize_ue8m0(x, 32)
            g.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                cq.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(cs, expected_s, rtol=0, atol=0)
        print("QUANT PASS", rows, width, magnitude, flush=True)
for rows, width, outputs in [
    (0, 128, 128),
    (1, 4096, 2048),
    (8, 128, 128),
    (64, 1024, 256),
]:
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(outputs, width, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones(outputs // 32, width // 32, device="cuda")
    before = w.view(torch.uint8).clone()
    q, s = quantize_ue8m0(x, 32)
    ref = (
        (q.float().view(rows, width // 32, 32) * s.unsqueeze(-1)).view(rows, width)
        @ w.float().t()
    ).bfloat16()
    y = rocm_v41_block_fp8_linear(x, w, [32, 32], scales)
    torch.testing.assert_close(y, ref, rtol=0.02, atol=0.25)
    torch.testing.assert_close(w.view(torch.uint8), before, rtol=0, atol=0)
    if rows:
        for _ in range(3):
            rocm_v41_block_fp8_linear(x, w, [32, 32], scales)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cy = rocm_v41_block_fp8_linear(x, w, [32, 32], scales)
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(cy, ref, rtol=0.02, atol=0.25)
    print("LINEAR PASS", rows, width, outputs, flush=True)
print("ALL ROCM STOCK-QUANT ADAPTER CHECKS PASSED", flush=True)
