"""Validate the installed FP4 split-buffer store/read paths on MI355X."""

import torch
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool

pool = DeepSeekV4IndexerPool.__new__(DeepSeekV4IndexerPool)
pool._infx_v41_hip = True
pool.use_fp4_indexer = True
pool.index_k_rne = True
pool.start_layer = 0
pool.page_size = 64
pool.index_k_payload_buffer = [
    torch.zeros(2, 1, 4, 64, 16, device="cuda", dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2
    )
]
pool.index_k_scale_buffer = [torch.zeros(2, 1, 4, 64, device="cuda", dtype=torch.uint8)]
values = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    device="cuda",
    dtype=torch.bfloat16,
).repeat(8)
exponents = torch.tensor([0, 1, -1, 2, -2, 3], device="cuda")
rows = (values.repeat(6, 1) * torch.exp2(exponents.float()).unsqueeze(-1)).bfloat16()
slots = torch.tensor([1, 16, 63, 64, 65, 127], device="cuda", dtype=torch.int64)
# FP4 E2M1 nibble codes, positive zero retained for both zero entries.
expected = (
    torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE],
        device="cuda",
        dtype=torch.uint8,
    )
    .repeat(8)
    .repeat(6, 1)
)
pool.set_index_fp4(0, slots, rows)
payload, scale = pool.get_index_k_fp4(0, slots)
torch.testing.assert_close(payload.view(torch.uint8), expected, rtol=0, atol=0)
expected_scale = (exponents + 127).to(torch.uint8).unsqueeze(-1).expand(-1, 4)
torch.testing.assert_close(
    scale.view(torch.uint8).view(-1, 4), expected_scale, rtol=0, atol=0
)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    pool.set_index_fp4(0, slots, rows)
    captured, captured_scale = pool.get_index_k_fp4(0, slots)
graph.replay()
torch.cuda.synchronize()
torch.testing.assert_close(captured.view(torch.uint8), expected, rtol=0, atol=0)
torch.testing.assert_close(
    captured_scale.view(torch.uint8).view(-1, 4), expected_scale, rtol=0, atol=0
)
print("Installed FP4 split indexer store/read exact bytes and graph PASS", flush=True)
