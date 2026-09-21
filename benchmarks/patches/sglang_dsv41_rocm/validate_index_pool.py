"""Validate the installed FP4 split-buffer store/read paths on MI355X."""

import torch
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool

pool = DeepSeekV4IndexerPool.__new__(DeepSeekV4IndexerPool)
pool._infx_v41_hip = True
pool.use_fp4_indexer = True
pool.index_k_rne = True
pool.start_layer = 0
pool.page_size = 256
pool.index_k_payload_buffer = [
    torch.zeros(2, 1, 4, 256, 16, device="cuda", dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2
    )
]
pool.index_k_scale_buffer = [
    torch.zeros(2, 1, 4, 256, device="cuda", dtype=torch.uint8)
]
values = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    device="cuda",
    dtype=torch.bfloat16,
).repeat(8)
rows = values.repeat(3, 1)
slots = torch.tensor([1, 255, 257], device="cuda", dtype=torch.int64)
# FP4 E2M1 nibble codes, positive zero retained for both zero entries.
expected = (
    torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE],
        device="cuda",
        dtype=torch.uint8,
    )
    .repeat(8)
    .repeat(3, 1)
)
pool.set_index_fp4(0, slots, rows)
payload, scale = pool.get_index_k_fp4(0, slots)
torch.testing.assert_close(payload.view(torch.uint8), expected, rtol=0, atol=0)
assert (scale.view(torch.uint8) == 127).all()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    pool.set_index_fp4(0, slots, rows)
    captured, captured_scale = pool.get_index_k_fp4(0, slots)
graph.replay()
torch.cuda.synchronize()
torch.testing.assert_close(captured.view(torch.uint8), expected, rtol=0, atol=0)
assert (captured_scale.view(torch.uint8) == 127).all()
print("Installed FP4 split indexer store/read exact bytes and graph PASS", flush=True)
