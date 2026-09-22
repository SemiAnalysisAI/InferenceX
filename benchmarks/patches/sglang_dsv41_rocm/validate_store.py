"""Check installed V4.1 pool stores above 2 GiB, with owned overflow guard memory."""

import json

import torch
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_flashmla
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool


def check(page_size):
    stride = KVLayout.V4.page_bytes(page_size)
    first_high = (2**31 + stride - 1) // stride
    pages = [first_high - 2, first_high, first_high + 1]
    guard_size = 2**31
    backing = torch.empty(
        guard_size + (pages[-1] + 1) * stride, dtype=torch.uint8, device="cuda"
    )
    cache = backing[guard_size:].view(-1, stride)
    reference = torch.full((1, stride), 165, dtype=torch.uint8, device="cuda")
    row = (torch.arange(512, device="cuda", dtype=torch.float32) - 256).bfloat16()[None]
    triton_fused_store_flashmla(
        row, reference, torch.zeros(1, dtype=torch.int64, device="cuda"), page_size
    )
    pool = DeepSeekV4SingleKVPool.__new__(DeepSeekV4SingleKVPool)
    pool._infx_v41_hip = True
    pool.kv_layout = KVLayout.V4
    pool.page_size = page_size
    pool.kv_buffer = [cache]
    for page in pages:
        loc = torch.tensor([page * page_size], dtype=torch.int64, device="cuda")
        wrapped = ((page * stride + 2**31) % 2**32) - 2**31
        guard_start = guard_size + wrapped
        assert 0 <= guard_start <= backing.numel() - stride
        for capture in (False, True):
            cache[page].fill_(165)
            if wrapped < 0:
                backing[guard_start : guard_start + stride].fill_(165)
            if capture:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    pool.set_key_buffer_fused(0, loc, row)
                cache[page].fill_(165)
                if wrapped < 0:
                    backing[guard_start : guard_start + stride].fill_(165)
                graph.replay()
            else:
                pool.set_key_buffer_fused(0, loc, row)
            torch.cuda.synchronize()
            # Compare the complete page, including untouched padding, FP8 bytes,
            # UE8M0 scales and BF16 RoPE, to the stock store at a safe address.
            torch.testing.assert_close(cache[page], reference[0], rtol=0, atol=0)
            if wrapped < 0:
                assert bool(
                    torch.all(backing[guard_start : guard_start + stride] == 165)
                )
            if capture:
                del graph
        print(
            json.dumps(
                {
                    "page_size": page_size,
                    "slot": int(loc.item()),
                    "byte_offset": page * stride,
                    "eager_graph_exact": True,
                }
            ),
            flush=True,
        )
    # Other model pools retain the original store path; verify its safe output.
    pool._infx_v41_hip = False
    pool.kv_buffer = [torch.full_like(reference, 165)]
    pool.set_key_buffer_fused(0, torch.zeros(1, dtype=torch.int64, device="cuda"), row)
    torch.testing.assert_close(pool.kv_buffer[0], reference, rtol=0, atol=0)


for size in [64, 128, 256]:
    check(size)
    torch.cuda.empty_cache()
print("Installed V4.1 pool wide-offset stores and unchanged fallback PASS", flush=True)
