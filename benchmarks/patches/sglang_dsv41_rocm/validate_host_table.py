"""Run after install.py in the pinned ROCm image on an allocated GPU."""

import os

import torch

from sglang.srt.layers import engram


class SingleRank:
    rank_in_group = 0

    def barrier(self):
        pass

    def broadcast_object(self, value, src):
        return value


assert torch.version.hip is not None
torch.manual_seed(41)
# This tiny standalone test has no model checkpoint or page cache to release.
engram._drop_page_cache_once = lambda reason: None
rows, dim, block = 1024, 512, 32
weight = torch.randn(rows, dim).to(torch.float8_e4m3fn)
scale_bytes = torch.randint(122, 132, (rows, dim // block), dtype=torch.uint8)
dequantized = (
    weight.float().reshape(rows, dim // block, block)
    * torch.exp2(scale_bytes.float() - 127).unsqueeze(-1)
).flatten(-2).bfloat16()
w_bytes = weight.numel()
for layout in ("per_rank", "shared"):
    table = engram._HostTable(
        layout, w_bytes + scale_bytes.numel(), "infx-owned-engram-test", SingleRank()
    )
    table.bytes[:w_bytes].copy_(weight.view(torch.uint8).flatten())
    table.bytes[w_bytes:].copy_(scale_bytes.flatten())
    embedding = engram.EngramEmbedding.__new__(engram.EngramEmbedding)
    torch.nn.Module.__init__(embedding)
    embedding.dim, embedding.tp_size = dim, 1
    embedding.row_start, embedding.rows = 0, rows
    embedding.host_table = table
    embedding.weight = torch.nn.Parameter(
        table.bytes[:w_bytes].view(torch.float8_e4m3fn).view(rows, dim),
        requires_grad=False,
    )
    embedding.scale = torch.nn.Parameter(
        table.bytes[w_bytes:].view(torch.float8_e8m0fnu).view(rows, dim // block),
        requires_grad=False,
    )
    assert embedding.weight.device.type == "cpu"
    assert embedding.weight.data_ptr() == table.bytes.data_ptr()
    assert embedding._table_pointer(embedding.weight) == table.device_ptr
    assert embedding._table_pointer(embedding.scale) == table.device_ptr + w_bytes
    for shape in ((0,), (4,), (16, 8)):
        ids_cpu = torch.randint(0, rows, shape)
        ids = ids_cpu.cuda()
        expected = dequantized[ids_cpu]
        actual = embedding(ids)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        if ids.numel():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = embedding(ids)
            captured.zero_()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured.cpu(), expected, rtol=0, atol=0)
            del graph, captured
        print(layout, shape, "stock class eager/graph exact PASS", flush=True)
    del embedding
    status = torch.cuda.cudart().cudaHostUnregister(table.bytes.data_ptr())
    assert int(status) == 0
    del table.bytes
    table.mm.close()
    if table.fd is not None:
        os.close(table.fd)
print("All HIP host-table pointer checks PASS", flush=True)
