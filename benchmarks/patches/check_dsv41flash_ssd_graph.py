"""GPU preflight: real SSD staging must refresh fixed buffers on every replay.

Runs inside the pinned, patched serving image before model startup. Tiny mapped
FP8 tables exercise deduplication, unowned heads, masks, changing IDs, and graph
shape transitions without loading the checkpoint.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vllm.models.deepseek_v4_1.common.engram import (
    EngramDiskStager,
    ParallelEngramEmbedding,
    _engram_disk_tensor,
    validate_disk_graph_config,
    engram_page_ranges,
)


def check() -> None:
    from vllm.config import ParallelConfig

    for size in (0, 1):
        validate_disk_graph_config(SimpleNamespace(parallel_config=ParallelConfig(ubatch_size=size)))
    for options in ({"ubatch_size": 2}, {"enable_dbo": True}, {"pipeline_parallel_size": 2}, {"data_parallel_size": 2}):
        try:
            validate_disk_graph_config(SimpleNamespace(parallel_config=ParallelConfig(**options)))
        except ValueError as error:
            assert "SSD graph staging requires" in str(error), str(error)
        else:
            raise AssertionError(f"unsupported configuration accepted: {options}")
    assert engram_page_ranges(np.array([], dtype=np.int64), 256, 8448) == []
    assert engram_page_ranges(np.array([0, 1, 0, 32]), 256, 8448) == [(0, 4096), (8192, 256)]
    assert engram_page_ranges(np.array([15, 15]), 264, 16384) == [(0, 8192)]
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix="engram-graph-") as directory:
        layers = []
        reference = []
        for idx in range(2):
            # Construct a controlled small shard while exercising the actual
            # production gather, H2D and dequantization methods.
            embed = ParallelEngramEmbedding.__new__(ParallelEngramEmbedding)
            torch.nn.Module.__init__(embed)
            embed.dim = 256
            embed.block_size = 32
            embed.head_start = 2 * idx
            embed.part_n_hash_cols = 2
            embed.n_hash_cols = 3  # rank 1 has one padded/unowned head
            embed.vocab_start_idx = 7 * idx
            embed.vocab_end_idx = 7 * (idx + 1)
            embed._stage = None
            embed._disk_h2d_done = None
            embed._disk_finalized = True
            embed.disk_offload_dir = directory
            values = ((torch.arange(7 * 256).reshape(7, 256) + idx) % 9 - 4).float()
            weights = values.to(torch.float8_e4m3fn)
            scales = (torch.arange(7 * 8).reshape(7, 8) % 3 + 126).to(torch.uint8)
            weight_path = str(Path(directory) / f"table{idx}.weight")
            scale_path = str(Path(directory) / f"table{idx}.scale")
            weights.view(torch.uint8).numpy().tofile(weight_path)
            scales.numpy().tofile(scale_path)
            embed.weight = torch.nn.Parameter(
                _engram_disk_tensor(weight_path, (7, 256), torch.float8_e4m3fn, False),
                requires_grad=False,
            )
            embed.weight_scale_inv = torch.nn.Parameter(
                _engram_disk_tensor(scale_path, (7, 8), torch.uint8, False),
                requires_grad=False,
            )
            layers.append(SimpleNamespace(
                embed_tokens=embed, layer_hash_index=idx,
                staged_rows=torch.zeros((64, 2, 256), dtype=torch.bfloat16, device="cuda"),
            ))
            reference.append(values * torch.pow(2., scales.float() - 127).repeat_interleave(32, 1))

        stager = EngramDiskStager(layers, 64, 2, 3, cache_bytes=2 * 264)
        addresses = [layer.staged_rows.data_ptr() for layer in layers]
        transfers = [layer.embed_tokens._disk_transfer for layer in layers]
        transfer_addresses = [(t.weights.data_ptr(), t.scales.data_ptr(), t.device_indices.data_ptr()) for t in transfers]
        graphs = {}
        for count in (1, 6, 64, 1, 64, 6, 1):
            if count not in graphs:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = torch.stack([layer.staged_rows[:count] for layer in layers])
                    output = output * stager.mask[:count][None, :, None, None]
                graphs[count] = graph, output
            for shift in (0, 0, 3, 11):
                ids = (np.arange(count * 2 * 3).reshape(count, 2, 3) + shift) % 15 - 1
                # Duplicates stress inverse mapping; alternating calls change
                # every table row, exposing a stale capture even at c1.
                if count > 1:
                    ids[1] = ids[0]
                keep = np.arange(count) % 3 != 2
                hashes = torch.tensor(ids, dtype=torch.int32, device="cuda")
                mask = torch.tensor(keep, device="cuda")
                stager.stage(hashes, mask)
                graph, output = graphs[count]
                graph.replay()
                expected = torch.zeros((2, count, 2, 256), dtype=torch.bfloat16)
                for idx, layer in enumerate(layers):
                    embed = layer.embed_tokens
                    for token in range(count):
                        for head in range(2):
                            global_head = embed.head_start + head
                            if global_head >= 3 or not keep[token]:
                                continue
                            row = int(ids[token, idx, global_head])
                            if embed.vocab_start_idx <= row < embed.vocab_end_idx:
                                expected[idx, token, head] = reference[idx][row - embed.vocab_start_idx]
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                assert [layer.staged_rows.data_ptr() for layer in layers] == addresses
                assert [(t.weights.data_ptr(), t.scales.data_ptr(), t.device_indices.data_ptr()) for t in transfers] == transfer_addresses
        assert sum(t.hits for t in transfers) > 0
        assert sum(t.misses for t in transfers) > 0
        assert sum(t.evictions for t in transfers) > 0
        stager.pool.shutdown(wait=True)
        print("Engram SSD full-graph fixed-buffer preflight: 28 cache-hit/miss/eviction replays passed exactly")


if __name__ == "__main__":
    check()
