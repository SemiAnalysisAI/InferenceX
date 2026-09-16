"""Verify the installed SSD Engram implementation refreshes rows on graph replay."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.models.deepseek_v4_1.common import engram as implementation


def check_device(device: int, directory: str) -> None:
    with torch.cuda.device(device):
        # Only distributed process-group metadata is stubbed. The table,
        # file-backed loading, gather, dequantization and replay are real.
        with (
            patch.object(
                implementation, "get_tensor_model_parallel_world_size", return_value=1
            ),
            patch.object(implementation, "get_engram_dp_size", return_value=1),
            patch.object(implementation, "engram_head_shard_rank", return_value=0),
        ):
            table = implementation.ParallelEngramEmbedding(
                num_embeddings=8,
                dim=32,
                head_sizes=(8,),
                cpu_offload=True,
                disk_offload_dir=directory,
            )
        table.weight.data.copy_(
            torch.arange(1, 9, dtype=torch.float32)[:, None].expand(8, 32)
        )
        table.weight_scale_inv.data.copy_(
            torch.tensor(
                [[127], [128], [126], [127], [127], [127], [127], [126]],
                dtype=torch.uint8,
            )
        )
        # Construct the lookup portion of Engram only; its projection and gate
        # weights are irrelevant to whether prepare_embeddings refreshes rows.
        layer = implementation.Engram.__new__(implementation.Engram)
        nn.Module.__init__(layer)
        layer.embed_tokens = table
        layer.staged_rows = torch.zeros((4, 1, 32), dtype=torch.bfloat16, device="cuda")
        layer._extra_staged_rows = []
        source_ids = torch.tensor(
            [[0], [2], [2], [-1]], dtype=torch.int64, device="cuda"
        )
        live_ids = source_ids.clone()
        observed = torch.empty_like(layer.staged_rows)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Warm the actual lookup and Triton dequantization before capture.
            layer.prepare_embeddings(live_ids)
            torch.testing.assert_close(
                layer.staged_rows.cpu(),
                torch.tensor([1, 1.5, 1.5, 0], dtype=torch.bfloat16)
                .reshape(4, 1, 1)
                .expand(4, 1, 32),
                rtol=0,
                atol=0,
            )
            stream.synchronize()
            capture = BreakableCUDAGraphCapture()
            with capture:
                # Hash IDs are produced by the preceding GPU graph segment,
                # as in the real model; retrieval must see these current IDs.
                live_ids.copy_(source_ids)
                layer.prepare_embeddings(live_ids)
                observed.copy_(layer.staged_rows)
            for ids, values in (
                ([7, 1, 0, 99], [4, 4, 1, 0]),
                ([3, 3, -1, 4], [4, 4, 0, 5]),
                ([0, 2, 2, -1], [1, 1.5, 1.5, 0]),
            ):
                source_ids.copy_(torch.tensor(ids, dtype=torch.int64).reshape(4, 1))
                capture.replay()
                stream.synchronize()
                expected = (
                    torch.tensor(values, dtype=torch.bfloat16)
                    .reshape(4, 1, 1)
                    .expand(4, 1, 32)
                )
                torch.testing.assert_close(observed.cpu(), expected, rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        print(f"Engram changing-ID replay PASS on CUDA device {device}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError("Engram replay check requires a CUDA GPU")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    for device in range(count):
        with tempfile.TemporaryDirectory(prefix="engram-replay-") as directory:
            check_device(device, directory)
    (args.result_dir / "engram_replay_check.json").write_text(
        json.dumps({"passed": True, "devices": count, "replays_per_device": 3})
    )


if __name__ == "__main__":
    main()
