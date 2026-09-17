"""Verify the installed GDS Engram implementation refreshes rows on graph replay."""

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
                disk_shard_id="replay",
            )
        table.weight.weight_loader(
            table.weight, torch.arange(1, 9, dtype=torch.float32)[:, None].expand(8, 32)
        )
        table.weight_scale_inv.weight_loader(
            table.weight_scale_inv,
            torch.tensor(
                [[127], [128], [126], [127], [127], [127], [127], [126]],
                dtype=torch.uint8,
            ),
        )
        # Construct the lookup portion of Engram only; its projection and gate
        # weights are irrelevant to whether prepare_embeddings refreshes rows.
        layer = implementation.Engram.__new__(implementation.Engram)
        nn.Module.__init__(layer)
        layer.embed_tokens = table
        layer.staged_rows = torch.zeros((4, 1, 32), dtype=torch.bfloat16, device="cuda")
        layer._extra_staged_rows = []
        source_ids = torch.tensor(
            [[0], [2], [2], [-1]], dtype=torch.int32, device="cuda"
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
                source_ids.copy_(torch.tensor(ids, dtype=torch.int32).reshape(4, 1))
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


def make_table(directory: str, shard: str, rank: int = 0):
    with (
        patch.object(
            implementation, "get_tensor_model_parallel_world_size", return_value=4
        ),
        patch.object(implementation, "get_engram_dp_size", return_value=1),
        patch.object(implementation, "engram_head_shard_rank", return_value=rank),
    ):
        return implementation.ParallelEngramEmbedding(
            num_embeddings=14,
            dim=32,
            head_sizes=(2, 2, 2, 2, 2, 2, 2),
            cpu_offload=True,
            disk_offload_dir=directory,
            disk_shard_id=shard,
        )


def load_table(table, value: float) -> None:
    table.weight.weight_loader(table.weight, torch.full((14, 32), value))
    table.weight_scale_inv.weight_loader(
        table.weight_scale_inv,
        torch.full((14, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
    )


def check_loading(directory: str) -> None:
    table = make_table(directory, "checkpoint")
    ids = torch.tensor([[0, 2, 4, 6, 8, 10, 12]], dtype=torch.int32, device="cuda")
    out = torch.empty((1, 2, 32), dtype=torch.bfloat16, device="cuda")
    table.weight.weight_loader(table.weight, torch.ones((14, 32)))
    try:
        table.lookup(ids, out)
    except RuntimeError as error:
        if "before both weights and scales loaded" not in str(error):
            raise
    else:
        raise AssertionError("Half-loaded table was served")
    table.weight_scale_inv.weight_loader(
        table.weight_scale_inv, torch.full((14, 1), 127, dtype=torch.uint8)
    )
    table.lookup(ids, out)
    torch.testing.assert_close(
        out.cpu(), torch.ones((1, 2, 32), dtype=torch.bfloat16), rtol=0, atol=0
    )
    # Concurrent writers must fail before modifying this engine's shard.
    try:
        make_table(directory, "checkpoint")
    except RuntimeError as error:
        if "shard is in use" not in str(error):
            raise
    else:
        raise AssertionError("Concurrent writer acquired the same shard")
    try:
        table.weight.weight_loader(table.weight, torch.full((14, 32), 9.0))
    except RuntimeError as error:
        if "requires a new model instance" not in str(error):
            raise
    else:
        raise AssertionError("Finalized COW table accepted a checkpoint rewrite")
    del table
    # Same dimensions, different checkpoint bytes: verify persistence as well
    # as lookup results so a private COW copy cannot conceal a stale disk file.
    table = make_table(directory, "checkpoint")
    load_table(table, 7)
    table.lookup(ids, out)
    torch.testing.assert_close(
        out.cpu(), torch.full((1, 2, 32), 7, dtype=torch.bfloat16), rtol=0, atol=0
    )
    weights = list(Path(directory).glob("*.weight.bin"))
    assert len(weights) == 1
    assert weights[0].read_bytes()[0] == 0x4E  # FP8 e4m3 encoding of 7.
    # Equal-sized layers must remain independent.
    other = make_table(directory, "other-layer")
    load_table(other, 2)
    other.lookup(ids, out)
    torch.testing.assert_close(
        out.cpu(), torch.full((1, 2, 32), 2, dtype=torch.bfloat16), rtol=0, atol=0
    )
    table.lookup(ids, out)
    torch.testing.assert_close(
        out.cpu(), torch.full((1, 2, 32), 7, dtype=torch.bfloat16), rtol=0, atol=0
    )
    # Final TP rank owns rows 12..13 and one real head plus a padded head.
    last = make_table(directory, "tp-boundary", rank=3)
    last.weight.weight_loader(
        last.weight, torch.arange(1, 15).float()[:, None].expand(14, 32)
    )
    last.weight_scale_inv.weight_loader(
        last.weight_scale_inv, torch.full((14, 1), 127, dtype=torch.uint8)
    )
    boundary_ids = torch.tensor(
        [
            [0, 2, 4, 6, 8, 10, 12],
            [0, 2, 4, 6, 8, 10, 13],
            [0, 2, 4, 6, 8, 10, -1],
            [0, 2, 4, 6, 8, 10, 14],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    boundary_out = torch.empty((4, 2, 32), dtype=torch.bfloat16, device="cuda")
    last.lookup(boundary_ids, boundary_out)
    expected = torch.tensor([[13, 0], [14, 0], [0, 0], [0, 0]], dtype=torch.bfloat16)
    torch.testing.assert_close(
        boundary_out.cpu(), expected[:, :, None].expand(4, 2, 32), rtol=0, atol=0
    )
    print(
        "Engram checkpoint reload, writer lock, layer isolation and TP boundary PASS",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--disk-dir", type=Path, required=True)
    args = parser.parse_args()
    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError("Engram replay check requires a CUDA GPU")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    for device in range(count):
        with tempfile.TemporaryDirectory(
            prefix="engram-replay-", dir=args.disk_dir
        ) as directory:
            check_device(device, directory)
        with (
            torch.cuda.device(device),
            tempfile.TemporaryDirectory(
                prefix="engram-load-", dir=args.disk_dir
            ) as directory,
        ):
            check_loading(directory)
    (args.result_dir / "engram_replay_check.json").write_text(
        json.dumps(
            {
                "passed": True,
                "devices": count,
                "replays_per_device": 3,
                "loading_checks": True,
            }
        )
    )


if __name__ == "__main__":
    main()
