#!/usr/bin/env python3
"""Microbenchmark the MiniMax M3 decode index-score kernel on ROCm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from vllm.models.minimax_m3.common.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    _decode_index_score_kernel,
)


def _csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _allocate_inputs(
    batch: int,
    seq_len: int,
    max_model_len: int,
) -> tuple[torch.Tensor, ...]:
    num_blocks = math.ceil(seq_len / SPARSE_BLOCK_SIZE)
    max_blocks = math.ceil(max_model_len / SPARSE_BLOCK_SIZE)
    score_stride = math.ceil(max_blocks / 16) * 16
    total_pages = batch * num_blocks

    idx_q = torch.randn(
        batch,
        1,
        128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    index_kv_cache = torch.randn(
        total_pages,
        SPARSE_BLOCK_SIZE,
        128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    page_ids = torch.randperm(total_pages, dtype=torch.int32, device="cuda")
    block_table = page_ids.view(batch, num_blocks)
    seq_lens = torch.full(
        (batch,),
        seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    score = torch.empty(
        1,
        batch,
        score_stride,
        dtype=torch.float32,
        device="cuda",
    )
    return idx_q, index_kv_cache, block_table, seq_lens, score


def _launch(
    tensors: tuple[torch.Tensor, ...],
    chunks: int,
    num_warps: int | None,
    num_stages: int | None,
) -> None:
    idx_q, index_kv_cache, block_table, seq_lens, score = tensors
    batch, num_idx_heads, head_dim = idx_q.shape
    launch_options = {}
    if num_warps is not None:
        launch_options["num_warps"] = num_warps
    if num_stages is not None:
        launch_options["num_stages"] = num_stages
    _decode_index_score_kernel[(batch, chunks)](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        0,
        1,
        head_dim**-0.5,
        1,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        num_kv_chunks=chunks,
        USE_PDL=False,
        **launch_options,
    )


def _benchmark(
    tensors: tuple[torch.Tensor, ...],
    chunks: int,
    num_warps: int | None,
    num_stages: int | None,
    warmup: int,
    repetitions: int,
) -> float:
    for _ in range(warmup):
        _launch(tensors, chunks, num_warps, num_stages)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        _launch(tensors, chunks, num_warps, num_stages)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repetitions


def _chunk_count(target_grid: int, batch: int) -> int:
    target = max(1, min(256, target_grid // batch))
    return 1 << (target.bit_length() - 1)


def _run_shape(
    label: str,
    batch: int,
    seq_len: int,
    max_model_len: int,
    configurations: list[tuple[str, int, int | None, int | None]],
    warmup: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    tensors = _allocate_inputs(batch, seq_len, max_model_len)
    actual_blocks = math.ceil(seq_len / SPARSE_BLOCK_SIZE)
    reference: torch.Tensor | None = None
    results: list[dict[str, Any]] = []

    for config_name, chunks, num_warps, num_stages in configurations:
        _launch(tensors, chunks, num_warps, num_stages)
        torch.cuda.synchronize()
        score = tensors[-1][..., :actual_blocks].clone()
        if reference is None:
            reference = score
            max_abs_error = 0.0
        else:
            max_abs_error = float((score - reference).abs().max().item())

        duration_us = _benchmark(
            tensors,
            chunks,
            num_warps,
            num_stages,
            warmup,
            repetitions,
        )
        result = {
            "label": label,
            "config": config_name,
            "batch": batch,
            "seq_len": seq_len,
            "max_model_len": max_model_len,
            "chunks": chunks,
            "num_warps": num_warps if num_warps is not None else "default",
            "num_stages": num_stages if num_stages is not None else "default",
            "grid_programs": batch * chunks,
            "blocks_per_chunk": math.ceil(actual_blocks / chunks),
            "duration_us": duration_us,
            "max_abs_error": max_abs_error,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    del tensors
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--m3-matrix", action="store_true")
    parser.add_argument("--compare-current", action="store_true")
    parser.add_argument("--chunks", type=_csv_ints, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warps", type=_csv_ints, default=[4])
    parser.add_argument("--stages", type=_csv_ints, default=[2])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.repetitions <= 0:
        parser.error("repetitions must be positive; warmup cannot be negative")

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    if args.m3_matrix:
        shapes = [
            ("1k-c1", 1, 1024, 2304),
            ("1k-c16", 16, 1024, 2304),
            ("1k-c256", 256, 1024, 2304),
            ("8k-c1", 1, 8192, 9472),
            ("8k-c16", 16, 8192, 9472),
            ("8k-c256", 256, 8192, 9472),
        ]
    else:
        if args.batch is None or args.seq_len is None or args.max_model_len is None:
            parser.error(
                "--batch, --seq-len, and --max-model-len are required "
                "unless --m3-matrix is used"
            )
        shapes = [("custom", args.batch, args.seq_len, args.max_model_len)]

    results: list[dict[str, Any]] = []
    for label, batch, seq_len, max_model_len in shapes:
        if batch <= 0 or max_model_len < seq_len:
            parser.error(
                "batch must be positive and max model length must cover seq len"
            )
        if args.compare_current:
            configurations = [
                ("current", _chunk_count(4096, batch), None, None),
                ("tuned", _chunk_count(2048, batch), 2, 3),
            ]
        else:
            configurations = [
                ("custom", chunks, num_warps, num_stages)
                for chunks in args.chunks
                for num_warps in args.warps
                for num_stages in args.stages
            ]
        results.extend(
            _run_shape(
                label,
                batch,
                seq_len,
                max_model_len,
                configurations,
                args.warmup,
                args.repetitions,
            )
        )

    output = {
        "device": torch.cuda.get_device_name(0),
        "results": results,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
