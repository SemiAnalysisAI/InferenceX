#!/usr/bin/env python3
"""Single-GPU vLLM block-copy benchmark; independent of the EP result schema."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from pathlib import Path


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def plan_cases(
    directions: list[str],
    block_sizes: list[int],
    counts: list[int],
    max_payload_bytes: int | None,
) -> tuple[list[dict], list[dict]]:
    """Bound copied payload before allocation and retain excluded-point provenance."""
    if max_payload_bytes is not None and max_payload_bytes <= 0:
        raise ValueError("max payload bytes must be positive")
    runnable, skipped = [], []
    for direction in directions:
        for block_bytes in block_sizes:
            for count in counts:
                if block_bytes <= 0 or count <= 0:
                    raise ValueError("block sizes and counts must be positive")
                point = {
                    "direction": direction,
                    "block_bytes": block_bytes,
                    "count": count,
                }
                if (
                    max_payload_bytes is not None
                    and block_bytes * count > max_payload_bytes
                ):
                    skipped.append({**point, "reason": "exceeds-max-payload-bytes"})
                else:
                    runnable.append(point)
    if not runnable:
        raise ValueError("no runnable points within the requested payload budget")
    return runnable, skipped


def block_pairs(count: int, layout: str, seed: int) -> list[list[int]]:
    """Use disjoint buffers and leave a destination guard block untouched."""
    if count <= 0:
        raise ValueError("block count must be positive")
    if layout == "contiguous":
        destinations = list(range(count))
    elif layout == "random":
        import random

        destinations = random.Random(seed).sample(range(count), count)
    else:
        raise ValueError(f"unknown layout: {layout}")
    return [[source, target] for source, target in enumerate(destinations)]


def summarize(samples_us: list[float], payload_bytes: int) -> dict:
    if not samples_us or any(not math.isfinite(x) or x <= 0 for x in samples_us):
        raise ValueError("timing samples must be finite and positive")
    ordered = sorted(samples_us)
    percentiles = {
        f"p{q}": ordered[math.ceil(q / 100 * len(ordered)) - 1]
        for q in (50, 90, 95, 99)
    }
    return {
        "sample_count": len(ordered),
        "percentiles_us": percentiles,
        "payload_gbps_at_latency_percentile": {
            key: payload_bytes / (value * 1000) for key, value in percentiles.items()
        },
        "samples_us": samples_us,
    }


def measure(operation, synchronize, warmup: int, iterations: int) -> list[float]:
    """Drained wall time includes Python/C++ submission and completion waiting."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    for _ in range(warmup):
        operation()
        synchronize()
    samples = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter_ns()
        operation()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return samples


def run_case(
    torch,
    swap_blocks,
    *,
    device: int,
    direction: str,
    block_bytes: int,
    count: int,
    layout: str,
    seed: int,
    warmup: int,
    iterations: int,
) -> dict:
    pairs = block_pairs(count, layout, seed)
    gpu = torch.device(f"cuda:{device}")
    src_device = "cpu" if direction == "h2d" else gpu
    dst_device = "cpu" if direction == "d2h" else gpu
    if direction not in ("h2d", "d2h", "d2d") or block_bytes <= 0:
        raise ValueError("invalid transfer direction or block size")
    # uint8 makes block_bytes literal for both the old inferred-size API and
    # the new explicit-size API. Two extra blocks exercise untouched memory.
    shape = (count + 2, block_bytes)
    source = torch.empty(
        shape, dtype=torch.uint8, device=src_device, pin_memory=src_device == "cpu"
    )
    destination = torch.empty(
        shape, dtype=torch.uint8, device=dst_device, pin_memory=dst_device == "cpu"
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reference = torch.randint(1, 256, shape, dtype=torch.uint8, generator=generator)
    source.copy_(reference)
    mapping = torch.tensor(pairs, dtype=torch.int64, device="cpu")
    expected = torch.zeros(shape, dtype=torch.uint8)
    for src, dst in pairs:
        expected[dst].copy_(reference[src])

    # Resolve the installed wrapper once, outside timing. Never retry an op
    # after TypeError: it may already have submitted device work.
    parameters = inspect.signature(swap_blocks).parameters
    if "block_size_in_bytes" in parameters:
        api = "explicit-block-size"

        def operation():
            swap_blocks(source, destination, block_bytes, mapping)
    elif len(parameters) == 3:
        api = "inferred-block-size"

        def operation():
            swap_blocks(source, destination, mapping)
    else:
        raise RuntimeError(f"unsupported swap_blocks signature: {parameters}")

    def synchronize():
        torch.cuda.synchronize(gpu)

    def check():
        synchronize()
        if not torch.equal(destination.cpu(), expected):
            raise RuntimeError(
                "swap_blocks correctness failed (copied or untouched blocks)"
            )
        if not torch.equal(source.cpu(), reference):
            raise RuntimeError("swap_blocks modified the source")

    destination.zero_()
    synchronize()
    operation()
    check()
    # Clear the successful preflight so the timed run cannot inherit its output.
    destination.zero_()
    samples = measure(operation, synchronize, warmup, iterations)
    check()
    payload_bytes = count * block_bytes
    return {
        "direction": direction,
        "block_bytes": block_bytes,
        "num_blocks": count,
        "layout": layout,
        "seed": seed,
        "payload_bytes": payload_bytes,
        "host_memory": "pinned" if direction != "d2d" else None,
        "api": api,
        "correctness_passed": True,
        "latency": summarize(samples, payload_bytes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=("h2d", "d2h", "d2d"),
        default=["h2d", "d2h", "d2d"],
    )
    parser.add_argument(
        "--block-bytes", nargs="+", type=positive_int, default=[4096, 65536, 1048576]
    )
    parser.add_argument(
        "--num-blocks", nargs="+", type=positive_int, default=[1, 16, 256]
    )
    parser.add_argument("--layout", choices=("contiguous", "random"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument(
        "--max-payload-bytes",
        type=positive_int,
        help="exclude points whose block_bytes * num_blocks exceeds this limit",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.device < 0:
        parser.error("warmup and device must be non-negative")
    try:
        points, skipped = plan_cases(
            args.directions, args.block_bytes, args.num_blocks, args.max_payload_bytes
        )
    except ValueError as exc:
        parser.error(str(exc))

    import torch
    import vllm
    from vllm._custom_ops import swap_blocks

    if not torch.cuda.is_available():
        raise RuntimeError(
            "swap_blocks requires a CUDA or ROCm GPU and matching vLLM build"
        )
    torch.cuda.set_device(args.device)
    result = {
        "schema": "collectivex-swap-blocks-v1",
        "operation": "vllm._custom_ops.swap_blocks",
        "timing": "drained-wall-clock-including-submission-and-synchronization",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "selection": {
            "requested_block_bytes": args.block_bytes,
            "requested_num_blocks": args.num_blocks,
            "max_payload_bytes": args.max_payload_bytes,
            "skipped_cases": skipped,
        },
        "runtime": {
            "sku": os.environ.get("COLLX_SHARD_SKU"),
            "torch": str(torch.__version__),
            "vllm": vllm.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "device": torch.cuda.get_device_name(args.device),
            "device_index": args.device,
            "image": os.environ.get("COLLECTIVEX_IMAGE"),
            "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA")
            or os.environ.get("GITHUB_SHA"),
        },
        "cases": [],
    }
    print(
        f"Selected {len(points)} points; excluded {len(skipped)} over budget",
        flush=True,
    )
    for point in points:
        row = run_case(
            torch,
            swap_blocks,
            device=args.device,
            **point,
            layout=args.layout,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result["cases"].append(row)
        print(
            f"{row['direction']} block_bytes={row['block_bytes']} blocks={row['num_blocks']}: "
            f"p50={row['latency']['percentiles_us']['p50']:.3f} us",
            flush=True,
        )
    # The caller supplies an existing output directory, including in containers.
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
