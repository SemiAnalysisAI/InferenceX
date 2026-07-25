#!/usr/bin/env python3
"""Measure the two TPU GEMM pilot shapes through OperatorX's JAX backend.

This is a methodology probe, not the production TPU runner. It deliberately
reports synchronized dispatch-to-ready latency separately from queued
steady-state throughput. XProf device duration remains the canonical
calibration measurement described in docs/TPU_BENCHMARK_METHODOLOGY.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def _percentile(samples: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ready_context(jax: Any, context: dict[str, Any]) -> None:
    arrays = [value for value in context.values() if hasattr(value, "block_until_ready")]
    jax.block_until_ready(arrays)


def _load_pilot_shapes(path: Path) -> list[dict[str, Any]]:
    shapes = json.loads(path.read_text())
    for shape in shapes:
        if shape.get("type") != "gemm":
            raise ValueError(f"pilot only accepts gemm entries, got {shape.get('type')!r}")
        args = shape["args"]
        if args.get("dtype_a") != "bf16" or args.get("dtype_b") != "bf16":
            raise ValueError("pilot currently requires bf16 inputs")
    return shapes


def _measure_shape(
    *,
    jax: Any,
    op_cls: Any,
    impl: Any,
    shape: dict[str, Any],
    warmup: int,
    sync_iters: int,
    batches: int,
    batch_iters: int,
    chiplet_peak_tflops: float,
) -> dict[str, Any]:
    op = op_cls(
        type="gemm",
        args=shape["args"],
        backend="jax",
        name=shape.get("name"),
    )
    context = impl.prepare(op)
    _ready_context(jax, context)

    for _ in range(warmup):
        impl.kernel(context)
        jax.block_until_ready(context["out"])

    synchronized_us: list[float] = []
    for _ in range(sync_iters):
        started_ns = time.perf_counter_ns()
        impl.kernel(context)
        jax.block_until_ready(context["out"])
        synchronized_us.append((time.perf_counter_ns() - started_ns) / 1_000.0)

    queued_us: list[float] = []
    enqueue_us: list[float] = []
    for _ in range(batches):
        outputs = []
        started_ns = time.perf_counter_ns()
        for _ in range(batch_iters):
            impl.kernel(context)
            outputs.append(context["out"])
        enqueued_ns = time.perf_counter_ns()
        # Retain and wait for every output. Waiting only for the final independent
        # dispatch does not establish that every earlier dispatch has completed.
        jax.block_until_ready(outputs)
        finished_ns = time.perf_counter_ns()
        enqueue_us.append((enqueued_ns - started_ns) / 1_000.0 / batch_iters)
        queued_us.append((finished_ns - started_ns) / 1_000.0 / batch_iters)

    args = shape["args"]
    flops = 2 * args["m"] * args["n"] * args["k"]
    queued_p50_us = statistics.median(queued_us)
    achieved_tflops = flops / (queued_p50_us * 1e-6) / 1e12
    return {
        "name": shape.get("name"),
        "shape": {"m": args["m"], "n": args["n"], "k": args["k"]},
        "dtype": "bf16",
        "backend": "jax",
        "device": str(jax.devices()[0]),
        "residency": "steady_state",
        "dispatch_to_ready_us": {
            "p10": _percentile(synchronized_us, 0.10),
            "p50": statistics.median(synchronized_us),
            "p90": _percentile(synchronized_us, 0.90),
            "samples": len(synchronized_us),
        },
        "queued_throughput_us": {
            "p10": _percentile(queued_us, 0.10),
            "p50": queued_p50_us,
            "p90": _percentile(queued_us, 0.90),
            "batches": len(queued_us),
            "iterations_per_batch": batch_iters,
        },
        "enqueue_us_per_op": {
            "p50": statistics.median(enqueue_us),
        },
        "achieved_tflops": achieved_tflops,
        "pct_of_chiplet_peak": achieved_tflops / chiplet_peak_tflops * 100.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testlist",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testlists" / "tpu_gemm_pilot.json",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sync-iters", type=int, default=10)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--batch-iters", type=int, default=64)
    parser.add_argument(
        "--chiplet-peak-tflops",
        type=float,
        default=1153.5,
        help="Ironwood BF16 peak per JAX-visible chiplet (2307 TFLOP/s per chip / 2).",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import jax

    from operatorx.core import Op
    from operatorx.runners.tpu.backends import jax as jax_backend

    if jax.default_backend() != "tpu":
        raise SystemExit(f"expected TPU backend, got {jax.default_backend()!r}")
    impl = next(candidate for candidate in jax_backend.IMPLS if candidate.op_type == "gemm")
    shapes = _load_pilot_shapes(args.testlist)

    with jax.default_device(jax.devices()[0]):
        rows = [
            _measure_shape(
                jax=jax,
                op_cls=Op,
                impl=impl,
                shape=shape,
                warmup=args.warmup,
                sync_iters=args.sync_iters,
                batches=args.batches,
                batch_iters=args.batch_iters,
                chiplet_peak_tflops=args.chiplet_peak_tflops,
            )
            for shape in shapes
        ]

    print(
        "name                 shape              dispatch_p50_us  "
        "queued_p50_us  enqueue_us/op  TFLOP/s  %chiplet_peak"
    )
    for row in rows:
        shape = row["shape"]
        shape_text = f"{shape['m']}x{shape['n']}x{shape['k']}"
        print(
            f"{row['name']:<20} {shape_text:<18} "
            f"{row['dispatch_to_ready_us']['p50']:>15.2f} "
            f"{row['queued_throughput_us']['p50']:>14.2f} "
            f"{row['enqueue_us_per_op']['p50']:>14.2f} "
            f"{row['achieved_tflops']:>8.2f} "
            f"{row['pct_of_chiplet_peak']:>13.2f}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
