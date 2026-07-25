#!/usr/bin/env python3
"""Measure the two TPU GEMM pilot shapes through OperatorX's JAX backend.

This is a methodology probe, not the production TPU runner. It validates sampled
numerical correctness and single-device placement, saves StableHLO and optimized
HLO, and reports synchronized dispatch-to-ready latency separately from queued
steady-state throughput. With --profile, XProf module duration is the canonical
device measurement described in docs/TPU_BENCHMARK_METHODOLOGY.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
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


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _device_metadata(device: Any) -> dict[str, Any]:
    return {
        "id": device.id,
        "platform": device.platform,
        "device_kind": device.device_kind,
        "process_index": device.process_index,
        "coords": list(getattr(device, "coords", ())),
        "core_on_chip": getattr(device, "core_on_chip", None),
        "description": str(device),
    }


def _validate_placement(context: dict[str, Any], device: Any) -> dict[str, Any]:
    placements = {}
    for name in ("A", "B", "out"):
        array = context[name]
        devices = sorted(array.devices(), key=lambda candidate: candidate.id)
        if devices != [device]:
            raise ValueError(f"{name} placed on {devices}, expected only {device}")
        placements[name] = [candidate.id for candidate in devices]
    return {
        "validated": True,
        "array_device_ids": placements,
        "input_shardings": str(context["fn"].input_shardings),
        "output_shardings": str(context["fn"].output_shardings),
    }


def _validate_correctness(
    *,
    jax: Any,
    context: dict[str, Any],
    m: int,
    n: int,
) -> dict[str, Any]:
    """Check sampled output cells against an independent NumPy FP32 dot."""
    import numpy as np

    row_indices = np.unique(np.array([0, min(1, m - 1), m - 1]))
    column_indices = np.unique(np.array([0, min(1, n - 1), n - 1]))
    a_rows = np.asarray(
        jax.device_get(context["A"][row_indices, :]),
        dtype=np.float32,
    )
    b_columns = np.asarray(
        jax.device_get(context["B"][:, column_indices]),
        dtype=np.float32,
    )
    actual = np.asarray(
        jax.device_get(
            context["out"][row_indices[:, None], column_indices[None, :]]
        ),
        dtype=np.float32,
    )
    expected = a_rows @ b_columns
    absolute_error = np.abs(actual - expected)
    relative_error = absolute_error / np.maximum(np.abs(expected), 1e-6)
    rtol = 0.01
    atol = 0.5
    passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    if not passed:
        raise ValueError(
            "sampled GEMM correctness check failed: "
            f"max_abs_error={absolute_error.max()}, "
            f"max_rel_error={relative_error.max()}"
        )
    return {
        "validated": True,
        "reference": "numpy_fp32_sampled_dot",
        "sampled_elements": int(actual.size),
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": float(absolute_error.max()),
        "max_rel_error": float(relative_error.max()),
    }


def _capture_hlo(
    *,
    jax_backend: Any,
    context: dict[str, Any],
    output_dir: Path,
    name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lowered = jax_backend.lower_gemm(context["A"], context["B"])
    stablehlo = lowered.as_text(dialect="stablehlo")
    optimized_hlo = context["fn"].as_text()
    if "stablehlo.dot_general" not in stablehlo:
        raise ValueError("StableHLO does not contain the expected dot_general")
    if "HloModule jit_dot" not in optimized_hlo:
        raise ValueError("optimized HLO is not the expected jit_dot module")

    algorithm = {
        "optimized_opcode": "convolution"
        if re.search(r"\bconvolution\(", optimized_hlo)
        else "unknown",
        "emitter": None,
        "window_config": None,
        "used_scoped_memory_configs": None,
        "cross_program_prefetch": "cross_program_prefetch_index" in optimized_hlo,
    }
    backend_config_lines = [
        line
        for line in optimized_hlo.splitlines()
        if "convolution_algorithm_config" in line and "backend_config=" in line
    ]
    if backend_config_lines:
        backend_config = json.loads(
            backend_config_lines[-1].split("backend_config=", 1)[1]
        )
        algorithm["emitter"] = backend_config.get(
            "convolution_algorithm_config", {}
        ).get("emitter")
        algorithm["window_config"] = backend_config.get("window_config")
        algorithm["used_scoped_memory_configs"] = backend_config.get(
            "used_scoped_memory_configs"
        )

    stablehlo_path = output_dir / f"{name}.stablehlo.mlir"
    optimized_hlo_path = output_dir / f"{name}.optimized_hlo.txt"
    stablehlo_path.write_text(stablehlo)
    optimized_hlo_path.write_text(optimized_hlo)
    return {
        "validated": True,
        "stablehlo_contains": "stablehlo.dot_general",
        "optimized_module": "jit_dot",
        "compiler_algorithm": algorithm,
        "stablehlo_path": str(stablehlo_path),
        "optimized_hlo_path": str(optimized_hlo_path),
    }


def _capture_xprof(
    *,
    jax: Any,
    impl: Any,
    context: dict[str, Any],
    profile_dir: Path,
    annotation_name: str,
    profile_iters: int,
) -> dict[str, Any]:
    from operatorx.runners.tpu.xprof import (
        find_perfetto_trace,
        parse_xla_module_durations,
    )

    profile_dir.mkdir(parents=True, exist_ok=True)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 1
    options.advanced_configuration = {
        "tpu_trace_mode": "TRACE_ONLY_XLA",
        "tpu_num_chips_to_profile_per_task": 1,
    }
    with jax.profiler.trace(
        profile_dir,
        create_perfetto_trace=True,
        profiler_options=options,
    ):
        with jax.profiler.StepTraceAnnotation(
            annotation_name,
            profile_iterations=profile_iters,
        ):
            outputs = []
            for _ in range(profile_iters):
                impl.kernel(context)
                outputs.append(context["out"])
            jax.block_until_ready(outputs)

    trace_path = find_perfetto_trace(profile_dir)
    return parse_xla_module_durations(
        trace_path,
        module_name="jit_dot",
        expected_samples=profile_iters,
        annotation_name=annotation_name,
    )


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
    artifacts_dir: Path | None,
    profile: bool,
    profile_iters: int,
    jax_backend: Any,
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

    placement = _validate_placement(context, jax.devices()[0])
    correctness = _validate_correctness(
        jax=jax,
        context=context,
        m=shape["args"]["m"],
        n=shape["args"]["n"],
    )
    hlo = None
    device_execution = None
    if artifacts_dir is not None:
        hlo = _capture_hlo(
            jax_backend=jax_backend,
            context=context,
            output_dir=artifacts_dir / "hlo",
            name=shape["name"],
        )
        if profile:
            device_execution = _capture_xprof(
                jax=jax,
                impl=impl,
                context=context,
                profile_dir=artifacts_dir / "profiles" / shape["name"],
                annotation_name=f"operatorx_{shape['name']}",
                profile_iters=profile_iters,
            )

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
    row = {
        "name": shape.get("name"),
        "shape": {"m": args["m"], "n": args["n"], "k": args["k"]},
        "dtype": "bf16",
        "backend": "jax",
        "device": _device_metadata(jax.devices()[0]),
        "placement": placement,
        "correctness": correctness,
        "hlo": hlo,
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
        "queued_achieved_tflops": achieved_tflops,
        "pct_of_chiplet_peak": achieved_tflops / chiplet_peak_tflops * 100.0,
    }
    if device_execution is not None:
        device_p50_us = device_execution["p50"]
        device_tflops = flops / (device_p50_us * 1e-6) / 1e12
        row["device_execution_us"] = device_execution
        row["device_achieved_tflops"] = device_tflops
        row["device_pct_of_chiplet_peak"] = (
            device_tflops / chiplet_peak_tflops * 100.0
        )
        row["queued_proxy_error_pct"] = (
            (queued_p50_us - device_p50_us) / device_p50_us * 100.0
        )
    return row


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
        "--artifacts-dir",
        type=Path,
        help="write StableHLO, optimized HLO, and optional XProf profiles here",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture and parse an XProf TRACE_ONLY_XLA profile for each shape",
    )
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument(
        "--chiplet-peak-tflops",
        type=float,
        default=1153.5,
        help="Ironwood BF16 peak per JAX-visible chiplet (2307 TFLOP/s per chip / 2).",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.profile and args.artifacts_dir is None:
        parser.error("--profile requires --artifacts-dir")

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
                artifacts_dir=args.artifacts_dir,
                profile=args.profile,
                profile_iters=args.profile_iters,
                jax_backend=jax_backend,
            )
            for shape in shapes
        ]

    print(
        "name                 shape              dispatch_p50_us  "
        "queued_p50_us  xprof_p50_us  queued_TF/s  xprof_TF/s  proxy_err%"
    )
    for row in rows:
        shape = row["shape"]
        shape_text = f"{shape['m']}x{shape['n']}x{shape['k']}"
        device_execution = row.get("device_execution_us")
        xprof_us = (
            f"{device_execution['p50']:.2f}" if device_execution else "-"
        )
        xprof_tflops = (
            f"{row['device_achieved_tflops']:.2f}"
            if "device_achieved_tflops" in row
            else "-"
        )
        proxy_error = (
            f"{row['queued_proxy_error_pct']:.2f}"
            if "queued_proxy_error_pct" in row
            else "-"
        )
        print(
            f"{row['name']:<20} {shape_text:<18} "
            f"{row['dispatch_to_ready_us']['p50']:>15.2f} "
            f"{row['queued_throughput_us']['p50']:>14.2f} "
            f"{xprof_us:>13} "
            f"{row['achieved_tflops']:>8.2f} "
            f"{xprof_tflops:>11} "
            f"{proxy_error:>10}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        run_metadata = {
            "software": {
                "python": platform.python_version(),
                "jax": _package_version("jax"),
                "jaxlib": _package_version("jaxlib"),
                "libtpu": _package_version("libtpu"),
            },
            "hostname": platform.node(),
            "argv": sys.argv,
            "backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "devices": [_device_metadata(device) for device in jax.devices()],
            "timing_contracts": {
                "device_execution_us": "XProf TRACE_ONLY_XLA module critical path",
                "queued_throughput_us": "retained-output batch completion per op",
                "dispatch_to_ready_us": "host dispatch through device readiness",
            },
            "residency": "steady_state",
        }
        args.json_out.write_text(
            json.dumps({"run": run_metadata, "rows": rows}, indent=2)
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
