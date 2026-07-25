#!/usr/bin/env python3
"""Profile many TPU GEMM shapes in one XProf session, then parse once."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import tpu_gemm_pilot as pilot


def _prepare_shape(
    *,
    jax: Any,
    op_cls: Any,
    impl: Any,
    shape: dict[str, Any],
    warmup: int,
    artifacts_dir: Path,
    jax_backend: Any,
) -> dict[str, Any]:
    op = op_cls(
        type="gemm",
        args=shape["args"],
        backend="jax",
        name=shape["name"],
    )
    context = impl.prepare(op)
    pilot._ready_context(jax, context)
    for _ in range(warmup):
        impl.kernel(context)
        jax.block_until_ready(context["out"])

    return {
        "shape": shape,
        "context": context,
        "placement": pilot._validate_placement(context, jax.devices()[0]),
        "correctness": pilot._validate_correctness(
            jax=jax,
            context=context,
            m=int(shape["args"]["m"]),
            n=int(shape["args"]["n"]),
        ),
        "hlo": pilot._capture_hlo(
            jax_backend=jax_backend,
            context=context,
            output_dir=artifacts_dir / "hlo",
            name=shape["name"],
        ),
    }


def _profile_all_shapes(
    *,
    jax: Any,
    impl: Any,
    prepared: list[dict[str, Any]],
    profile_dir: Path,
    profile_iters: int,
) -> list[dict[str, Any]]:
    profile_dir.mkdir(parents=True, exist_ok=True)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 1
    options.advanced_configuration = {
        "tpu_trace_mode": "TRACE_ONLY_XLA",
        "tpu_num_chips_to_profile_per_task": 1,
    }

    groups = []
    with jax.profiler.trace(
        profile_dir,
        create_perfetto_trace=True,
        profiler_options=options,
    ):
        for index, record in enumerate(prepared):
            name = record["shape"]["name"]
            annotation_name = f"operatorx_batch_{index:03d}_{name}"
            groups.append(
                {
                    "name": name,
                    "annotation_name": annotation_name,
                    "expected_samples": profile_iters,
                }
            )
            print(
                f"[profile {index + 1}/{len(prepared)}] {name}: "
                f"{profile_iters} dispatches",
                flush=True,
            )
            with jax.profiler.StepTraceAnnotation(
                annotation_name,
                profile_iterations=profile_iters,
            ):
                outputs = []
                for _ in range(profile_iters):
                    impl.kernel(record["context"])
                    outputs.append(record["context"]["out"])
                # This synchronization is the boundary that makes sequential
                # run_id grouping unambiguous in the final parse.
                jax.block_until_ready(outputs)
    return groups


def _apply_retention(
    profile_dir: Path,
    trace_path: Path,
    retention: str,
) -> None:
    if retention == "all":
        return
    for artifact in profile_dir.rglob("*"):
        if not artifact.is_file():
            continue
        if retention == "perfetto" and artifact == trace_path:
            continue
        artifact.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--testlist",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "testlists"
        / "tpu_gemm_sweep.json",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=21)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="output directory; default creates /tmp/operatorx-tpu-gemm-batch.*",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--profile-retention",
        choices=("all", "perfetto", "summary"),
        default="perfetto",
    )
    parser.add_argument("--chiplet-peak-tflops", type=float, default=1153.5)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.profile_iters <= 0:
        parser.error("--profile-iters must be positive")
    return args


def main() -> int:
    args = _parse_args()
    artifacts_dir = (
        args.artifacts_dir.expanduser().resolve()
        if args.artifacts_dir is not None
        else Path(tempfile.mkdtemp(prefix="operatorx-tpu-gemm-batch.", dir="/tmp"))
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_out = (
        args.json_out.expanduser().resolve()
        if args.json_out is not None
        else artifacts_dir / "results.json"
    )

    import jax
    from operatorx.core import Op
    from operatorx.runners.tpu.backends import jax as jax_backend
    from operatorx.runners.tpu.xprof import (
        find_perfetto_trace,
        parse_batched_xla_module_durations,
    )

    if jax.default_backend() != "tpu":
        raise SystemExit(f"expected TPU backend, got {jax.default_backend()!r}")
    shapes = pilot._load_pilot_shapes(args.testlist)
    names = [shape.get("name") for shape in shapes]
    if any(name is None for name in names) or len(names) != len(set(names)):
        raise SystemExit(f"every batch shape needs a unique name, got {names}")
    impl = next(
        candidate for candidate in jax_backend.IMPLS if candidate.op_type == "gemm"
    )

    prepared = []
    with jax.default_device(jax.devices()[0]):
        for index, shape in enumerate(shapes):
            args_shape = shape["args"]
            print(
                f"[prepare {index + 1}/{len(shapes)}] {shape['name']} "
                f"M={args_shape['m']} N={args_shape['n']} K={args_shape['k']}",
                flush=True,
            )
            prepared.append(
                _prepare_shape(
                    jax=jax,
                    op_cls=Op,
                    impl=impl,
                    shape=shape,
                    warmup=args.warmup,
                    artifacts_dir=artifacts_dir,
                    jax_backend=jax_backend,
                )
            )

        profile_dir = artifacts_dir / "profile"
        groups = _profile_all_shapes(
            jax=jax,
            impl=impl,
            prepared=prepared,
            profile_dir=profile_dir,
            profile_iters=args.profile_iters,
        )

    trace_path = find_perfetto_trace(profile_dir)
    parsed = parse_batched_xla_module_durations(
        trace_path,
        module_name="jit_dot",
        groups=groups,
    )

    rows = []
    for index, record in enumerate(prepared):
        shape = record["shape"]
        shape_args = shape["args"]
        device_execution = parsed["groups"][shape["name"]]
        device_execution["profile_retention"] = args.profile_retention
        if args.profile_retention == "summary":
            device_execution["trace_path"] = None
        flops = 2 * shape_args["m"] * shape_args["n"] * shape_args["k"]
        logical_bytes = 2 * (
            shape_args["m"] * shape_args["k"]
            + shape_args["k"] * shape_args["n"]
            + shape_args["m"] * shape_args["n"]
        )
        child_hlo = device_execution["child_hlo"]
        if child_hlo["model_flops"] != [flops]:
            raise ValueError(
                f"{shape['name']}: assigned XProf FLOPs "
                f"{child_hlo['model_flops']} != expected {flops}"
            )
        if child_hlo["raw_bytes_accessed"] != [logical_bytes]:
            raise ValueError(
                f"{shape['name']}: assigned XProf bytes "
                f"{child_hlo['raw_bytes_accessed']} != expected "
                f"{logical_bytes}"
            )
        device_tflops = flops / (device_execution["p50"] * 1e-6) / 1e12
        rows.append(
            {
                "name": shape["name"],
                "shape": {
                    "m": shape_args["m"],
                    "n": shape_args["n"],
                    "k": shape_args["k"],
                },
                "dtype": "bf16",
                "backend": "jax",
                "device": pilot._device_metadata(jax.devices()[0]),
                "placement": record["placement"],
                "correctness": record["correctness"],
                "hlo": record["hlo"],
                "residency": "steady_state",
                "device_execution_us": device_execution,
                "device_achieved_tflops": device_tflops,
                "device_pct_of_chiplet_peak": (
                    device_tflops / args.chiplet_peak_tflops * 100.0
                ),
                "profile_batch_index": index,
            }
        )

    result = {
        "run": {
            "software": {
                "python": platform.python_version(),
                "jax": pilot._package_version("jax"),
                "jaxlib": pilot._package_version("jaxlib"),
                "libtpu": pilot._package_version("libtpu"),
            },
            "hostname": platform.node(),
            "argv": sys.argv,
            "backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "devices": [
                pilot._device_metadata(device) for device in jax.devices()
            ],
            "timing_contract": (
                "one XProf TRACE_ONLY_XLA session; sequential shape blocks; "
                "module events assigned by XLA run_id after capture"
            ),
            "profile_sessions": 1,
            "profile_iters_per_shape": args.profile_iters,
            "shape_count": len(rows),
            "residency": "steady_state",
        },
        "profile_batch": {
            "assignment": parsed["assignment"],
            "total_module_events": parsed["total_module_events"],
            "trace_path": (
                None if args.profile_retention == "summary" else str(trace_path)
            ),
            "retention": args.profile_retention,
        },
        "rows": rows,
    }
    _apply_retention(
        profile_dir,
        trace_path,
        args.profile_retention,
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2))

    print("\nname                 samples    p50_us    p10_us    p90_us")
    for row in rows:
        timing = row["device_execution_us"]
        print(
            f"{row['name']:<20} {timing['samples']:>7} "
            f"{timing['p50']:>10.3f} {timing['p10']:>10.3f} "
            f"{timing['p90']:>10.3f}"
        )
    print(
        f"\none XProf session, {parsed['total_module_events']} module events "
        f"across {len(rows)} shapes"
    )
    print(f"wrote {json_out}")
    print(f"artifacts {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
