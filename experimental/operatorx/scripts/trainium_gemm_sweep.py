#!/usr/bin/env python3
"""Profile the TPU exploration GEMM corpus with NKI on one Trainium3 LNC=2.

This is a standalone NKI measurement path: it does not launch the kernel through
PyTorch/XLA.  The NKI compiler produces one NEFF for each distinct padded shape,
the NKI Spike runtime warms it up and captures one NTFF per sample, and
``neuron-profile`` supplies the device critical-path duration and hardware
counters.  Full compiler/profile artifacts are temporary by default; the compact
JSON output retains the measurements, validation gates, versions, and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_TESTLIST = Path(__file__).resolve().parents[1] / "testlists" / "tpu_gemm_sweep.json"
DEFAULT_SAMPLES = 21
DEFAULT_WARMUP = 10

# These are the block constraints of the OperatorX BF16 LNC=2 NKI kernel.
BLOCK_M = 2048
BLOCK_N_LNC2 = 2048
BLOCK_K = 1024

PROFILE_FIELDS = (
    "total_time",
    "total_active_time",
    "hardware_flops",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "tensor_engine_active_time",
    "dma_active_time",
    "hfu_estimated_percent",
    "mfu_estimated_percent",
    "mbu_estimated_percent",
    "neuroncore_cycle_count",
    "matmul_instruction_count",
    "tensor_engine_instruction_count",
    "instance_type",
    "profiler_version",
    "runtime_version",
    "driver_version",
    "collectives_version",
)


def ceil_to(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def executed_shape(row: dict[str, Any]) -> tuple[int, int, int]:
    args = row["args"]
    return (
        ceil_to(int(args["m"]), BLOCK_M),
        ceil_to(int(args["n"]), BLOCK_N_LNC2),
        ceil_to(int(args["k"]), BLOCK_K),
    )


def percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_us(samples_s: list[float]) -> dict[str, Any]:
    samples_us = [value * 1e6 for value in samples_s]
    return {
        "samples": samples_us,
        "count": len(samples_us),
        "p10": percentile(samples_us, 0.10),
        "p50": percentile(samples_us, 0.50),
        "p90": percentile(samples_us, 0.90),
        "min": min(samples_us),
        "max": max(samples_us),
    }


def compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {field: profile[field] for field in PROFILE_FIELDS if field in profile}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    names: set[str] = set()
    for row in rows:
        name = row.get("name")
        args = row.get("args", {})
        if not name or name in names:
            raise ValueError(f"every row needs a unique name; got {name!r}")
        names.add(name)
        if row.get("type") != "gemm":
            raise ValueError(f"{name}: only GEMM rows are supported")
        if (args.get("dtype_a"), args.get("dtype_b"), args.get("dtype_out")) != (
            "bf16",
            "bf16",
            "bf16",
        ):
            raise ValueError(f"{name}: this calibrated sweep requires BF16/BF16/BF16")
        for dimension in ("m", "n", "k"):
            if int(args[dimension]) <= 0:
                raise ValueError(f"{name}: {dimension} must be positive")
    return rows


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return (completed.stdout or completed.stderr).strip()


def environment_manifest() -> dict[str, Any]:
    neuron_ls_json = _command_output(["/opt/aws/neuron/bin/neuron-ls", "-j"])
    try:
        devices = json.loads(neuron_ls_json)
    except json.JSONDecodeError:
        devices = []
    for device in devices:
        processes = device.pop("neuron_processes", [])
        device["neuron_process_count"] = len(processes)
    neuron_ls_text = _command_output(["/opt/aws/neuron/bin/neuron-ls"])
    instance_match = re.search(r"instance-type:\s*(\S+)", neuron_ls_text)
    packages = {}
    for name in ("nki", "nkilib", "neuronx-cc", "torch-neuronx", "torch-xla"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "instance_type": instance_match.group(1) if instance_match else "unknown",
        "neuron_devices": devices,
        "logical_nc_config": 2,
        "logical_device_kind": "NC_v4d",
        "platform_target": "trn3",
        "packages": {
            **packages,
            "nkilib": (
                "bundled-unversioned"
                if importlib.util.find_spec("nkilib")
                else "not-installed"
            ),
        },
        "neuron_tools": _command_output(["/opt/aws/neuron/bin/neuron-profile", "--version"]),
        "neuronx_cc": _command_output(["neuronx-cc", "--version"]).splitlines()[0],
        "python": sys.version.split()[0],
    }


def validate_summary(
    summary: dict[str, Any], *, m: int, n: int, k: int
) -> None:
    expected_flops = 2 * m * n * k
    observed_flops = int(summary.get("hardware_flops", -1))
    # neuron-profile's derived hardware counter can omit a small boundary-tile
    # contribution even when the compiled GEMM and output are complete. Keep
    # this as a tight diagnostic gate rather than treating it as an exact ABI.
    relative_flops_error = abs(observed_flops - expected_flops) / expected_flops
    if relative_flops_error > 1e-4:
        raise RuntimeError(
            "profile hardware FLOPs do not match the executed padded GEMM: "
            f"expected {expected_flops}, got {observed_flops} "
            f"({relative_flops_error:.6%} relative error)"
        )
    if float(summary.get("total_time", 0.0)) <= 0.0:
        raise RuntimeError("profile total_time must be positive")
    if int(summary.get("matmul_instruction_count", 0)) <= 0:
        raise RuntimeError("profile contains no matmul instructions")
    if int(summary.get("hbm_read_bytes", 0)) <= 0 or int(summary.get("hbm_write_bytes", 0)) <= 0:
        raise RuntimeError("profile contains no HBM traffic")


def _profile_executable(
    *,
    compiled: Any,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    artifacts_dir: Path,
    samples: int,
    warmup: int,
    executed: tuple[int, int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one compiled NEFF through NKI Spike and retain one NTFF per sample."""
    import numpy as np
    from nki.compiler.ncc_driver import extract_perf_metrics
    from nki.runtime import SpikeTensor

    model = compiled._ensure_loaded()
    spike_inputs = {name: SpikeTensor.from_numpy(value, name) for name, value in inputs.items()}
    spike_outputs = {
        name: SpikeTensor.from_numpy(value, name) for name, value in outputs.items()
    }

    for _ in range(warmup):
        model(spike_inputs, outputs=spike_outputs, save_trace=False)

    profiles = []
    traces = artifacts_dir / "profiles"
    traces.mkdir()
    for index in range(samples):
        ntff = traces / f"sample-{index:02d}.ntff"
        model(
            spike_inputs,
            outputs=spike_outputs,
            save_trace=True,
            ntff_name=str(ntff),
        )
        summary = extract_perf_metrics(compiled.neff_path, str(ntff), verbose=True)
        if summary is None:
            raise RuntimeError(f"failed to parse native NKI profile {ntff}")
        validate_summary(summary, m=executed[0], n=executed[1], k=executed[2])
        profiles.append(compact_profile(summary))

    for name, spike_tensor in spike_outputs.items():
        np.copyto(outputs[name], spike_tensor.numpy())

    return profiles, {
        "neff_sha256": sha256_file(Path(compiled.neff_path)),
        "neff_size_bytes": Path(compiled.neff_path).stat().st_size,
    }


def profile_padded_shape(
    *,
    executed: tuple[int, int, int],
    artifacts_dir: Path,
    samples: int,
    warmup: int,
) -> dict[str, Any]:
    """Compile, validate, and profile one distinct padded BF16 NKI GEMM."""
    import ml_dtypes
    import numpy as np
    from nki.framework.compiled import CompileKernel
    from operatorx.runners.trainium.backends.nkilib import _matmul_lnc2

    m, n, k = executed
    lhs_t = np.ones((k, m), dtype=ml_dtypes.bfloat16)
    rhs = np.ones((k, n), dtype=ml_dtypes.bfloat16)
    captured_profiles: list[dict[str, Any]] = []
    executable: dict[str, Any] = {}

    def native_profile_executor(compiled: Any, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        nonlocal captured_profiles, executable
        captured_profiles, executable = _profile_executable(
            compiled=compiled,
            inputs=inputs,
            outputs=outputs,
            artifacts_dir=artifacts_dir,
            samples=samples,
            warmup=warmup,
            executed=executed,
        )

    kernel = _matmul_lnc2[2]._to_subclass(
        CompileKernel,
        artifacts_dir=str(artifacts_dir),
        target="trn3",
        _executor=native_profile_executor,
    )
    output = kernel(lhs_t, rhs)

    compiler_info_path = artifacts_dir / "info.json"
    compiler_info = json.loads(compiler_info_path.read_text())
    if int(compiler_info.get("lnc", 0)) != 2:
        raise RuntimeError(
            f"compiled executable did not preserve LNC=2: {compiler_info}"
        )
    executable.update(
        {
            "lnc": 2,
            "compiler_info_sha256": sha256_file(compiler_info_path),
        }
    )

    check_cells = ((0, 0), (m // 2, n // 2), (m - 1, n - 1))
    expected = float(k)
    observed = [float(output[row, column]) for row, column in check_cells]
    max_abs_error = max(abs(value - expected) for value in observed)
    correctness = {
        "validated": max_abs_error == 0.0,
        "cells": [list(cell) for cell in check_cells],
        "expected": expected,
        "observed": observed,
        "max_absolute_error": max_abs_error,
    }
    if not correctness["validated"]:
        raise RuntimeError(f"NKI GEMM correctness failed for {executed}: {correctness}")

    latency = summarize_us([float(profile["total_time"]) for profile in captured_profiles])
    hardware_flops = 2 * m * n * k
    return {
        "executed_shape": {"m": m, "n": n, "k": k},
        "samples": captured_profiles,
        "device_execution_us": latency,
        "correctness": correctness,
        "placement": {
            "validated": True,
            "logical_nc_config": 2,
            "physical_neuroncores": 2,
            "logical_device_kind": "NC_v4d",
            "compiler_info_lnc": executable["lnc"],
        },
        "compiled_program": {
            "validated": True,
            "backend": "nki-standalone-spike",
            "kernel": "operatorx.runners.trainium.backends.nkilib._matmul_lnc2",
            "hardware_flops": hardware_flops,
            **executable,
        },
        "residency": "steady_state",
    }


def build_rows(
    test_rows: list[dict[str, Any]],
    measured: dict[tuple[int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in test_rows:
        args = row["args"]
        logical = (int(args["m"]), int(args["n"]), int(args["k"]))
        padded = executed_shape(row)
        measurement = measured[padded]
        logical_flops = 2 * logical[0] * logical[1] * logical[2]
        executed_flops = 2 * padded[0] * padded[1] * padded[2]
        result.append(
            {
                "name": row["name"],
                "shape": {"m": logical[0], "n": logical[1], "k": logical[2]},
                "executed_shape": {"m": padded[0], "n": padded[1], "k": padded[2]},
                "dtype": "bf16",
                "backend": "nkilib",
                "profile_backend": "nki-native-spike",
                "logical_flops": logical_flops,
                "executed_flops": executed_flops,
                "padding_flops_ratio": executed_flops / logical_flops,
                "logical_bytes": 2
                * (logical[0] * logical[2] + logical[2] * logical[1] + logical[0] * logical[1]),
                "executed_logical_bytes": 2
                * (padded[0] * padded[2] + padded[2] * padded[1] + padded[0] * padded[1]),
                **measurement,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testlist", type=Path, default=DEFAULT_TESTLIST)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--only", help="comma-separated shape names for smoke testing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.warmup < 0:
        raise SystemExit("--samples must be positive and --warmup non-negative")

    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

    rows = load_rows(args.testlist)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        rows = [row for row in rows if row["name"] in wanted]
        missing = wanted - {row["name"] for row in rows}
        if missing:
            raise SystemExit(f"unknown --only names: {sorted(missing)}")

    if args.profile_root:
        profile_root = args.profile_root.resolve()
        profile_root.mkdir(parents=True, exist_ok=False)
        owns_profile_root = False
    else:
        profile_root = Path(tempfile.mkdtemp(prefix="operatorx-trainium-gemm-"))
        owns_profile_root = True

    try:
        distinct_shapes = sorted({executed_shape(row) for row in rows})
        measured: dict[tuple[int, int, int], dict[str, Any]] = {}
        for index, shape in enumerate(distinct_shapes, start=1):
            shape_dir = profile_root / f"m{shape[0]}-n{shape[1]}-k{shape[2]}"
            shape_dir.mkdir()
            print(f"[{index}/{len(distinct_shapes)}] NKI profile M={shape[0]} N={shape[1]} K={shape[2]}", flush=True)
            measured[shape] = profile_padded_shape(
                executed=shape,
                artifacts_dir=shape_dir,
                samples=args.samples,
                warmup=args.warmup,
            )

        output = {
            "schema_version": 1,
            "methodology": {
                "canonical_metric": "NKI Spike NTFF total_time",
                "samples_per_executable": args.samples,
                "warmup_executions": args.warmup,
                "backend": "nkilib",
                "profile_backend": "nki-native-spike",
                "lnc": 2,
                "residency": "steady_state",
                "full_artifacts_retained": bool(args.keep_artifacts),
                "testlist": args.testlist.name,
            },
            "environment": environment_manifest(),
            "rows": build_rows(rows, measured),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote compact results: {args.json_out}")
        print(f"temporary profile root: {profile_root}")
        return 0
    finally:
        if not args.keep_artifacts and (owns_profile_root or args.profile_root):
            # profile_root is either our mkdtemp or an explicit, newly-created
            # directory.  Never delete a pre-existing caller directory.
            shutil.rmtree(profile_root, ignore_errors=True)
            print(f"removed full NKI profiles and compiler logs: {profile_root}")


if __name__ == "__main__":
    raise SystemExit(main())
