"""Join a video-generation job's per-concurrency H3 results to its native power package.

One srt-slurm job serves the model once and runs the H3 client at every CONC_LIST
concurrency. Each cell leaves the client's run bundle (``h3/gpu/c<N>/baseline/run.json``)
and its measured window (``h3/gpu/c<N>/benchmark_concurrency_<N>.json``, also published as
the telemetry window). This module writes one aggregate per concurrency in the runner
workspace, validates its window against the power package with the shared multinode
validator (the server is a recipe service, so its GPUs carry the service's name as their
role, and energy is normalized per valid clip), and summarizes the point.

Invalid or missing power never publishes as measured: the aggregate records
``power_valid=0`` with the validation sidecar's reasons; REQUIRE_POWER fails the point.
An incomplete cell (valid clips != scheduled) fails the point regardless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from infx.results.power import multinode
from infx.results.power.audit import audit_summary
from infx.results.power.common import _write_json_atomic

SCENARIO_TYPE = "video-generation"
SERVICE = "h3-server"
H3_DIR = "h3"
POWER_DIR = "power"
BENCHMARK_STEM = "benchmark_concurrency_{concurrency}"
IDENTITY = ("cluster", "model", "image", "framework", "precision", "spec")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _server_gpu_uuids(logs_root: Path, gpu_count: int) -> tuple[str, ...]:
    """The actual server assignment, sealed in both native H3 launch records."""
    server = logs_root / H3_DIR / "server"
    spec = _read(server / "spec.json").get("gpu_uuids")
    launch = _read(server / "launch.json").get("gpu_uuids")
    if (
        not isinstance(spec, list)
        or len(spec) != gpu_count
        or any(not isinstance(uuid, str) or not uuid for uuid in spec)
        or len(set(spec)) != gpu_count
        or not isinstance(launch, list)
        or len(launch) != gpu_count
        or any(not isinstance(uuid, str) or not uuid for uuid in launch)
        or set(launch) != set(spec)
    ):
        raise ValueError("server spec and launch GPU UUIDs are missing or disagree")
    return tuple(spec)


def build_aggregate(
    run: Mapping[str, Any],
    bench: Mapping[str, Any],
    *,
    concurrency: int,
    gpu_count: int,
    identity: Mapping[str, str],
    run_path: str,
    run_sha256: str,
) -> dict[str, Any]:
    """The flat per-concurrency result the dashboard and power validator consume."""
    summary = run.get("summary") or {}
    serving = run.get("serving") or {}
    measurement = run.get("measurement") or {}
    plan = run.get("plan") or {}
    if measurement.get("concurrency") != concurrency or bench.get("max_concurrency") != concurrency:
        raise ValueError(f"cell c{concurrency}: run and window disagree with the concurrency")
    if bench.get("completed") != summary.get("valid") or bench.get("scheduled") != summary.get(
        "scheduled"
    ):
        raise ValueError(f"cell c{concurrency}: window counts do not match the run summary")
    valid_per_second = summary.get("valid_clips_per_second")
    return {
        "schema_version": 1,
        "scenario_type": SCENARIO_TYPE,
        "evidence_kind": "srt_slurm_h3_service",
        **{key: identity.get(key) for key in IDENTITY},
        "model_id": plan.get("model_id"),
        "model_revision": plan.get("model_revision"),
        "runtime": (run.get("configuration") or {}).get("runtime"),
        "runtime_revision": (run.get("configuration") or {}).get("runtime_revision"),
        "num_gpus": gpu_count,
        "is_multinode": False,
        "disagg": False,
        "plan_id": run.get("plan_id"),
        "plan_sha256": run.get("plan_sha256"),
        "generation": plan.get("generation"),
        "max_concurrency": concurrency,
        "serving_mode": serving.get("mode"),
        "scheduled": summary.get("scheduled"),
        "completed": summary.get("completed"),
        "valid": summary.get("valid"),
        "failed": summary.get("failed"),
        "technical_success_rate": summary.get("technical_success_rate"),
        "latency_median_seconds": summary.get("latency_median_seconds"),
        "latency_mean_seconds": summary.get("latency_mean_seconds"),
        "latency_max_seconds": summary.get("latency_max_seconds"),
        "client_ready_p50_seconds": (serving.get("client_ready_latency_seconds") or {}).get("p50"),
        "valid_clips_per_second": valid_per_second,
        "clips_per_hour": valid_per_second * 3600
        if isinstance(valid_per_second, (int, float))
        else None,
        "valid_video_seconds_per_second": serving.get("valid_video_seconds_per_second"),
        "measurement_wall_seconds": measurement.get("wall_seconds"),
        "benchmark_start_time_unix": bench.get("benchmark_start_time_unix"),
        "benchmark_end_time_unix": bench.get("benchmark_end_time_unix"),
        "duration": bench.get("duration"),
        "energy_normalization": bench.get("energy_normalization"),
        "run_status": run.get("status"),
        "run_path": run_path,
        "run_sha256": run_sha256,
        "ci": {
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.environ.get("RUNNER_NAME"),
        },
    }


def _complete(aggregate: Mapping[str, Any]) -> bool:
    return (
        aggregate.get("run_status") == "complete"
        and bool(aggregate.get("scheduled"))
        and aggregate.get("valid") == aggregate.get("scheduled")
    )


def run(
    *,
    logs_root: Path,
    workspace: Path,
    result_filename: str,
    concurrencies: Sequence[int],
    gpu_count: int,
    expected_producer_sha: str,
    identity: Mapping[str, str],
    require_power: bool,
    service: str = SERVICE,
) -> int:
    """Validate every cell; return 1 when any cell is incomplete, or invalid under REQUIRE_POWER."""
    power_dir = logs_root / POWER_DIR
    try:
        server_gpu_uuids = _server_gpu_uuids(logs_root, gpu_count)
    except (OSError, ValueError) as error:
        print(f"[video] unusable server GPU identity: {error}", file=sys.stderr)
        server_gpu_uuids = ()
    rc = 0
    cells: dict[str, Any] = {}
    for concurrency in concurrencies:
        cell = logs_root / H3_DIR / "gpu" / f"c{concurrency}"
        run_path = cell / "baseline" / "run.json"
        bench = cell / f"{BENCHMARK_STEM.format(concurrency=concurrency)}.json"
        agg = workspace / f"{result_filename}_conc{concurrency}.json"
        sidecar = workspace / f"power_validation_{result_filename}_conc{concurrency}.json"
        try:
            aggregate = build_aggregate(
                _read(run_path), _read(bench), concurrency=concurrency, gpu_count=gpu_count,
                identity=identity, run_path=run_path.relative_to(logs_root).as_posix(),
                run_sha256=_sha256(run_path),
            )  # fmt: skip
        except (OSError, ValueError) as error:
            print(f"[video] c{concurrency}: unusable H3 result: {error}", file=sys.stderr)
            cells[str(concurrency)] = {"status": "missing", "error": str(error)}
            rc = 1
            continue
        _write_json_atomic(agg, aggregate)
        step = multinode.run(
            power_dir, bench, agg,
            prefill_gpus=0, decode_gpus=0, aggregate_gpus=0, service_gpus={service: gpu_count},
            expected_producer_sha=expected_producer_sha, logs_root=logs_root,
            server_gpu_uuids=server_gpu_uuids, validation_result=sidecar,
            require_power=require_power,
        )  # fmt: skip
        patched = _read(agg)
        patched.update(audit_summary(_read(sidecar), sidecar.name))
        _write_json_atomic(agg, patched)
        complete = _complete(patched)
        if not complete:
            print(
                f"[video] c{concurrency}: {patched.get('valid')}/{patched.get('scheduled')} valid "
                f"clips (status {patched.get('run_status')}); the point is incomplete",
                file=sys.stderr,
            )
        cells[str(concurrency)] = {
            "status": "complete" if complete else "incomplete",
            "result": agg.name,
            "power_validation": sidecar.name,
            "power_valid": patched.get("power_valid"),
            "latency_median_seconds": patched.get("latency_median_seconds"),
            "valid_clips_per_second": patched.get("valid_clips_per_second"),
            "avg_total_gpu_power_w": patched.get("avg_total_gpu_power_w"),
            "joules_per_successful_query": patched.get("joules_per_successful_query"),
        }
        rc = rc or step or int(not complete)
    summary = {
        "schema_version": 1,
        "scenario_type": SCENARIO_TYPE,
        **{key: identity.get(key) for key in IDENTITY},
        "num_gpus": gpu_count,
        "concurrencies": [int(c) for c in concurrencies],
        "cells": cells,
        "status": "complete" if rc == 0 else "failed",
    }
    _write_json_atomic(workspace / f"{result_filename}.json", summary)
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--result-filename", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--expected-producer-sha", required=True)
    parser.add_argument("--service", default=SERVICE)
    parser.add_argument("--require-power", action="store_true")
    for key in IDENTITY:
        parser.add_argument(f"--{key}", default="")
    args = parser.parse_args(argv)
    if args.gpu_count <= 0 or len(set(args.concurrencies)) != len(args.concurrencies):
        parser.error("--gpu-count must be positive and --concurrencies distinct")
    return run(
        logs_root=args.logs_root.resolve(), workspace=args.workspace.resolve(),
        result_filename=args.result_filename, concurrencies=args.concurrencies,
        gpu_count=args.gpu_count, expected_producer_sha=args.expected_producer_sha,
        identity={key: getattr(args, key) for key in IDENTITY},
        require_power=args.require_power, service=args.service,
    )  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
