"""Thin srt-slurm adapters over the existing H3 supervisor and client.

srt-slurm owns allocation, container lifecycle, readiness and teardown. These
helpers only bind GPU identity from the job environment and call the retained
measurement code. Media validation, A/B compare, telemetry and result contracts
stay in the existing modules.

The native InferenceX path runs ``serve_role`` as a recipe service and
``run_srt_client`` as the custom benchmark: one server, every concurrency in
turn, each publishing its measured window for srt-slurm's power telemetry.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .mvp_gpu_job import _server_argv, cuda_devices, run_gpu_job, validate_gpu_job
from .mvp_runner import run_plan

BENCHMARK_STEM = "benchmark_concurrency_{concurrency}"
CELL_DIR = "gpu/c{concurrency}"
SERVER_DIR = "server"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bind_visible_gpus(spec: dict) -> dict:
    """Fill ``gpu_uuids`` from the container's visible devices before supervision."""
    frozen = dict(spec)
    if frozen.get("gpu_vendor") == "amd":
        from .mvp_amd_gpu import hip_devices

        devices = hip_devices()
    else:
        devices = cuda_devices()
    if not devices:
        raise RuntimeError("srt-slurm H3 job has no visible GPUs")
    frozen["gpu_uuids"] = devices
    frozen.setdefault("allocation", {})
    frozen["allocation"] = {
        **frozen["allocation"],
        "mode": frozen["allocation"].get("mode") or "dedicated_ci",
        "label": frozen["allocation"].get("label")
        or f"srt-slurm {os.environ.get('SLURM_JOB_ID', 'local')}",
    }
    return validate_gpu_job(frozen)


def run_srt_gpu_job(spec: dict, output: Path) -> dict:
    """Run the retained A/B (or serving) supervisor inside an srt-slurm service."""
    return run_gpu_job(bind_visible_gpus(spec), output)


def serve_role(
    spec: dict, role: str, port: int, *, host: str = "127.0.0.1", record: Path | None = None
) -> int:
    """Start one diffusion server and block until the process exits or is signaled.

    ``record`` receives the bound spec (real GPU UUIDs) and the launch the service made,
    the server-side evidence the result bundle pairs with each client run.
    """
    frozen = bind_visible_gpus(spec)
    frozen["port"] = port
    if role not in ("baseline", "candidate"):
        raise ValueError(f"unsupported H3 server role: {role}")
    argv = _server_argv(frozen, role, host=host)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # The server child re-checks the CUDA driver's UUID order against this list before
    # importing SGLang; the service container already exposes exactly the bound devices.
    env["VGBENCH_GPU_UUIDS"] = ",".join(frozen["gpu_uuids"])
    if record is not None:
        _write(record / "spec.json", frozen)
        _write(record / "launch.json", {
            "role": role, "host": host, "port": port, "argv": argv, "started_at": _now(),
            "hostname": socket.gethostname(), "gpu_uuids": frozen["gpu_uuids"],
            "slurm_job_id": env.get("SLURM_JOB_ID"), "slurm_step_id": env.get("SLURM_STEP_ID"),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        })  # fmt: skip
    process = subprocess.Popen(argv, env=env)

    def _stop(signum, frame):
        process.send_signal(signal.SIGTERM)

    previous = {sig: signal.signal(sig, _stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def _client_spec(spec: dict) -> dict:
    frozen = dict(spec)
    if not frozen.get("gpu_uuids"):
        frozen["gpu_uuids"] = ["GPU-00000000-0000-0000-0000-000000000001"]
    return validate_gpu_job(frozen)


def _run(frozen: dict, endpoint: str, output: Path) -> dict:
    serving = frozen.get("serving") or {}
    return run_plan(
        frozen["plan"],
        output,
        endpoint=endpoint,
        runtime="sglang",
        runtime_revision=frozen["baseline"]["revision"],
        hardware_label=",".join(frozen["gpu_uuids"]),
        model_revision=frozen["model"]["revision"],
        timeout_seconds=frozen["limits"]["request_seconds"],
        serving_concurrency=serving.get("concurrency"),
        delivery_deadline_seconds=serving.get("delivery_deadline_seconds"),
    )


def run_srt_client(spec: dict, endpoint: str, output: Path) -> dict:
    """Run the retained H3 client once, at the spec's serving load, into ``output``."""
    return _run(_client_spec(spec), endpoint, output)


def benchmark_window(run: dict, *, utc_minus_monotonic: float) -> dict:
    """The power-measurement contract of one run: its measured block on the Unix clock.

    The client measures on the monotonic clock; the offset captured in the same process
    maps it onto the head node's Unix clock that srt-slurm samples telemetry with.
    ``completed`` counts technically valid clips, the denominator energy is normalized by.
    """
    measurement, summary = run["measurement"], run["summary"]
    start = measurement["start_monotonic_seconds"] + utc_minus_monotonic
    end = measurement["end_monotonic_seconds"] + utc_minus_monotonic
    if not end > start:
        raise ValueError("measured block has no positive duration")
    return {
        "schema_version": 1,
        "benchmark_type": "video-generation",
        "max_concurrency": run["measurement"]["concurrency"],
        "benchmark_start_time_unix": start,
        "benchmark_end_time_unix": end,
        "duration": end - start,
        "completed": summary["valid"],
        "scheduled": summary["scheduled"],
        "valid": summary["valid"],
        "failed": summary["failed"],
        "energy_normalization": "clips",
        "clock": {"utc_minus_monotonic_seconds": utc_minus_monotonic},
        "run_path": "baseline/run.json",
        "run_sha256": hashlib.sha256(
            json.dumps(run, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def run_srt_matrix(
    spec: dict, endpoint: str, output: Path, concurrencies: list[int], *, window_dir: Path | None
) -> dict[int, dict]:
    """Run every concurrency against one ready server, each as its own measured cell.

    Cell ``N`` lands under ``output/gpu/cN``: the client's run bundle in ``baseline/`` and
    its window contract beside it. With ``window_dir`` (srt-slurm's
    ``SRT_MEASUREMENT_WINDOW_DIR``) the contract is also published as the completed
    telemetry window for that concurrency.
    """
    if not concurrencies or len(set(concurrencies)) != len(concurrencies):
        raise ValueError("concurrencies must be distinct and non-empty")
    frozen = _client_spec(spec)
    serving = frozen.get("serving") or {}
    results: dict[int, dict] = {}
    for concurrency in concurrencies:
        current = {**frozen, "serving": {**serving, "concurrency": concurrency}}
        current = validate_gpu_job(current)
        cell = output / CELL_DIR.format(concurrency=concurrency)
        before = time.time() - time.monotonic()
        run = _run(current, endpoint, cell / "baseline")
        after = time.time() - time.monotonic()
        offset = (before + after) / 2
        contract = benchmark_window(run, utc_minus_monotonic=offset)
        contract["clock"]["offset_spread_seconds"] = abs(after - before)
        result = cell / f"{BENCHMARK_STEM.format(concurrency=concurrency)}.json"
        _write(result, contract)
        if window_dir is not None:
            from infx.results.power.window import write_window

            write_window(result, concurrency, window_dir)
        results[concurrency] = run
    return results
