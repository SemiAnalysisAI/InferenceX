"""Thin srt-slurm adapters over the existing H3 supervisor and client.

srt-slurm owns allocation, container lifecycle, readiness and teardown. These
helpers only bind GPU identity from the job environment and call the retained
measurement code. Media validation, A/B compare, telemetry and result contracts
stay in the existing modules.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from .mvp_gpu_job import _server_argv, cuda_devices, run_gpu_job, validate_gpu_job
from .mvp_runner import run_plan


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


def serve_role(spec: dict, role: str, port: int) -> int:
    """Start one diffusion server and block until the process exits or is signaled."""
    frozen = bind_visible_gpus(spec)
    frozen["port"] = port
    if role not in ("baseline", "candidate"):
        raise ValueError(f"unsupported H3 server role: {role}")
    argv = _server_argv(frozen, role)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
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


def run_srt_client(spec: dict, endpoint: str, output: Path) -> dict:
    """Run the retained H3 client against a ready srt-slurm service endpoint."""
    frozen = dict(spec)
    if not frozen.get("gpu_uuids"):
        frozen["gpu_uuids"] = ["GPU-00000000-0000-0000-0000-000000000001"]
    frozen = validate_gpu_job(frozen)
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
