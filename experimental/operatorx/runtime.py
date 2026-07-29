"""Capture a snapshot of the runtime environment for one smoke-test invocation.

Generic host info (hostname, importlib.metadata library versions, git sha,
whitelisted env, run id) lives here. Platform-specific driver/runtime probes
live alongside their kernel runners at ``operatorx/runners/<platform>/runtime.py``
and expose a ``collect() -> dict[str, str]`` function whose output is merged
into RunInfo.software. Each probe is fail-soft.

Env-passed (no detection): cluster, container_image, instance_type via
``$OPERATORX_CLUSTER`` / ``$OPERATORX_CONTAINER_IMAGE`` / ``$OPERATORX_INSTANCE_TYPE``.
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import operatorx.runners as _runners_pkg
from operatorx.core.run import RunInfo

_ENV_EXACT = {
    "WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
    "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
    "SLURM_JOB_ID", "SLURM_NODELIST",
}
_ENV_PREFIXES = ("NCCL_", "RCCL_", "NEURON_", "XLA_", "JAX_", "TPU_", "PYTORCH_", "OPERATORX_")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_run_id(cluster: str | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:5]
    return f"{ts}-{cluster or 'unknown'}-{short}"


def _git_sha() -> str | None:
    try:
        repo = Path(__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, check=True, capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _operatorx_version() -> str:
    try:
        return importlib_metadata.version("operatorx")
    except Exception:
        return "0.0.0"


def _generic_software() -> dict[str, str]:
    # Host-level only: python version + platform driver/runtime probes.
    # Per-backend library versions are merged in by main.py via the backend's
    # own ``versions()`` function.
    return {"python": sys.version.split()[0]}


def _platform_software() -> dict[str, str]:
    # Each subpackage of operatorx.runners is a platform. The convention is
    # that it ships a `runtime.collect() -> dict[str, str]`.
    out: dict[str, str] = {}
    for info in pkgutil.iter_modules(_runners_pkg.__path__):
        if not info.ispkg:
            continue
        try:
            mod = importlib.import_module(
                f"operatorx.runners.{info.name}.runtime"
            )
        except Exception:
            continue
        try:
            out.update(mod.collect())
        except Exception:
            pass
    return out


def _collect_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _ENV_EXACT or any(k.startswith(p) for p in _ENV_PREFIXES):
            out[k] = v
    return out


def runtime_snapshot() -> RunInfo:
    cluster = os.environ.get("OPERATORX_CLUSTER") or None
    container_image = os.environ.get("OPERATORX_CONTAINER_IMAGE") or None
    instance_type = os.environ.get("OPERATORX_INSTANCE_TYPE") or None

    software = _generic_software()
    software.update(_platform_software())

    started_at = utc_now_iso()
    return RunInfo(
        id=_make_run_id(cluster),
        started_at=started_at,
        finished_at=started_at,
        cluster=cluster,
        operatorx_version=_operatorx_version(),
        operatorx_git_sha=_git_sha(),
        hostname=socket.gethostname(),
        instance_type=instance_type,
        software=software,
        container_image=container_image,
        env=_collect_env(),
    )
