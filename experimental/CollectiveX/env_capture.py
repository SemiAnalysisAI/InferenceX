#!/usr/bin/env python3
"""CollectiveX spike — Layer-0 environment + topology capture.

Emits a JSON document describing the node a collective benchmark ran on, so
every result is provenance-tagged and a B200-vs-GB200 comparison is defensible.
Standard library only (so it runs in any minimal container, and off-GPU it
degrades gracefully instead of crashing). torch is used only if importable.

Usage:
    python env_capture.py --out results/env_b200-dgxc.json
    python env_capture.py --redact --out env.json   # hash hostnames/IPs/UUIDs

Importable:
    from env_capture import capture_environment
    env = capture_environment(redact=False)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys

SCHEMA_VERSION = 1

# Env vars worth recording — transport/tuning knobs that change what a
# collective actually does (esp. the GB200 MNNVL flags vs B200).
ENV_PREFIXES = ("NCCL_", "NVSHMEM_", "MC_", "UCX_", "SGLANG_DEEPEP", "DEEPEP_")
ENV_EXACT = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "SLURM_JOB_ID",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_JOB_PARTITION",
    # Image identity — set by the launcher so the bundle records what ran.
    "COLLECTIVEX_IMAGE",
    "COLLECTIVEX_IMAGE_DIGEST",
)


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    """Run a command, return stdout (stripped) or None if unavailable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _redact(value: str | None) -> str | None:
    """Stable short hash so artifacts can be shared without leaking
    hostnames / IPs / GPU UUIDs / IB GUIDs while staying joinable."""
    if not value:
        return value
    return "redacted-" + hashlib.sha256(value.encode()).hexdigest()[:12]


def _gpus(redact: bool) -> dict:
    """GPU inventory via nvidia-smi (None fields off-GPU)."""
    info: dict = {"source": None, "count": None, "devices": []}
    q = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    if q is None:
        return info
    info["source"] = "nvidia-smi"
    devices = []
    for line in q.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        name, uuid, mem_mib, cc, bus = parts[:5]
        devices.append(
            {
                "name": name,
                "uuid": _redact(uuid) if redact else uuid,
                "memory_total_mib": int(mem_mib) if mem_mib.isdigit() else mem_mib,
                "compute_capability": cc,
                "pci_bus_id": _redact(bus) if redact else bus,
            }
        )
    info["count"] = len(devices)
    info["devices"] = devices
    return info


def _driver_cuda() -> dict:
    out = _run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    driver = out.splitlines()[0].strip() if out else None
    # `nvidia-smi` (no args) prints the CUDA driver-API version in its header.
    cuda = None
    header = _run(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", header)
        if m:
            cuda = m.group(1)
    return {"driver_version": driver, "cuda_version": cuda}


def _torch_info() -> dict:
    """NCCL / torch build info — only if torch is importable in this env."""
    info: dict = {"available": False}
    try:
        import torch  # type: ignore
    except Exception:
        return info
    info["available"] = True
    info["torch_version"] = torch.__version__
    try:
        info["cuda_runtime"] = torch.version.cuda
    except Exception:
        info["cuda_runtime"] = None
    try:
        if torch.cuda.is_available():
            nccl = torch.cuda.nccl.version()
            # version() returns an int (e.g. 22304) or a tuple, depending on build.
            info["nccl_version"] = (
                ".".join(map(str, nccl)) if isinstance(nccl, tuple) else nccl
            )
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{cc[0]}.{cc[1]}"
    except Exception as exc:  # pragma: no cover - hardware dependent
        info["error"] = repr(exc)
    return info


def _topology(redact: bool) -> dict:
    """GPU/NIC topology matrix + a fingerprint to gate comparability.

    The fingerprint is a hash of the structural part of `nvidia-smi topo -m`
    (the connection legend), so two nodes with the same wiring share a key
    even if absolute device IDs differ."""
    topo = _run(["nvidia-smi", "topo", "-m"])
    if topo is None:
        return {"source": None, "matrix": None, "fingerprint": None}
    # Fingerprint the link-type tokens (NV#, NODE, SYS, PIX, PXB, ...) only —
    # ignore GPU/NIC labels and whitespace so it's placement-stable.
    tokens = re.findall(r"\b(NV\d+|NODE|SYS|PIX|PXB|PHB|X)\b", topo)
    fingerprint = hashlib.sha256(" ".join(tokens).encode()).hexdigest()[:16]
    return {
        "source": "nvidia-smi topo -m",
        # The matrix can contain hostnames in some setups; redact wholesale.
        "matrix": ("<redacted>" if redact else topo),
        "fingerprint": fingerprint,
    }


def _rdma(redact: bool) -> dict:
    """RDMA/IB device presence — names only, GUIDs redactable."""
    devices: list[str] = []
    listing = _run(["ibv_devinfo", "-l"])
    if listing:
        for line in listing.splitlines()[1:]:  # first line is a count
            name = line.strip()
            if name:
                devices.append(name)
    elif _run(["ibstat", "-l"]):
        devices = [d.strip() for d in _run(["ibstat", "-l"]).splitlines() if d.strip()]
    return {
        "available": bool(devices),
        "devices": [_redact(d) if redact else d for d in devices],
    }


def _env_vars() -> dict:
    out = {}
    for k, v in os.environ.items():
        if k in ENV_EXACT or any(k.startswith(p) for p in ENV_PREFIXES):
            out[k] = v
    return dict(sorted(out.items()))


def capture_environment(redact: bool = False, timestamp: str | None = None) -> dict:
    """Return a JSON-serializable environment/provenance record."""
    host = socket.gethostname()
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": timestamp or _dt.datetime.now().astimezone().isoformat(),
        "redacted": redact,
        "host": _redact(host) if redact else host,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),  # x86_64 vs aarch64 (B200 vs GB200)
            "python": sys.version.split()[0],
        },
        "gpus": _gpus(redact),
        "driver": _driver_cuda(),
        "torch": _torch_info(),
        "topology": _topology(redact),
        "rdma": _rdma(redact),
        "env": _env_vars(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX Layer-0 environment capture")
    ap.add_argument("--out", help="write JSON here (default: stdout)")
    ap.add_argument(
        "--redact",
        action="store_true",
        help="hash hostnames / IPs / GPU UUIDs / IB GUIDs for shareable artifacts",
    )
    ap.add_argument(
        "--timestamp",
        help="ISO timestamp to stamp (default: now); pass one for reproducible bundles",
    )
    args = ap.parse_args()

    env = capture_environment(redact=args.redact, timestamp=args.timestamp)
    blob = json.dumps(env, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(blob + "\n")
        # A one-line human summary to stdout (the JSON is the artifact).
        g = env["gpus"]
        print(
            f"env -> {args.out} | machine={env['platform']['machine']} "
            f"gpus={g['count']} topo_fp={env['topology']['fingerprint']}"
        )
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
