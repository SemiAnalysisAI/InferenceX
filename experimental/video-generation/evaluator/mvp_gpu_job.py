"""Bounded, locally supervised H3 GPU measurements on a trusted Linux runner.

This is NOT a sandbox for untrusted runtime code. The operator must provision
the source trees, environments, weights, and (for CI acceptance) a dedicated
allocation. UUID file locks are cooperative, not a GPU scheduler. No container,
foreign PID, port owner, or GPU is ever killed/reset by this module.

The supervisor retains each child session leader until its process group has
drained. That unreaped leader reserves the group ID; PID/start-time/session
checks precede every group signal. Escaped or invisible GPU processes are a
cleanup/attribution failure, never permission to kill an unrelated process.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import io
import json
import math
import os
import platform
import re
import signal
import socket
import stat
import subprocess
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .mvp_runner import canonical_json_bytes, preview_plan, validate_plan


VERSION = "0.1.0"
_ROLES = ("baseline", "candidate")
_SHA = re.compile(r"[0-9a-f]{64}")
_REV = re.compile(r"[0-9a-f]{40}")
_GPU = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_LAUNCH = """import os
from evaluator.mvp_gpu_job import cuda_devices
if cuda_devices() != os.environ["VGBENCH_GPU_UUIDS"].split(","):
    raise RuntimeError("Runtime CUDA device UUIDs differ from assigned GPUs")
from sglang.cli.main import main
main()
"""
_IDENTITY = (
    "import importlib.util,importlib.metadata,json,os,platform;"
    "s=importlib.util.find_spec('sglang');"
    "print(json.dumps({'python_version':platform.python_version(),"
    "'sglang_module':s.origin if s else None,"
    "'cpu_native_thread_limits':{name:os.environ.get(name) for name in "
    "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},"
    "'compilation_worker_limit':os.environ.get('MAX_JOBS'),"
    "'packages':{d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()}}))"
)


class JobCancelled(RuntimeError):
    """A signal or whole-job watchdog cancelled further work."""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("supervised operation exceeded its deadline")


def _hash(path: Path, deadline: float | None = None, cancelled: threading.Event | None = None) -> str:
    digest = hashlib.sha256()
    _check_deadline(deadline)
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("hash input must be a regular file, not a FIFO/device/socket")
        while True:
            _check_deadline(deadline)
            if cancelled is not None and cancelled.is_set():
                raise JobCancelled("cancelled while verifying pinned files")
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _write(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read(path: Path, limit: int = 32 * 1024 * 1024) -> dict:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("JSON evidence must be a regular file, not a FIFO/device/socket")
        if info.st_size > limit:
            raise ValueError("JSON evidence exceeds the bounded input size")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("JSON evidence grew beyond the bounded input size")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON evidence key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON evidence")

    result = json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(result, dict):
        raise ValueError("JSON evidence must be an object")
    return result


def _keys(value: Any, expected: set[str], label: str, optional: set[str] | None = None) -> None:
    if not isinstance(value, dict) or set(value) - expected - (optional or set()) or expected - set(value):
        raise ValueError(f"{label} requires exactly the supported explicit fields")


def _absolute(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"{label} must be an absolute, traversal-free path")
    if Path(value) == Path("/"):
        raise ValueError(f"{label} cannot be the filesystem root")


def _number(value: Any, label: str, minimum: float, maximum: float, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum or (integer and not isinstance(value, int)):
        raise ValueError(f"{label} must be a finite {'integer' if integer else 'number'} in [{minimum}, {maximum}]")


def validate_gpu_job(spec: dict) -> dict:
    """Pure validation: no filesystem probes, network, GPU calls, or execution."""
    frozen = json.loads(canonical_json_bytes(spec))
    _keys(frozen, {"schema_version", "job_id", "authorization", "allocation", "gpu_uuids", "port", "lock_directory", "baseline", "candidate", "model", "server", "plan", "policy", "limits"}, "GPU job", optional={"serving"})
    if "serving" in frozen:
        from .mvp_serving import settings
        load = frozen["serving"]
        _keys(load, {"concurrency"}, "serving", optional={"mode", "delivery_deadline_seconds"})
        if load.get("mode", "closed_loop") != "closed_loop":
            raise ValueError("only closed_loop serving load is supported")
        if load["concurrency"] is None:
            raise ValueError("serving requires explicit concurrency")
        frozen["serving"] = settings(load["concurrency"], load.get("delivery_deadline_seconds"))
    if frozen["schema_version"] != VERSION:
        raise ValueError("unsupported GPU job schema_version")
    if not isinstance(frozen["job_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", frozen["job_id"]):
        raise ValueError("job_id must be a bounded safe identifier")
    authorization = frozen["authorization"]
    _keys(authorization, {"compute_approved", "model_license_reviewed", "approval_reference"}, "authorization")
    if any(not isinstance(authorization[field], bool) for field in ("compute_approved", "model_license_reviewed")) or not isinstance(authorization["approval_reference"], str) or len(authorization["approval_reference"]) > 2000:
        raise ValueError("authorization requires explicit booleans and a bounded approval-reference string")
    allocation = frozen["allocation"]
    _keys(allocation, {"mode", "label"}, "allocation")
    if allocation["mode"] not in {"cooperative_shared", "dedicated_ci"} or not isinstance(allocation["label"], str) or not allocation["label"].strip() or len(allocation["label"]) > 500:
        raise ValueError("allocation requires an explicit operator-declared supported mode and label")
    devices = frozen["gpu_uuids"]
    if not isinstance(devices, list) or not 1 <= len(devices) <= 8 or any(not isinstance(item, str) or not _GPU.fullmatch(item) for item in devices) or len(set(devices)) != len(devices):
        raise ValueError("gpu_uuids must contain 1–8 distinct full NVIDIA GPU UUIDs; MIG is unsupported")
    _number(frozen["port"], "port", 1024, 65535, True)
    _absolute(frozen["lock_directory"], "lock_directory")
    for role in _ROLES:
        runtime = frozen[role]
        _keys(runtime, {"python", "source", "revision", "source_sha256"}, role)
        _absolute(runtime["python"], f"{role}.python")
        _absolute(runtime["source"], f"{role}.source")
        if not isinstance(runtime["revision"], str) or not _REV.fullmatch(runtime["revision"]):
            raise ValueError(f"{role}.revision must be an immutable lowercase commit")
        if not isinstance(runtime["source_sha256"], str) or not _SHA.fullmatch(runtime["source_sha256"]):
            raise ValueError(f"{role}.source_sha256 must freeze the tracked source-file manifest")
    model = frozen["model"]
    _keys(model, {"path", "revision", "files"}, "model")
    _absolute(model["path"], "model.path")
    frozen["plan"] = validate_plan(frozen["plan"])
    if model["revision"] != frozen["plan"]["model_revision"]:
        raise ValueError("staged model revision must match the frozen plan")
    files = model["files"]
    if not isinstance(files, list) or not files or len(files) > 20000:
        raise ValueError("model.files requires a nonempty bounded complete file manifest")
    names = set()
    for entry in files:
        _keys(entry, {"path", "size_bytes", "sha256"}, "model file")
        path = entry["path"]
        if not isinstance(path, str) or not path or "\\" in path or "\x00" in path or Path(path).is_absolute() or ".." in Path(path).parts or path in names:
            raise ValueError("model file paths must be unique relative traversal-free names")
        names.add(path)
        _number(entry["size_bytes"], "model file size", 0, 2**50, True)
        if not isinstance(entry["sha256"], str) or not _SHA.fullmatch(entry["sha256"]):
            raise ValueError("each model file requires its SHA256")
    if not any(name.endswith((".safetensors", ".pt", ".bin")) for name in names):
        raise ValueError("model manifest contains no weight files")
    frozen["model"]["files"] = sorted(files, key=lambda item: item["path"])
    server = frozen["server"]
    _keys(server, {"ulysses_degree", "tp_size", "encoder_parallel", "performance_mode"}, "server",
          optional={"dit_cpu_offload", "layerwise_offload"})
    _number(server["ulysses_degree"], "server.ulysses_degree", 1, len(devices), True)
    _number(server["tp_size"], "server.tp_size", 1, len(devices), True)
    if len(devices) % server["ulysses_degree"] or len(devices) % server["tp_size"]:
        raise ValueError("server requires dividing Ulysses and tensor-parallel degrees")
    if server["encoder_parallel"] not in {"auto", "fold", "replicate"} or server["performance_mode"] not in {"manual", "speed", "memory"}:
        raise ValueError("unsupported explicit encoder parallelism or performance mode")
    if "layerwise_offload" in server:
        # This is one documented lossless placement, not a general argument
        # passthrough. Its transfer from 2x5090 to 2xH200 remains experimental.
        if "dit_cpu_offload" in server:
            raise ValueError("layerwise offload must not also set coarse dit_cpu_offload")
        layerwise = server["layerwise_offload"]
        _keys(layerwise, {"components", "prefetch_size", "resident_layers"}, "server.layerwise_offload")
        _number(layerwise["prefetch_size"], "layerwise prefetch_size", 1, 1, True)
        _number(layerwise["resident_layers"], "layerwise resident_layers", 20, 20, True)
        if (layerwise["components"] != ["dit", "text_encoder", "vae"]
                or len(devices) != 2 or server["tp_size"] != 2 or server["ulysses_degree"] != 1
                or server["encoder_parallel"] != "auto" or server["performance_mode"] != "memory"):
            raise ValueError("only the explicit two-GPU TP2/Ulysses1 lossless layerwise profile is supported")
    elif not isinstance(server.get("dit_cpu_offload"), bool):
        raise ValueError("server requires an explicit coarse offload boolean or supported layerwise profile")
    limits = frozen["limits"]
    ranges = {
        "job_seconds": (5, 86400), "startup_seconds": (0.1, 43200),
        "request_seconds": (0.1, 86400), "cleanup_seconds": (0.1, 120),
        "telemetry_interval_seconds": (0.1, 60), "command_seconds": (0.1, 60),
        "max_idle_memory_mib": (0, 5000),
    }
    _keys(limits, set(ranges), "limits")
    for name, (minimum, maximum) in ranges.items():
        _number(limits[name], f"limits.{name}", minimum, maximum)
    if limits["job_seconds"] <= 2 * limits["cleanup_seconds"] or limits["startup_seconds"] >= limits["job_seconds"] or limits["request_seconds"] >= limits["job_seconds"]:
        raise ValueError("whole-job budget must leave explicit cleanup time and bound each phase")
    if limits["job_seconds"] / limits["telemetry_interval_seconds"] > 100000:
        raise ValueError("telemetry plan exceeds 100000 samples")
    if frozen["plan"]["warmup_runs"] < 1:
        raise ValueError("controlled GPU measurements require a separately recorded warmup")
    from .mvp_compare import _policy
    frozen["policy"] = _policy(frozen["policy"])
    if frozen.get("serving") and frozen["policy"]["calibration_status"] == "operator_calibrated":
        raise ValueError("serving load requires an uncalibrated policy; serial calibration cannot qualify concurrent delivery metrics")
    memory_gate = frozen["policy"].get("max_memory_increase_fraction")
    if memory_gate is not None:
        _number(memory_gate, "max_memory_increase_fraction", 0, 100)
    return frozen


def _server_argv(spec: dict, role: str) -> list[str]:
    args = [spec[role]["python"], "-c", _LAUNCH, "serve", "--model-type", "diffusion",
            "--model-path", spec["model"]["path"], "--model-id", spec["plan"]["model_id"],
            "--revision", spec["model"]["revision"], "--model-variant", "fl2va",
            "--num-gpus", str(len(spec["gpu_uuids"])), "--ulysses-degree", str(spec["server"]["ulysses_degree"]),
            "--tp-size", str(spec["server"]["tp_size"]), "--encoder-parallel", spec["server"]["encoder_parallel"],
            "--performance-mode", spec["server"]["performance_mode"],
            "--host", "127.0.0.1", "--port", str(spec["port"]), "--enable-torch-compile", "false"]
    if "layerwise_offload" in spec["server"]:
        layerwise = spec["server"]["layerwise_offload"]
        args.extend(["--layerwise-offload-components", ",".join(layerwise["components"]),
                     "--dit-offload-prefetch-size", str(layerwise["prefetch_size"]),
                     "--dit-layerwise-resident-layers", str(layerwise["resident_layers"])])
    else:
        args.extend(["--dit-cpu-offload", str(spec["server"]["dit_cpu_offload"]).lower()])
    return args


def preview_gpu_job(spec: dict) -> dict:
    frozen = validate_gpu_job(spec)
    return {
        "evidence_kind": "gpu_job_preview_no_execution", "spec_sha256": _digest(frozen),
        "job_id": frozen["job_id"], "gpu_uuids": frozen["gpu_uuids"],
        "authorization": frozen["authorization"], "authorization_verification": "operator assertion; not proof of model-license rights",
        "allocation": frozen["allocation"], "allocation_verification": "operator-declared prerequisite, not scheduler attestation",
        "commands": {role: _server_argv(frozen, role) for role in _ROLES},
        "workload": preview_plan(frozen["plan"]), "limits": frozen["limits"],
        "sequence": ["verify pinned files", "acquire UUID locks", "verify idle", "baseline startup/warmup/measure/cleanup", "candidate startup/warmup/measure/cleanup", "compare"],
        "warnings": ["Trusted Linux runner and visible GPU-process PIDs are required.", "No GPU lock can enforce exclusive access against non-cooperating processes.", "No GPU work, model download, server startup, or CI acceptance occurs in preview."],
    }


def _command(argv: list[str], *, timeout: float, cwd: Path | None = None, env: dict | None = None) -> bytes:
    # Only bounded read-only utilities and a trusted, fixed Python identity probe.
    result = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode or len(result.stdout) > 8 * 1024 * 1024:
        raise RuntimeError(f"bounded identity/telemetry command failed: {Path(argv[0]).name}")
    return result.stdout


def source_file_manifest(source: Path, *, timeout: float = 10, deadline: float | None = None, cancelled: threading.Event | None = None) -> dict:
    """Generated build metadata is part of the pinned runtime identity."""
    source = Path(source).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("runtime source is not a directory")
    revision = _command(["git", "-C", str(source), "rev-parse", "HEAD"], timeout=timeout).decode().strip()
    dirty = _command(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], timeout=timeout)
    if dirty:
        raise ValueError("runtime source must be a clean committed checkout without untracked files")
    raw = _command(["git", "-C", str(source), "ls-files", "-z"], timeout=timeout)
    names = sorted(name.decode("utf-8") for name in raw.split(b"\x00") if name)
    if not names or len(names) > 100000:
        raise ValueError("runtime tracked-file inventory is empty or oversized")
    # setuptools-scm generates this ignored module, and SGLang imports it at
    # runtime. Pin its bytes while continuing to reject other ignored Python.
    generated = "python/sglang/_version.py"
    if generated not in names and ((source / generated).exists() or (source / generated).is_symlink()):
        names = sorted([*names, generated])
    entries = []
    for name in names:
        _check_deadline(deadline)
        target = (source / name).resolve(strict=True)
        if not target.is_relative_to(source) or not target.is_file():
            raise ValueError("runtime tracked file escapes checkout or is not a regular file")
        entries.append({"path": name, "size_bytes": target.stat().st_size, "sha256": _hash(target, deadline, cancelled)})
    # Ignored source files can shadow installed modules despite a clean git tree.
    package = source / "python" / "sglang"
    if not (package / "cli" / "main.py").is_file():
        raise ValueError("source does not contain the supported SGLang CLI layout")
    tracked = set(names)
    if any(path.relative_to(source).as_posix() not in tracked for path in package.rglob("*.py")):
        raise ValueError("untracked/ignored Python files exist inside the runtime package")
    return {"source": str(source), "revision": revision, "source_sha256": _digest(entries), "files": entries}


def _model_manifest(spec: dict, deadline: float, cancelled: threading.Event | None = None) -> dict:
    root = Path(spec["model"]["path"]).resolve(strict=True)
    allowed_blob_root = None
    if root.parent.name == "snapshots" and root.name == spec["model"]["revision"]:
        blobs = root.parent.parent / "blobs"
        if blobs.is_symlink():
            raise ValueError("HF blob directory cannot itself be a symlink")
        if blobs.exists():
            allowed_blob_root = blobs.resolve(strict=True)
            if allowed_blob_root != blobs or not allowed_blob_root.is_dir():
                raise ValueError("HF blob directory is not the canonical same-cache directory")
    entries = spec["model"]["files"]
    actual = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        _check_deadline(deadline)
        if any((Path(directory) / name).is_symlink() for name in directories):
            raise ValueError("model directory symlinks are unsupported")
        actual.update((Path(directory) / name).relative_to(root).as_posix() for name in files)
    if actual != {entry["path"] for entry in entries}:
        raise ValueError("staged model file inventory differs from the complete frozen manifest")
    for entry in entries:
        target = (root / entry["path"]).resolve(strict=True)
        if not target.is_file() or not (target.is_relative_to(root) or (allowed_blob_root and target.is_relative_to(allowed_blob_root))):
            raise ValueError("model file escapes the snapshot and its validated HF blob directory")
        if target.stat().st_size != entry["size_bytes"] or _hash(target, deadline, cancelled) != entry["sha256"]:
            raise ValueError("staged model file hash or size differs from frozen manifest")
    return {"path": str(root), "revision": spec["model"]["revision"], "manifest_sha256": _digest(entries),
            "verified_files": len(entries), "total_bytes": sum(item["size_bytes"] for item in entries),
            "verification": "complete file inventory and SHA256 before launch; not hardware attestation"}


def cuda_devices() -> list[str]:
    # Resolve the current mask through the driver without creating a CUDA context;
    # nvidia-smi ordinals alone do not establish CUDA device identity.
    import ctypes
    cuda = ctypes.CDLL("libcuda.so.1")
    def ok(code):
        if code != 0:
            raise RuntimeError("CUDA driver inventory failed: " + str(code))
    ok(cuda.cuInit(0))
    count = ctypes.c_int()
    ok(cuda.cuDeviceGetCount(ctypes.byref(count)))
    values = []
    for ordinal in range(count.value):
        device, raw = ctypes.c_int(), (ctypes.c_ubyte * 16)()
        ok(cuda.cuDeviceGet(ctypes.byref(device), ordinal))
        ok(cuda.cuDeviceGetUuid(ctypes.byref(raw), device))
        values.append("GPU-" + str(uuid.UUID(bytes=bytes(raw))))
    return values


def _runtime_env(source: str, gpu_uuids: list[str], nonce: str, cache: Path) -> dict[str, str]:
    # Do not inherit authentication, PYTHONPATH, remote endpoints, LD_PRELOAD, or
    # performance overrides. Never overwrite HOME/CODEX_HOME or user caches.
    result = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
    result.update({"PYTHONPATH": os.pathsep.join((str(Path(source) / "python"), str(Path(__file__).resolve().parent.parent))), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   # Host core discovery ignores container CPU/PID budgets in some native libraries.
                   # Keep the same explicit import/runtime thread policy for both arms and clients.
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                   "MAX_JOBS": "2",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "VGBENCH_LAUNCH_NONCE": nonce, "XDG_CACHE_HOME": str(cache), "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
                   # SGLang's native JIT/FlashInfer defaults do not follow XDG.
                   "SGLANG_CACHE_DIR": str(cache / "sglang"), "SGLANG_JIT_CACHE_DIR": str(cache / "sglang" / "jit"),
                   "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
                   "HF_HOME": str(cache / "huggingface"), "HF_HUB_CACHE": str(cache / "huggingface" / "hub"),
                   "TRITON_CACHE_DIR": str(cache / "triton"), "CUDA_CACHE_PATH": str(cache / "cuda")})
    return result


def _source_identity(spec: dict, role: str, env: dict, deadline: float, cancelled: threading.Event | None = None) -> dict:
    declared = spec[role]
    observed = source_file_manifest(Path(declared["source"]), timeout=spec["limits"]["command_seconds"], deadline=deadline, cancelled=cancelled)
    if any(observed[field] != declared[field] for field in ("revision", "source_sha256")):
        raise ValueError(f"{role} observed source revision/tree differs from pinned specification")
    executable = Path(declared["python"]).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("runtime Python executable is unavailable")
    data = _command([declared["python"], "-c", _IDENTITY], cwd=Path(declared["source"]), env=env,
                    timeout=min(spec["limits"]["command_seconds"], max(0.01, deadline - time.monotonic())))
    identity = json.loads(data)
    module = Path(identity.get("sglang_module") or "").resolve(strict=True)
    expected_module = (Path(declared["source"]) / "python" / "sglang" / "__init__.py").resolve(strict=True)
    if module != expected_module or not identity.get("packages", {}).get("torch"):
        raise ValueError("runtime identity probe did not resolve pinned source with an installed torch dependency")
    return {key: value for key, value in observed.items() if key != "files"} | identity | {
        "python": declared["python"], "python_sha256": _hash(executable, deadline),
        "environment_control": "allowlisted supervisor environment; no inherited runtime overrides",
    }


def _proc_identity(pid: int) -> dict | None:
    try:
        content = Path(f"/proc/{pid}/stat").read_text()
        # comm can itself contain spaces and parentheses; fields after its last
        # ')' begin with state (field 3), not process name tokens.
        fields = content[content.rfind(")") + 2:].split()
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "pgid": int(fields[2]),
                "session_id": int(fields[3]), "start_ticks": int(fields[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def _diagnostic_proc_file(pid: int, name: str) -> str:
    """Bounded reads of only the two non-command-line diagnostic proc files."""
    if name not in {"status", "cgroup"}:
        raise ValueError("unsupported process diagnostic file")
    with Path(f"/proc/{pid}/{name}").open(encoding="utf-8") as stream:
        value = stream.read(65537)
    if len(value) > 65536:
        raise ValueError("process diagnostic file exceeds size limit")
    return value


def _unowned_process_diagnostic(pid: int) -> dict:
    """Best-effort attribution evidence only; never establishes ownership.

    PID disappearance, reuse and permission failures are retained explicitly.
    No command line, environment, process name or exception text is recorded.
    UID/cgroup observations are not atomic with stat; the second stat detects
    some lifetime races but cannot prove provenance for an already exited PID.
    """
    def failure(error: Exception) -> dict:
        status = ("missing" if isinstance(error, (FileNotFoundError, ProcessLookupError))
                  else "permission_denied" if isinstance(error, PermissionError)
                  else "unavailable")
        return {"status": status, "error_type": type(error).__name__}

    def identity() -> dict:
        try:
            value = _proc_identity(pid)
            return ({"status": "observed", **{key: value[key] for key in
                     ("pid", "ppid", "pgid", "session_id", "start_ticks", "state")}}
                    if value is not None else {"status": "missing"})
        except Exception as error:
            return failure(error)

    result = {"pid": pid, "observed_at": _now(), "diagnostic_only": True,
              "ownership_established": False, "identity": identity()}
    try:
        status = _diagnostic_proc_file(pid, "status")
        match = re.search(r"^Uid:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", status, re.MULTILINE)
        if match is None:
            raise ValueError("UID fields unavailable")
        result["uid"] = {"status": "observed", **dict(zip(
            ("real", "effective", "saved", "filesystem"), map(int, match.groups())))}
    except Exception as error:
        result["uid"] = failure(error)
    try:
        entries = []
        for line in _diagnostic_proc_file(pid, "cgroup").splitlines():
            fields = line.split(":", 2)
            if len(fields) != 3 or not fields[0].isdigit() or not fields[2].startswith("/"):
                raise ValueError("cgroup membership unavailable")
            entries.append({"hierarchy_id": int(fields[0]), "controllers": fields[1], "path": fields[2]})
        if not entries:
            raise ValueError("cgroup membership unavailable")
        result["cgroup"] = {"status": "observed", "entries": entries}
    except Exception as error:
        result["cgroup"] = failure(error)
    result["identity_after"] = identity()
    before, after = result["identity"], result["identity_after"]
    if before["status"] == after["status"] == "observed":
        result["lifetime_check"] = ("same_start_ticks" if before["start_ticks"] == after["start_ticks"]
                                    else "pid_reused_during_reads")
    elif before["status"] == "observed" and after["status"] == "missing":
        result["lifetime_check"] = "disappeared_during_reads"
    else:
        result["lifetime_check"] = "unverified"
    return result


def _group_members(pgid: int) -> list[dict]:
    result = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            identity = _proc_identity(int(name))
            if identity and identity["pgid"] == pgid and identity["state"] != "Z":
                result.append(identity)
    return result


class OwnedProcess:
    """Own a new child session; never reap its leader before group cleanup."""

    def __init__(self, argv: list[str], *, cwd: Path, env: dict, stdout: Path, stderr: Path, nonce: str):
        self._mutex = threading.Lock()
        self._closed = False
        self.returncode = None
        self.receipt = None
        self.log_paths = (stdout, stderr)
        with stdout.open("xb") as out, stderr.open("xb") as err:
            self.process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                            stdout=out, stderr=err, start_new_session=True, close_fds=True)
        identity = _proc_identity(self.process.pid)
        if not identity or identity["ppid"] != os.getpid() or identity["pgid"] != self.process.pid or identity["session_id"] != self.process.pid:
            # Popen's exact child PID may be terminated, but never an unverified
            # process group. This branch cannot authorize any foreign PID.
            self.process.terminate()
            self.process.wait(timeout=1)
            raise RuntimeError("could not establish ownership of child session")
        self.identity = {**identity, "launch_nonce": nonce}

    def running(self) -> bool:
        current = _proc_identity(self.identity["pid"])
        return bool(current and current["start_ticks"] == self.identity["start_ticks"] and current["state"] != "Z")

    def check_output_budget(self):
        if any(path.stat().st_size > 128 * 1024 * 1024 for path in self.log_paths):
            raise RuntimeError("owned process output exceeded the 128 MiB per-stream safety limit")

    def owns(self, pid: int) -> bool:
        return self.observe(pid) is not None

    def observe(self, pid: int) -> dict | None:
        current = _proc_identity(pid)
        return current if current and current["pgid"] == self.identity["pgid"] and current["session_id"] == self.identity["session_id"] else None

    def _signal(self, sig: int) -> None:
        current = _proc_identity(self.identity["pid"])
        if not current or any(current[field] != self.identity[field] for field in ("pid", "pgid", "session_id", "start_ticks")):
            raise RuntimeError("refusing process-group signal: leader identity no longer matches")
        members = _group_members(self.identity["pgid"])
        if any(item["session_id"] != self.identity["session_id"] for item in members):
            raise RuntimeError("refusing process-group signal: unexpected session membership")
        if members:
            os.killpg(self.identity["pgid"], sig)

    def close(self, timeout: float | None = None, *, deadline: float | None = None) -> dict:
        end = deadline if deadline is not None else time.monotonic() + float(timeout)
        acquired = self._mutex.acquire(timeout=max(0, end - time.monotonic()))
        if not acquired:
            return {"status": "failed", "remaining_owned_pids": None, "reason": "shared cleanup deadline expired while another owner cleanup was active"}
        try:
            if self._closed:
                return self.receipt
            remaining_seconds = max(0, end - time.monotonic())
            self._signal(signal.SIGTERM if remaining_seconds > 0 else signal.SIGKILL)
            grace = time.monotonic() + remaining_seconds / 2
            while _group_members(self.identity["pgid"]) and time.monotonic() < grace:
                time.sleep(min(0.05, max(0, grace - time.monotonic())))
            if _group_members(self.identity["pgid"]):
                self._signal(signal.SIGKILL)
            while _group_members(self.identity["pgid"]) and time.monotonic() < end:
                time.sleep(min(0.05, max(0, end - time.monotonic())))
            remaining = [item["pid"] for item in _group_members(self.identity["pgid"])]
            self.receipt = {"status": "failed" if remaining else "clean", "remaining_owned_pids": remaining,
                            "scope": "exact verified child session/process group only"}
            if not remaining:
                self.returncode = self.process.wait(timeout=max(0.001, end - time.monotonic()))
                self._closed = True
            return self.receipt
        finally:
            self._mutex.release()


class _Supervisor:
    def __init__(self, limits: dict):
        self.total_deadline = time.monotonic() + limits["job_seconds"]
        self.deadline = self.total_deadline - 2 * limits["cleanup_seconds"]
        self.cleanup_seconds = limits["cleanup_seconds"]
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.reason = None
        self.processes: list[OwnedProcess] = []
        self.lock = threading.Lock()
        self.previous = {}
        self.watchdog = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("GPU supervisor must run on the main Python thread for signal handling")
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._on_signal)
        self.watchdog.start()
        return self

    def _on_signal(self, sig, frame):
        self.reason = f"received signal {sig}"
        self.cancelled.set()

    def _watch(self):
        while not self.finished.wait(0.1):
            if time.monotonic() >= self.deadline:
                self.reason = self.reason or "whole-job work deadline exceeded; cleanup reserve entered"
                self.cancelled.set()
            if self.cancelled.is_set():
                with self.lock:
                    processes = list(self.processes)
                for process in reversed(processes):
                    try:
                        process.close(deadline=self.total_deadline)
                    except (OSError, RuntimeError, subprocess.TimeoutExpired):
                        pass  # Main-thread cleanup records failures and quarantines.
                return

    def check(self):
        if self.cancelled.is_set():
            raise JobCancelled(self.reason or "job cancelled")
        _check_deadline(self.deadline)

    def spawn(self, argv: list[str], **kwargs) -> OwnedProcess:
        with self.lock:
            self.check()
            process = OwnedProcess(argv, **kwargs)
            self.processes.append(process)
            return process

    def __exit__(self, exc_type, exc, tb):
        self.finished.set()
        self.watchdog.join(timeout=0.2)
        for process in reversed(self.processes):
            try:
                process.close(deadline=min(self.total_deadline, time.monotonic() + self.cleanup_seconds))
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                pass
        for sig, previous in self.previous.items():
            signal.signal(sig, previous)


class GpuLease:
    def __init__(self, directory: Path, gpu_uuids: list[str], job_id: str):
        self.directory = directory
        self.gpus = sorted(gpu_uuids)
        self.job_id = job_id
        self.handles = []

    def __enter__(self):
        import fcntl
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError("lock directory must be owned by this UID and not group/world writable")
        try:
            for device in self.gpus:
                if (self.directory / (device + ".blocked.json")).exists():
                    raise RuntimeError("GPU lease is quarantined after unresolved cleanup; operator inspection required")
                descriptor = os.open(self.directory / (device + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                handle = os.fdopen(descriptor, "r+b")
                self.handles.append(handle)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def quarantine(self, reason: str):
        for device in self.gpus:
            path = self.directory / (device + ".blocked.json")
            # Do not replace a quarantine another operator already recorded.
            with path.open("xb") as stream:
                stream.write(canonical_json_bytes({"job_id": self.job_id, "at": _now(), "reason": reason}))

    def __exit__(self, *_):
        import fcntl
        for handle in reversed(self.handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        self.handles.clear()


class GpuProbe:
    def __init__(self, devices: list[str], timeout: float):
        self.devices = devices
        self.timeout = timeout

    def power_configuration(self, *, deadline: float | None = None) -> dict:
        fields = "uuid,power.limit,enforced.power.limit,power.default_limit,power.max_limit"
        result = {"observed_at": _now(), "query": fields, "status": "unavailable", "gpus": []}
        try:
            _check_deadline(deadline)
            timeout = self.timeout if deadline is None else min(self.timeout, max(0.001, deadline - time.monotonic()))
            raw = _command(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits",
                            "--id=" + ",".join(self.devices)], timeout=timeout).decode()
            result["raw"] = raw
            for row in csv.reader(io.StringIO(raw)):
                if len(row) != 5:
                    raise ValueError("incomplete power configuration")
                device = {"uuid": row[0].strip()}
                for key, value in zip(("configured_limit_w", "enforced_limit_w", "default_limit_w", "maximum_limit_w"), row[1:]):
                    try:
                        watts = float(value)
                    except ValueError:
                        watts = None
                    device[key] = watts if watts is not None and math.isfinite(watts) and watts > 0 else None
                result["gpus"].append(device)
            if sorted(d["uuid"] for d in result["gpus"]) != sorted(self.devices):
                raise ValueError("power configuration GPU inventory mismatch")
            result["status"] = "recorded"
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired, TimeoutError) as error:
            result.update(error=str(error), gpus=[])
        return result

    def snapshot(self, *, deadline: float | None = None) -> dict:
        def budget():
            _check_deadline(deadline)
            return self.timeout if deadline is None else min(self.timeout, max(0.001, deadline - time.monotonic()))
        fields = "index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version,power.draw,temperature.gpu,mig.mode.current"
        query_start = time.monotonic()
        query_utc = _now()
        raw = _command(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits", "--id=" + ",".join(self.devices)], timeout=budget())
        query_end = time.monotonic()
        rows = list(csv.reader(io.StringIO(raw.decode())))
        gpus = []
        for row in rows:
            row = [item.strip() for item in row]
            if len(row) != 10 or row[1] not in self.devices or row[9] not in {"Disabled", "[N/A]", "N/A", "[Not Supported]"}:
                raise RuntimeError("GPU inventory is incomplete, changed, or MIG-enabled")
            device = {"index": int(row[0]), "uuid": row[1], "name": row[2], "memory_total_mib": float(row[3]),
                      "memory_used_mib": float(row[4]), "utilization_percent": float(row[5]), "driver_version": row[6], "mig_mode": row[9]}
            for position, name in ((7, "power_watts"), (8, "temperature_celsius")):
                try:
                    device[name] = float(row[position])
                except ValueError:
                    device[name] = None
            if any(not math.isfinite(device[key]) or device[key] < 0 for key in ("memory_total_mib", "memory_used_mib", "utilization_percent")):
                raise RuntimeError("required GPU telemetry is not finite")
            gpus.append(device)
        if sorted(item["uuid"] for item in gpus) != sorted(self.devices):
            raise RuntimeError("observed GPU UUID inventory does not match leased devices")
        raw = _command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits", "--id=" + ",".join(self.devices)], timeout=budget())
        apps = []
        for row in csv.reader(io.StringIO(raw.decode())):
            row = [item.strip() for item in row]
            if len(row) != 3 or row[0] not in self.devices or not row[1].isdigit():
                raise RuntimeError("GPU compute process inventory is unavailable or malformed")
            try:
                memory = float(row[2])
            except ValueError:
                memory = None
            apps.append({"gpu_uuid": row[0], "pid": int(row[1]), "memory_used_mib": memory})
        return {"at": _now(), "monotonic_seconds": time.monotonic(), "gpus": gpus, "compute_apps": apps,
                "power_query": {"start_utc": query_utc, "start_monotonic_seconds": query_start,
                                "end_monotonic_seconds": query_end, "field": "power.draw"}}


def _idle(snapshot: dict, maximum: float) -> bool:
    return not snapshot["compute_apps"] and all(item["memory_used_mib"] <= maximum for item in snapshot["gpus"])


def summarize_gpu_samples(samples: list[dict], devices: list[str], interval: float, errors: list[str] | None = None,
                          *, window_start: float | None = None, window_end: float | None = None, command_seconds: float = 1) -> dict:
    """Pure telemetry aggregation shared by live recording and saved verification."""
    errors = errors or []
    measurement = [sample for sample in samples if sample["phase"] == "measurement"]
    peaks = {device: max((item["memory_used_mib"] for sample in samples for item in sample["gpus"] if item["uuid"] == device), default=None) for device in devices}
    measured_peaks = {device: max((item["memory_used_mib"] for sample in measurement for item in sample["gpus"] if item["uuid"] == device), default=None) for device in devices}
    owned = {device: sum(any(app["gpu_uuid"] == device for app in sample["owned_compute_apps"]) for sample in measurement) for device in devices}
    intervals = [right["monotonic_seconds"] - left["monotonic_seconds"] for left, right in zip(measurement, measurement[1:])]
    max_gap = max(interval * 3, interval + 2 * command_seconds + 0.1)
    window_valid = window_start is not None and window_end is not None and window_end > window_start
    coverage = bool(window_valid and measurement and measurement[0]["monotonic_seconds"] - window_start <= max_gap
                    and window_end - measurement[-1]["monotonic_seconds"] <= max_gap
                    and all(0 < gap <= max_gap for gap in intervals))
    return {"sample_count": len(samples), "measurement_sample_count": len(measurement), "errors": errors,
            "gpu_identity": [{key: item[key] for key in ("uuid", "index", "name", "memory_total_mib", "driver_version", "mig_mode")} for item in samples[0]["gpus"]] if samples else [],
            "observed_memory_peak_mib_by_gpu": peaks, "measurement_observed_memory_peak_mib_by_gpu": measured_peaks,
            "observed_owned_compute_by_gpu": owned, "requested_interval_seconds": interval,
            "maximum_observed_sample_gap_seconds": max(intervals) if intervals else None,
            "maximum_allowed_sample_gap_seconds": max_gap, "cadence_coverage_qualified": coverage,
            "measurement_window_start_monotonic_seconds": window_start, "measurement_window_end_monotonic_seconds": window_end,
            "memory_semantics": "maximum observed device-used VRAM samples during this role, not exact framework allocator peaks",
            "measurement_window": "client workload including separately recorded warmup requests; not GPU kernel timing",
            "qualified": bool(len(measurement) >= 2 and not errors and all(owned.values()) and coverage)}


class _Sampler:
    def __init__(self, probe: GpuProbe, owner: OwnedProcess, path: Path, interval: float, *, initial_sample: dict | None = None):
        self.probe, self.owner, self.path, self.interval = probe, owner, path, interval
        self.phase = "startup"
        self.window_start = None
        self.window_end = None
        self.done = threading.Event()
        self.failed = threading.Event()
        self.samples = []
        self.errors = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.path.touch(exist_ok=False)
        if initial_sample is not None:
            if initial_sample.get("compute_apps") != []:
                raise ValueError("Prelaunch sample must have an empty compute inventory")
            sample = {**initial_sample, "phase": "startup", "owned_compute_apps": [], "unowned_compute_apps": [],
                      "observation": "verified_idle_before_runtime_launch"}
            self.samples.append(sample)
            self.path.write_bytes(canonical_json_bytes(sample) + b"\n")

    def _loop(self):
        while not self.done.is_set():
            try:
                sample = self.probe.snapshot()
                if self.done.is_set():
                    return  # Never append evidence after a timed-out stop/hash.
                sample["phase"] = self.phase
                sample["owned_compute_apps"] = []
                sample["unowned_compute_apps"] = []
                for app in sample["compute_apps"]:
                    observation_status = "not_owned"
                    try:
                        observed = self.owner.observe(app["pid"])
                    except PermissionError:
                        observed = None
                        observation_status = "permission_denied"
                    if observed is None:
                        sample["unowned_compute_apps"].append({
                            **app, "ownership_observation": observation_status,
                            "process_diagnostic": _unowned_process_diagnostic(app["pid"]),
                        })
                    else:
                        sample["owned_compute_apps"].append({**app, "process_identity": observed})
                with self.path.open("ab") as stream:
                    stream.write(canonical_json_bytes(sample) + b"\n")
                    stream.flush()
                self.samples.append(sample)
                if sample["unowned_compute_apps"]:
                    raise RuntimeError("foreign or PID-namespace-invisible GPU process detected; no ownership established")
            except Exception as error:
                if self.done.is_set():
                    return
                self.errors.append(str(error) if isinstance(error, RuntimeError) else type(error).__name__)
                self.failed.set()
                return
            self.done.wait(self.interval)

    def start(self):
        self.thread.start()

    def begin_measurement(self):
        self.window_start = time.monotonic()
        self.phase = "measurement"

    def end_measurement(self):
        self.window_end = time.monotonic()
        self.phase = "cleanup"

    def bracket_completion(self, deadline: float) -> bool:
        """Retain the next ordinary sample before ending the owned runtime."""
        boundary = time.monotonic()
        while time.monotonic() < deadline and not self.failed.is_set():
            if self.samples and self.samples[-1]["monotonic_seconds"] >= boundary:
                return True
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        return False

    def stop(self, *, deadline: float):
        self.done.set()
        self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            self.errors.append("telemetry worker did not terminate within bounded command deadline")
            self.failed.set()

    def summary(self) -> dict:
        return summarize_gpu_samples(self.samples, self.probe.devices, self.interval, self.errors,
                                     window_start=self.window_start, window_end=self.window_end, command_seconds=self.probe.timeout)


def _port_available(port: int):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))


def _health(port: int, timeout: float) -> bool:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        ready = response.status == 200
        response.close()
        return ready
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _owned_listener(owner: OwnedProcess, port: int) -> bool:
    # Linux socket inodes bind the responding loopback listener to the launched
    # process group; a foreign server racing for the port is never benchmarked.
    inodes = set()
    for row in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = row.split()
        if len(fields) > 9 and fields[1] == f"0100007F:{port:04X}" and fields[3] == "0A":
            inodes.add(fields[9])
    for member in _group_members(owner.identity["pgid"]):
        try:
            for link in Path(f"/proc/{member['pid']}/fd").iterdir():
                try:
                    target = os.readlink(link)
                except (FileNotFoundError, PermissionError):
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    return True
        except (FileNotFoundError, PermissionError):
            continue
    return False


class _AttemptMonitor:
    """Preempt native decoder hangs using the client's durable attempt ledger."""

    def __init__(self, journal: Path, seconds: float, concurrency: int = 1):
        self.journal, self.seconds = journal, seconds
        self.offset = 0
        self.concurrency = concurrency
        self.active = {}
        self.transport = set()
        self.waiting = set()
        self.seen = set()

    def check(self):
        try:
            with self.journal.open("rb") as stream:
                stream.seek(self.offset)
                while True:
                    line = stream.readline(1024 * 1024 + 1)
                    if not line:
                        break
                    if len(line) > 1024 * 1024:
                        raise RuntimeError("attempt ledger record exceeds bounded monitor size")
                    if not line.endswith(b"\n"):
                        break
                    self.offset = stream.tell()
                    event = json.loads(line)
                    slot = event.get("slot_id")
                    if event.get("event") == "attempt_started":
                        if not isinstance(slot, str) or slot in self.seen or len(self.transport) >= self.concurrency:
                            raise RuntimeError("client attempts exceed declared concurrency or reuse a slot")
                        self.seen.add(slot)
                        self.transport.add(slot)
                        self.active[slot] = time.monotonic()
                    elif event.get("event") == "transport_finished":
                        if slot not in self.transport:
                            raise RuntimeError("transport completion does not match active slot")
                        self.transport.remove(slot)
                        self.active.pop(slot)
                        self.waiting.add(slot)
                    elif event.get("event") == "validation_started":
                        if slot not in self.waiting:
                            raise RuntimeError("validation does not follow completed transport")
                        self.waiting.remove(slot)
                        self.active[slot] = time.monotonic()
                    elif event.get("event") == "attempt_finished":
                        slot = event.get("record", {}).get("slot_id")
                        if slot not in self.active and slot not in self.waiting:
                            raise RuntimeError("attempt ledger completion does not match active slot")
                        self.active.pop(slot, None)
                        self.transport.discard(slot)
                        self.waiting.discard(slot)
        except FileNotFoundError:
            pass
        if any(time.monotonic() - start >= self.seconds for start in self.active.values()):
            raise TimeoutError("hard supervised per-attempt deadline exceeded; stopping owned runtime and client")


def _wait_client(process: OwnedProcess, owner: OwnedProcess | None, sampler: _Sampler | None, supervisor: _Supervisor, deadline: float, monitor: _AttemptMonitor | None = None):
    while process.running():
        supervisor.check()
        process.check_output_budget()
        _check_deadline(deadline)
        if owner is not None and not owner.running():
            raise RuntimeError("owned runtime exited during measurement")
        if owner is not None:
            owner.check_output_budget()
        if sampler is not None and sampler.failed.is_set():
            raise RuntimeError("GPU telemetry/ownership failed during measurement")
        if monitor is not None:
            monitor.check()
        time.sleep(0.05)
    cleanup = process.close(supervisor.cleanup_seconds)
    if cleanup["status"] != "clean":
        raise RuntimeError("owned client process group failed to drain")
    return process.returncode


def _cleanup_role(owner: OwnedProcess, sampler: _Sampler, probe: GpuProbe, limits: dict, *, deadline: float | None = None) -> dict:
    end = deadline if deadline is not None else time.monotonic() + limits["cleanup_seconds"]
    sampler.end_measurement()
    sampler.done.set()
    receipt = owner.close(deadline=end)
    sampler.stop(deadline=end)
    idle_after = False
    snapshot = None
    observed = {(app["gpu_uuid"], app["pid"]): app["process_identity"]
                for sample in sampler.samples for app in sample.get("owned_compute_apps", [])}
    waited_for = set()
    while time.monotonic() < end:
        snapshot = probe.snapshot(deadline=end)
        if _idle(snapshot, limits["max_idle_memory_mib"]):
            idle_after = True
            break
        # NVML can retain exited ranks while the driver releases their memory.
        # Wait only for previously attributed identities, never unknown/reused PIDs.
        foreign = False
        for app in snapshot["compute_apps"]:
            expected = observed.get((app["gpu_uuid"], app["pid"]))
            current = _proc_identity(app["pid"])
            if expected is None or (current is not None and any(current[key] != expected[key] for key in ("pid", "pgid", "session_id", "start_ticks"))):
                foreign = True
                break
            waited_for.add(app["pid"])
        if foreign:
            break
        time.sleep(min(0.2, max(0, end - time.monotonic())))
    return {**receipt, "status": "clean" if receipt["status"] == "clean" and idle_after else "failed",
            "idle_after": idle_after, "gpu_after": snapshot, "waited_for_driver_pids": sorted(waited_for),
            "reason": "owned group drained and leased devices idle" if idle_after else "GPU resources not verified idle after exact-owned cleanup"}


def _role(spec: dict, label: str, directory: Path, supervisor: _Supervisor, probe: GpuProbe, receipt: dict) -> dict:
    metadata = directory / "supervisor" / label
    metadata.mkdir(parents=True)
    cache = metadata / "cache"
    cache.mkdir()
    nonce = uuid.uuid4().hex
    env = _runtime_env(spec[label]["source"], spec["gpu_uuids"], nonce, cache)
    role = receipt["roles"][label] = {"status": "preflight", "source_identity": None, "cleanup": {"status": "not_started"}, "run_path": f"{label}/run.json"}
    _write(directory / "gpu-job.json", receipt)
    identity = _source_identity(spec, label, env, supervisor.deadline, supervisor.cancelled)
    role["source_identity"] = identity
    snapshot = probe.snapshot()
    role["gpu_before"] = snapshot
    if not _idle(snapshot, spec["limits"]["max_idle_memory_mib"]):
        raise RuntimeError("leased GPUs have existing compute/memory use; no runtime started")
    role["power_configuration_before"] = probe.power_configuration(deadline=supervisor.deadline)
    # This SGLang revision expects NVML ordinals. The child checks the CUDA
    # driver's UUID order before importing SGLang; NVML/CUDA order may differ.
    indices = {device["uuid"]: device["index"] for device in snapshot["gpus"]}
    selected = [indices[device] for device in spec["gpu_uuids"]]
    if len(set(selected)) != len(selected) or any(type(index) is not int or index < 0 for index in selected):
        raise RuntimeError("GPU inventory has invalid or duplicate runtime indices")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))
    env["VGBENCH_GPU_UUIDS"] = ",".join(spec["gpu_uuids"])
    role["runtime_gpu_binding"] = {"cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"], "expected_gpu_uuids": spec["gpu_uuids"],
                                   "verification": "child CUDA driver UUID check before SGLang import"}
    _port_available(spec["port"])
    owner = None
    sampler = None
    client = None
    try:
        boot_started = time.monotonic()
        role["startup_timing_window"] = {"start_monotonic_seconds": boot_started, "start_utc": _now(),
                                        "end_monotonic_seconds": None}
        owner = supervisor.spawn(_server_argv(spec, label), cwd=metadata, env=env,
                                 stdout=metadata / "runtime.stdout.log", stderr=metadata / "runtime.stderr.log", nonce=nonce)
        role.update(status="starting", process_identity=owner.identity, launch_argv=_server_argv(spec, label), started_at=_now())
        _write(directory / "gpu-job.json", receipt)
        sampler = _Sampler(probe, owner, metadata / "telemetry.jsonl", spec["limits"]["telemetry_interval_seconds"],
                           initial_sample=snapshot)
        sampler.start()
        ready_deadline = min(supervisor.deadline, time.monotonic() + spec["limits"]["startup_seconds"])
        while True:
            supervisor.check()
            _check_deadline(ready_deadline)
            if not owner.running() or sampler.failed.is_set():
                raise RuntimeError("runtime exited or GPU ownership/telemetry failed before readiness")
            owner.check_output_budget()
            if _owned_listener(owner, spec["port"]) and _health(spec["port"], min(1, max(0.01, ready_deadline - time.monotonic()))):
                break
            time.sleep(0.1)
        role["startup_seconds"] = time.monotonic() - boot_started
        role["startup_timing_window"]["end_monotonic_seconds"] = boot_started + role["startup_seconds"]
        role["status"] = "measuring"
        sampler.begin_measurement()
        client_env = _runtime_env("", [], nonce, cache)
        client_env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        hardware = ",".join(spec["gpu_uuids"])
        argv = [sys.executable, "-m", "evaluator.cli", "run", str(directory / "plan.json"), "--runtime", "sglang",
                "--endpoint", f"http://127.0.0.1:{spec['port']}", "--runtime-revision", spec[label]["revision"],
                "--hardware-label", hardware, "--model-revision", spec["model"]["revision"],
                "--timeout-seconds", str(spec["limits"]["request_seconds"]), "--output", str(directory / label), "--execute"]
        if spec.get("serving"):
            argv.extend(["--serving-concurrency", str(spec["serving"]["concurrency"])])
            if spec["serving"]["delivery_deadline_seconds"] is not None:
                argv.extend(["--delivery-deadline-seconds", str(spec["serving"]["delivery_deadline_seconds"])])
        client = supervisor.spawn(argv, cwd=directory, env=client_env, stdout=metadata / "client.stdout.json",
                                  stderr=metadata / "client.stderr.log", nonce=nonce)
        role["client_process_identity"] = client.identity
        _write(directory / "gpu-job.json", receipt)
        slot_count = len(spec["plan"]["cases"]) * spec["plan"]["repetitions"] + spec["plan"]["warmup_runs"]
        phase_count = 2 if spec.get("serving") else 1
        client_deadline = min(supervisor.deadline, time.monotonic() + phase_count * slot_count * spec["limits"]["request_seconds"])
        code = _wait_client(client, owner, sampler, supervisor, client_deadline,
                            _AttemptMonitor(directory / label / "events.jsonl", spec["limits"]["request_seconds"], spec.get("serving", {}).get("concurrency", 1)))
        role["client_exit_code"] = code
        role["power_completion_bracket_recorded"] = sampler.bracket_completion(
            min(supervisor.deadline, time.monotonic() + 3 * spec["limits"]["telemetry_interval_seconds"]))
        run_path = directory / label / "run.json"
        if run_path.is_file():
            run = _read(run_path)
            role["run_sha256"] = _hash(run_path)
            role["run_summary"] = run.get("summary")
            if run.get("evidence_kind") not in {"operator_endpoint", "live_h3"} or run.get("plan_sha256") != _digest(spec["plan"]):
                raise RuntimeError("supervised client returned non-endpoint or wrong-plan evidence")
            role["run_status"] = run.get("status")
            finalized = bool(run.get("finished_at") and run.get("status") in {"complete", "partial", "failed"})
            role["status"] = "complete" if code in {0, 1} and finalized else "failed"
        else:
            role["status"] = "failed"
        if role["status"] != "complete":
            raise RuntimeError("supervised client did not finalize the scheduled workload")
    finally:
        cleanup_end = min(supervisor.total_deadline, time.monotonic() + spec["limits"]["cleanup_seconds"])
        if client is not None:
            try:
                role["client_cleanup"] = client.close(deadline=cleanup_end)
            except Exception as error:
                role["client_cleanup"] = {"status": "failed", "reason": type(error).__name__}
        if owner is not None and sampler is not None:
            try:
                role["cleanup"] = _cleanup_role(owner, sampler, probe, spec["limits"], deadline=cleanup_end)
            except Exception as error:
                role["cleanup"] = {"status": "failed", "idle_after": False, "reason": f"cleanup verification failed: {type(error).__name__}"}
            role["telemetry_summary"] = sampler.summary()
            role["telemetry_path"] = (metadata / "telemetry.jsonl").relative_to(directory).as_posix()
            role["telemetry_sha256"] = _hash(metadata / "telemetry.jsonl")
        elif owner is not None:
            role["cleanup"] = owner.close(deadline=cleanup_end)
            role["cleanup"].update(status="failed", idle_after=False, reason="telemetry did not start; GPU idle unverified")
        role["finished_at"] = _now()
        role["power_configuration_after"] = probe.power_configuration(deadline=supervisor.total_deadline)
        partial = directory / label / "run.json"
        if partial.is_file():
            role["run_sha256"] = _hash(partial, supervisor.total_deadline)
            role["run_summary"] = _read(partial).get("summary")
        _write(directory / "gpu-job.json", receipt)
    if role["cleanup"]["status"] != "clean":
        raise RuntimeError("owned runtime cleanup did not establish idle GPUs")
    if role.get("client_cleanup", {}).get("status") != "clean":
        raise RuntimeError("owned benchmark client cleanup did not qualify")
    if not role["telemetry_summary"]["qualified"]:
        raise RuntimeError("measurement lacks complete telemetry and visible owned compute on every selected GPU")
    if label == "baseline" and role.get("run_status") != "complete":
        raise RuntimeError("baseline workload contains failed/invalid slots; candidate was not started")
    # Files are verified again after execution; changed source cannot qualify.
    after = source_file_manifest(Path(spec[label]["source"]), timeout=spec["limits"]["command_seconds"], deadline=supervisor.deadline, cancelled=supervisor.cancelled)
    if after["source_sha256"] != role["source_identity"]["source_sha256"] or after["revision"] != role["source_identity"]["revision"]:
        raise RuntimeError("runtime source changed while executing")
    return role


def _calibration(spec: dict, current: dict | None = None, *, deadline: float | None = None) -> tuple[bool, str]:
    if spec["policy"]["calibration_status"] != "operator_calibrated":
        return False, "policy is not calibrated; measured hardware results remain useful but regression acceptance is inconclusive"
    if current is None or deadline is None:
        return False, "calibrated CI requires independently verified current and prior raw GPU evidence"
    from .mvp_gpu_evidence import verify_calibration
    return verify_calibration(spec, current, deadline=deadline)


def _gate(spec: dict, receipt: dict, comparison: dict | None, *, verified_evidence: dict | None = None, deadline: float | None = None) -> dict:
    reasons = []
    if verified_evidence is None:
        reasons.append("raw run/media/telemetry/comparison evidence has not been verified")
    if receipt.get("status") != "complete" or receipt.get("failures"):
        reasons.append("GPU supervision did not finish without infrastructure failures")
    measured = receipt.get("measurement_status") == "complete"
    if not measured:
        reasons.append("controlled GPU measurement is incomplete")
    if receipt.get("evidence_kind") != "controlled_h3_gpu":
        reasons.append("receipt does not contain controlled GPU execution evidence")
    if comparison is None:
        reasons.append("verified paired comparison is unavailable")
    elif comparison.get("measurement", {}).get("performance_mode") != "same_configuration_class_regression":
        reasons.append("performance comparison is descriptive or incomparable, not a passing CI gate")
    if comparison is not None and (comparison.get("evidence_kind") not in {"operator_endpoint", "live_h3"} or comparison.get("plan_sha256") != _digest(spec["plan"])):
        reasons.append("comparison evidence is fixture/imported/mixed or does not match the frozen workload")
    roles = [receipt.get("roles", {}).get(role, {}) for role in _ROLES]
    for label, role in zip(_ROLES, roles):
        if role.get("cleanup", {}).get("status") != "clean" or not role.get("telemetry_summary", {}).get("qualified"):
            reasons.append("cleanup or observed GPU telemetry requirements did not qualify")
        if not role.get("source_identity") or not role.get("process_identity"):
            reasons.append("observed runtime source or owned process identity is missing")
        elif any(role["source_identity"].get(field) != spec[label][field] for field in ("revision", "source_sha256")):
            reasons.append("observed source identity does not match pinned role")
        gpu_ids = [item.get("uuid") for item in role.get("telemetry_summary", {}).get("gpu_identity", [])]
        if sorted(gpu_ids) != sorted(spec["gpu_uuids"]):
            reasons.append("observed GPUs do not match selected UUIDs")
    if all(role.get("source_identity") for role in roles):
        for field in ("python_sha256", "python_version", "packages"):
            left, right = roles[0]["source_identity"].get(field), roles[1]["source_identity"].get(field)
            if field == "packages":
                left = {key: value for key, value in (left or {}).items() if key != "sglang"}
                right = {key: value for key, value in (right or {}).items() if key != "sglang"}
            if left != right:
                reasons.append(f"baseline/candidate runtime dependency {field} differs")
        if roles[0].get("telemetry_summary", {}).get("gpu_identity") != roles[1].get("telemetry_summary", {}).get("gpu_identity"):
            reasons.append("observed hardware/driver identity differs")
    try:
        calibrated, calibration_reason = _calibration(spec, verified_evidence, deadline=deadline)
    except Exception as error:
        calibrated = False
        detail = str(error) if isinstance(error, (ValueError, RuntimeError, TimeoutError)) else type(error).__name__
        calibration_reason = f"calibration evidence unavailable or invalid: {detail}"
    if not calibrated:
        reasons.append(calibration_reason)
    if spec["allocation"]["mode"] != "dedicated_ci":
        reasons.append("cooperative shared-node allocation is not dedicated CI isolation")
    memory_change = None
    if all(role.get("telemetry_summary", {}).get("qualified") for role in roles):
        left = roles[0]["telemetry_summary"]["observed_memory_peak_mib_by_gpu"]
        right = roles[1]["telemetry_summary"]["observed_memory_peak_mib_by_gpu"]
        if all(isinstance(left.get(device), (int, float)) and left[device] > 0 and isinstance(right.get(device), (int, float)) and right[device] >= 0 for device in spec["gpu_uuids"]):
            memory_change = max(right[device] / left[device] - 1 for device in spec["gpu_uuids"])
    memory_threshold = spec["policy"].get("max_memory_increase_fraction")
    memory_failed = memory_change is not None and memory_threshold is not None and memory_change > memory_threshold
    if memory_threshold is None or memory_change is None:
        reasons.append("an explicit sampled-memory regression gate with complete observations is required for CI acceptance")
    detected = bool(memory_failed or (comparison and comparison.get("overall_status") == "fail"))
    result = "fail" if detected else ("inconclusive" if reasons or not comparison or comparison.get("overall_status") != "pass" else "pass")
    return {"regression_status": result, "ci_accepted": result == "pass", "release_qualified": False,
            "acceptance_reasons": list(dict.fromkeys(reasons)), "calibration": {"verified": calibrated, "reason": calibration_reason},
            "memory_increase_fraction": memory_change, "memory_threshold": memory_threshold,
            "memory_gate_semantics": "worst per-GPU relative change in observed sampled device-used VRAM maxima; not exact allocator peaks",
            "memory_gate_status": "fail" if memory_failed else ("pass" if memory_change is not None and memory_threshold is not None else "inconclusive")}


def _require_linux():
    if platform.system() != "Linux" or not Path("/proc/self/stat").is_file():
        raise RuntimeError("controlled GPU execution requires Linux /proc ownership checks; preview is cross-platform")


def run_gpu_job(spec: dict, output_dir: Path, *, serving_smoke: bool = False) -> dict:
    """Launch owned H3 sessions, or one serving smoke session; Linux-only.

    Calling this function is execution authorization. CLI callers must put an
    explicit --execute barrier in front of it. No provisioning/download occurs.
    The result's measurement_status, regression_status, and ci_accepted are
    intentionally independent; an uncalibrated hardware measurement can finish
    successfully without claiming that CI acceptance or release was earned.
    Calibrated policies undergo prior-evidence/cell verification before weight
    hashing or any GPU lease/probe. Observed environment equivalence and evidence
    integrity are checked again after measurement; later drift can still reject
    acceptance after compute was consumed.
    """
    spec = validate_gpu_job(spec)
    if serving_smoke and not spec.get("serving"):
        raise ValueError("single-runtime smoke requires an explicit serving load")
    approval = spec["authorization"]
    if not approval["compute_approved"] or not approval["model_license_reviewed"] or not approval["approval_reference"].strip():
        raise ValueError("GPU execution requires explicit compute approval and model-license review with an approval reference; no work started")
    _require_linux()
    directory = Path(output_dir).absolute()
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("GPU evidence output must be a new directory")
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / "spec.json", spec)
    _write(directory / "plan.json", spec["plan"])
    _write(directory / "policy.json", spec["policy"])
    receipt = {"schema_version": VERSION, "bundle_type": "controlled_serving_smoke" if serving_smoke else "controlled_gpu_job", "job_id": spec["job_id"],
               "execution_id": uuid.uuid4().hex,
               "evidence_kind": "no_gpu_measurement", "status": "running", "measurement_status": "incomplete",
               "regression_status": "inconclusive", "ci_accepted": False, "release_qualified": False,
               "spec_sha256": _digest(spec), "plan_sha256": _digest(spec["plan"]), "started_at": _now(), "finished_at": None,
               "allocation": spec["allocation"], "allocation_verification": "operator-declared prerequisite; UUID locks and process checks are cooperative, not scheduler attestation",
               "authorization": spec["authorization"], "authorization_verification": "operator assertion, not independent legal or allocation verification",
               "roles": {}, "failures": [], "comparison_path": None, "cleanup_status": "not_started",
               "supervisor_source_sha256": _hash(Path(__file__)), "trust_boundary": "trusted runtime code and operator-provisioned runner; not a hostile-code sandbox"}
    _write(directory / "gpu-job.json", receipt)
    comparison = None
    verification_deadline = time.monotonic() + spec["limits"]["job_seconds"]
    try:
        with _Supervisor(spec["limits"]) as supervisor:
            verification_deadline = supervisor.deadline
            from .mvp_gpu_evidence import preflight_calibration
            try:
                receipt["calibration_preflight"] = preflight_calibration(
                    spec, directory, started_at=receipt["started_at"],
                    execution_id=receipt["execution_id"], deadline=supervisor.deadline,
                )
            except Exception as error:
                detail = str(error) if isinstance(error, (ValueError, RuntimeError, TimeoutError)) else type(error).__name__
                receipt["calibration_preflight"] = {"status": "failed", "performed_before_gpu_lease": True, "reason": detail}
                _write(directory / "gpu-job.json", receipt)
                raise
            _write(directory / "gpu-job.json", receipt)
            receipt["model_identity"] = _model_manifest(spec, supervisor.deadline, supervisor.cancelled)
            supervisor.check()
            probe = GpuProbe(spec["gpu_uuids"], spec["limits"]["command_seconds"])
            with GpuLease(Path(spec["lock_directory"]), spec["gpu_uuids"], spec["job_id"]) as lease:
                try:
                    for role in (("baseline",) if serving_smoke else _ROLES):
                        supervisor.check()
                        _role(spec, role, directory, supervisor, probe, receipt)
                        receipt["evidence_kind"] = "controlled_h3_gpu"
                        _write(directory / "gpu-job.json", receipt)
                    # Model mutation during measurement invalidates the pinned identity.
                    if _model_manifest(spec, supervisor.deadline, supervisor.cancelled) != receipt["model_identity"]:
                        raise RuntimeError("staged model identity changed during measurement")
                    receipt.update(measurement_status="complete", evidence_kind="controlled_h3_gpu", cleanup_status="clean")
                    _write(directory / "gpu-job.json", receipt)
                finally:
                    if any(role.get("cleanup", {}).get("status") == "failed" for role in receipt["roles"].values()):
                        receipt["cleanup_status"] = "failed"
                        lease.quarantine("GPU idle or exact-owned cleanup was not established; inspect before clearing quarantine")
                    elif receipt["roles"] and all(role.get("cleanup", {}).get("status") == "clean" for role in receipt["roles"].values()):
                        receipt["cleanup_status"] = "clean"
            if serving_smoke:
                receipt["status"] = "complete"
            else:
                # Native decode/comparison is also a supervised child with the same
                # global work deadline, so a wedged codec cannot retain GPU jobs.
                metadata = directory / "supervisor"
                client_env = _runtime_env("", [], uuid.uuid4().hex, metadata / "compare-cache")
                client_env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
                compare = supervisor.spawn([sys.executable, "-m", "evaluator.cli", "compare", str(directory / "baseline"), str(directory / "candidate"), "--policy", str(directory / "policy.json")],
                                           cwd=directory, env=client_env, stdout=directory / "comparison.json", stderr=metadata / "compare.stderr.log", nonce=uuid.uuid4().hex)
                code = _wait_client(compare, None, None, supervisor, supervisor.deadline)
                if code not in {0, 1, 2}:
                    raise RuntimeError("supervised comparison process exited abnormally")
                comparison = _read(directory / "comparison.json")
                if comparison.get("bundle_type") != "mvp_comparison":
                    raise RuntimeError("supervised comparison did not produce its validated contract")
                receipt.update(comparison_path="comparison.json", comparison_sha256=_hash(directory / "comparison.json"), status="complete")
    except (Exception, KeyboardInterrupt) as error:
        receipt["status"] = "aborted" if isinstance(error, (JobCancelled, KeyboardInterrupt, TimeoutError)) else "failed"
        receipt["failures"].append(str(error) if isinstance(error, (ValueError, RuntimeError, TimeoutError)) else type(error).__name__)
    receipt["finished_at"] = _now()
    _write(directory / "gpu-job.json", receipt)
    verified = None
    try:
        if receipt["status"] == "complete":
            from .mvp_gpu_evidence import verify_measurement_job
            verified = verify_measurement_job(directory, deadline=verification_deadline, serving_smoke=serving_smoke)
        if serving_smoke:
            receipt["measurement_verified"] = verified is not None
        else:
            receipt.update(_gate(spec, receipt, comparison, verified_evidence=verified, deadline=verification_deadline))
    except Exception as error:
        receipt.update(regression_status="inconclusive", ci_accepted=False,
                       acceptance_reasons=[f"acceptance evidence could not be verified: {type(error).__name__}"])
    receipt["finished_at"] = _now()
    _write(directory / "gpu-job.json", receipt)
    return receipt


def evaluate_gpu_job(jobdir: Path, *, verification_timeout_seconds: float = 60) -> dict:
    """Recheck saved hashes and fail-closed acceptance without starting GPU work.

    These hashes provide integrity within the trusted-runner threat model, not
    cryptographic proof that an arbitrary supplied directory came from a GPU.
    """
    directory = Path(jobdir).resolve(strict=True)
    _number(verification_timeout_seconds, "verification_timeout_seconds", 0.01, 600)
    deadline = time.monotonic() + verification_timeout_seconds
    from .mvp_gpu_evidence import verify_measurement_job
    verified = verify_measurement_job(directory, deadline=deadline)
    receipt, spec, comparison = verified["receipt"], verified["spec"], verified["comparison"]
    return {**receipt, **_gate(spec, receipt, comparison, verified_evidence=verified, deadline=deadline)}
