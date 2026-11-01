"""Opt-in, request-correlated timing for the pinned H3 singleton runtime.

The two-file runtime patch calls this stdlib-only module in HTTP and scheduler
processes. No GPU synchronization, tracing backend, or network call is added.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time


VERSION = "1.0.0"
BASE_REVISION = "71de97b264b04dcd514cf904003028aefe9775c8"
PATCH = Path(__file__).parents[1] / "runtime-patches/sglang-71de97b-h3-server-timing.patch"
PATCHED_FILES = {
    "python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py": "27c4c13ad1417161d0b3f9f9cfaa4a259c86bdc3614450a43e8e07f87c37ed05",
    "python/sglang/multimodal_gen/runtime/managers/scheduler.py": "1cfb5d6db81dd8cfb1eea7444971f92140639fc98bb8b4a882a2ab2ba65cab6b",
}
STAGES = ("http_received", "http_accepted", "scheduler_dispatch", "forward_finished", "media_ready")
DURATIONS = {
    "prequeue_seconds": ("http_received", "http_accepted"),
    "queue_delay_seconds": ("http_accepted", "scheduler_dispatch"),
    "execution_seconds": ("scheduler_dispatch", "forward_finished"),
    "postprocess_seconds": ("forward_finished", "media_ready"),
    "server_ready_latency_seconds": ("http_received", "media_ready"),
}
_SAFE_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.:-]{0,199}")
_MAX_BYTES = 4 * 1024 * 1024


def identity() -> dict:
    return {"schema_version": VERSION, "base_runtime_revision": BASE_REVISION,
            "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "patched_runtime_files": dict(PATCHED_FILES)}


def validate_source(source: Path) -> None:
    for name, expected in PATCHED_FILES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("server timing requires the exact committed H3 instrumentation patch")


@lru_cache(maxsize=1)
def _clock_id() -> str:
    # Linux boot + time namespace identify a shared monotonic clock across the
    # HTTP and GPU-worker processes. Never subtract these from client clocks.
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    namespace = os.readlink("/proc/self/ns/time")
    return f"linux:{boot}:{namespace}:CLOCK_MONOTONIC"


def emit(request_id: str, event: str, **fields) -> None:
    path = os.environ.get("VGBENCH_SERVER_TIMING_PATH")
    if not path:
        return
    timestamp = time.monotonic_ns()
    if not isinstance(request_id, str) or not _SAFE_ID.fullmatch(request_id) or event not in STAGES:
        raise ValueError("invalid H3 server timing event identity")
    value = {"schema_version": VERSION, "request_id": request_id, "event": event,
             "monotonic_ns": timestamp, "clock_id": _clock_id(), "pid": os.getpid(),
             "instance_id": os.environ["VGBENCH_SERVER_TIMING_INSTANCE"], **fields}
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        # One small O_APPEND write per event. The supervisor owns/fsyncs the
        # final ledger; per-request fsync would perturb the measured workload.
        if os.write(descriptor, payload) != len(payload):
            raise OSError("incomplete H3 timing event write")
    finally:
        os.close(descriptor)


@contextmanager
def forward(requests: list, *, replica_id: int, leader: bool):
    enabled = bool(os.environ.get("VGBENCH_SERVER_TIMING_PATH")) and leader
    if enabled:
        if len(requests) != 1 or requests[0].num_outputs_per_prompt != 1:
            raise ValueError("H3 timing patch only qualifies singleton forward execution")
        request_id = requests[0].request_id
        emit(request_id, "scheduler_dispatch", observed_batch_size=len(requests), replica_id=replica_id)
    try:
        yield
    finally:
        if enabled:
            emit(request_id, "forward_finished")


def read_events(path: Path) -> list[dict]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("server timing ledger must be a regular file")
        data = stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError("server timing ledger exceeds its size bound")
    # Another request may be writing the final line. Its incomplete event is
    # unavailable until the next read, never a fabricated complete timestamp.
    events = [json.loads(line) for line in data.split(b"\n")[:-1]]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("server timing events must be objects")
    return events


def derive(events: list[dict], request_id: str, instance_id: str) -> dict | None:
    selected = [row for row in events if row.get("request_id") == request_id]
    if not selected:
        return None
    stages = {}
    clock_ids = set()
    for row in selected:
        stage, ns = row.get("event"), row.get("monotonic_ns")
        if (row.get("schema_version") != VERSION or row.get("instance_id") != instance_id
                or stage not in STAGES or stage in stages or type(ns) is not int or ns <= 0
                or not isinstance(row.get("clock_id"), str) or not row["clock_id"]):
            raise ValueError("invalid, duplicate or mismatched server timing event")
        stages[stage] = row
        clock_ids.add(row["clock_id"])
    ordered = [stages[name]["monotonic_ns"] for name in STAGES if name in stages]
    if len(clock_ids) != 1 or ordered != sorted(ordered):
        raise ValueError("server timing clocks differ or stages are reversed")
    dispatch = stages.get("scheduler_dispatch", {})
    batch, replica = dispatch.get("observed_batch_size"), dispatch.get("replica_id")
    if dispatch and (type(batch) is not int or batch != 1 or type(replica) is not int or replica < 0):
        raise ValueError("invalid observed H3 batch or replica identity")
    result = {"schema_version": VERSION, "status": "complete" if len(stages) == len(STAGES) else "partial",
              "request_id": request_id, "instance_id": instance_id, "clock_id": next(iter(clock_ids)),
              "clock": "time.monotonic_ns; same Linux boot and time namespace; nanoseconds",
              "observed_batch_size": batch, "replica_id": replica,
              "timestamps_ns": {name: stages.get(name, {}).get("monotonic_ns") for name in STAGES}}
    for name, (begin, end) in DURATIONS.items():
        result[name] = ((stages[end]["monotonic_ns"] - stages[begin]["monotonic_ns"]) / 1e9
                        if begin in stages and end in stages else None)
    return result


def collect(request_id: str | None) -> dict | None:
    path = os.environ.get("VGBENCH_SERVER_TIMING_PATH")
    if not path or request_id is None:
        return None
    try:
        return derive(read_events(Path(path)), request_id, os.environ["VGBENCH_SERVER_TIMING_INSTANCE"])
    except (OSError, ValueError, TypeError, KeyError):
        # Preserve the video outcome. The independent evidence verifier rejects
        # malformed timing rather than making generation appear to have failed.
        return {"schema_version": VERSION, "status": "invalid", "request_id": request_id}


def verify_evidence(directory: Path, role: dict, run: dict, *, required: bool = False) -> None:
    evidence = role.get("server_timing_evidence")
    if not evidence:
        if required or any(record.get("server_timings") is not None for record in run["records"]):
            raise ValueError("server timings lack a supervisor-owned evidence ledger")
        return
    if (evidence.get("schema_version") != VERSION or evidence.get("base_runtime_revision") != BASE_REVISION
            or evidence.get("patch_sha256") != hashlib.sha256(PATCH.read_bytes()).hexdigest()
            or evidence.get("patched_runtime_files") != PATCHED_FILES
            or any(not isinstance(evidence.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", evidence[key])
                   for key in ("patch_sha256", "helper_sha256"))
            or evidence.get("instance_id") != role.get("process_identity", {}).get("launch_nonce")):
        raise ValueError("server timing instrumentation identity is missing or inconsistent")
    raw = evidence.get("path")
    if (not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts
            or "\\" in raw):
        raise ValueError("invalid server timing ledger path")
    path = (directory / raw).resolve(strict=True)
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("server timing ledger escapes its job directory")
    events = read_events(path)
    if path.stat().st_size > _MAX_BYTES or hashlib.sha256(path.read_bytes()).hexdigest() != evidence.get("sha256"):
        raise ValueError("server timing ledger hash mismatch")
    for record in run["records"]:
        expected = derive(events, record.get("job_id"), evidence["instance_id"])
        if record.get("server_timings") != expected:
            raise ValueError("server timings differ from request-correlated raw events")
        if required and record.get("status") == "succeeded" and (expected or {}).get("status") != "complete":
            raise ValueError("successful request lacks complete server timing evidence")
