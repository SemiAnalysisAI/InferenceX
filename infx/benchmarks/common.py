"""Filesystem and child-process boundaries shared by prepared clients."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from .spec import PreparedFile, RuntimeSpec, secret_environment_key


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(file: PreparedFile) -> Path:
    path = Path(file.path)
    if not path.is_file() or sha256_file(path) != file.sha256:
        raise ValueError(f"prepared file missing or changed: {path}")
    return path


def verify_snapshot_assets(
    runtime: RuntimeSpec,
    repository: str,
    *,
    expected_revision: str | None,
    only_snapshot: bool,
    repo_type: Literal["dataset", "model"] = "dataset",
) -> str:
    cache = Path(runtime.env["HF_HUB_CACHE"]) / (f"{repo_type}s--" + repository.replace("/", "--"))
    reference = cache / "refs" / "main"
    revision = reference.read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"prepared {repo_type} revision must be an immutable snapshot SHA")
    if expected_revision is not None and revision != expected_revision:
        raise ValueError(f"offline {repo_type} main ref does not match the prepared snapshot")
    snapshot = cache / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(f"prepared {repo_type} snapshot is unavailable")
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"prepared {repo_type} snapshot is empty")
    bound = {Path(asset.path).resolve() for asset in runtime.assets}
    if any(path.resolve() not in bound for path in (reference, *files)):
        raise ValueError(
            f"{repo_type} snapshot/ref contains content absent from the prepared asset list"
        )
    if only_snapshot and [path for path in snapshot.parent.iterdir() if path.is_dir()] != [
        snapshot
    ]:
        raise ValueError(f"pilot {repo_type} cache must contain only the prepared snapshot")
    return revision


def verify_model_snapshot_assets(
    runtime: RuntimeSpec,
    repository: str,
    *,
    expected_revision: str,
    expected_snapshot: Path,
) -> Path:
    """Bind nominal tokenizer lookup to the exact canonical serving snapshot."""
    revision = verify_snapshot_assets(
        runtime,
        repository,
        expected_revision=expected_revision,
        only_snapshot=False,
        repo_type="model",
    )
    cache = Path(runtime.env["HF_HUB_CACHE"]) / ("models--" + repository.replace("/", "--"))
    snapshot = (cache / "snapshots" / revision).resolve(strict=True)
    if snapshot != expected_snapshot.resolve(strict=True):
        raise ValueError("offline model cache snapshot differs from the serving model snapshot")
    return snapshot


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def read_json(path: Path) -> Any:
    return decode_json(path.read_text())


def decode_json(text: str) -> Any:
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    require_finite(value)
    return value


def require_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite numeric value")
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    if isinstance(value, list):
        for item in value:
            require_finite(item)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_endpoint(endpoint: str) -> str:
    url = urlsplit(endpoint)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        raise ValueError("endpoint must be an HTTP(S) origin without credentials or a path")
    # Accessing port also rejects malformed or out-of-range port values.
    _ = url.port
    return endpoint.rstrip("/")


def child_environment(runtime: RuntimeSpec) -> dict[str, str]:
    env = dict(os.environ)
    for key in (*runtime.env_unset, "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    # Ambient AIPerf settings can silently change the registered scenario.
    for key in list(env):
        if key.startswith("AIPERF_") or secret_environment_key(key):
            env.pop(key)
    env.update(runtime.env)
    return env


def run_child(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    log: Path,
    timeout_seconds: float,
    terminate_grace_seconds: float,
) -> dict[str, Any]:
    """Run one foreground client, forwarding cancellation to its entire process group."""
    stopped_by: int | None = None

    def stop(signum: int, _frame: Any) -> None:
        nonlocal stopped_by
        if stopped_by is None:
            stopped_by = signum

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    child: subprocess.Popen[bytes] | None = None
    try:
        with log.open("xb") as output:
            child = subprocess.Popen(
                list(argv),
                env=dict(env),
                cwd=cwd,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            cleanup_deadline: float | None = None
            while child.poll() is None:
                if stopped_by is not None or time.monotonic() - started >= timeout_seconds:
                    timed_out = stopped_by is None
                    with _process_gone_ok():
                        os.killpg(child.pid, signal.SIGTERM)
                    # One deadline; repeated TERM does not restart or reenter cleanup.
                    cleanup_deadline = time.monotonic() + terminate_grace_seconds
                    while child.poll() is None and time.monotonic() < cleanup_deadline:
                        time.sleep(0.05)
                    # The leader may have exited while descendants still hold artifact files.
                    with _process_gone_ok():
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    break
                time.sleep(0.05)
            orphaned_descendants = False
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                pass
            else:
                orphaned_descendants = True
                with _process_gone_ok():
                    os.killpg(child.pid, signal.SIGTERM)
                deadline = cleanup_deadline or time.monotonic() + terminate_grace_seconds
                while time.monotonic() < deadline:
                    try:
                        os.killpg(child.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.05)
                with _process_gone_ok():
                    os.killpg(child.pid, signal.SIGKILL)
            return {
                "returncode": child.returncode,
                "cancelled_by_signal": stopped_by,
                "timed_out": timed_out,
                "orphaned_descendants": orphaned_descendants,
            }
    finally:
        if child is not None and child.poll() is None:
            with _process_gone_ok():
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _process_gone_ok() -> Any:
    from contextlib import suppress

    return suppress(ProcessLookupError)


def child_failed(status: Mapping[str, Any]) -> bool:
    return bool(
        status["returncode"] != 0
        or status["cancelled_by_signal"]
        or status["timed_out"]
        or status["orphaned_descendants"]
    )
