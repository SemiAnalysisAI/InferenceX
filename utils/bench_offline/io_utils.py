"""Small file and process helpers shared by the offline benchmark."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    """Append one JSONL record while serializing writers across MPI ranks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            fd = os.open(
                path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(fd, payload[offset:])
                    if written == 0:
                        raise OSError("JSONL append made no progress")
                    offset += written
            finally:
                os.close(fd)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def read_locked_text(path: Path) -> str:
    """Read a JSONL snapshot without observing an in-flight append."""
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_SH)
        try:
            try:
                return path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return ""
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def tail_text(path: Path, max_bytes: int = 2_000_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        return stream.read().decode("utf-8", errors="replace")
