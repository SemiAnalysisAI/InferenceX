"""Integrity-checked, disposable mmap snapshots around the unchanged pinned client.

Only the owning UID shares snapshots. A client receives an independent copy, never
a hardlink to shared data. Duplicate cold preparation is permitted on contention;
canonical publication is always locked and atomic.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import read_json, sha256_file, write_json


@contextmanager
def _lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    with path.open("a+b") as stream:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("derived-cache lock contention deadline exceeded") from None
                time.sleep(0.02)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _inventory(root: Path) -> dict[str, str]:
    if root.is_symlink():
        raise ValueError("derived cache snapshot must not be a symlink")
    inventory: dict[str, str] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or re.fullmatch(r"[0-9a-f]{32}", entry.name) is None:
            continue
        if entry.is_symlink():
            raise ValueError("derived cache entries must not be symlinks")
        manifest = read_json(entry / "manifest.json")
        if manifest.get("cache_key") != entry.name or manifest.get("compressed") is not False:
            raise ValueError(
                "derived cache has an unexpected key or unsupported compressed payload"
            )
        for name in ("dataset.dat", "index.dat"):
            payload = entry / name
            if not payload.is_file() or payload.stat().st_size == 0:
                raise ValueError("derived cache payload is incomplete")
        for path in sorted(entry.rglob("*")):
            if path.is_symlink():
                raise ValueError("derived cache payload must not be a symlink")
            if path.is_file():
                inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    if not inventory:
        raise ValueError("derived cache has no complete client-produced entry")
    return inventory


class MmapCache:
    def __init__(
        self,
        base: Path,
        contract: dict[str, Any],
        *,
        lock_timeout_seconds: float,
        validate_manifests: Callable[[Path], None] | None = None,
    ) -> None:
        if not base.is_absolute():
            raise ValueError("mmap cache base must be explicit and absolute")
        identity = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        private = base / f"infx-uid-{os.getuid()}"
        private.mkdir(mode=0o700, parents=True, exist_ok=True)
        mode = private.stat()
        if private.is_symlink() or mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) & 0o077:
            raise ValueError(
                "client cache namespace must be private and owned by the executing UID"
            )
        self.root = private / identity
        self.root.mkdir(mode=0o700, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("derived cache namespace must not be a symlink")
        self.canonical = self.root / "complete"
        self.run_dir = self.root / f"run-{uuid.uuid4().hex}"
        self.lock_timeout_seconds = lock_timeout_seconds
        self.validate_manifests = validate_manifests
        self.events: list[str] = []

    def _check(self) -> dict[str, str]:
        receipt = read_json(self.canonical / "infx-integrity.json")
        current = _inventory(self.canonical)
        if self.validate_manifests is not None:
            self.validate_manifests(self.canonical)
        if receipt != {"schema_version": 1, "files": current}:
            raise ValueError("derived cache payload differs from its integrity receipt")
        return current

    def _quarantine(self) -> None:
        self.canonical.rename(self.root / f"quarantine-{uuid.uuid4().hex}")
        self.events.append("invalid snapshot quarantined; rebuilding from authoritative inputs")

    def prepare(self) -> Path:
        try:
            with _lock(self.root / "publication.lock", self.lock_timeout_seconds):
                if self.canonical.exists():
                    try:
                        expected = self._check()
                    except (OSError, ValueError, TypeError, KeyError):
                        self._quarantine()
                    else:
                        shutil.copytree(self.canonical, self.run_dir, copy_function=shutil.copyfile)
                        (self.run_dir / "infx-integrity.json").unlink()
                        if _inventory(self.run_dir) != expected:
                            raise ValueError("derived cache changed while copying")
                        self.events.append("verified snapshot copied without hardlinks")
                        return self.run_dir
        except TimeoutError:
            self.events.append("lock contention: preparing an independent cold cache")
        self.run_dir.mkdir(mode=0o700)
        return self.run_dir

    def publish(self) -> None:
        """Call only after the foreground client and its result validators succeeded."""
        try:
            inventory = _inventory(self.run_dir)
            if self.validate_manifests is not None:
                self.validate_manifests(self.run_dir)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.events.append(f"cache publication skipped: {exc}")
            return
        try:
            with _lock(self.root / "publication.lock", self.lock_timeout_seconds):
                if self.canonical.exists():
                    try:
                        self._check()
                    except (OSError, ValueError, TypeError, KeyError):
                        self._quarantine()
                    else:
                        self.events.append("another completed snapshot already published")
                        return
                write_json(
                    self.run_dir / "infx-integrity.json", {"schema_version": 1, "files": inventory}
                )
                # This is an owned directory populated by one exited client. Rename
                # publishes payload and integrity receipt together, never a half entry.
                self.run_dir.rename(self.canonical)
                self.events.append("verified snapshot published atomically")
        except TimeoutError:
            self.events.append("cache publication skipped on lock contention")

    def close(self) -> None:
        # Quarantined shared evidence remains for diagnosis; only this run's copy is removed.
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
