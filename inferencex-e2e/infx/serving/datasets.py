"""Lock-pinned file datasets of the serving scenarios: read the lock, fetch and verify one file.

    python -m infx.serving.datasets <name>   # local path on stdout, or one ERROR line, exit 1

Token: GREENLAB_HF_TOKEN when set; unset means anonymous (never the runner's HF_TOKEN).
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

LOCK = Path(__file__).resolve().parents[2] / "benchmarks/single_node/serving/datasets/lock.yaml"
FIELDS = {
    "repo": r"[^/\s]+/[^/\s]+",
    "revision": r"[0-9a-f]{40}",
    "file": r"\S+",
    "sha256": r"[0-9a-f]{64}",
}


class DatasetError(ValueError):
    """A lock or download problem, as one line."""


@dataclass(frozen=True)
class LockEntry:
    name: str
    repo: str
    revision: str
    file: str
    sha256: str
    entries: int | None = None
    private: bool = False


def load_lock(path: Path = LOCK) -> dict[str, LockEntry]:
    """Every pinned dataset by name; raise ``DatasetError`` on the first malformed entry."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise DatasetError(f"lock: cannot read {path.name} ({error})".replace("\n", " ")) from None
    if not isinstance(raw, dict):
        raise DatasetError("lock: must map dataset names to entries")
    lock = {}
    for name, entry in raw.items():
        keys = set(entry) if isinstance(entry, dict) else set()
        if not set(FIELDS) <= keys or keys - {*FIELDS, "entries", "private"}:
            raise DatasetError(
                f"lock: {name} needs exactly {', '.join(FIELDS)} (and optional entries, private)"
            )
        for field, pattern in FIELDS.items():
            if not re.fullmatch(pattern, str(entry[field])):
                raise DatasetError(
                    f"lock: {name}.{field} {entry[field]!r} does not match {pattern}"
                )
        entries = entry.get("entries")
        if entries is not None and (
            isinstance(entries, bool) or not isinstance(entries, int) or entries < 1
        ):
            raise DatasetError(f"lock: {name}.entries must be a positive integer")
        private = entry.get("private", False)
        if not isinstance(private, bool):
            raise DatasetError(f"lock: {name}.private must be true or false")
        lock[name] = LockEntry(name, *(str(entry[field]) for field in FIELDS), entries, private)
    return lock


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, lock: dict[str, LockEntry] | None = None) -> Path:
    """Download dataset ``name`` into the HF cache, check its sha256 and return its path."""
    entry = (load_lock() if lock is None else lock).get(name)
    if entry is None:
        raise DatasetError(f"dataset {name}: not in {LOCK.name}")
    token = os.environ.get("GREENLAB_HF_TOKEN") or None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    where = f"{entry.repo}@{entry.revision}"
    unreadable = (
        f"dataset {name}: {where} not readable; GREENLAB_HF_TOKEN is {'set' if token else 'unset'}"
    )
    try:
        path = Path(
            hf_hub_download(
                entry.repo,
                entry.file,
                revision=entry.revision,
                repo_type="dataset",
                token=token or False,
            )
        )
    except RepositoryNotFoundError:  # includes GatedRepoError
        raise DatasetError(unreadable) from None
    except HfHubHTTPError as error:
        if getattr(getattr(error, "response", None), "status_code", None) in (401, 403):
            raise DatasetError(unreadable) from None
        raise
    actual = sha256(path)
    if actual != entry.sha256:
        raise DatasetError(
            f"dataset {name}: sha256 mismatch for {where}/{entry.file}: "
            f"expected {entry.sha256}, got {actual}"
        )
    if os.environ.get("RESULT_DIR"):
        Path(os.environ["RESULT_DIR"], "dataset.txt").write_text(
            f"{name} {where} file={entry.file} sha256={actual} path={path}\n"
        )
    return path


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m infx.serving.datasets <name>", file=sys.stderr)
        return 2
    try:
        print(fetch(argv[0]))
    except DatasetError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
