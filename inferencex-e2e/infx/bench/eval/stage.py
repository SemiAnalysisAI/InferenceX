"""Copying an eval's allow-listed artifacts out of its results directory."""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path

# The workflows upload exactly these names from the workspace root; collectors read them there.
ARTIFACTS = (
    "results*.json",
    "*_report.json",
    "*_results.jsonl",
    "*_artifacts.tar.gz",
    "sample*.jsonl",
)
_EXTENSIONS = (".tar.gz", ".jsonl", ".json")


def copy(results_dir: Path, destination: Path, *, suffix: str = "") -> list[Path]:
    """Copy the artifacts anywhere under ``results_dir`` into ``destination``; return them."""
    destination.mkdir(parents=True, exist_ok=True)
    staged = []
    for source in sorted(results_dir.rglob("*")):
        if source.is_file() and any(fnmatch.fnmatchcase(source.name, glob) for glob in ARTIFACTS):
            target = _target(destination, source.name, suffix)
            shutil.copyfile(source, target)
            staged.append(target)
    return staged


def _target(destination: Path, name: str, suffix: str) -> Path:
    # Batched runs share one destination, so a suffixed name never replaces another.
    if not suffix:
        return destination / name
    extension = next(extension for extension in _EXTENSIONS if name.endswith(extension))
    stem = name[: -len(extension)]
    target = destination / f"{stem}{suffix}{extension}"
    count = 2
    while target.exists():
        target = destination / f"{stem}{suffix}_{count}{extension}"
        count += 1
    return target
