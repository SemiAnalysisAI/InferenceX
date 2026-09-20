"""Standalone Python 3.11 probe; deliberately does not import the Python 3.12 wrapper."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_runtime(distributions: list[str], dataset_loader: str | None) -> dict[str, Any]:
    def normalized(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    installed = sorted(
        (normalized(dist.metadata["Name"]), dist.version)
        for dist in importlib.metadata.distributions()
    )
    if len({name for name, _ in installed}) != len(installed):
        raise ValueError("installed runtime has ambiguous duplicate distributions")
    entries = sorted(
        (entry.group, entry.name, entry.value, entry.dist.metadata["Name"])
        for entry in importlib.metadata.entry_points(group="aiperf.plugins")
    )
    selected = (
        {normalized(name) for name in distributions}
        | {normalized(entry[3]) for entry in entries}
        | {name for name, _ in installed}
    )
    distribution_files = {}
    for name in sorted(selected):
        dist = importlib.metadata.distribution(name)
        if not dist.files:
            raise ValueError(f"installed distribution has no file manifest: {name}")
        files = {}
        for entry in sorted(dist.files):
            if entry.suffix == ".pyc" or "__pycache__" in entry.parts:
                continue
            path = Path(dist.locate_file(entry))
            if not path.is_file():
                raise ValueError(f"installed distribution file missing: {name}/{entry}")
            files[str(entry)] = _file_digest(path)
        raw_url = dist.read_text("direct_url.json")
        direct_url = json.loads(raw_url) if raw_url else None
        url = urlsplit(direct_url.get("url", "")) if direct_url else None
        if url is not None and (url.username is not None or url.password is not None):
            raise ValueError(
                "installed source metadata contains credentials and cannot be published"
            )
        distribution_files[name] = {
            "version": dist.version,
            "files": files,
            "direct_url": direct_url,
        }
    resolution = None
    if dataset_loader is not None:
        # Import only inside the selected installed child, never from a checkout.
        with contextlib.redirect_stdout(sys.stderr):
            from aiperf.plugin import plugins

            entry = plugins.get_entry("public_dataset_loader", dataset_loader)
            resolution = entry.model_dump(mode="json", exclude={"loaded_class"})
    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "python_build": list(platform.python_build()),
        "python_sha256": _file_digest(Path(sys.executable).resolve()),
        "python_paths": {
            "executable": str(Path(sys.executable).absolute()),
            "executable_resolved": str(Path(sys.executable).resolve()),
            "prefix": str(Path(sys.prefix).absolute()),
            "base_prefix": str(Path(sys.base_prefix).absolute()),
        },
        "installed": installed,
        "distributions": distribution_files,
        "aiperf_plugin_entry_points": entries,
        "dataset_resolution": resolution,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", action="append", required=True)
    parser.add_argument("--dataset-loader")
    args = parser.parse_args()
    print(json.dumps(inspect_runtime(args.distribution, args.dataset_loader), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
