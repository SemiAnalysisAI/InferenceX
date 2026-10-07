"""Verify the CPU-prepared AMD exporter and reuse its exact squash bytes.

The install command is stdlib-only so an allocated host can populate its local cache
before srt-slurm starts the exporter. It never imports from a container registry.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infx.clusters.slurm import SquashPolicy


@dataclass(frozen=True)
class PreparedExporter:
    """Verified source and the explicit cache file the exporter must start from."""

    source: Path
    destination: Path
    sha256: str
    node_local: bool


def file_sha256(path: Path) -> str:
    """Hash without loading the image into memory; host Python 3.10 is supported."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_exporter(source: Path, destination: Path, expected_sha256: str) -> None:
    """Publish identical verified bytes atomically, never replacing another image."""
    if file_sha256(source) != expected_sha256:
        raise ValueError(f"Prepared exporter checksum mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".amd-dme-", dir=destination.parent) as staging:
            staged = Path(staging) / source.name
            shutil.copyfile(source, staged)
            if file_sha256(staged) != expected_sha256:
                raise ValueError("Prepared exporter changed while being copied")
            with contextlib.suppress(FileExistsError):
                os.link(staged, destination)
    if file_sha256(destination) != expected_sha256:
        raise ValueError(f"Refusing to replace a different cached exporter: {destination}")


def stage_exporter(
    artifact: Path,
    image: str,
    policy: SquashPolicy,
    expected_sha256: str,
    provenance_output: Path,
) -> PreparedExporter:
    """Prepare shared storage now, or an explicit allocated-host copy for local storage."""
    from infx.launch.backends.slurm.squash import squash_path

    artifact = artifact.resolve(strict=True)
    subprocess.run(["sha256sum", "--check", "SHA256SUMS"], cwd=artifact, check=True)
    source = artifact / "amd-exporter.sqsh"
    provenance = json.loads((artifact / "provenance.json").read_text())
    if not image.endswith("@" + provenance["image"]["upstream_reported_registry_digest"]):
        raise ValueError("Prepared exporter does not match the configured image digest")
    if file_sha256(source) != expected_sha256:
        raise ValueError("Prepared exporter does not match the caller's pinned squash checksum")
    destination = squash_path(image, policy)
    node_local = policy.visibility == "node-local"
    if not node_local:
        install_exporter(source, destination, expected_sha256)
    shutil.copyfile(artifact / "provenance.json", provenance_output)
    return PreparedExporter(source, destination, expected_sha256, node_local)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("sha256")
    args = parser.parse_args()
    try:
        install_exporter(args.source, args.destination, args.sha256)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Verified prepared AMD exporter: {args.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
