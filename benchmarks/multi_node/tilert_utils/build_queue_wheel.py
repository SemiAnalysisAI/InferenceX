#!/usr/bin/env python3
"""Build InferenceX's queueing backport from the official TileRT post2 wheel."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


UPSTREAM_VERSION = "0.1.5.post2"
PATCHED_VERSION = "0.1.5.post2+inferencex.1"
UPSTREAM_WHEEL = "tilert-0.1.5.post2-cp312-cp312-manylinux_2_28_x86_64.whl"
UPSTREAM_URL = (
    "https://github.com/tile-ai/TileRT/releases/download/"
    f"v{UPSTREAM_VERSION}/{UPSTREAM_WHEEL}"
)
UPSTREAM_SHA256 = "e65b876ccfc1a419b0047a6d6b395f619ea35c15194ad1892f171c78476fe407"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_upstream(destination: Path) -> None:
    with urllib.request.urlopen(UPSTREAM_URL) as response:  # noqa: S310 (fixed URL)
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    actual = sha256(destination)
    if actual != UPSTREAM_SHA256:
        raise RuntimeError(
            f"upstream wheel SHA256 mismatch: expected {UPSTREAM_SHA256}, got {actual}"
        )


def update_metadata(unpacked: Path) -> None:
    old_dist_info = unpacked / f"tilert-{UPSTREAM_VERSION}.dist-info"
    new_dist_info = unpacked / f"tilert-{PATCHED_VERSION}.dist-info"
    old_dist_info.rename(new_dist_info)
    metadata = new_dist_info / "METADATA"
    text = metadata.read_text()
    old_version = f"Version: {UPSTREAM_VERSION}\n"
    if text.count(old_version) != 1:
        raise RuntimeError("expected exactly one upstream Version field in METADATA")
    metadata.write_text(text.replace(old_version, f"Version: {PATCHED_VERSION}\n"))


def build(output_dir: Path) -> Path:
    patch = Path(__file__).with_name("patches") / "tilert-0.1.5.post2-queue.patch"
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tilert-queue-wheel-") as temporary:
        work = Path(temporary)
        upstream = work / UPSTREAM_WHEEL
        unpacked = work / "unpacked"
        download_upstream(upstream)
        with zipfile.ZipFile(upstream) as archive:
            archive.extractall(unpacked)
        subprocess.run(
            ["patch", "-p1", "--batch", "--forward", "-i", str(patch)],
            cwd=unpacked,
            check=True,
        )
        update_metadata(unpacked)
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", "--dest-dir", str(output_dir), "."],
            cwd=unpacked,
            check=True,
        )
    wheels = list(output_dir.glob("tilert-0.1.5.post2+inferencex.1-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one patched wheel, found {len(wheels)}")
    return wheels[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    wheel = build(args.output_dir.resolve())
    print(f"{sha256(wheel)}  {wheel}")


if __name__ == "__main__":
    main()
