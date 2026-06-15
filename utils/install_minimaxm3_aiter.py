#!/usr/bin/env python3
"""Install the pinned MI300X AITER build used by the M3 fusion experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Wheel:
    distribution: str
    version: str
    filename: str
    url: str
    sha256: str


WHEELS = (
    Wheel(
        distribution="flydsl",
        version="0.1.9.dev20260529+d7ae22d",
        filename=(
            "flydsl-0.1.9.dev20260529+d7ae22d-cp312-cp312-"
            "manylinux_2_27_x86_64.whl"
        ),
        url=(
            "https://rocm.frameworks-devreleases.amd.com/whl-staging/"
            "gfx942-gfx950/flydsl-0.1.9.dev20260529%2Bd7ae22d-"
            "cp312-cp312-manylinux_2_27_x86_64.whl"
        ),
        sha256="8f3de2a83c7f3775962edb4ac929612956c4b4bb358c0ea5fa97643d3397791d",
    ),
    Wheel(
        distribution="amd-aiter",
        version="0.1.15.post1+rocm7.2.manylinux.2.28",
        filename=(
            "amd_aiter-0.1.15.post1+rocm7.2.manylinux.2.28-"
            "cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        ),
        url=(
            "https://github.com/ROCm/aiter/releases/download/v0.1.15.post1/"
            "amd_aiter-0.1.15.post1%2Brocm7.2.manylinux.2.28-"
            "cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        ),
        sha256="c4ccc510f223f36d9ca69f70cd817e373869e2caac98465e6ab527d9b2cfc1e1",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_wheels_match() -> bool:
    for wheel in WHEELS:
        try:
            installed = importlib.metadata.version(wheel.distribution)
        except importlib.metadata.PackageNotFoundError:
            return False
        if installed != wheel.version:
            return False
    return True


def download_wheel(wheel: Wheel, destination: Path, attempts: int = 3) -> None:
    request = urllib.request.Request(
        wheel.url,
        headers={"User-Agent": "InferenceX-MiniMaxM3-AITER-installer"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                with destination.open("wb") as file:
                    shutil.copyfileobj(response, file, length=1024 * 1024)
            actual_sha256 = sha256_file(destination)
            if actual_sha256 != wheel.sha256:
                raise RuntimeError(
                    f"{wheel.filename}: expected sha256 {wheel.sha256}, "
                    f"got {actual_sha256}"
                )
            return
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(attempt * 2)


def verify_installation() -> None:
    for wheel in WHEELS:
        installed = importlib.metadata.version(wheel.distribution)
        if installed != wheel.version:
            raise RuntimeError(
                f"{wheel.distribution}: expected {wheel.version}, got {installed}"
            )

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import inspect; import aiter; "
                "from aiter.dist.device_communicators.custom_all_reduce "
                "import CustomAllreduce; "
                "assert 'use_1stage' in "
                "inspect.signature(CustomAllreduce.fused_ar_rms).parameters"
            ),
        ],
        check=True,
    )


def install_wheels() -> None:
    if installed_wheels_match():
        verify_installation()
        print("Pinned MiniMax M3 AITER wheels are already installed")
        return

    torch_version = importlib.metadata.version("torch")
    with tempfile.TemporaryDirectory(prefix="inferencex-m3-aiter-") as temp_dir:
        paths: list[Path] = []
        for wheel in WHEELS:
            path = Path(temp_dir) / wheel.filename
            print(f"Downloading {wheel.distribution} {wheel.version}")
            download_wheel(wheel, path)
            paths.append(path)

        for wheel, path in zip(WHEELS, paths, strict=True):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--no-deps",
                    "--force-reinstall",
                    str(path),
                ],
                check=True,
            )
            print(f"Installed {wheel.distribution} {wheel.version}")

    if importlib.metadata.version("torch") != torch_version:
        raise RuntimeError("Pinned AITER installation unexpectedly changed torch")
    verify_installation()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that the pinned wheels are already installed",
    )
    args = parser.parse_args()

    if args.check:
        if not installed_wheels_match():
            raise SystemExit("Pinned MiniMax M3 AITER wheels are not installed")
        verify_installation()
        print("Pinned MiniMax M3 AITER wheels validated")
        return
    install_wheels()


if __name__ == "__main__":
    main()
