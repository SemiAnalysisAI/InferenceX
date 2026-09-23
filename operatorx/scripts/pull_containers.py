#!/usr/bin/env python3
"""Import container images from containers.toml as enroot squash files.

Run on the cluster head node before submit_smoke.py. Idempotent — already-imported
images are skipped. The actual import runs inside srun on a compute node, matching
the cluster convention.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.10 fallback (e.g. b300 login)
    import tomli as tomllib  # type: ignore

import os

SQUASH_DIR = Path(os.environ.get("OPERATORX_SQUASH_DIR", "/home/sa-shared/containers"))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "containers.toml"
PARTITION = os.environ.get("OPERATORX_PARTITION", "gpu-2")
ACCOUNT = os.environ.get("OPERATORX_ACCOUNT")  # None -> omit --account
QOS = os.environ.get("OPERATORX_QOS")          # None -> omit --qos


def safe_name(image: str) -> str:
    out = image
    for ch in "/:@#":
        out = out.replace(ch, "_")
    return out


def squash_path(image: str) -> Path:
    return SQUASH_DIR / f"{safe_name(image)}.sqsh"


def unique_images(platform: str) -> set[str]:
    data = tomllib.loads(MANIFEST.read_text())
    return {entry["image"] for entry in data.get(platform, {}).values()}


def import_image(image: str) -> int:
    sqsh = squash_path(image)
    if sqsh.exists():
        print(f"[skip] {image} -> {sqsh}")
        return 0
    SQUASH_DIR.mkdir(parents=True, exist_ok=True)

    sqsh_q = shlex.quote(str(sqsh))
    lock_q = shlex.quote(f"{sqsh}.lock")
    image_q = shlex.quote(image)
    inner = (
        f"exec 9>{lock_q} && "
        f"flock -w 600 9 || {{ echo 'lock timeout'; exit 1; }} && "
        f"if unsquashfs -l {sqsh_q} >/dev/null 2>&1; then "
        f"echo '[skip] became valid during lock wait'; "
        f"else rm -f {sqsh_q} && enroot import -o {sqsh_q} docker://{image_q}; fi"
    )

    cmd = ["srun", f"--partition={PARTITION}", "--gres=gpu:1", "--time=60",
           f"--job-name=pull-{safe_name(image)[:40]}"]
    if ACCOUNT:
        cmd.append(f"--account={ACCOUNT}")
    if QOS:
        cmd.append(f"--qos={QOS}")
    cmd += ["bash", "-c", inner]
    print(f"[pull] {image} -> {sqsh}")
    return subprocess.run(cmd).returncode


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <platform>")
        print("  e.g., nvidia | amd | tpu | trainium")
        return 2
    platform = argv[1]
    images = unique_images(platform)
    if not images:
        print(f"no entries for platform={platform!r} in {MANIFEST}")
        return 1
    print(f"{len(images)} unique image(s) to ensure for {platform}:")
    for img in sorted(images):
        print(f"  {img}")
    print()
    rc = 0
    for image in sorted(images):
        if import_image(image) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
