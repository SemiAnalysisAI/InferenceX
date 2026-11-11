"""Prove the image runs the PR head's Python on the merge-base wheel's compiled artifacts.

Reads the PR checkout at /src with the file lists the build stage wrote from git.
"""

import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import vllm

SOURCE = Path("/src")
SITE = Path(vllm.__file__).resolve().parent.parent
MERGE_BASE = os.environ["MERGE_BASE"]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def lines(path: str) -> set[str]:
    return {line for line in Path(path).read_text().splitlines() if line}


def main() -> int:
    print("installed vllm", vllm.__version__, "at", SITE)
    meta_url = f"https://wheels.vllm.ai/{MERGE_BASE}/cu130/vllm/metadata.json"
    with urllib.request.urlopen(meta_url) as response:
        wheels = json.load(response)
    wheel = next(
        w for w in wheels if w["package_name"] == "vllm" and "x86_64" in w["platform_tag"]
    )
    local = Path("/tmp/merge-base.whl")
    urllib.request.urlretrieve(urllib.parse.urljoin(meta_url, wheel["path"]), local)
    print("merge-base wheel:", wheel["filename"], "sha256", digest(local))
    archive = zipfile.ZipFile(local)
    shipped = set(archive.namelist())

    tracked = {path for path in lines("/tracked.txt") if path.endswith(".py")}
    changed = {
        path
        for path in lines("/pr-files.txt")
        if path.startswith("vllm/") and path.endswith(".py") and path in tracked
    }
    installed = {
        str(path.relative_to(SITE)) for path in (SITE / "vllm").rglob("*.py") if path.is_file()
    }

    # Every tracked module the official wheel ships, or the PR touches, comes from the PR head.
    missing = sorted({path for path in tracked if path in shipped or path in changed} - installed)
    differing = sorted(
        path for path in installed & tracked if digest(SOURCE / path) != digest(SITE / path)
    )
    extra = sorted(installed - tracked - shipped - {"vllm/_version.py"})
    print(
        f"PR head modules: {len(tracked)} tracked, {len(changed)} changed by the PR; "
        f"installed: {len(installed)}; missing: {len(missing)}; "
        f"differing from the PR head: {len(differing)}; unexpected: {len(extra)}"
    )
    for label, paths in (("missing", missing), ("differing", differing), ("unexpected", extra)):
        for path in paths[:20]:
            print(f"  {label}: {path}")

    binaries = sorted(name for name in shipped if name.endswith(".so") or name == "vllm/vllm-rs")
    mismatched = []
    for name in binaries:
        target = SITE / name
        if not target.exists():
            mismatched.append(f"absent {name}")
        elif digest(target) != hashlib.sha256(archive.read(name)).hexdigest():
            mismatched.append(f"differs {name}")
    print(
        f"compiled artifacts in the merge-base wheel: {len(binaries)}; "
        f"not identical in the image: {len(mismatched)}"
    )
    for item in mismatched[:20]:
        print("  ", item)
    local.unlink()

    ampere = SITE / "vllm/models/deepseek_v41/nvidia/specialized_decode/ampere/__init__.py"
    print("SM80 kernels installed:", ampere.is_file())
    ok = ampere.is_file() and not (missing or differing or extra or mismatched)
    print("VERIFY", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
