#!/usr/bin/env python3
"""Limit vLLM's ROCm GPU profiler to local rank 0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import stat
import tempfile
from pathlib import Path


RELATIVE_PATH = "vllm/v1/worker/gpu_worker.py"
PATCH_MARKER = "InferenceX MI300X profiling: keep ROCTracer on rank 0."
EXPECTED_SOURCE_SHA256 = (
    "61eb2c62031337ca131d5610b5d5b2998630e159b48e1066b8d80bb68b4594ef"
)
EXPECTED_PATCHED_SHA256 = (
    "2e3eade99ba27dee35edf6d404a3490e57b2b288b8916fef514ac20da7d40a8c"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patch_source(source: str) -> str:
    """Keep ROCTracer on rank 0 while preserving profiler collectives."""
    if PATCH_MARKER in source:
        return source

    anchor = '                        activities=["CPU", "CUDA"],'
    replacement = """                        # InferenceX MI300X profiling: keep ROCTracer on rank 0.
                        # CPU-only peers preserve collective profiler control.
                        activities=(
                            ["CPU", "CUDA"]
                            if self.local_rank == 0
                            else ["CPU"]
                        ),"""
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"{RELATIVE_PATH}: expected one source anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def find_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Unable to locate the installed vLLM package")
    return Path(spec.origin).resolve().parent.parent


def _write_atomic(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temp_file:
        temp_file.write(text)
        temp_path = Path(temp_file.name)
    temp_path.chmod(mode)
    os.replace(temp_path, path)


def apply_runtime_patch(vllm_root: Path, check_only: bool = False) -> None:
    path = vllm_root / RELATIVE_PATH
    source = path.read_text(encoding="utf-8")
    source_sha = _sha256(source)

    if PATCH_MARKER in source:
        if source_sha != EXPECTED_PATCHED_SHA256:
            raise RuntimeError(
                f"{RELATIVE_PATH}: patched source fingerprint mismatch; "
                f"expected {EXPECTED_PATCHED_SHA256}, got {source_sha}"
            )
        patched = source
    else:
        if source_sha != EXPECTED_SOURCE_SHA256:
            raise RuntimeError(
                f"{RELATIVE_PATH}: source fingerprint mismatch; "
                f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha}"
            )
        patched = patch_source(source)
        patched_sha = _sha256(patched)
        if patched_sha != EXPECTED_PATCHED_SHA256:
            raise RuntimeError(
                f"{RELATIVE_PATH}: generated patch fingerprint mismatch; "
                f"expected {EXPECTED_PATCHED_SHA256}, got {patched_sha}"
            )

    compile(patched, str(path), "exec")
    if not check_only:
        _write_atomic(path, patched)

    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    action = "validated" if check_only else "applied"
    print(
        f"MI300X rank-0 profiler runtime patch {action} for vLLM {version}; "
        f"sha256={_sha256(patched)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.vllm_root.resolve() if args.vllm_root else find_vllm_root()
    apply_runtime_patch(root, check_only=args.check)


if __name__ == "__main__":
    main()
