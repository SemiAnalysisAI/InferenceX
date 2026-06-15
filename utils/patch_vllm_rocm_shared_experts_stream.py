#!/usr/bin/env python3
"""Enable vLLM's existing shared-expert auxiliary stream on ROCm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import stat
import tempfile
from pathlib import Path


RELATIVE_PATH = "vllm/model_executor/layers/fused_moe/runner/shared_experts.py"
EXPECTED_SOURCE_SHA256 = (
    "5e06a945729ceac5c033f0876119cc395fd22da12c70a9cd07a32fbb2be4def0"
)
EXPECTED_PATCHED_SHA256 = (
    "f47474638508ac2a4dad361cf7c704a2cdd151a83defb35c4e2c14c6a06e1687"
)
SOURCE_ANCHOR = """        should_run_shared_in_aux_stream = (
            current_platform.is_cuda()
"""
PATCHED_ANCHOR = """        should_run_shared_in_aux_stream = (
            current_platform.is_cuda_alike()
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patch_source(source: str) -> str:
    """Apply the one-line ROCm eligibility change from vLLM PR #38665."""
    if PATCHED_ANCHOR in source:
        return source
    count = source.count(SOURCE_ANCHOR)
    if count != 1:
        raise RuntimeError(f"{RELATIVE_PATH}: expected one source anchor, found {count}")
    return source.replace(SOURCE_ANCHOR, PATCHED_ANCHOR, 1)


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

    if source_sha == EXPECTED_PATCHED_SHA256:
        patched = source
    else:
        if source_sha != EXPECTED_SOURCE_SHA256:
            raise RuntimeError(
                f"{RELATIVE_PATH}: source fingerprint mismatch; "
                f"expected {EXPECTED_SOURCE_SHA256} or "
                f"{EXPECTED_PATCHED_SHA256}, got {source_sha}"
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
        f"ROCm shared-expert stream runtime patch {action} for vLLM {version}; "
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
