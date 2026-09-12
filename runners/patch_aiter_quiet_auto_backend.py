#!/usr/bin/env python3
"""Keep AITER automatic backend fallback quiet; warn once for explicit Gluon.

Runtime equivalent of https://github.com/ROCm/aiter/pull/5471.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

OLD_IMPORT = """import itertools

import torch
"""
NEW_IMPORT = """import itertools
from functools import lru_cache

import torch
"""
OLD_HELPER = """_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
"""
NEW_HELPER = """_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


@lru_cache(maxsize=1)
def _warn_gluon_fallback_once():
    _LOGGER.warning(
        "Gluon was explicitly requested for moe_gemm_a16w4 but is not supported "
        "on this GPU; using Triton."
    )


def _is_gluon_available():
"""
OLD_DISPATCH = """    if backend in (None, "gluon"):
        if _is_gluon_available():
            backend = "gluon"
        else:
            _LOGGER.warning("GLUON backend not available. Using TRITON backend!!!")
            backend = "triton"

    backend = backend.lower()
"""
NEW_DISPATCH = """    if backend in (None, "gluon"):
        if _is_gluon_available():
            backend = "gluon"
        else:
            if backend == "gluon":
                _warn_gluon_fallback_once()
            backend = "triton"

    backend = backend.lower()
"""
REPLACEMENTS = (
    (OLD_IMPORT, NEW_IMPORT),
    (OLD_HELPER, NEW_HELPER),
    (OLD_DISPATCH, NEW_DISPATCH),
)


def installed_backend_path() -> Path:
    """Locate the backend from the installed AITER root without importing it."""
    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("aiter package is not installed")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    return package_root / "ops/triton/moe/moe_op_gemm_a16w4.py"


def patch_backend(backend_path: Path) -> bool:
    """Preflight all supported blocks before writing; return whether changed."""
    source_bytes = backend_path.read_bytes()
    source = source_bytes.decode("utf-8")
    states = [(source.count(old), source.count(new)) for old, new in REPLACEMENTS]
    if all(state == (0, 1) for state in states):
        changed = False
    elif all(state == (1, 0) for state in states):
        changed = True
    else:
        raise RuntimeError(f"partially patched or unsupported AITER backend at {backend_path}")

    # Reject stray/modified pieces even if the supported blocks also exist.
    expected_markers = 0 if changed else 2
    if any(
        source.count(marker) != expected_markers
        for marker in ("lru_cache", "_warn_gluon_fallback_once")
    ):
        raise RuntimeError(f"partially patched or unsupported AITER backend at {backend_path}")

    patched = source
    if changed:
        for old, new in REPLACEMENTS:
            patched = patched.replace(old, new)
    patched_bytes = patched.encode("utf-8")
    compile(patched_bytes, str(backend_path), "exec")
    if changed:
        backend_path.write_bytes(patched_bytes)
    return changed


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [BACKEND_PATH]", file=sys.stderr)
        return 2

    try:
        backend_path = (
            Path(argv[1]) if len(argv) == 2 else installed_backend_path()
        ).resolve()
        before = hashlib.sha256(backend_path.read_bytes()).hexdigest()
        changed = patch_backend(backend_path)
        after = hashlib.sha256(backend_path.read_bytes()).hexdigest()
    except (OSError, RuntimeError, SyntaxError, UnicodeError) as error:
        print(f"ERROR: failed to patch AITER quiet auto backend: {error}", file=sys.stderr)
        return 1

    state = "Patch applied" if changed else "Already patched"
    print(f"{state}: AITER quiet auto backend at {backend_path}")
    print(f"SHA256 before={before} after={after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
