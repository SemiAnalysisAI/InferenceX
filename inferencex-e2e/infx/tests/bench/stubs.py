"""Stub executables for the ``infx.bench`` tests."""

from __future__ import annotations

from pathlib import Path


def executable(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` as an executable script; return ``path``."""
    path.write_text(text)
    path.chmod(0o755)
    return path
