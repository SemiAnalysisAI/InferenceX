"""The one package installer: ``uv`` from ``PATH``, else Astral's standalone build."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from infx.bench import env, proc

INSTALLER = "https://astral.sh/uv/install.sh"


def find(install_dir: Path | None = None) -> str:
    """Return a ``uv`` executable, installing it into ``install_dir`` when absent.

    Rootless enroot containers cannot mutate dpkg, so the standalone installer is used.
    """
    found = shutil.which("uv")
    if found:
        return found
    install_dir = install_dir or Path(tempfile.gettempdir()) / "inferencex-uv"
    uv = install_dir / "uv"
    if not (uv.is_file() and os.access(uv, os.X_OK)):
        install_dir.mkdir(parents=True, exist_ok=True)
        environ = {**os.environ, "UV_INSTALL_DIR": str(install_dir), "UV_NO_MODIFY_PATH": "1"}
        proc.call(["sh", "-c", f"curl -LsSf {INSTALLER} | sh"], environ)
    if not (uv.is_file() and os.access(uv, os.X_OK)):
        raise env.BenchError(f"uv installation did not create {uv}")
    return str(uv)


def pip(*args: str, python: str = sys.executable) -> list[str]:
    """``uv pip <args>`` aimed at ``python``'s environment."""
    return [find(), "pip", *args, "--python", python]
