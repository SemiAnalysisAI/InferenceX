"""The isolated AIPerf runtime: a per-job uv venv with ``utils/aiperf`` installed editable."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from infx.bench import proc

UV_INSTALLER = "https://astral.sh/uv/install.sh"
# AIPerf plus what the post-run AgentX processing and plots import.
DEPENDENCIES = (
    "numpy>=1.24",
    "pandas>=2.0.0",
    "aiohttp>=3.10",
    "transformers>=4.46",
    "xlsxwriter>=3.2.1",
    "tqdm>=4.66",
    "datasets>=4.7.0",
    "tiktoken",
    "matplotlib",
    "huggingface_hub[cli]>=0.25.0",
    "urllib3",
    "requests",
)


@dataclass(frozen=True)
class Runtime:
    """Paths of one job's AIPerf runtime directory."""

    root: Path

    @classmethod
    def for_job(cls, environ: Mapping[str, str]) -> Runtime:
        """``AIPERF_RUNTIME_DIR``, else ``inferencex-agentic-<job>`` in the temp directory."""
        explicit = environ.get("AIPERF_RUNTIME_DIR")
        if explicit:
            return cls(Path(explicit).absolute())
        # The process id survives the re-exec into the venv, so both phases agree.
        job = environ.get("SLURM_JOB_ID") or str(os.getpid())
        return cls(Path(tempfile.gettempdir()) / f"inferencex-agentic-{job}")

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def aiperf(self) -> Path:
        return self.venv / "bin" / "aiperf"

    @property
    def hf(self) -> Path:
        return self.venv / "bin" / "hf"

    def active(self) -> bool:
        """Whether this process already runs the venv's python."""
        return Path(sys.executable) == self.python

    def exec_python(self, args: Sequence[str], environ: Mapping[str, str]) -> NoReturn:
        """Replace this process with the venv's python running ``args``."""
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(self.python, [str(self.python), *args], environ)  # noqa: S606


def bootstrap(runtime: Runtime, python_version: str, aiperf_source: Path) -> int:
    """Create ``runtime``'s venv from scratch; return a shell-style status."""
    uv = _uv(runtime.root / "uv" / "bin")
    if uv is None:
        return 1
    shutil.rmtree(runtime.venv, ignore_errors=True)
    cache = runtime.root / "uv-cache"
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "UV_CACHE_DIR": str(cache)}
    # AIPerf dropped Python 3.10, which some ROCm images still ship; uv fetches a pinned build.
    rc = proc.call([uv, "venv", "--python", python_version, str(runtime.venv)], env)
    if rc:
        return rc
    install = [uv, "pip", "install", "--python", str(runtime.python), "-e", str(aiperf_source)]
    retries = {"UV_HTTP_TIMEOUT": "120", "UV_HTTP_RETRIES": "3"}
    if proc.call([*install, *DEPENDENCIES], {**env, **retries}):
        print(
            "ERROR: benchmark client dependency bootstrap failed; inspect network/package "
            "resolution before recipe repairs",
            file=sys.stderr,
        )
        return 1
    if not (_executable(runtime.aiperf) and _executable(runtime.hf)):
        print(
            f"ERROR: isolated AIPerf environment is incomplete at {runtime.venv}", file=sys.stderr
        )
        return 1
    return 0


def _uv(install_dir: Path) -> str | None:
    """``uv`` from ``PATH``, else Astral's installer (rootless enroot cannot mutate dpkg)."""
    found = shutil.which("uv")
    if found:
        return found
    uv = install_dir / "uv"
    if not _executable(uv):
        install_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "UV_INSTALL_DIR": str(install_dir), "UV_NO_MODIFY_PATH": "1"}
        proc.call(["sh", "-c", f"curl -LsSf {UV_INSTALLER} | sh"], env)
    if not _executable(uv):
        print(f"ERROR: uv installation did not create {uv}", file=sys.stderr)
        return None
    return str(uv)


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
