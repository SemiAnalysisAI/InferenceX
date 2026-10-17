"""The isolated AgentX runtime: a per-job uv venv with the pinned PyPI release."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from infx.bench import proc, uv
from infx.bench.env import BenchError

PYTHON_VERSION = "3.11"
REQUIREMENTS = Path(__file__).with_name("requirements.txt")
VENDOR = Path(__file__).with_name("vendor")
HARNESS_WHEEL = VENDOR / "agentx_harness-1.0.11-py3-none-any.whl"
EXCLUDE_NEWER = "2026-10-17T00:00:00Z"
"""Both experiment arms resolve the harness dependencies against the same index snapshot."""


@dataclass(frozen=True)
class Runtime:
    """Paths of one job's AgentX runtime directory."""

    root: Path

    @classmethod
    def for_job(cls, environ: Mapping[str, str]) -> Runtime:
        """``AIPERF_RUNTIME_DIR``, else ``inferencex-agentic-<job>`` in the temp directory."""
        explicit = environ.get("AIPERF_RUNTIME_DIR")
        if explicit:
            return cls(Path(explicit).absolute())
        job = environ.get("SLURM_JOB_ID") or str(os.getpid())
        return cls(Path(tempfile.gettempdir()) / f"inferencex-agentic-{job}")

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def agentx(self) -> Path:
        return self.venv / "bin" / "agentx"

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


def bootstrap(runtime: Runtime) -> int:
    """Create ``runtime``'s venv from scratch; return ``uv venv``'s status, else 0."""
    uv_exe = uv.find(runtime.root / "uv" / "bin")
    shutil.rmtree(runtime.venv, ignore_errors=True)
    cache = runtime.root / "uv-cache"
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "UV_CACHE_DIR": str(cache)}
    rc = proc.call([uv_exe, "venv", "--python", PYTHON_VERSION, str(runtime.venv)], env)
    if rc:
        return rc
    install = [uv_exe, "pip", "install", "--python", str(runtime.python)]
    install += ["--no-config", "--exclude-newer", EXCLUDE_NEWER, "-r", str(REQUIREMENTS), str(HARNESS_WHEEL)]
    if proc.call(install, {**env, "UV_HTTP_TIMEOUT": "120", "UV_HTTP_RETRIES": "3"}):
        raise BenchError(
            "benchmark client dependency bootstrap failed; inspect network/package "
            "resolution before recipe repairs"
        )
    if not (_executable(runtime.agentx) and _executable(runtime.hf)):
        raise BenchError(f"isolated AgentX environment is incomplete at {runtime.venv}")
    _record_harness(runtime, uv_exe)
    return 0


def _record_harness(runtime: Runtime, uv_exe: str) -> None:
    """Write the installed harness identity beside the results; refuse to replay with another arm."""
    out = Path(os.environ["RESULT_DIR"]).absolute() / "x1l_harness_fingerprint.json"
    script = Path(__file__).with_name("x1l_fingerprint.py")
    argv = [str(runtime.python), "-P", str(script), str(out), str(VENDOR / "ARM.json"), str(HARNESS_WHEEL), uv_exe]
    if proc.call(argv, dict(os.environ)):
        raise BenchError(f"installed AgentX harness does not match {VENDOR / 'ARM.json'}; see {out}")


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
