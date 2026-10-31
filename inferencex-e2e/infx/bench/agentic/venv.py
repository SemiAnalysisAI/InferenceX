"""The isolated AIPerf runtime: a per-job uv venv with the pinned PyPI nightly.

The nightly (not the 0.13.0 release) is required: released 0.13.0 rejects the
serving scenarios' ``phases[].concurrency.warmupRequestsPerLane`` and
``traceIdleGapCapSeconds`` keys."""

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
        job = environ.get("SLURM_JOB_ID") or str(os.getpid())
        return cls(Path(tempfile.gettempdir()) / f"inferencex-agentic-{job}")

    @property
    def venv(self) -> Path:
        """The venv directory under ``root``."""
        return self.root / "venv"

    @property
    def python(self) -> Path:
        """The venv's python interpreter."""
        return self.venv / "bin" / "python"

    @property
    def aiperf(self) -> Path:
        """The venv's ``aiperf`` executable."""
        return self.venv / "bin" / "aiperf"

    @property
    def hf(self) -> Path:
        """The venv's ``hf`` executable."""
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
    install += ["--no-config", "-r", str(REQUIREMENTS)]
    if proc.call(install, {**env, "UV_HTTP_TIMEOUT": "120", "UV_HTTP_RETRIES": "3"}):
        raise BenchError(
            "benchmark client dependency bootstrap failed; inspect network/package "
            "resolution before recipe repairs"
        )
    if not (_executable(runtime.aiperf) and _executable(runtime.hf)):
        raise BenchError(f"isolated AIPerf environment is incomplete at {runtime.venv}")
    return 0


def _executable(path: Path) -> bool:
    """Whether ``path`` is a file this process may execute."""
    return path.is_file() and os.access(path, os.X_OK)
