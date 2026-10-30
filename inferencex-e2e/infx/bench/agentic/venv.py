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
        """The venv directory under ``root``."""
        return self.root / "venv"

    @property
    def python(self) -> Path:
        """The venv's python interpreter."""
        return self.venv / "bin" / "python"

    @property
    def agentx(self) -> Path:
        """The venv's ``agentx`` executable."""
        return self.venv / "bin" / "agentx"

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
    if ref := os.environ.get("AGENTX_HARNESS_REF", "dac6e191afa5c4c64a85adb7569a89a711b953de"):
        install += ["--no-config", str(build_harness(runtime, uv_exe, ref, env))]
        install += ["datasets>=4.7.0"]
    else:
        install += ["--no-config", "-r", str(REQUIREMENTS)]
    if proc.call(install, {**env, "UV_HTTP_TIMEOUT": "120", "UV_HTTP_RETRIES": "3"}):
        raise BenchError(
            "benchmark client dependency bootstrap failed; inspect network/package "
            "resolution before recipe repairs"
        )
    if not (_executable(runtime.agentx) and _executable(runtime.hf)):
        raise BenchError(f"isolated AgentX environment is incomplete at {runtime.venv}")
    return 0


HARNESS_REPO = "https://github.com/SemiAnalysisAI/agentx-harness.git"


def build_harness(runtime: Runtime, uv_exe: Path, ref: str, env: Mapping[str, str]) -> Path:
    """Build the AgentX wheel from ``ref`` of the harness repository (diagnostic runs only)."""
    source = runtime.root / "agentx-harness-src"
    dist = runtime.root / "agentx-harness-dist"
    shutil.rmtree(source, ignore_errors=True)
    shutil.rmtree(dist, ignore_errors=True)
    print(f"Building AgentX harness from {HARNESS_REPO}@{ref}", flush=True)
    path_env = {**env, "PATH": f"{Path(uv_exe).parent}:{env.get('PATH', '')}"}
    steps = (
        ["git", "clone", "--filter=blob:none", "--no-checkout", HARNESS_REPO, str(source)],
        ["git", "-C", str(source), "checkout", "--detach", ref],
        [
            str(uv_exe), "run", "--no-project", "--with", "tomlkit", "--with", "packaging",
            str(source / "tools" / "build_agentx_release.py"),
            "--tag", "agentx-v1.0.9", "--out-dir", str(dist),
        ],
    )
    for step in steps:
        if proc.call(step, path_env):
            raise BenchError(f"AgentX harness source build failed at: {' '.join(step)}")
    wheels = sorted(dist.glob("agentx_harness-*.whl"))
    if len(wheels) != 1:
        raise BenchError(f"expected one AgentX wheel in {dist}, found {wheels}")
    return wheels[0]


def _executable(path: Path) -> bool:
    """Whether ``path`` is a file this process may execute."""
    return path.is_file() and os.access(path, os.X_OK)
