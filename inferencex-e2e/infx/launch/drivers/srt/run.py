"""One srt-slurm launch: its request, the Slurm backend, and the environment srtctl runs with."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from infx.config import repository_root
from infx.launch.backends.slurm import SlurmBackend
from infx.launch.context import Launch, LaunchError
from infx.launch.policy import runtime_env
from infx.launch.request import LaunchRequest, RequestError

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.slurm import SrtSlurmSettings
    from infx.launch.lifecycle import Lifecycle
    from infx.launch.request import SrtRequest


@dataclass
class SrtRun:
    """One srt-slurm launch and the environment its children (git, uv, make, srtctl) get."""

    cluster: Cluster
    backend: SlurmBackend
    request: SrtRequest
    life: Lifecycle
    env: dict[str, str]
    srt: SrtSlurmSettings

    @classmethod
    def create(cls, launch: Launch, request: SrtRequest) -> SrtRun:
        """Children get the runtime settings, then the cluster's ``env`` and ``srt-slurm.env``."""
        backend = slurm_backend(launch)
        srt = backend.settings.srt_slurm
        if srt is None:
            raise LaunchError(f"cluster {launch.cluster.id!r} has no slurm.srt-slurm settings")
        env = runtime_env(launch.cluster, request)
        env.update(launch.cluster.env)
        env.update(srt.env)
        # Slurm jobs build their own venv; the launcher's must not leak into them.
        env.pop("VIRTUAL_ENV", None)
        root = str(repository_root())
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
        # Read by srt-slurm's post-benchmark eval.
        env["INFMAX_WORKSPACE"] = str(request.workspace)
        return cls(launch.cluster, backend, request, launch.life, env, srt)

    @property
    def workspace(self) -> Path:
        return self.request.workspace

    def prepend_path(self, directory: Path) -> None:
        self.env["PATH"] = os.pathsep.join(filter(None, (str(directory), self.env.get("PATH"))))


def require(request: LaunchRequest, *names: str) -> None:
    """Fail unless every variable is set: inputs only some lanes need."""
    missing = [name for name in names if not request.env.get(name)]
    if missing:
        raise RequestError.missing(*missing)


def slurm_backend(launch: Launch) -> SlurmBackend:
    if not isinstance(launch.backend, SlurmBackend):
        raise TypeError(
            f"{launch.path} needs the Slurm backend, got {type(launch.backend).__name__}"
        )
    return launch.backend
