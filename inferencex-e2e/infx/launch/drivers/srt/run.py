"""One srt-slurm launch and the environment its children (git, uv, make, srtctl) run with."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from infx.config import repository_root
from infx.launch.backends.slurm import SlurmBackend, cli
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
    cluster: Cluster
    backend: SlurmBackend
    request: SrtRequest
    life: Lifecycle
    env: dict[str, str]
    srt: SrtSlurmSettings
    account: str | None  # the Slurm account jobs are submitted under

    @classmethod
    def create(
        cls, launch: Launch, request: SrtRequest, job_env: Mapping[str, str] | None = None
    ) -> SrtRun:
        """Children get the runtime settings: the cluster's, ``srt-slurm.env``, then ``job_env``."""
        backend = slurm_backend(launch)
        srt = backend.settings.srt_slurm
        if srt is None:
            raise LaunchError(f"cluster {launch.cluster.id!r} has no slurm.srt-slurm settings")
        env = runtime_env(launch.cluster, request, srt.env, job_env or {})
        # Slurm jobs build their own venv; the launcher's must not leak into them.
        env.pop("VIRTUAL_ENV", None)
        root = str(repository_root())
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
        # Read by srt-slurm's post-benchmark eval.
        env["INFMAX_WORKSPACE"] = str(request.workspace)
        # Without an account, srtctl falls back to SLURM_ACCOUNT, then to "default", which
        # some clusters reject; the user's Slurm default comes before that.
        account = backend.settings.account or env.get("SLURM_ACCOUNT") or cli.default_account()
        return cls(launch.cluster, backend, request, launch.life, env, srt, account)

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
