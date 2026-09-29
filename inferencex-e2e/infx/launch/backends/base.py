"""The backend interface: how drivers run containers and follow jobs on any scheduler.

A backend adapts one scheduler (a cluster record's ``scheduler:``) and serves one launch.
Drivers name what runs; the backend owns how: images, delivering the checkout and volumes,
returning outputs, following logs, reading job state, and cancelling every job it creates
when the launch's :class:`~infx.launch.lifecycle.Lifecycle` ends.
"""

from __future__ import annotations

import fnmatch
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.base import SchedulerSettings
    from infx.launch.lifecycle import Lifecycle
    from infx.launch.request import LaunchRequest


class BackendError(RuntimeError):
    """A scheduler, image, or storage operation failed; reported without a traceback."""


class JobState(StrEnum):
    """A job's state in scheduler-neutral terms."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class JobStatus:
    """What a backend knows about a job; ``raw`` is the scheduler's reading, for diagnostics.

    ``exit_code`` is the command's exit status as a shell reports it (128 + N after
    signal N) when the scheduler reports one. Only ``state`` decides success: a job
    that did not succeed can still report 0 (Slurm reads ``CANCELLED|0:0`` so).
    """

    state: JobState
    raw: str
    exit_code: int | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the job finished successfully."""
        return self.state is JobState.SUCCEEDED


@dataclass(frozen=True)
class Job:
    """A job a backend follows: one it started, or one another tool submitted.

    Backends return their own subclasses; drivers only pass them back.
    """

    id: str


@dataclass(frozen=True)
class Image:
    """An image made available to a backend's jobs; they start from ``reference``."""

    name: str
    reference: str


@dataclass(frozen=True)
class Mount:
    """Cluster volume ``volume`` (a ``<scheduler>.volumes`` name) seen at ``target``.

    The backend resolves the name to its own volume spec. With ``create`` it makes the
    volume's directory before the container starts.
    """

    volume: str
    target: PurePosixPath
    create: bool = False


@dataclass(frozen=True)
class Container:
    """One containerized command on one node, with the checkout at ``workdir``.

    ``workspace`` is delivered at ``workdir``: bind-mounted, or copied without the
    entries whose names match an ``exclude`` pattern (fnmatch, at any depth).
    ``outputs`` are the workspace-relative paths the launch needs back, which
    :meth:`Backend.fetch_outputs` returns. ``required_paths`` must be readable where
    the container runs, or :meth:`Backend.run_container` fails before starting it.
    ``env`` extends the environment described by :meth:`Backend.container_env`; it may
    carry credentials, which a backend must expose no more widely than the launching
    environment does.
    """

    image: Image
    command: Sequence[str]
    gpus: int
    time_limit_min: int
    workspace: Path
    workdir: PurePosixPath
    env: Mapping[str, str] = field(default_factory=dict)
    mounts: Sequence[Mount] = ()
    required_paths: Sequence[PurePosixPath] = ()
    outputs: Sequence[PurePosixPath] = ()
    exclude: Sequence[str] = ()


# Variables that describe the launching host rather than the workload, as fnmatch
# patterns: its session and locale, its Python/uv/conda interpreters and linker paths,
# the GitHub runner and its tokens, and its GPU selection. In a container they would
# name host paths or a host interpreter, or leak runner credentials. Bash's own state
# is host-specific too, and exporting a readonly one (UID, SHELLOPTS, ...) fails.
HOST_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "PWD", "OLDPWD", "SHELL", "SHLVL", "TERM", "TMPDIR",
    "HOSTNAME", "_", "LANG", "LC_*", "LD_*", "PYTHON*", "VIRTUAL_ENV", "CONDA_*", "UV_*",
    "GITHUB_*", "RUNNER_*", "ACTIONS_*", "XDG_*", "SSH_*", "*_VISIBLE_DEVICES",
    "BASH*", "SHELLOPTS", "UID", "EUID", "PPID",
)  # fmt: skip
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Backend(ABC):
    """Runs containers and follows jobs on one cluster's scheduler for one launch."""

    # Host state of this backend's scheduler beyond HOST_ENV (e.g. its own client
    # configuration), as fnmatch patterns that never reach a container.
    host_env: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cluster: Cluster, request: LaunchRequest, life: Lifecycle) -> None:
        """Serve ``request`` on ``cluster``; register cleanups on ``life``."""
        self.cluster = cluster
        self.request = request
        self.life = life

    @classmethod
    def container_env(cls, env: Mapping[str, str]) -> dict[str, str]:
        """The part of the launching environment ``env`` a container receives.

        Every non-empty variable a shell can export, except host state (HOST_ENV and
        :attr:`host_env`). Backends that pass the launching environment through as is
        give containers a superset of this; workloads rely only on this part.
        """
        patterns = (*HOST_ENV, *cls.host_env)
        return {
            name: value
            for name, value in env.items()
            if value
            and _SHELL_NAME.fullmatch(name)
            and not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        }

    @abstractmethod
    def prepare_image(self, image: str) -> Image:
        """Make the registry image ``image`` available to this backend's containers."""

    @abstractmethod
    def run_container(self, container: Container) -> Job:
        """Start ``container`` and return its job at once, while it still runs.

        Every backend returns without waiting for the command to finish; callers follow
        it with :meth:`stream_logs` and read its outcome with :meth:`state`. The job is
        cancelled when the launch ends.
        """

    @abstractmethod
    def stream_logs(self, job: Job) -> None:
        """Print the job's log as it grows until the job ends.

        Raises ``BackendError`` when the job ended without producing its log.
        """

    @abstractmethod
    def state(self, job: Job) -> JobStatus:
        """Return the job's status: pending or running while it is, else how it ended.

        After :meth:`stream_logs` returns this is the final status; backends wait
        briefly for their accounting to record the end.
        """

    @abstractmethod
    def fetch_outputs(self, job: Job, dest: Path) -> Path:
        """Return a local directory holding the job's outputs (``Container.outputs``).

        Jobs that write where this host reads return that directory as is; otherwise
        the outputs are copied into ``dest`` at their workspace-relative paths (missing
        ones skipped), which is returned. Idempotent, so a driver registers it as a
        cleanup right after :meth:`run_container`: cleanups run in reverse order, so
        the outputs come back before the job is cancelled, on every exit path.
        """

    @abstractmethod
    def cancel(self, job: Job) -> None:
        """Stop the job if the scheduler still holds it; an ended job is left alone."""

    @classmethod
    @abstractmethod
    def cleanup(cls, settings: SchedulerSettings | None, runner: str) -> None:
        """Remove what earlier launches on ``runner`` left behind and wait until it is gone.

        Each backend maps the runner name to the names or labels it gives its jobs.
        ``settings`` is the cluster's scheduler record, or None when the runner's
        cluster cannot be resolved; a backend that needs it to find its jobs does
        nothing then. A no-op on hosts without this scheduler.
        """
