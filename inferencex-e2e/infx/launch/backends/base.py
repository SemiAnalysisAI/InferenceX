"""The backend interface: how drivers run containers and follow jobs on any scheduler.

A backend serves one launch on one cluster's scheduler. Drivers say what runs; the backend
owns images, delivering the checkout and volumes, returning outputs, following logs and job
state, and cancelling every job it starts when the launch's Lifecycle ends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.base import SchedulerSettings
    from infx.launch.lifecycle import Lifecycle
    from infx.launch.request import LaunchRequest


class BackendError(RuntimeError):
    """A scheduler, image, or storage operation failed; reported without a traceback."""


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class JobStatus:
    """``raw`` is the scheduler's own reading, for diagnostics.

    ``exit_code`` is the shell-style status (128 + N after signal N) when the scheduler
    reports one. Only ``state`` decides success: Slurm reports a cancelled job as ``0:0``.
    """

    state: JobState
    raw: str
    exit_code: int | None = None

    @property
    def succeeded(self) -> bool:
        return self.state is JobState.SUCCEEDED


@dataclass(frozen=True)
class Job:
    """A job a backend follows; backends return their own subclasses, drivers pass them back."""

    id: str


@dataclass(frozen=True)
class Image:
    """An image a backend's jobs start from ``reference``."""

    name: str
    reference: str


@dataclass(frozen=True)
class Mount:
    """Cluster volume ``volume`` seen at ``target``; with ``create`` its directory is made first."""

    volume: str
    target: PurePosixPath
    create: bool = False


@dataclass(frozen=True)
class Container:
    """One containerized command on one node, with the checkout ``workspace`` at ``workdir``.

    A copy-in backend leaves out entries matching an ``exclude`` pattern (fnmatch, any depth).
    ``outputs`` are the workspace-relative paths ``fetch_outputs`` returns. Each of
    ``required_paths`` must be readable where the container runs, or it never starts.
    Each container receives the launching environment's non-empty, exportable variables
    except host state (paths, interpreters, runner credentials), with ``env`` over them;
    workloads rely on no more, though a backend may pass more. ``env`` may carry
    credentials, which a backend must expose no more widely than the launching environment
    does.
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


class Backend(ABC):
    """Runs containers and follows jobs on one cluster's scheduler for one launch."""

    def __init__(self, cluster: Cluster, request: LaunchRequest, life: Lifecycle) -> None:
        self.cluster = cluster
        self.request = request
        self.life = life

    @abstractmethod
    def prepare_image(self, image: str) -> Image:
        """Make the registry image ``image`` available to this backend's containers."""

    @abstractmethod
    def run_container(self, container: Container) -> Job:
        """Start ``container`` and return without waiting; the job is cancelled when the launch ends."""

    @abstractmethod
    def stream_logs(self, job: Job) -> None:
        """Print the job's log until the job ends; raise ``BackendError`` if it ended without one."""

    @abstractmethod
    def state(self, job: Job) -> JobStatus:
        """The job's status; final once ``stream_logs`` has returned."""

    @abstractmethod
    def fetch_outputs(self, job: Job, dest: Path) -> Path:
        """A local directory holding the job's outputs at their workspace-relative paths.

        Returns the job's own directory when this host can read it, else copies the outputs
        into ``dest``. Idempotent, so drivers register it as a cleanup right after
        ``run_container``: cleanups run LIFO, so outputs come back before the job is cancelled.
        """

    @classmethod
    @abstractmethod
    def cleanup(cls, settings: SchedulerSettings | None, runner: str) -> None:
        """Remove what earlier launches on ``runner`` left behind and wait until it is gone.

        ``settings`` is None when the runner's cluster cannot be resolved. A no-op on hosts
        without this scheduler.
        """
