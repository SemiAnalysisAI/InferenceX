"""A scheduler backend that runs each container as a local process.

It is what a new backend is: this module, a settings model registered in
``infx.clusters.SCHEDULERS``, one ``infx.launch.backends.BACKENDS`` entry, and a cluster
record with ``scheduler: fake``. Unlike Slurm/Pyxis, volumes are claims it provisions
under its own root, the checkout is copied in without the excluded entries, only the
declared outputs are copied back, and containers get no host state.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from pydantic import Field

from infx.clusters.base import SchedulerSettings, Volume
from infx.launch.backends.base import (
    Backend,
    BackendError,
    Container,
    Image,
    Job,
    JobState,
    JobStatus,
)


class FakeVolume(Volume):
    """A volume claim the backend provisions under ``<root>/volumes/<claim>``."""

    claim: str


class FakeSettings(SchedulerSettings):
    """``fake:``: where the backend keeps volumes and, per namespace, containers."""

    root: Path
    namespace: str
    volumes: dict[str, FakeVolume] = Field(default_factory=dict)


def containers_dir(settings: FakeSettings) -> Path:
    return settings.root / settings.namespace / "containers"


@dataclass(frozen=True)
class FakeJob(Job):
    process: subprocess.Popen = field(compare=False)
    workdir: Path
    outputs: tuple[PurePosixPath, ...]


class FakeBackend(Backend):
    host_env = ("FAKE_HOST_*",)
    cleaned: list[tuple[FakeSettings | None, str]] = []

    def prepare_image(self, image: str) -> Image:
        return Image(image, f"fake://{image}")

    def run_container(self, container: Container) -> Job:
        settings = self.cluster.scheduler_settings
        parent = containers_dir(settings)
        parent.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(dir=parent))
        shutil.copytree(
            container.workspace, workdir, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*container.exclude),
        )  # fmt: skip
        views = [(workdir, container.workdir)]
        for mount in container.mounts:
            volume = settings.volumes.get(mount.volume)
            if volume is None:
                raise BackendError(f"cluster {self.cluster.id!r} has no volume {mount.volume!r}")
            source = settings.root / "volumes" / volume.claim
            if mount.create:
                source.mkdir(parents=True, exist_ok=True)
            views.append((source, mount.target))

        def host(value: str) -> str:
            for source, target in sorted(views, key=lambda view: -len(view[1].parts)):
                if PurePosixPath(value).is_relative_to(target):
                    return str(source / PurePosixPath(value).relative_to(target))
            return value

        for path in container.required_paths:
            if not os.access(host(str(path)), os.R_OK):
                raise BackendError(f"readiness-blocked: {path} is unavailable")
        env = {
            name: host(value)
            for name, value in {**self.container_env(self.request.env), **container.env}.items()
        }
        process = subprocess.Popen(list(container.command), cwd=workdir, env={**env, "PATH": os.defpath})
        job = FakeJob(str(process.pid), process, workdir, tuple(container.outputs))
        self.life.callback(self.cancel, job)
        return job

    def stream_logs(self, job: Job) -> None:
        job.process.wait()

    def state(self, job: Job) -> JobStatus:
        rc = job.process.poll()
        if rc is None:
            return JobStatus(JobState.RUNNING, "process running")
        return JobStatus(JobState.SUCCEEDED if rc == 0 else JobState.FAILED, f"exit {rc}", rc)

    def fetch_outputs(self, job: Job, dest: Path) -> Path:
        for output in job.outputs:
            source = job.workdir / output
            if source.is_dir():
                shutil.copytree(source, dest / output, dirs_exist_ok=True)
            elif source.exists():
                (dest / output).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest / output)
        return dest

    def cancel(self, job: Job) -> None:
        if job.process.poll() is None:
            job.process.kill()
            job.process.wait()

    @classmethod
    def cleanup(cls, settings: FakeSettings | None, runner: str) -> None:
        cls.cleaned.append((settings, runner))
