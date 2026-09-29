"""A scheduler backend that runs each container as a local process.

A new backend is this much: its module, a settings model in ``infx.clusters.SCHEDULERS``,
an ``infx.launch.backends.BACKENDS`` entry, and a cluster record naming the scheduler.
Unlike Slurm/Pyxis it reaches volumes by claim, copies the checkout in without the excluded
entries, copies back only the declared outputs, and passes containers no host state.
"""

import fnmatch
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
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

# Launching-host state as fnmatch patterns. In a container these would name host paths or
# a host interpreter, or leak runner credentials; exporting bash's readonly ones fails.
HOST_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "PWD", "OLDPWD", "SHELL", "SHLVL", "TERM", "TMPDIR",
    "HOSTNAME", "_", "LANG", "LC_*", "LD_*", "PYTHON*", "VIRTUAL_ENV", "CONDA_*", "UV_*",
    "GITHUB_*", "RUNNER_*", "ACTIONS_*", "XDG_*", "SSH_*", "*_VISIBLE_DEVICES",
    "BASH*", "SHELLOPTS", "UID", "EUID", "PPID",
)  # fmt: skip
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def container_env(env: Mapping[str, str]) -> dict[str, str]:
    """The part of the launching environment every container receives (see ``Container``)."""
    return {
        name: value
        for name, value in env.items()
        if value
        and _SHELL_NAME.fullmatch(name)
        and not any(fnmatch.fnmatchcase(name, pattern) for pattern in HOST_ENV)
    }


class FakeVolume(Volume):
    claim: str


class FakeSettings(SchedulerSettings):
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
            for name, value in {**container_env(self.request.env), **container.env}.items()
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
