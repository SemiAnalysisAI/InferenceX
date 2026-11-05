"""The native backend: a container's command runs as a process group on the runner host.

Any argument or environment value that is an absolute container path is replaced with the host
path behind it. Paths a command builds at run time are not translated, so commands must reach
files through those values.

An image is a tar archive pinned by digest (``<url>@sha256:<hex>``). Its top directory goes first
on the job's ``PATH``, followed by a venv of the clients' packages, which no archive includes.

Each runner's running job is recorded under the cluster's root so that ``cleanup`` can stop it.
"""

from __future__ import annotations

import calendar
import fcntl
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, TextIO, override

from infx.bench.proc import (
    GROUP_RECORD_ENV,
    REPO_ROOT,
    echo,
    group_recorder,
    recorded_groups,
    status,
)
from infx.bench.serve import STOP_GRACE_S
from infx.clusters.native import NativeSettings, native_settings, split_image
from infx.launch.backends.base import (
    Backend,
    BackendError,
    Container,
    Image,
    Job,
    JobState,
    JobStatus,
    container_env,
    mounted_host_path,
)
from infx.launch.context import LaunchError

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.base import SchedulerSettings
    from infx.launch.lifecycle import Lifecycle
    from infx.launch.request import LaunchRequest

HOST_LOCK = ".host.lock"
FETCH_TIMEOUT_S = 60
TERM_GRACE_S = STOP_GRACE_S + 30.0
KILL_WAIT_S = 10.0
POLL_S = 0.1
START_SLACK_S = 5.0
SYSTEM_PATH = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")


@dataclass(frozen=True)
class NativeJob(Job):
    """A process group on the runner host, led by ``process`` and ended by ``deadline``."""

    process: subprocess.Popen[bytes] = field(compare=False, repr=False)
    outputs: Path
    deadline: float


def _native_job(job: Job) -> NativeJob:
    """``job`` as the native job it must be."""
    if not isinstance(job, NativeJob):
        raise TypeError(f"not a native job: {job!r}")
    return job


def _record_path(settings: NativeSettings, runner: str) -> Path:
    """The file recording ``runner``'s running job, one JSON line per process group.

    The job's own group comes first, then each group the job starts in a session of its own.
    """
    return settings.jobs / f"{runner}.jsonl"


def _host_mounts(
    container: Container, settings: NativeSettings
) -> list[tuple[Path, PurePosixPath]]:
    """Where this host sees ``container``'s workdir and mounts, creating ``create`` mounts.

    Raises ``BackendError`` for an unknown volume or an unreadable ``required_paths`` entry.
    """
    mounts = [(container.workspace, container.workdir)]
    for mount in container.mounts:
        source = settings.path(mount.volume)
        if source is None:
            raise BackendError(f"this cluster has no volume {mount.volume!r}")
        if mount.create:
            source.mkdir(parents=True, exist_ok=True)
        mounts.append((source, mount.target))
    for path in container.required_paths:
        if not os.access(mounted_host_path(mounts, path), os.R_OK):
            raise BackendError(f"readiness-blocked: {path} is unavailable on this host")
    return mounts


def _host_value(mounts: Sequence[tuple[Path, PurePosixPath]], value: str) -> str:
    """The host path of ``value`` if it is an absolute container path, else ``value`` unchanged."""
    return str(mounted_host_path(mounts, PurePosixPath(value))) if value.startswith("/") else value


def _start_time(pid: int) -> float | None:
    """When process ``pid`` started, to the second, or None when there is no such process."""
    ps = subprocess.run(
        ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
        env={"LC_ALL": "C", "TZ": "UTC0"},
    )
    try:
        return calendar.timegm(time.strptime(ps.stdout.strip(), "%a %b %d %H:%M:%S %Y"))
    except ValueError:
        return None


def _signal_group(pgid: int, sig: int) -> bool:
    """Send ``sig`` to process group ``pgid``; return whether any process is left in it.

    macOS answers EPERM for a group whose members have all exited but are not yet reaped.
    """
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(pgid: int, leader: subprocess.Popen[bytes] | None = None) -> None:
    """SIGTERM process group ``pgid``, SIGKILL what outlives ``TERM_GRACE_S``, and wait.

    ``TERM_GRACE_S`` outlasts the ``STOP_GRACE_S`` a job's supervisor gives its server to stop.
    ``leader`` is reaped while waiting, since a zombie leader would keep the group alive.
    """
    for sig, wait_s in ((signal.SIGTERM, TERM_GRACE_S), (signal.SIGKILL, KILL_WAIT_S)):
        if not _signal_group(pgid, sig):
            return
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if leader is not None:
                leader.poll()
            if not _signal_group(pgid, 0):
                return
            time.sleep(POLL_S)
    raise BackendError(f"process group {pgid} survived SIGKILL")


def _is_same_group(pgid: int, started_at: float) -> bool:
    """Whether ``pgid`` still identifies the group recorded as started at ``started_at``.

    It does not if a process that started more than ``START_SLACK_S`` from that time now has the
    id; the slack allows for ps reporting start times to the second. It does if no process has
    the id, because a group can outlive its leader and its id is not reused until it is gone.
    """
    leader_start = _start_time(pgid)
    return leader_start is None or abs(leader_start - started_at) <= START_SLACK_S


def _stop_recorded(record: Path) -> None:
    """Stop every group in ``record`` that still has a process in it.

    A group is left alone if its id now belongs to a different process.
    """
    for pgid, started_at in recorded_groups(record):
        if not _is_same_group(pgid, started_at):
            print(
                f"[native] Process {pgid} is no longer the one recorded; leaving it alone",
                flush=True,
            )
        elif _signal_group(pgid, 0):
            print(f"[native] Stopping process group {pgid}, which is still running", flush=True)
            _stop_group(pgid)


def _unpack(url: str, digest: str, target: Path) -> None:
    """Fetch the archive at ``url``, check its digest and unpack its top directory as ``target``."""
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".unpack-") as scratch:
        archive = Path(scratch) / "archive"
        sha256 = hashlib.sha256()
        print(f"[native] Fetching image {url}", flush=True)
        try:
            with (
                urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as response,  # noqa: S310
                archive.open("wb") as out,
            ):
                while chunk := response.read(1 << 20):
                    sha256.update(chunk)
                    out.write(chunk)
        except OSError as error:
            raise BackendError(f"fetching image {url} failed: {error}") from None
        if sha256.hexdigest() != digest:
            raise LaunchError(f"image {url} has sha256 {sha256.hexdigest()}, not {digest}")
        unpacked = Path(scratch) / "unpacked"
        unpacked.mkdir()
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(unpacked, filter="data")
        except (OSError, tarfile.TarError) as error:
            raise LaunchError(f"cannot unpack image {url}: {error}") from None
        entries = list(unpacked.iterdir())
        if len(entries) != 1:
            raise LaunchError(f"image {url} must unpack to one directory")
        entries[0].rename(target)


def _remove_tree(root: Path) -> None:
    """Remove the tree at ``root``, if there is one, read-only directories included."""
    if root.exists():
        for directory, _, _ in os.walk(root):
            Path(directory).chmod(0o700)
        shutil.rmtree(root)


def _remove_job(settings: NativeSettings, runner: str) -> None:
    """Stop what ``runner``'s job left running, then delete its record and directory."""
    record = _record_path(settings, runner)
    _stop_recorded(record)
    record.unlink(missing_ok=True)
    _remove_tree(settings.jobs / runner)


class NativeBackend(Backend):
    """Host processes on the runner's own machine."""

    def __init__(self, cluster: Cluster, request: LaunchRequest, life: Lifecycle) -> None:
        """Serve ``request`` on ``cluster``; refuse a host with another OS or without uv."""
        super().__init__(cluster, request, life)
        self.settings = native_settings(cluster)
        self.job_dir = self.settings.jobs / request.runner_name
        self.record = _record_path(self.settings, request.runner_name)
        if sys.platform != cluster.os:
            raise LaunchError(f"cluster {cluster.id!r} runs {cluster.os}, not {sys.platform}")
        uv = shutil.which("uv", path=request.env.get("PATH", ""))
        if uv is None:
            raise LaunchError(
                "native jobs build their clients' venv with uv, which is not on this host's PATH"
            )
        self.uv = Path(uv).absolute()
        self._timed_out = False
        self._host_slot: TextIO | None = None

    @override
    def prepare_image(self, image: str) -> Image:
        """Unpack ``image``'s archive unless that was already done, and return the ``Image``.

        The digest fixes the archive's bytes, so an earlier unpack can be reused. The ``Image``
        reference is the unpacked directory, which goes first on a job's ``PATH``.
        """
        try:
            url, digest = split_image(image)
        except ValueError as error:
            raise LaunchError(str(error)) from None
        unpacked = self.settings.images / digest
        if not unpacked.is_dir():
            self.settings.images.mkdir(parents=True, exist_ok=True)
            _unpack(url, digest, unpacked)
        return Image(image, str(unpacked))

    @override
    def run_container(self, container: Container) -> NativeJob:
        """Build the job's clients' venv, then start the command as a new process group.

        Both are recorded so they can be stopped later. The command runs in the checkout. Like a
        container, it gets its own home and temporary directories, which are removed with it.
        """
        mounts = _host_mounts(container, self.settings)
        self._host_slot = self._take_host_slot()
        try:
            _remove_tree(self.job_dir)
            self.record.unlink(missing_ok=True)
            for name in ("home", "tmp"):
                (self.job_dir / name).mkdir(parents=True)
            venv = self._build_venv()
            env = {
                "PATH": os.pathsep.join((container.image.reference, str(venv), *SYSTEM_PATH)),
                "HOME": str(self.job_dir / "home"),
                "TMPDIR": str(self.job_dir / "tmp"),
                **{
                    name: _host_value(mounts, value)
                    for name, value in {**container_env(self.request.env), **container.env}.items()
                },
                GROUP_RECORD_ENV: str(self.record),
            }
            argv = [_host_value(mounts, arg) for arg in container.command]
            echo(argv, env)
            try:
                process = self._start_group(argv, env, container.workspace)
            except OSError as error:
                raise BackendError(f"cannot run {argv[0]}: {error}") from None
        except BaseException:
            self._drop_job()
            raise
        job = NativeJob(
            str(process.pid),
            process=process,
            outputs=container.workspace,
            deadline=time.monotonic() + container.time_limit_min * 60,
        )
        self.life.event.set(nodes=socket.gethostname())
        self.life.callback(self.cancel, job)
        return job

    @override
    def stream_logs(self, job: Job) -> None:
        """Wait for the job, which writes to our output, and stop it at its time limit."""
        job = _native_job(job)
        with self.life.event.stage("run"):
            try:
                job.process.wait(timeout=max(0.0, job.deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self._timed_out = True
                print(
                    f"WARNING: native job {job.id} reached its time limit; stopping it",
                    file=sys.stderr,
                    flush=True,
                )
                _stop_group(job.process.pid, job.process)

    @override
    def state(self, job: Job) -> JobStatus:
        """The job's exit status; an ended job's failure is recorded as the launch's.

        A job stopped at its time limit counts as cancelled whatever status it exited with.
        """
        job = _native_job(job)
        rc = job.process.wait()
        code = status(rc)
        if self._timed_out:
            self.life.event.fail(
                "JobTimeout", f"native job {job.id} hit its time limit", stage="run"
            )
            return JobStatus(JobState.CANCELLED, f"exit {rc}", exit_code=code)
        if rc == 0:
            return JobStatus(JobState.SUCCEEDED, f"exit {rc}", exit_code=code)
        self.life.event.fail("JobFailed", f"native job {job.id} exited {code}", stage="run")
        return JobStatus(JobState.FAILED, f"exit {rc}", exit_code=code)

    @override
    def fetch_outputs(self, job: Job, dest: Path) -> Path:
        """Jobs write into the checkout this host reads, so nothing is copied."""
        return _native_job(job).outputs

    def cancel(self, job: Job) -> None:
        """Stop the job and any process groups it started, then delete what it left on disk."""
        job = _native_job(job)
        _stop_group(job.process.pid, job.process)
        job.process.wait()
        self._drop_job()

    def _drop_job(self) -> None:
        """Stop the job's recorded groups that still run, and free the host's job slot.

        The job's record and directory are deleted.
        """
        _remove_job(self.settings, self.request.runner_name)
        if self._host_slot is not None:
            self._host_slot.close()

    def _start_group(
        self, argv: Sequence[str], env: Mapping[str, str], cwd: Path
    ) -> subprocess.Popen[bytes]:
        """Start ``argv`` as a new process group, added to the job's record before it runs."""
        with group_recorder(self.record) as recorder:
            return subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=recorder,  # noqa: PLW1509 - the launcher runs no other thread
            )

    def _build_venv(self) -> Path:
        """Build the clients' venv in the job's directory and return its ``bin``.

        It holds the ``client`` group as this checkout's ``uv.lock`` pins it. ``bin`` also links to
        uv, which the AgentX and eval clients use to install their own packages.

        The build keeps the launcher's ``HOME`` and ``UV_*`` settings so that it uses the host's uv
        cache.
        """
        venv = self.job_dir / "venv"
        env = {
            "PATH": os.pathsep.join(SYSTEM_PATH),
            "TMPDIR": str(self.job_dir / "tmp"),
            **{
                name: value
                for name, value in self.request.env.items()
                if name == "HOME" or name.startswith("UV_")
            },
            "UV_PROJECT_ENVIRONMENT": str(venv),
        }
        print(f"[native] Building the clients' venv {venv} from uv.lock", flush=True)
        argv = [
            str(self.uv), "sync", "--quiet", "--frozen", "--only-group", "client",
            "--project", str(REPO_ROOT), "--python", sys.executable,
        ]  # fmt: skip
        uv_process = self._start_group(argv, env, self.job_dir)
        try:
            rc = uv_process.wait()
        except BaseException:
            _stop_group(uv_process.pid, uv_process)
            raise
        if rc:
            raise BackendError(f"building the clients' venv failed (exit {rc})")
        (venv / "bin" / "uv").symlink_to(self.uv)
        return venv / "bin"

    def _take_host_slot(self) -> TextIO:
        """Take this host's one native job slot for the launch, or refuse while it is taken.

        Native jobs share the server port, so a client could measure a server left running by
        another job. The slot is a lock held until the job is gone. A killed launcher releases its
        lock but can leave its job running, so the job records are checked as well.
        """
        self.settings.jobs.mkdir(parents=True, exist_ok=True)
        slot = (self.settings.jobs / HOST_LOCK).open("a")
        try:
            fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BackendError("another native job runs on this host") from None
        for record in self.settings.jobs.glob("*.jsonl"):
            for pgid, started_at in recorded_groups(record):
                if _is_same_group(pgid, started_at) and _signal_group(pgid, 0):
                    raise BackendError(
                        f"{record.stem}'s native job still runs on this host; "
                        f"python -m infx.launch cleanup on {record.stem} stops it"
                    )
        return slot

    @override
    @classmethod
    def cleanup(cls, settings: SchedulerSettings | None, runner: str) -> None:
        """Stop the groups of the job an earlier launch on ``runner`` left running.

        The job directory it left is then removed.
        """
        if not isinstance(settings, NativeSettings):
            print(f"No native job records here; nothing to clean up for {runner}")
            return
        print(f"[native] Stopping the process groups {runner}'s job left running", flush=True)
        _remove_job(settings, runner)
