"""The Slurm backend: salloc/srun with Pyxis containers, and jobs other tools submit.

A container runs as one ``srun`` step in a one-node ``salloc`` allocation that is cancelled
when the launch ends. The checkout and volumes are bind-mounted, so outputs need no
copying, and steps inherit the whole launching environment (``srun --export=ALL``). Jobs
are named after the runner, or :func:`srtctl_job_name` when srtctl submits them, which is
how :meth:`SlurmBackend.cleanup` finds leftovers. srt-slurm also uses
the Slurm-only operations below the generic ones.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, override

from infx.bench.proc import echo
from infx.clusters.slurm import HelperImage, SquashPolicy, slurm_settings
from infx.launch import proc
from infx.launch.backends.base import (
    Backend,
    BackendError,
    Container,
    Image,
    Job,
    JobState,
    JobStatus,
)
from infx.launch.backends.slurm import cli
from infx.launch.backends.slurm.squash import (
    ensure_image,
    registry_reference,
    reuse_or_registry,
    squash_path,
)
from infx.launch.request import RequestError

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.clusters.base import SchedulerSettings
    from infx.launch.lifecycle import Lifecycle
    from infx.launch.request import LaunchRequest

CANCEL_POLL_S = 10.0
# End states that lose the job to the cluster rather than fail its workload.
_LOST_STATES = frozenset({"BOOT_FAIL", "CANCELLED", "NODE_FAIL", "PREEMPTED"})


def srtctl_job_name(runner: str) -> str:
    """What the jobs srtctl submits for ``runner`` are named.

    Other repositories share the physical runner names and cancel jobs by them, so these
    long jobs carry a prefix that only this repository's cleanup cancels.
    """
    return f"inferencex-{runner}"


@dataclass(frozen=True)
class SlurmJob(Job):
    """A Slurm job: an allocation running one of our container steps, or an adopted job."""

    log: Path | None = None
    outputs: Path | None = None
    step: subprocess.Popen[bytes] | None = field(default=None, compare=False, repr=False)


def _slurm_job(job: Job) -> SlurmJob:
    """``job`` as the Slurm job it must be."""
    if not isinstance(job, SlurmJob):
        raise TypeError(f"not a Slurm job: {job!r}")
    return job


def _stop(step: subprocess.Popen[bytes]) -> None:
    """Stop a container step's ``srun`` client that is still running."""
    if step.poll() is None:
        step.terminate()
        try:
            step.wait(timeout=30)
        except subprocess.TimeoutExpired:
            step.kill()
            step.wait()


class SlurmBackend(Backend):
    """Slurm with Pyxis containers."""

    def __init__(self, cluster: Cluster, request: LaunchRequest, life: Lifecycle) -> None:
        """Serve ``request`` on the Slurm ``cluster``."""
        super().__init__(cluster, request, life)
        self.settings = slurm_settings(cluster)

    @override
    def prepare_image(self, image: str) -> Image:
        """Stage ``image`` for a container this backend runs.

        A valid shared squash costs no allocation, and a cold import runs in its own
        short step instead of holding the GPU allocation. Node-local squashes and
        all-nodes imports can only be staged inside the allocation, which
        :meth:`run_container` does.
        """
        squash = self.settings.squash
        if squash is None:
            return Image(image, registry_reference(image))
        policy = squash.policy()
        if policy.needs_job():
            return Image(image, str(squash_path(image, policy)))
        return Image(image, self._ensure(image, policy))

    @override
    def run_container(self, container: Container) -> SlurmJob:
        """Allocate one node, finish staging, check readiness, and start the ``srun`` step."""
        event = self.life.event
        mounts = [(container.workspace, container.workdir)]
        for mount in container.mounts:
            source = self._volume_path(mount.volume)
            if mount.create:
                source.mkdir(parents=True, exist_ok=True)
            mounts.append((source, mount.target))
        with event.stage("queue_wait"):
            allocation = self._allocate(container.gpus, container.time_limit_min)
        event.set(slurm_job_id=allocation.id)
        squash = self.settings.squash
        with event.stage("prepare"):
            if squash is not None and squash.policy().needs_job():
                ensure_image(container.image.name, squash.policy(), job=allocation)
            for path in container.required_paths:
                if cli.srun(allocation, ["test", "-r", str(_host_path(mounts, path))]) != 0:
                    raise BackendError(
                        f"readiness-blocked: {path} is unavailable on the allocated node"
                    )
        spec = cli.ContainerSpec(
            image=container.image.reference,
            mounts=[(str(source), str(target)) for source, target in mounts],
            workdir=str(container.workdir),
            env=dict(container.env),
        )
        rendered = spec.srun_args()
        extra = ["--mpi=none", *(arg for arg in self.settings.srun_args if arg not in rendered)]
        argv = cli.srun_argv(allocation, container.command, container=spec, extra=extra)
        echo(argv)
        step = subprocess.Popen(argv)
        self.life.callback(_stop, step)
        return SlurmJob(allocation.id, outputs=container.workspace, step=step)

    @override
    def stream_logs(self, job: Job) -> None:
        """Follow the job's log file; a container step already streams to our output."""
        job = _slurm_job(job)
        event = self.life.event
        if job.step is not None:
            with event.stage("run"):
                job.step.wait()
        elif job.log is None:
            raise BackendError(f"Slurm job {job.id} has no log to follow")
        else:
            with event.stage("queue_wait"):
                cli.wait_for_log(job, job.log)
            with event.stage("run"):
                cli.follow_log(job, job.log)

    @override
    def state(self, job: Job) -> JobStatus:
        """A container step's exit code; other jobs from squeue while listed, then accounting.

        An ended job's failure is reported and recorded as the launch's.
        """
        job = _slurm_job(job)
        if job.step is not None:
            rc = job.step.poll()
            if rc is None:
                return JobStatus(JobState.RUNNING, "srun step running")
            state = JobState.SUCCEEDED if rc == 0 else JobState.FAILED
            status = JobStatus(state, f"srun exit {rc}", rc if rc >= 0 else 128 - rc)
        elif (queued := cli.queue_state(job)) is not None:
            pending = queued in {"PENDING", "CONFIGURING"}
            return JobStatus(JobState.PENDING if pending else JobState.RUNNING, queued)
        else:
            status = cli.final_status(job)
        self._record_end(job, status)
        return status

    def _record_end(self, job: SlurmJob, status: JobStatus) -> None:
        event = self.life.event
        if status.nodes:
            event.set(nodes=status.nodes)
        if status.succeeded:
            return
        if job.step is not None:
            message = f"container step in Slurm job {job.id} exited {status.exit_code}"
            event.fail("JobFailed", message, stage="run")
        elif status.state in {JobState.FAILED, JobState.CANCELLED}:
            state, _, exit_code = status.raw.partition("|")
            ended = state.split(" ", 1)[0]
            kind = "JobFailed" if ended == "COMPLETED" else "Job" + ended.title().replace("_", "")
            message = f"Slurm job {job.id} ended with state={state} exit_code={exit_code}"
            event.fail(kind, message, stage="run", retriable=ended in _LOST_STATES)
        else:
            message = f"could not verify terminal Slurm status for job {job.id}"
            event.fail("JobStatusUnknown", message, stage="run", retriable=True)

    @override
    def fetch_outputs(self, job: Job, dest: Path) -> Path:
        """Jobs write to shared storage this host reads, so nothing is copied."""
        outputs = _slurm_job(job).outputs
        if outputs is None:
            raise BackendError(f"Slurm job {job.id} records no outputs")
        return outputs

    def cancel(self, job: Job, *, wait_s: float = 0.0) -> None:
        """``scancel`` the job if squeue lists it; with ``wait_s``, wait that long for it to go."""
        if not cli.is_active(job):
            return
        cli.cancel(job)
        deadline = time.monotonic() + wait_s
        while wait_s and cli.is_active(job):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                print(
                    f"WARNING: job {job.id} still present after {wait_s:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                return
            print(
                f"Waiting for job {job.id} to leave the queue ({remaining:.0f}s left)", flush=True
            )
            time.sleep(CANCEL_POLL_S)

    @override
    @classmethod
    def cleanup(cls, settings: SchedulerSettings | None, runner: str) -> None:
        """Cancel this user's jobs named after ``runner``, srtctl's included.

        Slurm finds them by user and name alone, so no settings are needed.
        """
        if shutil.which("squeue") is None:
            print(f"No Slurm scheduler here; nothing to clean up for {runner}")
            return
        names = (runner, srtctl_job_name(runner))
        print(f"[Slurm] Cleaning up jobs named {' and '.join(names)}")
        cli.cancel_named(names)

    def stage_image(
        self,
        image: str,
        *,
        framework: str | None = None,
        model_prefix: str | None = None,
        helper: HelperImage | None = None,
        single_node: bool = False,
    ) -> Image:
        """The image a job that srtctl submits starts ``image`` from.

        srtctl allocates that job itself, so nothing can be imported inside its
        allocation: images on node-local storage are handed to Pyxis by registry
        reference. Images are imported first only where ``squash.single-node-import``
        (single-node jobs) or ``squash.multi-node-import`` (multi-node jobs) says so;
        otherwise a valid squash is reused or Pyxis imports the image in the job. An
        ``unchecked`` image is handed over as its squash path, untouched.
        """
        squash = self.settings.squash
        if squash is None:
            return Image(image, registry_reference(image))
        policy = squash.helper_policy(helper) if helper else squash.policy(framework, model_prefix)
        if policy.import_mode == "unchecked":
            return Image(image, str(squash_path(image, policy)))
        if not (squash.single_node_import if single_node else squash.multi_node_import):
            return Image(image, reuse_or_registry(image, policy))
        if policy.visibility == "node-local":
            return Image(image, registry_reference(image))
        return Image(image, self._ensure(image, policy))

    def image_provenance(self, image: Image) -> str:
        """What exactly jobs started from ``image`` run, for audit records.

        A squash file is identified by its sha256 as ``sha256sum`` prints it
        (``<hex>  <path>``); a registry reference by itself, which pins the image only
        when it names a digest.
        """
        squash = Path(image.reference)
        if not squash.is_absolute():
            return image.reference
        try:
            with squash.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError as error:
            raise BackendError(f"squash of {image.name} is not readable: {error}") from None
        return f"{digest}  {squash}"

    def submit_batch(self, script: Path, *, gpus: int, time_min: int, log: Path) -> SlurmJob:
        """``sbatch`` ``script`` on one node from the checkout; cancelled when the launch ends."""
        settings = self.settings
        extra = ["--ntasks=1", *(["--exclusive"] if settings.exclusive else [])]
        job = cli.sbatch(
            script,
            self._resources(gpus, time_min),
            output=log,
            chdir=self.request.workspace,
            extra=[*extra, *settings.salloc_args],
        )
        self.life.callback(cli.cancel, job)
        return self._followed(SlurmJob(job.id, log=log))

    def attach(
        self, job_id: str, *, log: Path | None = None, outputs: Path | None = None
    ) -> SlurmJob:
        """Follow a job another tool submitted; the caller decides when to cancel it."""
        return self._followed(SlurmJob(job_id, log=log, outputs=outputs))

    def _followed(self, job: SlurmJob) -> SlurmJob:
        """``job`` as the launch's recorded Slurm job; its log explains later failures."""
        self.life.event.set(slurm_job_id=job.id)
        if job.log is not None:
            self.life.event.evidence(job.log)
        return job

    def stage_workspace(self, workspace: Path, staging: Path, *, exclude: Sequence[str]) -> Path:
        """A copy of ``workspace`` that compute nodes see: itself on Lustre, else rsynced into ``staging``."""
        try:
            fstype = proc.run(
                ["findmnt", "-n", "-o", "FSTYPE", "-T", workspace], capture=True
            ).stdout.strip()
        except OSError:
            fstype = ""
        if fstype == "lustre":
            print(
                f"Using the Lustre-backed checkout {workspace} as the jobs' workspace", flush=True
            )
            return workspace
        staging.mkdir(parents=True, exist_ok=True)
        rsync = [
            "rsync", "-a", "--delete", *(f"--exclude={pattern}" for pattern in exclude),
            f"{workspace}/", f"{staging}/",
        ]  # fmt: skip
        if rc := proc.run(rsync).returncode:
            raise BackendError(f"staging the workspace to {staging} failed (exit {rc})")
        print(f"Staged the node-local checkout to {staging} for compute nodes", flush=True)
        return staging

    def _volume_path(self, volume: str) -> Path:
        """Host path of one of the cluster's volumes."""
        path = self.settings.path(volume)
        if path is None:
            raise BackendError(f"cluster {self.cluster.id!r} has no volume {volume!r}")
        return path

    def _exclude(self) -> list[str]:
        """Nodes to keep out of our allocations; SALLOC_EXCLUDE takes a bad node out of rotation."""
        extra = self.request.env.get("SALLOC_EXCLUDE")
        return [*self.settings.exclude, *([extra] if extra else [])]

    def _resources(self, gpus: int, time_min: int) -> cli.Resources:
        """One node named after the runner, so the workflow cleanup finds it."""
        settings = self.settings
        return cli.Resources(
            partition=settings.partition,
            account=settings.account,
            time_min=time_min,
            job_name=self.request.runner_name,
            gres=settings.gres_for(gpus),
            exclude=self._exclude(),
        )

    def _allocate(self, gpus: int, time_min: int) -> Job:
        """``salloc`` one node and cancel it when the launch ends."""
        settings = self.settings
        extra = ["--exclusive"] if settings.exclusive else []
        extra += [f"--{name}={value}" for name, value in settings.cpu_directives().items()]
        job = cli.salloc(self._resources(gpus, time_min), extra=[*extra, *settings.salloc_args])
        self.life.callback(cli.cancel, job)
        return job

    def _ensure(self, image: str, policy: SquashPolicy) -> str:
        """Stage ``image`` before any allocation exists (a ``compute`` import runs its own step)."""
        alloc_args = self._import_step_args() if policy.import_mode == "compute" else ()
        return ensure_image(image, policy, job=None, alloc_args=alloc_args)

    def _import_step_args(self) -> list[str]:
        """Allocation flags of a standalone one-node import step (no GPUs)."""
        minutes = self.request.enroot_import_time_limit
        if minutes is None:
            raise RequestError.missing("ENROOT_IMPORT_TIME_LIMIT")
        settings = self.settings
        args = [f"--partition={settings.partition}"]
        if settings.account:
            args.append(f"--account={settings.account}")
        args += [f"--time={minutes}", f"--job-name={self.request.runner_name}"]
        if settings.exclude:
            args.append("--exclude=" + ",".join(settings.exclude))
        squash_args = settings.squash.import_step_args if settings.squash else ()
        return [*args, *squash_args]


def _host_path(mounts: Sequence[tuple[Path, PurePosixPath]], path: PurePosixPath) -> Path:
    """Where the node sees container path ``path``: through the longest matching mount."""
    for source, target in sorted(mounts, key=lambda mount: -len(mount[1].parts)):
        if path.is_relative_to(target):
            return source / path.relative_to(target)
    return Path(path)
