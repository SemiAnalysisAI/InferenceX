"""Thin, echoed wrappers around the Slurm CLI (salloc, srun, sbatch, squeue, sacct, scontrol, scancel)."""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from infx.launch import proc
from infx.launch.backends.base import BackendError, Job, JobState, JobStatus

_GRANTED = re.compile(r"Granted job allocation ([1-9][0-9]*)")


class SlurmError(BackendError):
    """A Slurm command failed or a job did not reach the state the caller needs."""


@dataclass(frozen=True)
class ContainerSpec:
    """Pyxis container options for one ``srun`` step.

    Steps never mount the home directory, run as remapped root, and skip the image
    entrypoint. ``env`` is appended to ``--export`` so the step sees it in the container.
    """

    image: str
    mounts: list[tuple[str, str]] = field(default_factory=list)
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def srun_args(self) -> list[str]:
        """Render the ``--container-*`` flags for this container."""
        args = [f"--container-image={self.image}"]
        if self.mounts:
            args.append(
                "--container-mounts=" + ",".join(f"{host}:{dest}" for host, dest in self.mounts)
            )
        args += ["--no-container-mount-home", "--container-remap-root"]
        if self.workdir:
            args.append(f"--container-workdir={self.workdir}")
        args.append("--no-container-entrypoint")
        return args


def _export_arg(export: str, env: dict[str, str]) -> str:
    """Build ``--export``; Slurm splits it on commas, so values must not contain one."""
    for name, value in env.items():
        if "," in value:
            raise ValueError(f"srun --export cannot carry {name}: value contains ','")
    return "--export=" + ",".join([export, *(f"{name}={value}" for name, value in env.items())])


@dataclass(frozen=True)
class Resources:
    """One-node resource flags shared by salloc and sbatch."""

    partition: str
    account: str | None
    time_min: int
    job_name: str
    gres: str | None = None
    exclude: Sequence[str] = ()

    def args(self) -> list[str]:
        """Render the flags."""
        args = [f"--partition={self.partition}"]
        if self.account:
            args.append(f"--account={self.account}")
        args += ["--nodes=1", f"--time={self.time_min}", f"--job-name={self.job_name}"]
        if self.gres:
            args.append(f"--gres={self.gres}")
        if self.exclude:
            args.append("--exclude=" + ",".join(self.exclude))
        return args


def salloc(resources: Resources, *, extra: Sequence[str] = ()) -> Job:
    """Allocate nodes with ``salloc --no-shell`` and return the granted job.

    Output is echoed to stderr; raises ``SlurmError`` if salloc fails or never
    prints ``Granted job allocation <id>``.
    """
    argv = ["salloc", *resources.args(), *extra, "--no-shell"]
    # The grant line is parsed, so pin the message locale.
    result = proc.run(argv, env={**os.environ, "LC_ALL": "C"}, capture=True)
    output = result.stdout + result.stderr
    sys.stderr.write(output)
    sys.stderr.flush()
    match = _GRANTED.search(output)
    if result.returncode != 0 or match is None:
        raise SlurmError(f"salloc failed to allocate a job (exit {result.returncode})")
    return Job(match.group(1))


def srun_argv(
    job: Job | None,
    argv: Sequence[str],
    *,
    container: ContainerSpec | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """The ``srun`` command for one step (inside ``job`` when given), exporting our environment.

    Without a job, srun makes its own allocation from the flags in ``extra``.
    """
    args = ["srun"]
    if job is not None:
        args.append(f"--jobid={job.id}")
    if container is not None:
        args += container.srun_args()
    args.append(_export_arg("ALL", container.env if container else {}))
    return [*args, *extra, *argv]


def srun(job: Job | None, argv: Sequence[str], *, extra: Sequence[str] = ()) -> int:
    """Run one host (non-container) step to completion, streaming its output; return its exit code."""
    return proc.run(srun_argv(job, argv, extra=extra)).returncode


def sbatch(
    script: Path,
    resources: Resources,
    *,
    output: Path,
    chdir: Path | None = None,
    extra: Sequence[str] = (),
) -> Job:
    """Submit ``script`` with ``sbatch --parsable --export=ALL`` and return the job.

    ``output`` receives the batch log (follow it with :func:`stream_log`).
    """
    argv = ["sbatch", "--parsable", *resources.args(), "--export=ALL", f"--output={output}"]
    if chdir is not None:
        argv.append(f"--chdir={chdir}")
    argv += [*extra, os.fspath(script)]
    result = proc.run(argv, capture=True)
    sys.stderr.write(result.stderr)
    # --parsable prints "<id>" or "<id>;<cluster>".
    job_id = result.stdout.strip().split(";", 1)[0]
    if result.returncode != 0 or not job_id.isdigit():
        raise SlurmError(f"sbatch failed to submit {script} (exit {result.returncode})")
    return Job(job_id)


def queue_state(job: Job, *, echo_command: bool = True) -> str | None:
    """The job's squeue state (PENDING, RUNNING, ...), or None once squeue no longer lists it.

    A failing or missing squeue counts as not listed.
    """
    argv = ["squeue", "-j", job.id, "--noheader", "--format=%i|%T"]
    try:
        result = proc.run(argv, capture=True, echo_command=echo_command)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        job_id, _, state = line.strip().partition("|")
        if job_id == job.id:
            return state
    return None


def is_active(job: Job) -> bool:
    """Whether ``job`` is still queued or running."""
    return queue_state(job) is not None


def cancel(job: Job) -> None:
    """``scancel`` the job, ignoring failures (it may already be gone)."""
    with contextlib.suppress(FileNotFoundError):
        proc.run(["scancel", job.id])


def cancel_named(names: Sequence[str], *, poll_s: float = 5.0) -> None:
    """``scancel`` this user's jobs named any of ``names`` and wait until squeue drops them."""
    user = os.environ.get("USER") or getpass.getuser()
    for name in names:
        proc.run(["scancel", f"--user={user}", f"--name={name}"])
    query = ["squeue", f"--user={user}", "--name=" + ",".join(names), "--noheader", "--format=%i"]
    while True:
        result = subprocess.run(query, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return
        print(
            f"Waiting for jobs {' '.join(result.stdout.split())} to leave the queue",
            file=sys.stderr,
        )
        time.sleep(poll_s)


def _query(argv: list[str]) -> str:
    """Captured stdout of a status query, or ``""`` when the command is unavailable or fails."""
    try:
        result = proc.run(argv, capture=True)
    except FileNotFoundError:
        return ""
    return result.stdout if result.returncode == 0 else ""


def _observe(job: Job) -> tuple[str, str]:
    """One (state, exit code) reading: allocation accounting, else the controller's record."""
    accounting = _query(["sacct", "-X", "-n", "-P", "-j", job.id, "--format=State,ExitCode"])
    first = accounting.splitlines()[0] if accounting.strip() else ""
    state, _, exit_code = first.partition("|")
    if state.strip():
        return state.strip(), exit_code.strip()
    # Some pools do not expose slurmdbd; the controller retains recent terminal
    # allocations. Require its exact JobId, state, and exit code.
    fields = dict(
        item.split("=", 1)
        for item in _query(["scontrol", "show", "job", "-o", job.id]).split()
        if "=" in item
    )
    if fields.get("JobId") != job.id:
        return "", ""
    return fields.get("JobState", ""), fields.get("ExitCode", "")


def job_status(state: str, exit_code: str) -> JobStatus:
    """Map a Slurm allocation reading onto the neutral states.

    Only ``COMPLETED`` with exit code ``0:0`` succeeds; any other state Slurm reports
    after the job left PENDING/CONFIGURING/RUNNING/COMPLETING is a failed end.
    """
    code, _, _ = exit_code.partition(":")
    raw = f"{state}|{exit_code}"
    exit_status = int(code) if code.isdigit() else None
    if state == "COMPLETED" and exit_code == "0:0":
        return JobStatus(JobState.SUCCEEDED, raw, 0)
    if not state:
        return JobStatus(JobState.UNKNOWN, raw)
    if state in {"PENDING", "CONFIGURING"}:
        return JobStatus(JobState.PENDING, raw)
    if state in {"RUNNING", "COMPLETING"}:
        return JobStatus(JobState.RUNNING, raw)
    if state.startswith("CANCELLED"):
        return JobStatus(JobState.CANCELLED, raw, exit_status)
    return JobStatus(JobState.FAILED, raw, exit_status)


_UNSETTLED = frozenset({JobState.UNKNOWN, JobState.PENDING, JobState.RUNNING})


def final_status(job: Job, *, attempts: int = 10, delay_s: float = 1.0) -> JobStatus:
    """Wait for the allocation's terminal state and return it (never raises).

    Only the top-level allocation is inspected (``sacct -X``), never service steps.
    Unsettled readings (none, pending, running) are retried because accounting lags;
    after ``attempts`` the last reading is returned. A non-success result is also
    reported on stderr.
    """
    state, exit_code = "", ""
    for attempt in range(attempts):
        state, exit_code = _observe(job)
        if job_status(state, exit_code).state not in _UNSETTLED:
            break
        if attempt + 1 < attempts:
            time.sleep(delay_s)
    status = job_status(state, exit_code)
    if status.state in _UNSETTLED:
        print(f"ERROR: could not verify terminal Slurm status for job {job.id}", file=sys.stderr)
    elif not status.succeeded:
        print(
            f"ERROR: Slurm job {job.id} ended with state={state} exit_code={exit_code}",
            file=sys.stderr,
        )
    return status


def stream_log(job: Job, path: Path, *, wait_s: float = 5.0, poll_s: float = 10.0) -> None:
    """Wait for ``path`` to appear, then follow it until ``job`` leaves the queue.

    The log is on network storage, where inotify does not work, so ``tail -F``
    polls it. Raises ``SlurmError`` (after printing ``scontrol show job``) if the
    job ends before creating the log.
    """
    while not path.exists():
        if queue_state(job, echo_command=False) is None:
            print(f"ERROR: job {job.id} failed before creating {path}", file=sys.stderr)
            with contextlib.suppress(FileNotFoundError):
                proc.run(["scontrol", "show", "job", job.id])
            raise SlurmError(f"job {job.id} ended before creating {path}")
        time.sleep(wait_s)

    # tail exits (after a final read) once the sentinel dies; it dies when the job does.
    sentinel = subprocess.Popen(["sleep", "2147483647"])
    print(f"Tailing {path}", file=sys.stderr, flush=True)
    tail_argv = ["tail", "-F", "-s", "2", "-n+1", str(path), f"--pid={sentinel.pid}"]
    proc.echo(tail_argv)
    tail = subprocess.Popen(tail_argv, stderr=subprocess.DEVNULL)
    try:
        while queue_state(job, echo_command=False) is not None:
            time.sleep(poll_s)
    finally:
        sentinel.kill()
        sentinel.wait()
        try:
            tail.wait(timeout=30)
        except subprocess.TimeoutExpired:
            tail.kill()
            tail.wait()
