"""Echoed wrappers around the Slurm CLI."""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from infx.bench.proc import echo
from infx.launch import proc
from infx.launch.backends.base import BackendError, Job, JobState, JobStatus

_GRANTED = re.compile(r"Granted job allocation ([1-9][0-9]*)")


class SlurmError(BackendError):
    """A Slurm command failed or a job did not reach the state the caller needs."""


@dataclass(frozen=True)
class ContainerSpec:
    """Pyxis options of one ``srun`` step: no home mount, remapped root, no image entrypoint."""

    image: str
    mounts: list[tuple[str, str]] = field(default_factory=list)
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def srun_args(self) -> list[str]:
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
    """``salloc --no-shell`` and return the granted job; raise ``SlurmError`` without a grant."""
    argv = ["salloc", *resources.args(), *extra, "--no-shell"]
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
    """One ``srun`` step exporting our environment; without ``job`` it allocates from ``extra``."""
    env = container.env if container else {}
    for name, value in env.items():
        if "," in value:
            raise ValueError(f"srun --export cannot carry {name}: value contains ','")
    args = ["srun"]
    if job is not None:
        args.append(f"--jobid={job.id}")
    if container is not None:
        args += container.srun_args()
    exported = [f"{name}={value}" for name, value in env.items()]
    return [*args, "--export=" + ",".join(["ALL", *exported]), *extra, *argv]


def srun(job: Job | None, argv: Sequence[str], *, extra: Sequence[str] = ()) -> int:
    """Run one host step to completion, streaming its output; return its exit code."""
    return proc.run(srun_argv(job, argv, extra=extra)).returncode


def sbatch(
    script: Path, resources: Resources, *, output: Path, chdir: Path, extra: Sequence[str] = ()
) -> Job:
    """``sbatch --parsable --export=ALL`` with the batch log at ``output``; return the job."""
    argv = [
        "sbatch", "--parsable", *resources.args(), "--export=ALL", f"--output={output}",
        f"--chdir={chdir}", *extra, os.fspath(script),
    ]  # fmt: skip
    result = proc.run(argv, capture=True)
    sys.stderr.write(result.stderr)
    job_id = result.stdout.strip().split(";", 1)[0]
    if result.returncode != 0 or not job_id.isdigit():
        raise SlurmError(f"sbatch failed to submit {script} (exit {result.returncode})")
    return Job(job_id)


def queue_state(job: Job, *, echo_command: bool = True) -> str | None:
    """The job's squeue state, or None once squeue no longer lists it (or cannot run)."""
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
    return queue_state(job) is not None


def cancel(job: Job) -> None:
    """``scancel`` the job, ignoring failures: it may already be gone."""
    with contextlib.suppress(FileNotFoundError):
        proc.run(["scancel", job.id])


def cancel_named(names: Sequence[str]) -> None:
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
        time.sleep(5)


def _query(argv: list[str], env: Mapping[str, str] | None = None) -> str:
    try:
        result = proc.run(argv, env=env, capture=True)
    except FileNotFoundError:
        return ""
    return result.stdout if result.returncode == 0 else ""


def default_account() -> str | None:
    """This user's Slurm default account, or None when accounting can't say."""
    user = os.environ.get("USER") or getpass.getuser()
    account = _query(["sacctmgr", "-nP", "show", "user", user, "format=DefaultAccount"]).strip()
    return account.splitlines()[0] if account else None


def _time(text: str) -> datetime | None:
    """A ``SLURM_TIME_FORMAT=%s`` timestamp; unset ones read ``Unknown``, ``None`` or 0."""
    text = text.strip()
    if not (text.isascii() and text.isdecimal()) or int(text) == 0:
        return None
    try:
        return datetime.fromtimestamp(int(text), UTC)
    except (OverflowError, OSError, ValueError):
        return None


def observe(job: Job) -> JobStatus:
    """One reading of the allocation: accounting (``sacct -X``), else the controller's record.

    Never raises; a job neither knows reads as ``UNKNOWN``.
    """
    env = {**os.environ, "SLURM_TIME_FORMAT": "%s"}
    argv = ["sacct", "-X", "-n", "-P", "-j", job.id]
    accounting = _query([*argv, "--format=State,ExitCode,NodeList,Submit,Start,End"], env)
    first = accounting.splitlines()[0] if accounting.strip() else ""
    state, exit_code, nodes, submitted, started, ended = [*first.split("|"), *[""] * 6][:6]
    if not state.strip():
        fields = dict(
            item.split("=", 1)
            for item in _query(["scontrol", "show", "job", "-o", job.id], env).split()
            if "=" in item
        )
        if fields.get("JobId") != job.id:
            fields = {}
        state, exit_code, nodes, submitted, started, ended = (
            fields.get(name, "")
            for name in ("JobState", "ExitCode", "NodeList", "SubmitTime", "StartTime", "EndTime")
        )
    nodes = nodes.strip()
    return replace(
        job_status(state.strip(), exit_code.strip()),
        nodes=None if nodes in {"", "None assigned", "(null)"} else nodes,
        submitted_at=_time(submitted),
        started_at=_time(started),
        ended_at=_time(ended),
    )


def job_status(state: str, exit_code: str) -> JobStatus:
    """Only ``COMPLETED`` with ``0:0`` succeeds; any other state after the job ran failed."""
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
    """The allocation's terminal status (``sacct -X``: never its steps); never raises.

    Accounting lags, so unsettled readings are retried; after ``attempts`` the last one is
    returned.
    """
    status = observe(job)
    for _ in range(attempts - 1):
        if status.state not in _UNSETTLED:
            break
        time.sleep(delay_s)
        status = observe(job)
    return status


def wait_for_log(job: Job, path: Path, *, wait_s: float = 5.0) -> None:
    """Wait until ``job`` creates ``path``.

    Raises ``SlurmError``, after printing ``scontrol show job``, if the job ends first.
    """
    while not path.exists():
        if queue_state(job, echo_command=False) is None:
            print(f"ERROR: job {job.id} failed before creating {path}", file=sys.stderr)
            with contextlib.suppress(FileNotFoundError):
                proc.run(["scontrol", "show", "job", job.id])
            raise SlurmError(f"job {job.id} ended before creating {path}")
        time.sleep(wait_s)


def follow_log(job: Job, path: Path) -> None:
    """Print ``path`` as it grows until ``job`` leaves the queue."""
    sentinel = subprocess.Popen(["sleep", "2147483647"])
    print(f"Tailing {path}", file=sys.stderr, flush=True)
    tail_argv = ["tail", "-F", "-s", "2", "-n+1", str(path), f"--pid={sentinel.pid}"]
    echo(tail_argv)
    tail = subprocess.Popen(tail_argv, stderr=subprocess.DEVNULL)
    # HACK (manual e2e only): also stream srtctl's engine logs as they appear, and aiperf's.
    # PD worker lines carry the role and worker index, e.g. [P0] / [D0].
    pid = sentinel.pid
    # Not streamed (the uploaded log keeps them): aiter's line per new GEMM shape, torch's
    # hipBLASLt fallback warning + its source line, the 10 s KV Transfer metrics.
    noise = (r"\[aiter\] shape is M:|bgemm_internal_cublaslt error|"
             r"^  out = F\.linear\(inp, weights, bias\)$|KV Transfer metrics:")
    engines = r'''while kill -0 "$2"; do
  for f in "$1"/*_agg_w*.out "$1"/*_prefill_w*.out "$1"/*_decode_w*.out; do
    [[ -e $f && $seen != *" $f "* ]] || continue
    seen+=" $f "; mode=${f%_w*}; mode=${mode##*_}; tag=${mode:0:1}${f##*_w}; tag=${tag%.out}
    [[ $mode == agg ]] && tag= || tag="[${tag^^}] "
    tail -qF -s 2 -n+1 --pid="$2" "$f" | grep --line-buffered -vE "$3" | sed -u "s/^/$tag/" &
  done
  sleep 2
done'''
    subprocess.Popen(["bash", "-c", engines, "engines", str(path.parent), str(pid), noise],
                     stderr=subprocess.DEVNULL)
    subprocess.Popen(["tail", "-F", "-s", "2", "-n+1", f"--pid={pid}", f"{path.parent}/benchmark.out"],
                     stderr=subprocess.DEVNULL)
    try:
        while queue_state(job, echo_command=False) is not None:
            time.sleep(10)
    finally:
        sentinel.kill()
        sentinel.wait()
        try:
            tail.wait(timeout=30)
        except subprocess.TimeoutExpired:
            tail.kill()
            tail.wait()
