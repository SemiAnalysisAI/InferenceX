"""simple-slurm job steps with the existing explicit salloc allocation lifecycle."""

from __future__ import annotations

import io
from contextlib import contextmanager, suppress
import fcntl
import json
import signal
import sys
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zipfile


HOST_EXPORTS = "HOME,PATH,USER,XDG_CACHE_HOME,ENROOT_CACHE_PATH"

# A host utility must also work before a container can read the staged source. Send a small
# stdlib-only zipapp over stdin, as the old probes sent probe.py and image-import code. This
# preserves login/compute filesystem independence without generating a shell program.
REMOTE_BOOTSTRAP = """import pathlib, runpy, sys, tempfile
with tempfile.TemporaryDirectory(prefix='collectivex-remote-') as directory:
    archive = pathlib.Path(directory) / 'remote.pyz'
    archive.write_bytes(sys.stdin.buffer.read())
    sys.argv[0] = str(archive)
    runpy.run_path(str(archive), run_name='__main__')
"""


def remote_archive() -> bytes:
    """Bundle compute-host utilities; Slurm and GPU dependencies stay outside the archive."""
    content = io.BytesIO()
    root = Path(__file__).parent
    with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.glob("*.py")):
            archive.write(path, f"runtime/{path.name}")
        archive.writestr("__main__.py", "from runtime.node import main\nraise SystemExit(main())\n")
    return content.getvalue()


class SlurmAllocation:
    """One allocated shard; persistent job IDs let workflow cleanup outlive its launcher."""

    def __init__(self, root: Path, env: dict[str, str]):
        self.root, self.env = root, env
        self.job_id: str | None = None

    def allocate(self, arguments: list[str], attempt: int = 1) -> str:
        """Obtain the same no-shell allocation and record it before any job step."""
        name = self.env.get("RUNNER_NAME", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
            name = "collectivex"
        suffix = "" if attempt == 1 else f"-a{attempt}"
        path = log_path(self.root, f"scheduler-allocation{suffix}")
        log("scheduler-request=submit")
        try:
            run(["salloc", f"--job-name={name}", *arguments, "--no-shell"], path=path, env=self.env)
        except subprocess.CalledProcessError:
            log_tail(path)
            raise
        match = re.search(r"Granted job allocation ([1-9][0-9]*)", path.read_text())
        if match is None:
            raise RuntimeError("could not resolve allocated JOB_ID from salloc")
        self.job_id = match[1]
        (self.root / "jobid").write_text(self.job_id + "\n")
        os.chmod(self.root / "jobid", 0o600)
        return self.job_id

    def step(
        self,
        options: dict,
        argv: list[str],
        path: Path,
        *,
        timeout: int | None = None,
        stdin: bytes | None = None,
        kill_after: int | None = 30,
    ) -> int:
        """Run a step through simple-slurm, retaining literal Pyxis and boolean options.

        simple-slurm 0.3.6 drops empty-valued srun flags and models sbatch's option set.
        Pass bare flags and plugin options through its command tail. Quote all data before
        its shell-string API; a real executable fixture tests spaces and shell metacharacters.
        """
        if not self.job_id or not re.fullmatch(r"[1-9][0-9]*", self.job_id):
            raise ValueError("a job step requires an existing allocation")
        env = dict(self.env)
        # The library constructs a queue helper even for srun. Our queue queries always use
        # an explicit format, so an unrelated inherited SQUEUE_FORMAT must not affect steps.
        env.pop("SQUEUE_FORMAT", None)
        with command_context(env, path, stdin):
            from simple_slurm import Slurm

            slurm = Slurm()
            extra = []
            for key, value in {"jobid": self.job_id, **options}.items():
                if value is None or value is False:
                    continue
                if value is True:
                    extra.append(f"--{key.replace('_', '-')}")
                elif f"--{key}" in slurm.parser._option_string_actions:
                    slurm.add_arguments(**{key: shlex.quote(str(value))})
                else:
                    extra.append(f"--{key.replace('_', '-')}={value}")
            executable = ["srun"]
            if timeout is not None:
                kill = ["-k", str(kill_after)] if kill_after is not None else []
                executable = ["timeout", *kill, str(timeout), *executable]
            return slurm.srun(
                shlex.join([*extra, *map(str, argv)]), srun_cmd=shlex.join(executable)
            )

    def host(self, nodes: int, command: list[str], path: Path, **options) -> int:
        """Run a stdlib node utility once per node, without accessing its filesystem first."""
        return self.step(
            {
                "nodes": nodes,
                "ntasks": nodes,
                "ntasks_per_node": 1,
                "chdir": "/tmp",
                "export": HOST_EXPORTS,
                "input": "all",
                **options,
            },
            ["python3", "-c", REMOTE_BOOTSTRAP, *command],
            path,
            stdin=remote_archive(),
        )

    def nodes(self) -> str:
        """Return the allocated host list for quarantine/retry, validating every name."""
        expression = run(
            ["squeue", "-h", "-j", self.job_id, "-o", "%N"], env=self.env
        ).stdout.strip()
        if not re.fullmatch(r"[][A-Za-z0-9._,-]+", expression):
            raise RuntimeError("invalid allocation node list")
        names = run(["scontrol", "show", "hostnames", expression], env=self.env).stdout.splitlines()
        if not names or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) for name in names
        ):
            raise RuntimeError("cannot identify nodes from a rejected allocation")
        return ",".join(names)

    def release(self) -> None:
        """Cancel only this allocation and retain its recovery record until it has stopped."""
        record = self.root / "jobid"
        job_id = self.job_id or (record.read_text().strip() if record.is_file() else "")
        if not job_id:
            return
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise RuntimeError("invalid cleanup allocation")
        with suppress(OSError, subprocess.SubprocessError):
            run(["scancel", job_id], env=self.env, check=False)
        for _ in range(30):
            try:
                state = run(["squeue", "-h", "-j", job_id, "-o", "%A"], env=self.env, check=False)
                finished = state.returncode == 0 and not state.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                finished = False
            if finished:
                record.unlink(missing_ok=True)
                self.job_id = None
                return
            time.sleep(1)
        raise RuntimeError("scheduled allocation did not terminate during cleanup")


def log(message: str) -> None:
    """Keep the launcher log prefix and its stdout/stderr separation."""
    print(f"[collectivex] {message}", file=sys.stderr, flush=True)


def log_path(root: Path, name: str) -> Path:
    """Create a private log; callers use separate names for retry evidence."""
    path = root / "logs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text("")
    return path


def log_tail(path: Path) -> None:
    """Report the last 100 lines without losing the complete private log."""
    if path.is_file() and path.stat().st_size:
        log("--- command log tail ---")
        print("\n".join(path.read_text(errors="replace").splitlines()[-100:]), file=sys.stderr)
        log("--- end command log tail ---")


def run(
    argv: list[str],
    *,
    path: Path | None = None,
    env: dict | None = None,
    cwd: Path | None = None,
    input: str | None = None,
    timeout: float | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Execute literal arguments, preserving failures and optional private output."""
    with path.open("a") if path else open(os.devnull, "w") as output:
        return subprocess.run(
            [str(value) for value in argv],
            cwd=cwd,
            env=env,
            input=input,
            text=True,
            stdin=subprocess.DEVNULL if input is None else None,
            stdout=output if path else subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=check,
        )


def write_json(path: Path, value: object) -> None:
    """Atomically publish private state used by a later cleanup process."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    try:
        with temporary.open("w") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(path: Path, timeout: float | None = None, *, private: bool = True):
    """Keep cache installation in one exclusive critical section across processes."""
    if private and path.is_symlink():
        raise RuntimeError(f"cache lock is unsafe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        if private:
            os.chmod(path, 0o600)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if deadline else 0))
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"cache lock timed out: {path}") from None
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def interrupted():
    """Translate launcher signals into the existing 128+signal exit status."""

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def command_context(env: dict[str, str], output: Path, stdin: bytes | None = None):
    """Scope simple-slurm's inherited environment and FDs to one synchronous call.

    The library's public srun API inherits these instead of accepting subprocess kwargs.
    Host orchestration is single-threaded; rank processes never enter this context.
    """
    import tempfile

    previous_env, previous_cwd = dict(os.environ), Path.cwd()
    saved = [os.dup(fd) for fd in (0, 1, 2)]
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.environ.clear()
        os.environ.update(env)
        with output.open("ab") as stream, tempfile.TemporaryFile() as incoming:
            if stdin is not None:
                incoming.write(stdin)
                incoming.seek(0)
            os.dup2(incoming.fileno(), 0)
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in enumerate(saved):
            os.dup2(original, fd)
            os.close(original)
        os.environ.clear()
        os.environ.update(previous_env)
        os.chdir(previous_cwd)


def existing_exclusions(expression: str) -> str:
    requested = subprocess.check_output(
        ["scontrol", "show", "hostnames", expression], text=True
    ).splitlines()
    current = set(
        subprocess.check_output(["sinfo", "-N", "-h", "-o", "%N"], text=True).splitlines()
    )
    if not current:
        raise ValueError("Slurm returned no nodes; refusing to discard exclusions")
    return ",".join(node for node in requested if node in current)
