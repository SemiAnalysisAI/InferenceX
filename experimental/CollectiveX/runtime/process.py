"""Process execution, private logs, locks, and recoverable JSON state."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


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


def run(argv: list[str], *, path: Path | None = None, env: dict | None = None,
        cwd: Path | None = None, input: str | None = None, timeout: float | None = None,
        check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Execute literal arguments, preserving failures and optional private output."""
    with path.open("a") if path else open(os.devnull, "w") as output:
        return subprocess.run(
            [str(value) for value in argv], cwd=cwd, env=env, input=input, text=True,
            stdin=subprocess.DEVNULL if input is None else None,
            stdout=output if path else subprocess.PIPE if capture else None, stderr=subprocess.STDOUT,
            timeout=timeout, check=check,
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

    previous = {sig: signal.signal(sig, stop) for sig in
                (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
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
