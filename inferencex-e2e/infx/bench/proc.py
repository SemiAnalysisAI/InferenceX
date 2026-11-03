"""Child processes: shell-style exit statuses, signal handling, and this checkout's PYTHONPATH."""

from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

from infx.bench.env import BenchError

if TYPE_CHECKING:
    from typing_extensions import Self

REPO_ROOT = Path(__file__).resolve().parents[2]
_SECRET_NAME = re.compile(r"TOKEN(?!S|IZER)|SECRET|API_?KEY|PASSWORD", re.IGNORECASE)


def status(returncode: int) -> int:
    """Shell-style status: death by signal N is 128 + N."""
    return 128 - returncode if returncode < 0 else returncode


def echo(argv: Sequence[str | os.PathLike[str]], env: Mapping[str, str] | None = None) -> None:
    """Print ``+ <argv>`` to stderr, masking the values of credential variables.

    Names like ``TOKENS`` and ``TOKENIZER`` hold counts and paths, so they are not masked.
    """
    text = shlex.join(map(os.fspath, argv))
    secrets = {
        value
        for source in (os.environ, env or {})
        for name, value in source.items()
        if _SECRET_NAME.search(name) and len(value) >= 4
    }
    for value in sorted(secrets, key=len, reverse=True):
        text = text.replace(value, "***")
    sys.stdout.flush()
    print(f"+ {text}", file=sys.stderr, flush=True)


def call(
    argv: Sequence[str],
    env: Mapping[str, str] | None = None,
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> int:
    """Run ``argv`` echoed like ``set -x``; 127 if it cannot start, 124 past ``timeout``."""
    echo(argv, env)
    try:
        return status(
            subprocess.run(argv, env=env, cwd=cwd, timeout=timeout, check=False).returncode
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR: {argv[0]} exceeded its {timeout:g}s deadline", file=sys.stderr)
        return 124
    except OSError as error:
        print(f"ERROR: cannot run {argv[0]}: {error}", file=sys.stderr)
        return 127


def tee(
    argv: Sequence[str],
    log: Path,
    env: Mapping[str, str] | None = None,
    *,
    session: DeferSignals | None = None,
    exit_timeout: float = 0.0,
) -> int:
    """``argv 2>&1 | tee log``; return argv's status, 127 if it cannot start.

    With ``session``, argv leads a new session that receives the signals ``session``
    defers, and every process left in it must exit within ``exit_timeout`` seconds of
    argv, or the survivors are killed and ``BenchError`` is raised.
    """
    echo(argv, env)
    with log.open("wb") as sink:
        try:
            child = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=session is not None,
            )
        except OSError as error:
            print(f"ERROR: cannot run {argv[0]}: {error}", file=sys.stderr, flush=True)
            return 127
        if session is not None:
            session.attach(child.pid)
        try:
            with child:
                output = cast("io.BufferedReader", child.stdout)
                while chunk := output.read1(1 << 16):
                    for stream in (sys.stdout.buffer, sink):
                        stream.write(chunk)
                        stream.flush()
        finally:
            if session is not None:
                session.attach(None)
    if session is not None:
        _await_empty_group(child.pid, exit_timeout)
    return status(child.returncode)


def _await_empty_group(pgid: int, timeout: float) -> None:
    """Return once process group ``pgid`` is empty; past ``timeout``, kill it and raise."""
    deadline = time.monotonic() + timeout
    while _group_alive(pgid):
        if time.monotonic() >= deadline:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            raise BenchError(
                f"client process group {pgid} still had live processes {timeout:g}s after "
                "the client exited; they were killed"
            )
        time.sleep(0.05)


def _group_alive(pgid: int) -> bool:
    """Whether any process is left in group ``pgid``."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pythonpath(environ: Mapping[str, str] = os.environ) -> str:
    """``PYTHONPATH`` with this checkout first; containers never install ``infx``."""
    root = str(REPO_ROOT)
    rest = [
        path for path in environ.get("PYTHONPATH", "").split(os.pathsep) if path not in {"", root}
    ]
    return os.pathsep.join([root, *rest])


SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class _Signals:
    """Handle ``SIGNALS`` for the ``with`` block and remember the first one received."""

    def __init__(self) -> None:
        """Start with no signal received and no saved handlers."""
        self.received: int | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> Self:
        """Install ``_handle`` for every signal in ``SIGNALS`` and return self."""
        for signum in SIGNALS:
            self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_: object) -> None:
        """Restore the handlers that were in place before the block."""
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        """Record ``signum`` unless an earlier signal was already recorded."""
        self.received = self.received or signum


class DeferSignals(_Signals):
    """Hold signals until the block ends; the foreground child in our group gets them itself.

    A child in its own session (``tee(session=...)``) does not, so it is sent each one.
    """

    def __init__(self) -> None:
        """Start with no signal received and no attached session."""
        super().__init__()
        self._session: int | None = None

    def attach(self, pgid: int | None) -> None:
        """Forward later signals to process group ``pgid``; ``None`` stops forwarding."""
        self._session = pgid
        if pgid is not None and self.received:
            self._forward(self.received)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        """Record ``signum`` and forward it to the attached session, if any."""
        super()._handle(signum, frame)
        self._forward(signum)

    def _forward(self, signum: int) -> None:
        """Send ``signum`` to the attached session, which may already have exited."""
        if self._session is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._session, signum)


class RelaySignals(_Signals):
    """Run commands one at a time, relaying signals; after one, nothing new starts."""

    def __init__(self) -> None:
        """Start with no running child."""
        super().__init__()
        self._child: subprocess.Popen[bytes] | None = None

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        """Record ``signum`` and forward it to the running child, if any."""
        super()._handle(signum, frame)
        if self._child is not None:
            self._child.send_signal(signum)

    def run(self, command: Sequence[str]) -> int:
        """Run ``command`` to completion; return its status."""
        if self.received:
            return 128 + self.received
        echo(command)
        try:
            self._child = child = subprocess.Popen(command)
        except OSError as error:
            print(f"ERROR: cannot run {command[0]}: {error}", file=sys.stderr, flush=True)
            return 127
        if self.received:
            child.send_signal(self.received)
        try:
            rc = status(child.wait())
        finally:
            self._child = None
        return 128 + self.received if rc == 0 and self.received else rc
