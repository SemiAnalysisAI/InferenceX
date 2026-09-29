"""A launch's cleanups, signal handling and exit code."""

from __future__ import annotations

import signal
import sys
import traceback
from collections.abc import Callable
from contextlib import ExitStack
from types import FrameType, TracebackType
from typing import Any, Self

# SIGHUP: start_runners.sh kills the runner's tmux session (RUNNER_SETUP.md, Gotchas).
_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class _Interrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class Lifecycle:
    """Runs a launch's cleanups LIFO on every exit path and owns its exit code.

    The first nonzero recorded code wins; a failed cleanup turns 0 into 1 and never raises.
    A signal during the body runs the cleanups, then raises ``SystemExit(128 + signum)``.
    Signals arriving during the cleanups are held off until they finish, then dropped.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._rc = 0
        self._cleanup_failed = False
        self._previous: dict[int, Any] = {}
        self._exiting = False
        self._signum = 0

    @property
    def returncode(self) -> int:
        if self._rc:
            return self._rc
        return 1 if self._cleanup_failed else 0

    def record(self, rc: int) -> None:
        if rc and not self._rc:
            self._rc = rc

    def callback(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> None:
        """Register a cleanup; one that raises or returns a nonzero int fails the launch."""

        def run() -> None:
            try:
                result = fn(*args, **kwargs)
            except Exception:  # noqa: BLE001 - every cleanup must get its turn
                traceback.print_exc()
                self._cleanup_failed = True
                return
            if isinstance(result, int) and not isinstance(result, bool) and result:
                print(
                    f"WARNING: cleanup {getattr(fn, '__name__', fn)!s} returned {result}",
                    file=sys.stderr,
                )
                self._cleanup_failed = True

        self._stack.callback(run)

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        # Inside __exit__ (at its first instruction, or delivered by pthread_sigmask for a
        # signal that arrived just before) no body is left to unwind, and raising there
        # would skip every cleanup.
        if self._exiting or (frame is not None and frame.f_code is _EXIT_CODE):
            self._signum = self._signum or signum
            return
        for sig in _SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        raise _Interrupted(signum)

    def __enter__(self) -> Self:
        for sig in _SIGNALS:
            self._previous[sig] = signal.signal(sig, self._on_signal)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self._exiting = True
        # Cleanups (scancel, artifact copies) must not be aborted midway.
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
        if isinstance(exc, _Interrupted):
            self._signum = exc.signum
        elif isinstance(exc, SystemExit):
            code = exc.code
            self.record(code if isinstance(code, int) else (0 if code is None else 1))
        elif exc is not None:
            self.record(1)
        if self._signum:
            print(f"Received signal {self._signum}; running cleanups", file=sys.stderr)
        try:
            self._stack.close()
        finally:
            # SIG_IGN drops the signals held off during the cleanups before the mask lifts.
            for sig in _SIGNALS:
                signal.signal(sig, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            for sig, previous in self._previous.items():
                signal.signal(sig, previous)
            self._previous.clear()
        if self._signum:
            raise SystemExit(128 + self._signum) from None
        return False


_EXIT_CODE = Lifecycle.__exit__.__code__
