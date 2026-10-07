"""A launch's cleanups, signal handling, exit code and job record."""

from __future__ import annotations

import signal
import sys
import traceback
from collections.abc import Callable
from contextlib import ExitStack
from types import FrameType, TracebackType
from typing import Any, Self

from infx.launch.event import JobEventBuilder

_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class _Interrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class Lifecycle:
    """Runs a launch's cleanups LIFO on every exit path and owns its exit code and ``event``.

    The first nonzero recorded code wins; a failed cleanup turns 0 into 1 and never raises.
    A signal during the body runs the cleanups, then raises ``SystemExit(128 + signum)``.
    Signals arriving during the cleanups are held off until they finish, then dropped.
    """

    def __init__(self, event: JobEventBuilder | None = None) -> None:
        self.event = event if event is not None else JobEventBuilder()
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
            name = getattr(fn, "__name__", fn)
            try:
                result = fn(*args, **kwargs)
            except Exception as error:  # noqa: BLE001 - every cleanup must get its turn
                traceback.print_exc()
                self.event.error(error, stage="cleanup", report=False)
                self._cleanup_failed = True
                return
            if isinstance(result, int) and not isinstance(result, bool) and result:
                message = f"cleanup {name!s} returned {result}"
                print(f"WARNING: {message}", file=sys.stderr)
                self.event.fail("CleanupFailed", message, stage="cleanup", report=False)
                self._cleanup_failed = True

        self._stack.callback(run)

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
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
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, _SIGNALS)
        if isinstance(exc, _Interrupted):
            self._signum = exc.signum
        elif isinstance(exc, SystemExit):
            code = exc.code
            code = code if isinstance(code, int) else (0 if code is None else 1)
            self.record(code)
            self.event.exited(code)
        elif exc is not None:
            self.event.error(exc, report=False)
            self.record(1)
        if self._signum:
            self.event.interrupted(self._signum)
            print(f"Received signal {self._signum}; running cleanups", file=sys.stderr)
        try:
            self._stack.close()
        finally:
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
