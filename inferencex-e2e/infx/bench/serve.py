"""``serve``: start one inference server, run clients against it once it is ready, then stop it.

Exits with the first failing client's status, 1 when the server exits or is not ready within its
budget, or 128 + N after signal N. With ``INFX_GROUP_RECORD`` naming a file, the server's process
group is appended to it before the server's program runs, so whoever named it can stop a server
that outlives this command.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import FrameType

from infx.bench import proc, server

STOP_GRACE_S = 60.0
GATE_POLL_S = 0.1


class _Interrupted(BaseException):
    """A signal arrived while the server was starting."""

    def __init__(self, signum: int) -> None:
        """Carry the received signal number ``signum``."""
        super().__init__(signum)
        self.signum = signum


@contextlib.contextmanager
def _interruptible() -> Iterator[None]:
    """Turn ``proc.SIGNALS`` into ``_Interrupted`` for the block, restoring the handlers after."""

    def interrupt(signum: int, _frame: FrameType | None) -> None:
        """Abandon the wait for the server."""
        raise _Interrupted(signum)

    previous = {signum: signal.signal(signum, interrupt) for signum in proc.SIGNALS}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def stop(child: subprocess.Popen[bytes]) -> None:
    """SIGTERM ``child``'s process group, then SIGKILL whatever is left after ``STOP_GRACE_S``.

    Forked engine workers keep the group alive after the leader exits. The leader stays a zombie
    until it is polled after a client or waited for here, so no other process can have its pid,
    which is the group's id. macOS answers EPERM rather than ESRCH for a group whose members have
    all exited.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        print(
            f"WARNING: {child.args[0]} still running {STOP_GRACE_S:g}s after SIGTERM",
            file=sys.stderr,
        )
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()


def _start_group(
    argv: Sequence[str], recorder: Callable[[], None] | None, name: str
) -> subprocess.Popen[bytes] | None:
    """Start ``argv`` in a process group of its own, or warn about ``name`` and return None."""
    try:
        return subprocess.Popen(
            argv,
            start_new_session=True,
            preexec_fn=recorder,  # noqa: PLW1509 - this command runs no other thread
        )
    except OSError as error:
        print(f"WARNING: could not start the {name}: {error}", file=sys.stderr, flush=True)
        return None


def _run_gate(argv: Sequence[str], timeout_s: float, relay: proc.RelaySignals) -> None:
    """Run the report-only ``argv`` in a recorded group of its own, whatever its status.

    Its whole group stops at ``timeout_s``, or at a signal, so that nothing it started keeps
    sending the server traffic during the clients.
    """
    proc.echo(argv)
    with proc.group_recorder(os.environ.get(proc.GROUP_RECORD_ENV)) as recorder:
        gate = _start_group(argv, recorder, "gate")
    if gate is None:
        return
    deadline = time.monotonic() + timeout_s
    try:
        while gate.poll() is None and not relay.received and time.monotonic() < deadline:
            time.sleep(GATE_POLL_S)
    finally:
        stop(gate)
    print(f"The gate exited {proc.status(gate.returncode)} (report-only)", flush=True)


def serve(
    server_argv: Sequence[str],
    ready_url: str,
    log: Path,
    clients: Sequence[Sequence[str]],
    ready_timeout_s: float | None = None,
    sampler_argv: Sequence[str] | None = None,
    gate_argv: Sequence[str] | None = None,
    gate_timeout_s: float = 0.0,
) -> int:
    """Run ``server_argv`` with its output in ``log``, then ``clients`` once it is ready.

    It is ready when ``ready_url`` answers within ``ready_timeout_s``. The run fails if the
    server exits before the clients finish, because a client can still pass after a dead server
    failed its requests. ``sampler_argv`` runs beside the server until the server stops, and its
    status does not affect the run. ``gate_argv`` runs once the server is ready, before the
    clients, for at most ``gate_timeout_s``; neither its status nor its stopping affects the run.

    A signal before the server is ready abandons the wait. A signal while a client runs is relayed
    to the client. A signal while the server is stopping is held until the stop finishes.
    """
    proc.echo(server_argv)
    with proc.DeferSignals() as deferred:
        with (
            log.open("ab") as sink,
            proc.group_recorder(os.environ.get(proc.GROUP_RECORD_ENV)) as recorder,
        ):
            child = subprocess.Popen(
                server_argv,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=recorder,  # noqa: PLW1509 - this command runs no other thread
            )
            sampler = (
                None if sampler_argv is None else _start_group(sampler_argv, recorder, "sampler")
            )
        try:
            with proc.RelaySignals() as relay:
                with _interruptible():
                    if deferred.received:
                        raise _Interrupted(deferred.received)
                    server.wait_ready(ready_url, pid=child.pid, log=log, timeout_s=ready_timeout_s)
                print(f"Server ready at {ready_url}", flush=True)
                if gate_argv is not None:
                    _run_gate(gate_argv, gate_timeout_s, relay)
                for client in clients:
                    rc = relay.run(client)
                    if (code := child.poll()) is not None:
                        print(
                            f"ERROR: the server exited with status {proc.status(code)} while "
                            f"the clients ran; see {log}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return rc or 1
                    if rc:
                        return rc
            return 0
        except _Interrupted as interrupted:
            return 128 + interrupted.signum
        except server.NotReadyError as error:
            print(f"ERROR: {error}; see {log}", file=sys.stderr, flush=True)
            return 1
        finally:
            if sampler is not None:
                stop(sampler)
            stop(child)


def main(argv: list[str]) -> int:
    """Run the ``serve`` command."""
    parser = argparse.ArgumentParser(prog="python3 -m infx.bench serve", description=__doc__)
    parser.add_argument(
        "--ready-url", required=True, help="URL that answers below 400 once the server is ready"
    )
    parser.add_argument("--log", type=Path, required=True, help="file receiving the server output")
    parser.add_argument("--ready-timeout", type=float, help="seconds the server has to be ready")
    parser.add_argument(
        "--client",
        type=json.loads,
        action="append",
        required=True,
        help="a client argv, a JSON list; repeated clients run in order until one fails",
    )
    parser.add_argument(
        "--sampler",
        type=json.loads,
        help="an argv, a JSON list, to run in its own process group beside the server",
    )
    parser.add_argument(
        "--gate",
        type=json.loads,
        help="a report-only argv, a JSON list, to run in its own process group before the clients",
    )
    parser.add_argument("--gate-timeout", type=float, help="seconds the gate may run")
    split = argv.index("--")
    args = parser.parse_args(argv[:split])
    if args.gate is not None and args.gate_timeout is None:
        parser.error("--gate needs --gate-timeout")
    return serve(
        argv[split + 1 :], args.ready_url, args.log, args.client, args.ready_timeout, args.sampler,
        args.gate, args.gate_timeout or 0.0,
    )  # fmt: skip
