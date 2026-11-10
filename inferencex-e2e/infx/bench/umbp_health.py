from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import selectors
import signal
import subprocess
import sys
from pathlib import Path

from infx.bench.env import BenchError


def process_identity(pid: int) -> tuple[str, int]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return fields[0], int(fields[19])


def record_pool(pid: int, destination: Path) -> None:
    state, started = process_identity(pid)
    if state in {"Z", "X"}:
        raise BenchError(f"UMBP pool {pid} exited before readiness")
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": pid, "start_ticks": started}))
    temporary.replace(destination)


class FailureIntervals:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.streak = {"store": 0, "load": 0}

    def observe(self, line: bytes) -> str | None:
        match = re.search(rb"KV Transfer metrics:.*?UMBPStoreConnector=(\{[^\r\n]*?\})", line)
        if match is None:
            return None
        try:
            counters = ast.literal_eval(match.group(1).decode())
        except (SyntaxError, ValueError) as error:
            raise BenchError("Cannot parse UMBP health counters") from error
        for operation in self.streak:
            failed = counters.get(f"{operation}_failed")
            if failed is None:
                continue
            if not isinstance(failed, int) or failed < 0:
                raise BenchError(f"Invalid UMBP {operation} failure count: {failed}")
            self.streak[operation] = self.streak[operation] + 1 if failed else 0
            if self.streak[operation] >= self.limit:
                return f"UMBP {operation} failed in {self.streak[operation]} consecutive reports"
        return None


def supervise(
    command: list[str],
    pool_state: Path,
    poll_seconds: float,
    failure_intervals: int,
    stop_timeout: float,
) -> int:
    if not command or poll_seconds <= 0 or failure_intervals <= 0 or stop_timeout <= 0:
        raise BenchError("UMBP health supervision requires a command and positive limits")
    try:
        pool = json.loads(pool_state.read_text())
        pool_pid, pool_started = int(pool["pid"]), int(pool["start_ticks"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise BenchError(f"Invalid UMBP pool state: {pool_state}") from error

    def pool_error() -> str | None:
        try:
            state, started = process_identity(pool_pid)
        except (OSError, ValueError, IndexError):
            return f"UMBP pool {pool_pid} is no longer observable"
        if state in {"Z", "X"} or started != pool_started:
            return f"UMBP pool {pool_pid} exited or its PID was reused"
        return None

    failure = pool_error()
    if failure:
        raise BenchError(failure)
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    requested_signal = 0

    def forward(signum: int, _frame: object) -> None:
        nonlocal requested_signal
        requested_signal = signum
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signum)

    handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    reports = FailureIntervals(failure_intervals)
    pending = b""
    try:
        if child.stdout is None:
            raise BenchError("Worker output pipe unavailable")
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            while not requested_signal:
                failure = pool_error()
                if failure:
                    break
                ready = selector.select(poll_seconds)
                if ready:
                    chunk = os.read(child.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        failure = reports.observe(line)
                        if failure:
                            break
                    if failure:
                        break
                    if len(pending) > 1048576:
                        raise BenchError("Worker emitted an unterminated line over 1 MiB")
                elif child.poll() is not None:
                    break
        if failure:
            print(json.dumps({"event": "umbp_health_failed", "reason": failure}), flush=True)
            return 70
        if requested_signal:
            return 128 + requested_signal
        try:
            code = child.wait(timeout=stop_timeout)
        except subprocess.TimeoutExpired as error:
            raise BenchError("Worker closed output without exiting") from error
        return code if code >= 0 else 128 - code
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(timeout=stop_timeout)
        # Descendants can outlive their group leader after a serving-process failure.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        if child.stdout is not None:
            child.stdout.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-state", type=Path, required=True)
    parser.add_argument("--record-pid", type=int)
    parser.add_argument("--poll-seconds", type=float)
    parser.add_argument("--failure-intervals", type=int)
    parser.add_argument("--stop-timeout", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.record_pid is not None:
        record_pool(args.record_pid, args.pool_state)
        return 0
    if any(
        value is None for value in (args.poll_seconds, args.failure_intervals, args.stop_timeout)
    ):
        raise BenchError("Specify poll-seconds, failure-intervals and stop-timeout")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return supervise(
        command, args.pool_state, args.poll_seconds, args.failure_intervals, args.stop_timeout
    )
