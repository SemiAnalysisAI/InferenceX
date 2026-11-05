"""Ports and process liveness for tests that run real servers and process groups."""

import os
import socket
import time


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone(pid: int, timeout_s: float = 10.0) -> bool:
    """Whether ``pid`` is gone within ``timeout_s``; an orphan is reaped by init."""
    deadline = time.monotonic() + timeout_s
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not alive(pid)
