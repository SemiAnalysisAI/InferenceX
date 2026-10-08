"""Server readiness; ``wait --url URL [--pid PID] [--log FILE]`` blocks until it is ready."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from infx.bench import env

REQUEST_TIMEOUT_S = 10
POLL_S = 5.0


class NotReadyError(env.BenchError):
    """The server died or the readiness budget ran out."""


def http_status(url: str) -> int | None:
    """The response status, or ``None`` when no HTTP response arrived."""
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:  # noqa: S310
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (OSError, ValueError, http.client.HTTPException):
        return None


def http_json(url: str) -> object | None:
    """A successful JSON response body, or ``None``."""
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:  # noqa: S310
            return json.load(response)
    except (OSError, ValueError, http.client.HTTPException):
        return None


def process_alive(pid: int) -> bool:
    """A zombie waiting to be reaped is not a live server."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, IndexError):
        return True
    return state not in {"Z", "X"}


def wait_ready(
    url: str, *, pid: int | None = None, log: Path | None = None, poll_s: float = POLL_S
) -> None:
    """Poll ``url`` until it answers below 400, streaming ``log``; fail once ``pid`` exits."""
    offset = 0
    while True:
        if log is not None and log.exists():
            with log.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
            offset += len(chunk)
            sys.stdout.write(chunk.decode(errors="replace"))
            sys.stdout.flush()
        status = http_status(url)
        if status is not None and status < 400:
            return
        if pid is not None and not process_alive(pid):
            raise NotReadyError(f"process {pid} died before {url} became ready")
        time.sleep(poll_s)


def chat_route_budget(environ: Mapping[str, str] = os.environ) -> tuple[int, int]:
    """The workflow's chat-route readiness ``(timeout, stabilization)`` in seconds."""
    names = ("EVAL_ENDPOINT_READY_TIMEOUT_SECONDS", "EVAL_MODEL_STABILIZATION_SECONDS")
    env.require(*names, env=environ)
    return env.positive_int(names[0], environ), env.non_negative_int(names[1], environ)


def wait_chat_route(
    base_url: str, model: str, budget: tuple[int, int], *, poll_s: float = POLL_S
) -> None:
    """Wait until ``model`` is served and the chat route is mounted."""
    timeout_s, stabilization_s = budget
    chat_url = f"{base_url}/v1/chat/completions"
    start = time.monotonic()
    healthy_since: float | None = None
    next_report = 0.0
    while True:
        now = time.monotonic()
        models = http_json(f"{base_url}/v1/models")
        entries = models.get("data", []) if isinstance(models, dict) else []
        served = any(isinstance(entry, dict) and entry.get("id") == model for entry in entries)
        # A bare GET answering 401/403/405 proves the route is mounted.
        if served and http_status(chat_url) in {401, 403, 405}:
            break
        status = http_status(f"{base_url}/health")
        if status is None or status >= 400:
            healthy_since = None
        elif healthy_since is None:
            healthy_since = now
        # Some frontends list no model before the first request; sustained health counts.
        if healthy_since is not None and now - healthy_since >= stabilization_s:
            break
        elapsed = now - start
        if elapsed >= timeout_s:
            raise NotReadyError(
                f"chat endpoint for model {model!r} not ready within {timeout_s}s: {chat_url}"
            )
        if elapsed >= next_report:
            print(f"Waiting for {chat_url} ({model!r}): {int(elapsed)}/{timeout_s}s", flush=True)
            next_report += 60
        time.sleep(poll_s)
    print(f"OpenAI chat endpoint ready for model {model!r}: {chat_url}", flush=True)


def main(argv: list[str]) -> int:
    """Run the ``wait`` command."""
    parser = argparse.ArgumentParser(prog="python3 -m infx.bench wait")
    parser.add_argument("--url", required=True, help="health URL that answers below 400 when ready")
    parser.add_argument("--pid", type=int, help="server process; fail if it exits")
    parser.add_argument("--log", type=Path, help="server log to stream while waiting")
    args = parser.parse_args(argv)
    wait_ready(args.url, pid=args.pid, log=args.log)
    return 0
