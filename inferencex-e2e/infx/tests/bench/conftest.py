"""Fixtures shared by the ``infx.bench`` tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

Respond = Callable[[str, str], tuple[int, object]]


@pytest.fixture
def http_server() -> Iterator[Callable[[Respond], str]]:
    """Start local servers answering with ``respond(method, path) -> (status, json_body)``."""
    started: list[ThreadingHTTPServer] = []

    def start(respond: Respond) -> str:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._reply(*respond("GET", self.path))

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self._reply(*respond("POST", self.path))

            def _reply(self, status: int, body: object) -> None:
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_: object) -> None:
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, args=(0.01,), daemon=True).start()
        started.append(httpd)
        return f"http://127.0.0.1:{httpd.server_port}"

    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()
