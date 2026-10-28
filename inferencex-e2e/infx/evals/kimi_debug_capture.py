from __future__ import annotations

import base64
import json
import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--kimi-capture-dir", required=True)


def pytest_configure(config: pytest.Config) -> None:
    os.environ["KIMI_CAPTURE_DIR"] = config.getoption("--kimi-capture-dir")


def record(event: str, **fields: object) -> None:
    path = Path(os.environ["KIMI_CAPTURE_DIR"]) / (
        f"kimi_http_{os.environ.get('PYTEST_XDIST_WORKER', 'main')}_results.jsonl"
    )
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"time_ns": time.time_ns(), "event": event, **fields}) + "\n")


class CaptureStream(httpx.SyncByteStream):
    def __init__(self, stream: httpx.SyncByteStream, request_id: str) -> None:
        self.stream = stream
        self.request_id = request_id

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self.stream:
                record(
                    "chunk",
                    request_id=self.request_id,
                    base64=base64.b64encode(chunk).decode("ascii"),
                )
                yield chunk
            record("eof", request_id=self.request_id)
        except Exception as error:
            record("stream_error", request_id=self.request_id, error=repr(error))
            raise

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            record("close", request_id=self.request_id)


@pytest.fixture(autouse=True)
def capture_http(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    original = httpx.HTTPTransport.handle_request

    def handle(transport: httpx.HTTPTransport, message: httpx.Request) -> httpx.Response:
        request_id = uuid.uuid4().hex
        # Do not inject X-Request-Id: the router uses it as a consistent-hash key.
        record(
            "request",
            request_id=request_id,
            nodeid=request.node.nodeid,
            method=message.method,
            url=str(message.url),
            body=message.content.decode("utf-8"),
        )
        try:
            response = original(transport, message)
        except Exception as error:
            record("transport_error", request_id=request_id, error=repr(error))
            raise
        record(
            "response",
            request_id=request_id,
            status=response.status_code,
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower() in {"x-request-id", "content-type", "content-encoding"}
            },
        )
        response.stream = CaptureStream(response.stream, request_id)
        return response

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    record(
        "pytest",
        nodeid=report.nodeid,
        when=report.when,
        outcome=report.outcome,
        detail=str(report.longrepr) if report.failed else "",
    )
