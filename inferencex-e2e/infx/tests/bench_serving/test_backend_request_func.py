"""OpenAI streaming request functions against a local SSE server."""

import asyncio
import json
import socket

import aiohttp
import pytest
from aiohttp import web

from infx.bench_serving import backend_request_func as req

CHOICE = {
    "completions": {"choices": [{"text": "hi"}]},
    "chat/completions": {"choices": [{"delta": {"content": "hi"}}]},
}
REQUEST = {
    "completions": req.async_request_openai_completions,
    "chat/completions": req.async_request_openai_chat_completions,
}


def request_input(endpoint: str, port: int) -> req.RequestFuncInput:
    return req.RequestFuncInput(
        prompt="p",
        api_url=f"http://127.0.0.1:{port}/v1/{endpoint}",
        prompt_len=1,
        output_len=4,
        model="m",
    )


async def write_events(response: web.StreamResponse, events: list[object]) -> None:
    """Write ``bytes`` events verbatim, anything else as one ``data:`` event."""
    for event in events:
        if isinstance(event, bytes):
            await response.write(event)
            continue
        data = event if isinstance(event, str) else json.dumps(event)
        await response.write(f"data: {data}\n\n".encode())


def stream(endpoint: str, events: list[object]) -> req.RequestFuncOutput:
    """Serve ``events`` as one SSE stream on HTTP 200 and run the request function on it."""

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.read()
        response = web.StreamResponse()
        await response.prepare(request)
        await write_events(response, events)
        await response.write_eof()
        return response

    return call(endpoint, handler)


def call(endpoint: str, handler) -> req.RequestFuncOutput:
    """Run the request function for ``endpoint`` against a local server answering with ``handler``."""

    async def run() -> req.RequestFuncOutput:
        app = web.Application()
        app.router.add_post(f"/v1/{endpoint}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            return await REQUEST[endpoint](request_input(endpoint, port))
        finally:
            await runner.cleanup()

    return asyncio.run(run())


@pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])
def test_error_event_mid_stream_fails_the_request(endpoint):
    error = {"error": {"message": "context length exceeded", "code": 400}}

    output = stream(endpoint, [CHOICE[endpoint], error, "[DONE]"])

    assert output.success is False
    assert "context length exceeded" in output.error


def test_chat_stream_without_a_token_chunk_fails_the_request():
    usage = {"choices": [], "usage": {"completion_tokens": 0}}

    output = stream("chat/completions", [usage, "[DONE]"])

    assert output.success is False
    assert output.error.startswith("Never received a valid chunk")


@pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])
def test_sse_comment_and_event_lines_are_not_parsed_as_data(endpoint):
    keepalive = b": keep-alive\n\n"
    typed = b"event: message\n"

    output = stream(endpoint, [keepalive, typed, CHOICE[endpoint], keepalive, "[DONE]"])

    assert output.error == ""
    assert output.success is True
    assert output.generated_text == "hi"


ENDPOINTS = pytest.mark.parametrize("endpoint", ["completions", "chat/completions"])


@ENDPOINTS
def test_completed_request_records_its_send_and_return_times(endpoint):
    output = stream(endpoint, [CHOICE[endpoint], CHOICE[endpoint], "[DONE]"])

    assert output.success is True
    assert output.failure_class == ""
    assert output.http_status is None
    assert output.end_time - output.start_time >= output.latency >= output.ttft > 0


@ENDPOINTS
def test_non_200_response_is_classified_by_its_status(endpoint):
    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=503, text="overloaded")

    output = call(endpoint, handler)

    assert output.success is False
    assert (output.failure_class, output.http_status) == ("http_status", 503)
    assert output.ttft == 0
    assert output.end_time >= output.start_time


@ENDPOINTS
def test_error_event_after_a_token_keeps_the_observed_ttft(endpoint):
    error = {"error": {"message": "context length exceeded", "code": 400}}

    output = stream(endpoint, [CHOICE[endpoint], error, "[DONE]"])

    assert output.success is False
    assert output.failure_class == "stream_error"
    assert output.ttft > 0


@ENDPOINTS
def test_stream_without_a_token_chunk_is_an_empty_response(endpoint):
    output = stream(endpoint, ["[DONE]"])

    assert output.success is False
    assert output.failure_class == "empty_response"
    assert output.ttft == 0


@ENDPOINTS
def test_stream_cut_after_a_token_is_a_connection_failure_with_ttft(endpoint):
    async def handler(request: web.Request) -> web.StreamResponse:
        await request.read()
        response = web.StreamResponse()
        await response.prepare(request)
        await write_events(response, [CHOICE[endpoint]])
        request.transport.close()
        return response

    output = call(endpoint, handler)

    assert output.success is False
    assert output.failure_class == "connection"
    assert output.ttft > 0
    assert output.end_time - output.start_time >= output.ttft


@ENDPOINTS
def test_request_exceeding_the_client_timeout_is_a_timeout(endpoint, monkeypatch):
    monkeypatch.setattr(req, "AIOHTTP_TIMEOUT", aiohttp.ClientTimeout(total=0.2))

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.read()
        response = web.StreamResponse()
        await response.prepare(request)
        await asyncio.sleep(1)
        return response

    output = call(endpoint, handler)

    assert output.success is False
    assert output.failure_class == "timeout"
    assert output.end_time - output.start_time >= 0.2


@ENDPOINTS
def test_refused_connection_fails_before_any_byte(endpoint):
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]

    output = asyncio.run(REQUEST[endpoint](request_input(endpoint, port)))

    assert output.success is False
    assert output.failure_class == "connection"
    assert output.http_status is None
    assert output.ttft == 0
