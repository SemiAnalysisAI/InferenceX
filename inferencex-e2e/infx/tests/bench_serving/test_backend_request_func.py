"""OpenAI streaming request functions against a local SSE server."""

import asyncio
import json
import urllib.request

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


def stream(endpoint: str, events: list[object]) -> req.RequestFuncOutput:
    """Serve ``events`` as one SSE stream on HTTP 200 and run the request function on it.

    ``bytes`` events are written verbatim, anything else as one ``data:`` event.
    """

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.read()
        response = web.StreamResponse()
        await response.prepare(request)
        for event in events:
            if isinstance(event, bytes):
                await response.write(event)
                continue
            data = event if isinstance(event, str) else json.dumps(event)
            await response.write(f"data: {data}\n\n".encode())
        await response.write_eof()
        return response

    async def run() -> req.RequestFuncOutput:
        app = web.Application()
        app.router.add_post(f"/v1/{endpoint}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            return await REQUEST[endpoint](
                req.RequestFuncInput(
                    prompt="p",
                    api_url=f"http://127.0.0.1:{port}/v1/{endpoint}",
                    prompt_len=1,
                    output_len=4,
                    model="m",
                )
            )
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


@pytest.fixture
def netrc_path(monkeypatch, tmp_path):
    """Clear the proxy variables and point ``NETRC`` at a file that does not exist yet."""
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    # On macOS and Windows, getproxies() also reads the system proxy settings.
    monkeypatch.setattr(urllib.request, "getproxies", urllib.request.getproxies_environment)
    monkeypatch.setattr(urllib.request, "proxy_bypass", urllib.request.proxy_bypass_environment)
    path = tmp_path / "netrc"
    monkeypatch.setenv("NETRC", str(path))
    return path


def test_no_proxy_and_no_netrc_entry_leave_the_environment_untrusted(netrc_path):
    assert not req.env_changes_request("http://127.0.0.1:8000/v1/completions")


def test_a_proxy_for_the_url_scheme_is_trusted(netrc_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    assert req.env_changes_request("http://10.0.0.5:8000/v1/completions")
    assert not req.env_changes_request("https://10.0.0.5:8000/v1/completions")


def test_no_proxy_excludes_the_host(netrc_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    assert not req.env_changes_request("http://127.0.0.1:8000/v1/completions")


def test_a_netrc_entry_for_the_host_is_trusted(netrc_path):
    netrc_path.write_text("machine bench.internal login user password secret\n")
    assert req.env_changes_request("http://bench.internal:8000/v1/generate_stream")
    assert not req.env_changes_request("http://127.0.0.1:8000/v1/generate_stream")


def test_a_proxy_from_the_environment_still_carries_the_request(netrc_path, monkeypatch):
    """The request reaches the server through ``HTTP_PROXY`` for a host that does not resolve."""
    targets = []

    async def proxy(request: web.Request) -> web.StreamResponse:
        targets.append(str(request.url))
        await request.read()
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(f"data: {json.dumps(CHOICE['completions'])}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def run() -> req.RequestFuncOutput:
        app = web.Application()
        app.router.add_post("/v1/completions", proxy)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{runner.addresses[0][1]}")
        try:
            return await req.async_request_openai_completions(
                req.RequestFuncInput(
                    prompt="p",
                    api_url="http://bench.invalid/v1/completions",
                    prompt_len=1,
                    output_len=4,
                    model="m",
                )
            )
        finally:
            await runner.cleanup()

    output = asyncio.run(run())
    assert output.success, output.error
    assert output.generated_text == "hi"
    assert targets == ["http://bench.invalid/v1/completions"]
