"""OpenAI streaming request functions against a local SSE server."""

import asyncio
import json

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
