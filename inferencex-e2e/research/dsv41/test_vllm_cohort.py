import asyncio
import json

import pytest

from research.dsv41.vllm_cohort import stream_request


class Response:
    status = 200

    def __init__(self, chunks):
        self.chunks = chunks
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def __aiter__(self):
        for chunk in self.chunks:
            yield f"data: {json.dumps(chunk)}\n".encode()
        yield b"data: [DONE]\n"


class Session:
    def __init__(self, response):
        self.response = response

    def post(self, *_args, **_kwargs):
        return self.response


@pytest.mark.parametrize("output_length, success", [(3, True), (4, False)])
def test_stream_counts_delta_token_ids_and_retains_final_usage(
    monkeypatch, output_length, success
):
    monkeypatch.setenv("MODEL", "test-model")
    response = Response(
        [
            {"choices": [{"token_ids": [10, 11]}]},
            {"choices": [{"token_ids": [12]}]},
            {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 3}},
        ]
    )
    record = {"events": [], "success": False, "meta": {}, "error": ""}
    asyncio.run(
        stream_request(
            Session(response), "http://test", [1] * 8, output_length, record, 0
        )
    )
    assert [event[1] for event in record["events"]] == [2, 3]
    assert record["meta"] == {"prompt_tokens": 8, "completion_tokens": 3}
    assert record["success"] is success
    assert record["error"] == ("" if success else "Incorrect completion length")
