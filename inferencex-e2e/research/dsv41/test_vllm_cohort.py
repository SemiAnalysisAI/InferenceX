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
def test_stream_uses_continuous_usage_without_recounting_final_chunk(
    monkeypatch, output_length, success
):
    monkeypatch.setenv("MODEL", "test-model")
    response = Response(
        [
            {
                "choices": [{"text": "a"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2},
            },
            {
                "choices": [{"text": "b"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            },
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


def test_native_admission_waits_for_all_headers_before_generation(
    tmp_path, monkeypatch
):
    from research.dsv41.vllm_cohort import admit_wave

    monkeypatch.setenv("MODEL", "test-model")

    async def exercise():
        class Reply:
            status = 200

            def __init__(self, network, kind, ordinal=0):
                self.network, self.kind, self.ordinal = network, kind, ordinal
                self.content = self

            async def __aenter__(self):
                if self.kind == "completion":
                    await asyncio.sleep(0.001 * self.ordinal)
                    self.network.headers += 1
                return self

            async def __aexit__(self, *_):
                return None

            async def text(self):
                return json.dumps({"is_paused": self.network.paused})

            async def __aiter__(self):
                await self.network.resumed.wait()
                for count in (2, 4):
                    yield (
                        "data: "
                        + json.dumps(
                            {"usage": {"prompt_tokens": 8, "completion_tokens": count}}
                        )
                        + "\n"
                    ).encode()
                yield b"data: [DONE]\n"

        class Network:
            def __init__(self):
                self.paused, self.headers, self.requests = False, 0, 0
                self.resumed = asyncio.Event()

            def post(self, url, **_kwargs):
                if "/pause?" in url:
                    self.paused = True
                    return Reply(self, "control")
                if url.endswith("/resume"):
                    if self.headers != 2:
                        raise RuntimeError(
                            "Native test server refuses premature release"
                        )
                    self.paused = False
                    self.resumed.set()
                    return Reply(self, "control")
                self.requests += 1
                return Reply(self, "completion", self.requests)

            def get(self, _url):
                return Reply(self, "control")

        network = Network()
        records, tasks = await admit_wave(
            network,
            "http://test",
            [([1] * 8, 8, 4, None)] * 2,
            4,
            output=tmp_path,
            label="test",
            dp_size=2,
            barrier=True,
            settle_seconds=0,
            timeout_seconds=1,
        )
        await asyncio.gather(*tasks)
        return records

    records = asyncio.run(exercise())
    assert all(r["success"] for r in records)
    assert [[e[1] for e in r["events"]] for r in records] == [[2, 4], [2, 4]]
    marker = json.loads((tmp_path / "admission-test.json").read_text())
    assert marker["headers_received"] == 2
    assert marker["last_header_monotonic_s"] <= marker["resume_start_monotonic_s"]


@pytest.mark.parametrize("second_rank_batch", [2, 1])
def test_profile_validation_checks_actual_generation_batch(tmp_path, second_rank_batch):
    from research.dsv41.vllm_cohort import validate_cohort_profiles

    paths = []
    for rank, batch in enumerate((2, second_rank_batch)):
        path = tmp_path / f"dp{rank}_pp0_tp0_dcp0_ep{rank}_rank0.trace.json"
        path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {"cat": "kernel", "name": "real_work", "ts": 0, "dur": 2},
                        {
                            "cat": "user_annotation",
                            "name": f"execute_12_context_0(sq0)_generation_{batch}(sq12)",
                        },
                    ]
                }
            )
        )
        paths.append(path)
    if second_rank_batch == 1:
        with pytest.raises(RuntimeError, match="No full-batch GPU execution"):
            validate_cohort_profiles(paths, 2, 2, 4)
    else:
        result = validate_cohort_profiles(paths, 2, 2, 4)
        assert [(r["dp"], r["tp"], r["full_batch_execute_scopes"]) for r in result] == [
            (0, 0, 1),
            (1, 0, 1),
        ]


@pytest.mark.parametrize("rank", [None, 2])
def test_native_token_endpoint_streams_without_text_or_prompt_echo(monkeypatch, rank):
    import aiohttp
    from aiohttp import web

    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setenv("VLLM_COHORT_TOKEN_API", "1")

    async def exercise():
        async def native_generate(request):
            body = await request.json()
            params = body.get("sampling_params", {})
            if (
                body.get("token_ids") != [11, 12, 13]
                or body.get("return_token_ids") is not False
                or params.get("detokenize") is not False
                or params.get("max_tokens") != 3
                or params.get("ignore_eos") is not True
                or request.headers.get("X-data-parallel-rank")
                != (None if rank is None else "2")
            ):
                return web.Response(status=400, text="Invalid native token request")
            chunks = [
                {
                    "choices": [{"token_ids": [41]}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
                {
                    "choices": [{"token_ids": [42, 43]}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 3},
                },
            ]
            return web.Response(
                text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n",
                content_type="text/event-stream",
            )

        app = web.Application()
        app.router.add_post("/inference/v1/generate", native_generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        record = {"events": [], "success": False, "meta": {}, "error": ""}
        try:
            async with aiohttp.ClientSession() as session:
                await stream_request(
                    session, f"http://127.0.0.1:{port}", [11, 12, 13], 3, record, rank
                )
        finally:
            await runner.cleanup()
        return record

    result = asyncio.run(exercise())
    assert result["success"] is True
    assert [event[1] for event in result["events"]] == [1, 3]
    assert result["meta"] == {"prompt_tokens": 3, "completion_tokens": 3}
