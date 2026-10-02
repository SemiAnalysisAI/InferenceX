import asyncio
import gzip
import json

from research.dsv41.sharded_client import sharded_wave


def test_shards_preserve_order_and_collect_metrics_during_delayed_stream(
    tmp_path, monkeypatch
):
    import aiohttp
    from aiohttp import web

    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setenv("VLLM_COHORT_TOKEN_API", "1")

    async def exercise():
        paused = False
        resumed = asyncio.Event()

        async def pause(_request):
            nonlocal paused
            paused = True
            return web.json_response({})

        async def state(_request):
            return web.json_response({"is_paused": paused})

        async def resume(_request):
            nonlocal paused
            paused = False
            resumed.set()
            return web.json_response({})

        async def metrics(_request):
            return web.Response(text='vllm:num_requests_running{engine="0"} 2\n')

        async def generate(request):
            body = await request.json()
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await resumed.wait()
            # One shard is slow; the other shard and metrics remain independent.
            if request.headers.get("X-data-parallel-rank") == "0":
                await asyncio.sleep(1.2)
            item = {
                "usage": {
                    "prompt_tokens": len(body["token_ids"]),
                    "completion_tokens": 3,
                }
            }
            await response.write(("data: " + json.dumps(item) + "\n\n").encode())
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/pause", pause)
        app.router.add_get("/is_paused", state)
        app.router.add_post("/resume", resume)
        app.router.add_get("/metrics", metrics)
        app.router.add_post("/inference/v1/generate", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = f"http://127.0.0.1:{runner.addresses[0][1]}"
        try:
            async with aiohttp.ClientSession() as session:
                return await sharded_wave(
                    session,
                    base,
                    [([11], 1, 3, None), ([22, 23], 2, 3, None)],
                    3,
                    output=tmp_path,
                    dp_size=2,
                    label="test",
                    settle_seconds=0.01,
                    timeout_seconds=20,
                )
        finally:
            await runner.cleanup()

    records = asyncio.run(exercise())
    assert [r["meta"]["prompt_tokens"] for r in records] == [1, 2]
    assert all(r["success"] and r["events"][-1][1] == 3 for r in records)
    assert records[1]["end"] < records[0]["end"]
    with gzip.open(tmp_path / "test-metrics.jsonl.gz", "rt") as handle:
        samples = [json.loads(line) for line in handle]
    during = [
        r for r in samples if records[1]["end"] < r["monotonic_s"] < records[0]["end"]
    ]
    assert len(during) >= 1
    assert during[0]["metrics"] == ['vllm:num_requests_running{engine="0"} 2']
    assert all(r["request_start_monotonic_s"] <= r["monotonic_s"] for r in samples)
