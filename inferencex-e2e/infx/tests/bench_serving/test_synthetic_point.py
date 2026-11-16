"""The real client, aggregation and collector against the scripted local server."""

import asyncio
import json
import signal
import subprocess

import pytest

from infx.bench_serving import benchmark_serving as client
from synthetic_point import MODEL, TOLERANCE_MS, Reply, ScriptedServer, child_env, client_command
from synthetic_point import produce, write_tokenizer

IDENTITY = {"hw": "synthetic-test", "framework": "synthetic-test", "precision": "synthetic-test",
            "model_prefix": "synthetic-test"}


def assert_matches_script(block, expected):
    """Every attempt of ``block`` is what its scripted reply should have produced."""
    columns = block["columns"]
    assert {len(column) for column in columns.values()} == {len(expected["attempts"])}
    for index, attempt in enumerate(expected["attempts"]):
        got = {name: column[index] for name, column in columns.items()}
        assert block["outcomes"][got["outcome"]] == attempt["outcome"], index
        assert got["output_len"] == attempt["output_len"], index
        assert got["completion_evidence"] == attempt["completion_evidence"], index
        for name in ("ttft_ms", "tpot_ms", "e2el_ms"):
            if attempt[name] is None:
                assert got[name] is None, (index, name)
            else:
                assert -TOLERANCE_MS["below"] <= got[name] - attempt[name] <= TOLERANCE_MS["above"], (
                    index, name, got[name], attempt[name])
    counts = block["counts"]
    assert {name: counts[name] for name in expected["counts"] if name != "failed_by_class"} == {
        name: value for name, value in expected["counts"].items() if name != "failed_by_class"}
    assert {name: n for name, n in counts["failed_by_class"].items() if n} == (
        expected["counts"]["failed_by_class"])
    assert counts["dispatched"] == counts["completed"] + counts["failed"] == len(columns["outcome"])


def test_scripted_curve_survives_the_client_the_aggregation_and_the_collector(tmp_path):
    manifest = produce(tmp_path / "out", IDENTITY)

    out = tmp_path / "out"
    assert manifest["synthetic"] is True
    points = {point["conc"]: point for point in manifest["points"]}
    assert sorted(points) == [1, 2, 4, 8]
    rows = {row["conc"]: row for row in json.loads((out / manifest["collected"]).read_text())}
    assert sorted(rows) == [1, 2, 4]
    for conc, point in points.items():
        expected = point["expected"]
        raw = json.loads((out / point["raw"]).read_text())
        aggregate = json.loads((out / point["aggregate"]).read_text())
        block = raw["request_records"]
        assert raw["synthetic"] == "true"
        assert aggregate["model"] == MODEL
        assert block["schema_version"] == manifest["request_records_schema_version"] == 1
        assert block["accounting"] == {"complete": True, "reasons": []}
        assert_matches_script(block, expected)
        assert expected["duration_s"]["min"] <= raw["duration"] <= expected["duration_s"]["max"]
        assert block["window"]["duration"] == raw["duration"] == aggregate["duration"]
        assert aggregate["completed"] == raw["completed"] == expected["counts"]["completed"]
        assert aggregate["request_throughput"] == raw["request_throughput"]
        assert aggregate["benchmark_outcome"]["status"] == expected["gate_status"]
        failed = expected["gate_status"] == "failed"
        assert (point["client_exit_code"], point["aggregate_exit_code"]) == ((1, 1) if failed else (0, 0))
        assert (point["uploaded_as_bmk"] is None) is failed
        carried = aggregate["request_records"]
        if expected["columns_omitted"]:
            assert "columns" not in carried
            assert carried["columns_omitted"] == {
                "reason": "attempts_exceed_aggregate_limit", "attempts": 40, "limit": 20}
            assert carried["counts"] == block["counts"]
            assert carried["accounting"] == block["accounting"]
        else:
            assert carried == block
        if not failed:
            assert rows[conc] == aggregate
    assert points[8]["expected"]["counts"]["failed_by_class"] == {
        "connection": 8, "empty_response": 8, "http_status": 8, "stream_error": 8}
    truncated = points[1]["expected"]["attempts"][85]
    assert (truncated["outcome"], truncated["completion_evidence"]) == ("incomplete_stream", 0)


def request(**overrides):
    return {
        "backend": "openai", "model_id": MODEL, "model_name": None, "tokenizer": None,
        "input_requests": [("w1 w2", 2, 4, None)] * 4, "logprobs": None, "best_of": 1,
        "request_rate": float("inf"), "burstiness": 1.0, "disable_tqdm": True, "num_warmups": 0,
        "profile": False, "selected_percentile_metrics": ["ttft"], "selected_percentiles": [99],
        "ignore_eos": True, "goodput_config_dict": {}, "max_concurrency": 2, "lora_modules": None,
        **overrides,
    }


@pytest.mark.parametrize("cancelled", ["the run", "one request"])
def test_cancellation_during_the_measured_requests_ends_the_run_without_a_result(cancelled):
    async def run(server):
        run = asyncio.create_task(client.benchmark(**request(
            api_url=f"{server.base_url}/v1/completions", base_url=server.base_url)))
        await asyncio.to_thread(server.streaming.wait)
        in_flight = asyncio.all_tasks() - {run, asyncio.current_task()}
        assert len(in_flight) == 4
        target = run if cancelled == "the run" else next(iter(in_flight))
        target.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

    with ScriptedServer([Reply("hang")] * 2, batch=2, warmups=0) as server:
        asyncio.run(run(server))


@pytest.mark.parametrize("interrupt", [signal.SIGINT, signal.SIGTERM])
def test_interrupted_client_exits_nonzero_and_writes_no_result(tmp_path, interrupt):
    tokenizer = write_tokenizer(tmp_path / "tokenizer")
    result = tmp_path / "result.json"
    with ScriptedServer([Reply("hang")] * 10, batch=2, warmups=4) as server:
        command = client_command(server.base_url, tokenizer, 2, 20, result)
        process = subprocess.Popen(command, env=child_env(), stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            assert server.streaming.wait(timeout=120)
            process.send_signal(interrupt)
            returncode = process.wait(timeout=60)
        finally:
            process.kill()

    assert returncode != 0
    assert server.arrivals > 4
    assert sorted(path.name for path in tmp_path.iterdir()) == ["tokenizer"]
