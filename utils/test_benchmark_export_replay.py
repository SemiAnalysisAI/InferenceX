import asyncio
import json
from pathlib import Path

from aiohttp import web

from bench_serving.benchmark_export_replay import (
    load_replay_sessions,
    run_export_replay_benchmark,
)


def _count_tokens(text: str) -> int:
    return max(1, len((text or "").split())) if text else 0


def _multiturn_payload(runtime_stack_id: str = "standalone:sglang") -> dict:
    return {
        "adapter_id": "inferencex_multiturn",
        "exports": [
            {
                "trace_id": "trace-chat-1",
                "runtime_stack_id": runtime_stack_id,
                "hardware_profile_id": "nvidia:h200_sxm_141gb",
                "canonical_model_id": "qwen3_30b_a3b",
                "support_status": "supported",
                "benchmark_certification_status": "dataset_replay_verified",
                "session": {
                    "session_id": "session-chat-1",
                    "turns": [
                        {
                            "turn_idx": 0,
                            "turn_id": 0,
                            "messages": [
                                {
                                    "role": "user",
                                    "content_blocks": [
                                        {"type": "text", "text": "Investigate the flaky test."}
                                    ],
                                }
                            ],
                            "expected_output_tokens": 8,
                            "wait_before_ms": 0,
                        },
                        {
                            "turn_idx": 1,
                            "turn_id": 1,
                            "messages": [
                                {
                                    "role": "user",
                                    "content_blocks": [
                                        {"type": "text", "text": "Investigate the flaky test."}
                                    ],
                                },
                                {
                                    "role": "assistant",
                                    "content_blocks": [
                                        {"type": "text", "text": "I found a race in the setup."}
                                    ],
                                },
                                {
                                    "role": "tool",
                                    "content_blocks": [
                                        {"type": "log", "text": "pytest -k flaky_test -> failed"}
                                    ],
                                },
                            ],
                            "expected_output_tokens": 6,
                            "wait_before_ms": 10,
                        },
                    ],
                },
            }
        ],
    }


def _trace_replay_payload(runtime_stack_id: str = "standalone:trt_llm") -> dict:
    return {
        "adapter_id": "inferencex_trace_replay",
        "exports": [
            {
                "trace_id": "trace-replay-1",
                "runtime_stack_id": runtime_stack_id,
                "hardware_profile_id": "nvidia:b200_sxm_180gb",
                "canonical_model_id": "gpt_oss_120b",
                "support_status": "supported",
                "benchmark_certification_status": "dataset_replay_verified",
                "trace_metadata": {"session_id": "session-replay-1"},
                "events": [
                    {
                        "turn_id": 0,
                        "arrival_time_offset_ms": 0,
                        "input_messages": [
                            {
                                "role": "user",
                                "content_blocks": [
                                    {"type": "text", "text": "Summarize the incident report."}
                                ],
                            }
                        ],
                        "target_output_tokens": 7,
                    },
                    {
                        "turn_id": 1,
                        "arrival_time_offset_ms": 25,
                        "input_messages": [
                            {
                                "role": "user",
                                "content_blocks": [
                                    {"type": "text", "text": "Summarize the incident report."}
                                ],
                            },
                            {
                                "role": "assistant",
                                "content_blocks": [
                                    {"type": "text", "text": "The outage started after deploy."}
                                ],
                            },
                        ],
                        "target_output_tokens": 5,
                    },
                ],
            }
        ],
    }


async def _start_mock_server(
    sse_mode: str = "normal",
    metrics_text: str | None = None,
) -> tuple[web.AppRunner, str]:
    """Start a mock OpenAI-compatible server.

    sse_mode controls how SSE frames are written to the wire:
      - "normal": one data frame per write (default)
      - "multiline": multiple data frames packed into a single write
      - "split": a single data frame split across two writes
    """

    async def _stream_response(request: web.Request, chunks: list[dict]) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)

        if sse_mode == "multiline":
            # Pack ALL data frames into a single TCP write
            blob = b""
            for chunk in chunks:
                blob += f"data: {json.dumps(chunk)}\n\n".encode()
            blob += b"data: [DONE]\n\n"
            await response.write(blob)
        elif sse_mode == "split":
            # Split the first frame across two writes
            for idx, chunk in enumerate(chunks):
                frame = f"data: {json.dumps(chunk)}\n\n".encode()
                if idx == 0:
                    mid = len(frame) // 2
                    await response.write(frame[:mid])
                    await asyncio.sleep(0.005)
                    await response.write(frame[mid:])
                else:
                    await response.write(frame)
                    await asyncio.sleep(0.005)
            await response.write(b"data: [DONE]\n\n")
        else:
            for chunk in chunks:
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await asyncio.sleep(0.005)
            await response.write(b"data: [DONE]\n\n")

        await response.write_eof()
        return response

    async def chat_handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        # Verify the fallback from max_completion_tokens -> max_tokens.
        if "max_completion_tokens" in payload:
            return web.json_response({"error": "unsupported field"}, status=400)
        assert payload["messages"]
        return await _stream_response(
            request,
            [
                {"choices": [{"delta": {"content": "patched"}}]},
                {"usage": {"completion_tokens": 2}},
            ],
        )

    async def completions_handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        assert payload["prompt"].startswith("USER:")
        return await _stream_response(
            request,
            [
                {"choices": [{"text": "resolved"}]},
                {"usage": {"completion_tokens": 2}},
            ],
        )

    async def metrics_handler(_: web.Request) -> web.Response:
        return web.Response(
            text=metrics_text
            or (
                "vllm:gpu_cache_usage_perc 0.42\n"
                "vllm:cpu_cache_usage_perc 0.25\n"
                "sglang:cache_hit_rate 0.8\n"
            )
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_handler)
    app.router.add_post("/v1/completions", completions_handler)
    app.router.add_get("/metrics", metrics_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    sockets = getattr(site, "_server").sockets
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def test_load_replay_sessions_multiturn_chat(tmp_path: Path) -> None:
    export_file = tmp_path / "multiturn.json"
    export_file.write_text(json.dumps(_multiturn_payload()))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:sglang"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"qwen3_30b_a3b"},
        request_mode="auto",
        ignore_waits=False,
    )

    assert len(sessions) == 1
    assert sessions[0].request_mode == "chat"
    assert sessions[0].turns[1].wait_before_s == 0.01
    assert selection["support_statuses"] == ["supported"]
    assert selection["support_status_counts"] == {"supported": 1}
    assert selection["benchmark_certification_statuses"] == ["dataset_replay_verified"]
    assert selection["benchmark_certification_status_counts"] == {
        "dataset_replay_verified": 1
    }
    assert selection["request_mode_mix"] == {"chat": 1}


def test_load_replay_sessions_trace_replay_auto_uses_completions(tmp_path: Path) -> None:
    export_file = tmp_path / "trace_replay.json"
    export_file.write_text(json.dumps(_trace_replay_payload()))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:trt_llm"},
        hardware_profile_ids={"nvidia:b200_sxm_180gb"},
        canonical_model_ids={"gpt_oss_120b"},
        request_mode="auto",
    )

    assert len(sessions) == 1
    assert sessions[0].request_mode == "completions"
    assert sessions[0].turns[1].wait_before_s == 0.025
    assert sessions[0].turns[0].completion_prompt.startswith("USER:")
    assert selection["support_statuses"] == ["supported"]
    assert selection["benchmark_certification_statuses"] == ["dataset_replay_verified"]
    assert selection["request_mode_mix"] == {"completions": 1}


def test_load_replay_sessions_support_status_filter(tmp_path: Path) -> None:
    payload = _multiturn_payload()
    payload["exports"].append(
        {
            **payload["exports"][0],
            "trace_id": "trace-chat-preview",
            "support_status": "reviewed_preview",
        }
    )
    export_file = tmp_path / "multiturn_mixed_status.json"
    export_file.write_text(json.dumps(payload))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:sglang"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"qwen3_30b_a3b"},
        support_statuses={"supported"},
        request_mode="auto",
        ignore_waits=False,
    )

    assert [session.trace_id for session in sessions] == ["trace-chat-1"]
    assert selection["support_statuses"] == ["supported"]
    assert selection["support_status_counts"] == {"supported": 1}
    assert selection["benchmark_certification_statuses"] == ["dataset_replay_verified"]


def test_run_export_replay_benchmark_chat(tmp_path: Path) -> None:
    export_file = tmp_path / "multiturn.json"
    export_file.write_text(json.dumps(_multiturn_payload()))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:sglang"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"qwen3_30b_a3b"},
        request_mode="chat",
        ignore_waits=True,
    )

    async def _run() -> dict:
        runner, base_url = await _start_mock_server()
        try:
            return await run_export_replay_benchmark(
                sessions=sessions,
                selection_metadata=selection,
                model_id="Qwen/Qwen3-30B-A3B",
                model_name=None,
                chat_api_url=f"{base_url}/v1/chat/completions",
                completion_api_url=f"{base_url}/v1/completions",
                count_text_tokens=_count_tokens,
                max_concurrency=1,
                selected_percentiles=[99],
                disable_tqdm=True,
                num_warmup_sessions=1,
            )
        finally:
            await runner.cleanup()

    result = asyncio.run(_run())
    assert result["aggregate_metrics"]["completed_sessions"] == 1
    assert result["selection"]["request_mode_mix"] == {"chat": 1}
    assert result["server_metrics_summary"]["samples"] >= 0
    assert result["server_metrics_summary"]["gpu_cache_usage_peak"] == 0.42
    assert result["server_metrics_summary"]["cpu_cache_usage_peak"] == 0.25
    assert result["server_metrics_summary"]["gpu_cache_metric_name"] == "vllm:gpu_cache_usage_perc"
    assert result["server_metrics_summary"]["cpu_cache_metric_name"] == "vllm:cpu_cache_usage_perc"
    assert result["server_metrics_summary"]["cpu_cache_metric_available"] is True
    assert result["server_metrics_summary"]["observability_status"] == "direct_cpu_cache_metric"
    assert result["server_metrics_summary"]["kv_offload_observed"] is True


def test_run_export_replay_benchmark_completions(tmp_path: Path) -> None:
    export_file = tmp_path / "trace_replay.json"
    export_file.write_text(json.dumps(_trace_replay_payload()))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:trt_llm"},
        hardware_profile_ids={"nvidia:b200_sxm_180gb"},
        canonical_model_ids={"gpt_oss_120b"},
        request_mode="completions",
        ignore_waits=True,
    )

    async def _run() -> dict:
        runner, base_url = await _start_mock_server()
        try:
            return await run_export_replay_benchmark(
                sessions=sessions,
                selection_metadata=selection,
                model_id="gpt-oss-120b",
                model_name=None,
                chat_api_url=f"{base_url}/v1/chat/completions",
                completion_api_url=f"{base_url}/v1/completions",
                count_text_tokens=_count_tokens,
                max_concurrency=1,
                selected_percentiles=[99],
                disable_tqdm=True,
                num_warmup_sessions=0,
            )
        finally:
            await runner.cleanup()

    result = asyncio.run(_run())
    assert result["aggregate_metrics"]["completed_sessions"] == 1
    assert result["selection"]["request_mode_mix"] == {"completions": 1}


def test_run_export_replay_benchmark_sglang_token_usage_metrics(tmp_path: Path) -> None:
    export_file = tmp_path / "multiturn_sglang_metrics.json"
    export_file.write_text(json.dumps(_multiturn_payload(runtime_stack_id="standalone:sglang")))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:sglang"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"qwen3_30b_a3b"},
        request_mode="chat",
        ignore_waits=True,
    )

    async def _run() -> dict:
        runner, base_url = await _start_mock_server(
            metrics_text=(
                'sglang:token_usage{model_name="Qwen/Qwen3-30B-A3B"} 0.61\n'
                'sglang:cache_hit_rate{model_name="Qwen/Qwen3-30B-A3B"} 0.8\n'
            )
        )
        try:
            return await run_export_replay_benchmark(
                sessions=sessions,
                selection_metadata=selection,
                model_id="Qwen/Qwen3-30B-A3B",
                model_name=None,
                chat_api_url=f"{base_url}/v1/chat/completions",
                completion_api_url=f"{base_url}/v1/completions",
                count_text_tokens=_count_tokens,
                max_concurrency=1,
                selected_percentiles=[99],
                disable_tqdm=True,
                num_warmup_sessions=0,
            )
        finally:
            await runner.cleanup()

    result = asyncio.run(_run())
    summary = result["server_metrics_summary"]
    assert result["aggregate_metrics"]["completed_sessions"] == 1
    assert summary["samples"] >= 0
    assert summary["gpu_cache_usage_peak"] == 0.61
    assert summary["gpu_cache_metric_name"] == "sglang:token_usage"
    assert summary["cpu_cache_metric_name"] is None
    assert summary["cpu_cache_metric_available"] is False
    assert summary["cache_hit_rate_avg"] == 0.8
    assert summary["observability_status"] == "indirect_without_cpu_cache_metric"
    assert summary["kv_offload_observed"] is False


def test_sse_multiline_chunks(tmp_path: Path) -> None:
    """Verify replay works when the server packs multiple SSE frames into one TCP write."""
    export_file = tmp_path / "multiturn.json"
    export_file.write_text(json.dumps(_multiturn_payload()))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:sglang"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"qwen3_30b_a3b"},
        request_mode="chat",
        ignore_waits=True,
    )

    async def _run() -> dict:
        runner, base_url = await _start_mock_server(sse_mode="multiline")
        try:
            return await run_export_replay_benchmark(
                sessions=sessions,
                selection_metadata=selection,
                model_id="Qwen/Qwen3-30B-A3B",
                model_name=None,
                chat_api_url=f"{base_url}/v1/chat/completions",
                completion_api_url=f"{base_url}/v1/completions",
                count_text_tokens=_count_tokens,
                max_concurrency=1,
                selected_percentiles=[99],
                disable_tqdm=True,
                num_warmup_sessions=0,
            )
        finally:
            await runner.cleanup()

    result = asyncio.run(_run())
    assert result["aggregate_metrics"]["completed_sessions"] == 1


def test_sse_split_across_chunks(tmp_path: Path) -> None:
    """Verify replay works when a single SSE frame is split across TCP writes."""
    export_file = tmp_path / "multiturn.json"
    export_file.write_text(json.dumps(_multiturn_payload()))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:sglang"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"qwen3_30b_a3b"},
        request_mode="chat",
        ignore_waits=True,
    )

    async def _run() -> dict:
        runner, base_url = await _start_mock_server(sse_mode="split")
        try:
            return await run_export_replay_benchmark(
                sessions=sessions,
                selection_metadata=selection,
                model_id="Qwen/Qwen3-30B-A3B",
                model_name=None,
                chat_api_url=f"{base_url}/v1/chat/completions",
                completion_api_url=f"{base_url}/v1/completions",
                count_text_tokens=_count_tokens,
                max_concurrency=1,
                selected_percentiles=[99],
                disable_tqdm=True,
                num_warmup_sessions=0,
            )
        finally:
            await runner.cleanup()

    result = asyncio.run(_run())
    assert result["aggregate_metrics"]["completed_sessions"] == 1


def test_empty_content_no_phantom_itl(tmp_path: Path) -> None:
    """Verify that SSE chunks with empty/null content don't inflate ITL counts."""
    export_file = tmp_path / "multiturn.json"
    # Use a single-turn export to isolate ITL counting
    single_turn_payload = {
        "adapter_id": "inferencex_multiturn",
        "exports": [
            {
                "trace_id": "trace-itl-1",
                "runtime_stack_id": "standalone:sglang",
                "hardware_profile_id": "nvidia:h200_sxm_141gb",
                "canonical_model_id": "qwen3_30b_a3b",
                "support_status": "supported",
                "session": {
                    "session_id": "session-itl-1",
                    "turns": [
                        {
                            "turn_idx": 0,
                            "turn_id": 0,
                            "messages": [
                                {
                                    "role": "user",
                                    "content_blocks": [
                                        {"type": "text", "text": "Hello"}
                                    ],
                                }
                            ],
                            "expected_output_tokens": 4,
                            "wait_before_ms": 0,
                        },
                    ],
                },
            }
        ],
    }
    export_file.write_text(json.dumps(single_turn_payload))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:sglang"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"qwen3_30b_a3b"},
        request_mode="chat",
        ignore_waits=True,
    )

    async def _run() -> dict:
        # Custom server that sends empty-content chunks between real ones
        async def _chat_with_empty(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            if "max_completion_tokens" in payload:
                return web.json_response({"error": "unsupported"}, status=400)

            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            # Frame 1: real content
            await response.write(
                f'data: {{"choices": [{{"delta": {{"content": "hello"}}}}]}}\n\n'.encode()
            )
            await asyncio.sleep(0.005)
            # Frame 2: empty content (should not generate ITL entry)
            await response.write(
                f'data: {{"choices": [{{"delta": {{"content": ""}}}}]}}\n\n'.encode()
            )
            await asyncio.sleep(0.005)
            # Frame 3: null content (should not generate ITL entry)
            await response.write(
                f'data: {{"choices": [{{"delta": {{}}}}]}}\n\n'.encode()
            )
            await asyncio.sleep(0.005)
            # Frame 4: real content
            await response.write(
                f'data: {{"choices": [{{"delta": {{"content": " world"}}}}]}}\n\n'.encode()
            )
            await asyncio.sleep(0.005)
            # Usage frame
            await response.write(
                f'data: {{"usage": {{"completion_tokens": 2}}}}\n\n'.encode()
            )
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", _chat_with_empty)
        app.router.add_get("/metrics", lambda _: web.Response(text=""))

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host="127.0.0.1", port=0)
        await site.start()
        sockets = getattr(site, "_server").sockets
        port = sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"

        try:
            return await run_export_replay_benchmark(
                sessions=sessions,
                selection_metadata=selection,
                model_id="Qwen/Qwen3-30B-A3B",
                model_name=None,
                chat_api_url=f"{base_url}/v1/chat/completions",
                completion_api_url=f"{base_url}/v1/completions",
                count_text_tokens=_count_tokens,
                max_concurrency=1,
                selected_percentiles=[99],
                disable_tqdm=True,
                num_warmup_sessions=0,
            )
        finally:
            await runner.cleanup()

    result = asyncio.run(_run())
    agg = result["aggregate_metrics"]
    assert agg["completed_sessions"] == 1
    # With 2 real content chunks, ITL should have exactly 1 entry
    # (first content = TTFT, second content = 1 ITL). Empty/null chunks
    # must not inflate this count.
    turn_metrics = result["per_turn_metrics"]["turn_1"]
    assert turn_metrics["completed"] == 1


def test_actual_context_len_for_file_backed_assets(tmp_path: Path) -> None:
    """Verify that actual_context_len counts rendered payload tokens, not asset metadata."""
    payload = {
        "adapter_id": "inferencex_trace_replay",
        "exports": [
            {
                "trace_id": "test-asset-trace",
                "runtime_stack_id": "standalone:vllm",
                "hardware_profile_id": "nvidia:h200_sxm_141gb",
                "canonical_model_id": "gpt_oss_120b",
                "support_status": "reviewed_preview",
                "benchmark_certification_status": "dataset_replay_verified",
                "context_band": "xlc2_384k_512k",
                "trace_metadata": {
                    "session_id": "test-session",
                    "estimated_kv_bytes_peak": 27000000000,
                    "expected_offload_mode": "soft_offload",
                },
                "events": [
                    {
                        "event_id": "evt-0",
                        "trace_id": "test-asset-trace",
                        "session_id": "test-session",
                        "turn_id": 0,
                        "arrival_time_offset_ms": 0,
                        "input_messages": [
                            {
                                "role": "user",
                                "content_blocks": [
                                    {"type": "text", "text": "Analyze this codebase"},
                                    {
                                        "type": "table",
                                        "text": None,
                                        "asset_path": "synthetic_v0/context_assets/big_file.md",
                                        "asset_token_count": 500000,
                                        "asset_byte_count": 2500000,
                                    },
                                ],
                            }
                        ],
                        "output": {"output_token_count": 100},
                    }
                ],
            }
        ],
    }
    export_file = tmp_path / "asset_test.json"
    export_file.write_text(json.dumps(payload))

    sessions, _ = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:vllm"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"gpt_oss_120b"},
        request_mode="chat",
        ignore_waits=True,
    )

    assert len(sessions) == 1
    turn = sessions[0].turns[0]

    # Estimated context_len should include the 500k asset_token_count
    assert turn.context_len >= 500000

    # Actual context_len should be much smaller — just the rendered text
    # "[TABLE]" is ~1 token + "Analyze this codebase" is ~3 tokens
    assert turn.actual_context_len < 100
    assert turn.actual_context_len > 0

    # The gap proves the measurement works
    assert turn.context_len > turn.actual_context_len * 100


def test_depth_telemetry_in_benchmark_result(tmp_path: Path) -> None:
    """Verify depth_telemetry block is emitted in benchmark results."""
    export_file = tmp_path / "multiturn.json"
    export_file.write_text(json.dumps(_multiturn_payload()))

    sessions, selection = load_replay_sessions(
        export_file=str(export_file),
        count_text_tokens=_count_tokens,
        runtime_stack_ids={"standalone:sglang"},
        hardware_profile_ids={"nvidia:h200_sxm_141gb"},
        canonical_model_ids={"qwen3_30b_a3b"},
        request_mode="chat",
        ignore_waits=True,
    )

    async def _run() -> dict:
        runner, base_url = await _start_mock_server()
        try:
            return await run_export_replay_benchmark(
                sessions=sessions,
                selection_metadata=selection,
                model_id="Qwen/Qwen3-30B-A3B",
                model_name=None,
                chat_api_url=f"{base_url}/v1/chat/completions",
                completion_api_url=f"{base_url}/v1/completions",
                count_text_tokens=_count_tokens,
                max_concurrency=1,
                selected_percentiles=[99],
                disable_tqdm=True,
                num_warmup_sessions=0,
            )
        finally:
            await runner.cleanup()

    result = asyncio.run(_run())

    # depth_telemetry block must exist
    assert "depth_telemetry" in result
    dt = result["depth_telemetry"]
    assert "total_estimated_input_tokens" in dt
    assert "total_actual_input_tokens" in dt
    assert "max_actual_context_len_per_turn" in dt
    assert dt["total_actual_input_tokens"] > 0
    assert dt["max_actual_context_len_per_turn"] > 0

    # Aggregate metrics must also carry actual input tokens
    agg = result["aggregate_metrics"]
    assert "total_actual_input_tokens" in agg
    assert "max_actual_context_len_per_turn" in agg

    # Per-turn metrics should have actual context length
    for turn_key, turn_metrics in result["per_turn_metrics"].items():
        assert "mean_actual_context_len" in turn_metrics
