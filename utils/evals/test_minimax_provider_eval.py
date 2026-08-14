import builtins
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import minimax_provider_eval as mpe


def _language_response(content: str = "お正月に子どもへ渡します。") -> dict[str, Any]:
    return {
        "id": "language-response",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content, "reasoning": ""},
            }
        ],
    }


def _tool_response(
    *, arguments: str | None = None, finish_reason: str = "tool_calls"
) -> dict[str, Any]:
    if finish_reason != "tool_calls":
        return {
            "id": "tool-response",
            "choices": [
                {"finish_reason": finish_reason, "message": {"content": "done"}}
            ],
        }
    if arguments is None:
        arguments = json.dumps(
            {
                "patient_id": "P-009417",
                "med_list_path": "/mnt/clinical/emr/patients/P-009417/med_list_v3.json",
                "classification_scheme": "rxnorm_ingredient",
                "overlap_policy": "current_only",
                "strict_route_matching": True,
                "include_otc": True,
                "ignore_statuses": ["discontinued", "on_hold"],
                "output_format": "detailed_json",
            }
        )
    return {
        "id": "tool-response",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "flag_duplicate_therapies",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
            }
        ],
    }


def _scenario_response(
    content: str = "123, some-parameter, xyz, another-parameter",
) -> dict[str, Any]:
    return {
        "id": "scenario-response",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
    }


def _response_for(payload: dict[str, Any]) -> dict[str, Any]:
    tools = payload.get("tools", [])
    if not tools:
        return _language_response()
    tool_name = tools[0]["function"]["name"]
    if tool_name == "flag_duplicate_therapies":
        return _tool_response()
    assert tool_name == "example"
    return _scenario_response()


def _compatibility(output_dir: Path) -> dict[str, Any]:
    paths = list(output_dir.glob(mpe.COMPATIBILITY_GLOB))
    assert len(paths) == 1
    assert re.fullmatch(
        r"results_minimax_vendor_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}\.json",
        paths[0].name,
    )
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _native(output_dir: Path) -> dict[str, Any]:
    return json.loads(
        (output_dir / mpe.NATIVE_REPORT_FILENAME).read_text(encoding="utf-8")
    )


def _score(output_dir: Path) -> float:
    return _compatibility(output_dir)["results"][mpe.TASK_NAME][
        "exact_match,strict-match"
    ]


def _run(
    output_dir: Path,
    post: Any = _response_for,
    **kwargs: Any,
) -> bool:
    def http_post(**request: Any) -> Any:
        return post(request["payload"])

    return mpe.run_evaluation(
        base_url="http://127.0.0.1:8000/v1/",
        api_key="secret",
        model="MiniMax-M3",
        output_dir=output_dir,
        http_post=http_post,
        **kwargs,
    )


def test_success_writes_complete_native_and_compatibility_reports(
    tmp_path: Path,
) -> None:
    invocations: list[dict[str, Any]] = []

    def http_post(**request: Any) -> dict[str, Any]:
        invocations.append(request)
        return _response_for(request["payload"])

    output_dir = tmp_path / "output"
    assert mpe.run_evaluation(
        base_url="http://127.0.0.1:8000/v1/",
        api_key="secret",
        model="MiniMax-M3",
        output_dir=output_dir,
        http_post=http_post,
    )

    assert len(invocations) == 3
    assert [
        call["payload"].get("tools", [{}])[0].get("function", {}).get("name")
        if call["payload"].get("tools")
        else None
        for call in invocations
    ] == [None, "flag_duplicate_therapies", "example"]
    for call in invocations:
        assert call["url"] == "http://127.0.0.1:8000/v1/chat/completions"
        assert call["headers"]["Authorization"] == "Bearer secret"
        assert call["timeout_seconds"] <= mpe.DEFAULT_REQUEST_TIMEOUT_SECONDS
        assert call["payload"]["model"] == "MiniMax-M3"
        assert call["payload"]["temperature"] == 0
        assert call["payload"]["top_p"] == 1
        assert call["payload"]["max_tokens"] == 2048
        assert "data_index" not in call["payload"]
        assert "check_type" not in call["payload"]
        assert "expected_tool_call" not in call["payload"]

    native = _native(output_dir)
    assert native["verifier"] == mpe.ADAPTER_NAME
    assert native["task"] == mpe.TASK_NAME
    assert native["completed"] is True
    assert native["threshold"] == 1.0
    assert native["source"] == {
        "url": mpe.UPSTREAM_SOURCE,
        "ref": mpe.UPSTREAM_REF,
        "indices": [0, 71, 101],
    }
    assert native["summary"]["total"] == 3
    assert native["summary"]["passed_count"] == 3
    assert native["summary"]["overall_compatibility_score"] == 1.0
    assert native["metrics"] == {
        "Query-Success-Rate": 1.0,
        "ToolCalls-Trigger-Similarity": 1.0,
        "ToolCalls-Schema-Accuracy": 1.0,
        "Error-Only-Reasoning-Rate": 0.0,
        "Language-Following-Success-Rate": 1.0,
        "Scenario-Check-Pass-Rate": 1.0,
    }
    assert [result["data_index"] for result in native["results"]] == [0, 71, 101]
    assert [result["response"]["id"] for result in native["results"]] == [
        "language-response",
        "tool-response",
        "scenario-response",
    ]

    compatibility = _compatibility(output_dir)
    assert compatibility["result_format"] == mpe.RESULT_FORMAT
    assert compatibility["eval_adapter"] == mpe.ADAPTER_NAME
    assert compatibility["model_name"] == "MiniMax-M3"
    assert _score(output_dir) == 1.0
    assert compatibility["n-samples"][mpe.TASK_NAME] == {
        "original": 3,
        "effective": 3,
    }
    assert "secret" not in json.dumps([native, compatibility])


def test_schema_failure_fails_only_tool_case(tmp_path: Path) -> None:
    def post(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("tools", [{}])[0].get("function", {}).get("name") == (
            "flag_duplicate_therapies"
        ):
            return _tool_response(arguments="{}")
        return _response_for(payload)

    output_dir = tmp_path / "output"
    assert not _run(output_dir, post)
    native = _native(output_dir)
    tool_result = native["results"][1]
    assert tool_result["tool_calls_valid"] is False
    assert tool_result["failures"] == ["tool_call_schema"]
    assert native["metrics"]["ToolCalls-Schema-Accuracy"] == 0.0
    assert native["summary"]["passed_count"] == 2
    assert _score(output_dir) == pytest.approx(2 / 3)


def test_trigger_failure_uses_expected_label(tmp_path: Path) -> None:
    def post(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("tools", [{}])[0].get("function", {}).get("name") == (
            "flag_duplicate_therapies"
        ):
            return _tool_response(finish_reason="stop")
        return _response_for(payload)

    output_dir = tmp_path / "output"
    assert not _run(output_dir, post)
    native = _native(output_dir)
    assert native["results"][1]["failures"] == ["tool_call_trigger"]
    assert native["metrics"]["ToolCalls-Trigger-Similarity"] == 0.0
    assert native["summary"]["tool_calls_finish_stop"] == 1
    assert native["summary"]["stop_finish_stop"] == 1


def test_negative_trigger_requires_stop_finish_reason(tmp_path: Path) -> None:
    def post(payload: dict[str, Any]) -> dict[str, Any]:
        tools = payload.get("tools", [])
        if tools and tools[0]["function"]["name"] == "example":
            response = _scenario_response()
            response["choices"][0]["finish_reason"] = "length"
            return response
        return _response_for(payload)

    output_dir = tmp_path / "output"
    assert not _run(output_dir, post)
    native = _native(output_dir)
    assert native["results"][2]["failures"] == ["tool_call_trigger"]
    assert native["summary"]["stop_finish_stop"] == 0
    assert native["metrics"]["ToolCalls-Trigger-Similarity"] == 1.0


def test_language_failure_uses_pinned_cyrillic_range(tmp_path: Path) -> None:
    def post(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("tools"):
            return _language_response("Это ответ")
        return _response_for(payload)

    output_dir = tmp_path / "output"
    assert not _run(output_dir, post)
    native = _native(output_dir)
    assert native["results"][0]["language_following_checked"] is True
    assert native["results"][0]["language_following_valid"] is False
    assert native["results"][0]["failures"] == ["language_following"]
    assert native["metrics"]["Language-Following-Success-Rate"] == 0.0


def test_scenario_failure_uses_visible_first_occurrence_order(tmp_path: Path) -> None:
    def post(payload: dict[str, Any]) -> dict[str, Any]:
        tools = payload.get("tools", [])
        if tools and tools[0]["function"]["name"] == "example":
            return _scenario_response(
                "<think>123 some-parameter</think> xyz then some-parameter"
            )
        return _response_for(payload)

    output_dir = tmp_path / "output"
    assert not _run(output_dir, post)
    result = _native(output_dir)["results"][2]
    assert result["scenario_check_checked"] is True
    assert result["scenario_check_detail"] == {
        "expected": ["123", "some-parameter", "xyz", "another-parameter"],
        "actual": ["xyz", "some-parameter"],
    }
    assert result["scenario_check_valid"] is False
    assert result["failures"] == ["scenario_check"]


def test_transport_retries_once_then_preserves_success(tmp_path: Path) -> None:
    attempts = 0

    def http_post(**request: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("connection reset")
        return _response_for(request["payload"])

    output_dir = tmp_path / "output"
    assert mpe.run_evaluation(
        base_url="https://provider.example/v1",
        api_key="secret",
        model="MiniMax-M3",
        output_dir=output_dir,
        http_post=http_post,
    )
    assert attempts == 4
    assert _native(output_dir)["results"][0]["attempts"] == 2
    assert _score(output_dir) == 1.0


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    (
        (400, ValueError),
        (408, ValueError),
        (429, mpe.TransportError),
        (503, mpe.TransportError),
    ),
)
def test_default_http_post_retries_only_retryable_http_statuses(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    error_type: type[BaseException],
) -> None:
    def fail_request(*args: Any, **kwargs: Any) -> Any:
        raise mpe.urllib.error.HTTPError(
            "https://provider.example/v1/chat/completions",
            status_code,
            "request failed",
            {},
            io.BytesIO(b"provider rejected request"),
        )

    monkeypatch.setattr(mpe._NO_REDIRECT_OPENER, "open", fail_request)

    with pytest.raises(error_type):
        mpe._default_http_post(
            url="https://provider.example/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            payload={"model": "MiniMax-M3", "messages": []},
            timeout_seconds=1,
        )


def test_default_http_post_rejects_redirect_without_leaking_authorization() -> None:
    received_authorization: list[str | None] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def record_request(self) -> None:
            received_authorization.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices":[]}')

        do_GET = record_request
        do_POST = record_request

        def log_message(self, *args: Any) -> None:
            return None

    target_server = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_url = f"http://127.0.0.1:{target_server.server_address[1]}/credential-target"

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(302)
            self.send_header("Location", target_url)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            return None

    redirect_server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
    redirect_thread = threading.Thread(
        target=redirect_server.serve_forever,
        daemon=True,
    )
    target_thread.start()
    redirect_thread.start()
    try:
        with pytest.raises(ValueError, match="HTTP 302"):
            mpe._default_http_post(
                url=(
                    f"http://127.0.0.1:{redirect_server.server_address[1]}"
                    "/v1/chat/completions"
                ),
                headers={"Authorization": "Bearer secret"},
                payload={"model": "MiniMax-M3", "messages": []},
                timeout_seconds=1,
            )
    finally:
        redirect_server.shutdown()
        target_server.shutdown()
        redirect_server.server_close()
        target_server.server_close()
        redirect_thread.join()
        target_thread.join()

    assert received_authorization == []


@pytest.mark.parametrize(
    "malformed",
    (
        None,
        [],
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"finish_reason": "stop"}]},
        {
            "choices": [
                {"finish_reason": "stop", "message": {"content": {"not": "text"}}}
            ]
        },
        {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": {}}}]},
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "123, some-parameter, xyz, another-parameter",
                        "tool_calls": [{"id": "unexpected"}],
                    },
                }
            ]
        },
    ),
)
def test_malformed_chat_response_records_diagnostic_and_continues(
    tmp_path: Path, malformed: Any
) -> None:
    calls = 0

    def post(payload: dict[str, Any]) -> Any:
        nonlocal calls
        calls += 1
        return malformed

    output_dir = tmp_path / "output"
    assert not _run(output_dir, post)
    native = _native(output_dir)
    assert calls == 3
    assert len(native["results"]) == 3
    for result in native["results"]:
        assert result["status"] == "failed"
        assert result["failures"][0] == "query_failed"
        assert result["response"]["error"]["type"] in {"TypeError", "ValueError"}


def test_exhausted_transport_failure_does_not_skip_later_cases(tmp_path: Path) -> None:
    calls = 0

    def http_post(**request: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise mpe.TransportError("offline")
        return _response_for(request["payload"])

    output_dir = tmp_path / "output"
    assert not mpe.run_evaluation(
        base_url="https://provider.example/v1",
        api_key="secret",
        model="MiniMax-M3",
        output_dir=output_dir,
        http_post=http_post,
    )
    native = _native(output_dir)
    assert calls == 4
    assert len(native["results"]) == 3
    assert native["results"][0]["failures"] == [
        "query_failed",
        "language_following",
    ]
    assert native["results"][1]["case_passed"] is True
    assert native["results"][2]["case_passed"] is True
    assert native["metrics"]["Query-Success-Rate"] == pytest.approx(2 / 3)


def test_global_deadline_caps_attempts_and_publishes_partial_report(
    tmp_path: Path,
) -> None:
    now = [0.0]
    timeouts: list[float] = []

    def clock() -> float:
        return now[0]

    def http_post(**request: Any) -> dict[str, Any]:
        timeouts.append(request["timeout_seconds"])
        now[0] += 0.7
        return _response_for(request["payload"])

    output_dir = tmp_path / "output"
    assert not mpe.run_evaluation(
        base_url="https://provider.example/v1",
        api_key="secret",
        model="MiniMax-M3",
        output_dir=output_dir,
        timeout_seconds=1.0,
        request_timeout_seconds=180,
        http_post=http_post,
        clock=clock,
    )
    native = _native(output_dir)
    assert timeouts == pytest.approx([1.0, 0.3])
    assert len(native["results"]) == 3
    assert native["results"][0]["case_passed"] is True
    assert native["results"][1]["suite_timed_out"] is True
    assert native["results"][2]["suite_timed_out"] is True
    assert native["completed"] is False
    assert native["integration_error"]["type"] == "SuiteTimeoutError"
    compatibility = _compatibility(output_dir)
    assert _score(output_dir) == 0.0
    assert compatibility["n-samples"][mpe.TASK_NAME]["effective"] == 1
    assert compatibility["integration_error"]["type"] == "SuiteTimeoutError"


def test_default_http_post_bounds_a_drip_feed_body_by_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    class FakeSocket:
        def __init__(self) -> None:
            self.timeouts: list[float] = []

        def settimeout(self, timeout: float) -> None:
            self.timeouts.append(timeout)

    class DripResponse:
        def __init__(self) -> None:
            self.socket = FakeSocket()
            self.fp = type(
                "Raw", (), {"raw": type("Socket", (), {"_sock": self.socket})()}
            )()
            self.reads = 0

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            return self.read1(size)

        def read1(self, size: int) -> bytes:
            self.reads += 1
            now[0] += 0.02
            return b"x"

    response = DripResponse()
    monkeypatch.setattr(mpe.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        mpe._NO_REDIRECT_OPENER,
        "open",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(mpe.TransportError, match="deadline"):
        mpe._default_http_post(
            url="https://provider.example/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            payload={"model": "MiniMax-M3", "messages": []},
            timeout_seconds=0.05,
        )

    assert response.reads == 3
    assert response.socket.timeouts == pytest.approx([0.05, 0.03, 0.01])


def test_reasoning_only_response_is_always_checked(tmp_path: Path) -> None:
    def post(payload: dict[str, Any]) -> dict[str, Any]:
        if not payload.get("tools"):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "reasoning": "I could not answer",
                            "content": "",
                            "tool_calls": [],
                        },
                    }
                ]
            }
        return _response_for(payload)

    output_dir = tmp_path / "output"
    assert not _run(output_dir, post)
    native = _native(output_dir)
    assert native["results"][0]["error_only_reasoning"] is True
    assert native["results"][0]["failures"] == [
        "error_only_reasoning",
        "language_following",
    ]
    assert native["metrics"]["Error-Only-Reasoning-Rate"] == pytest.approx(1 / 3)


def test_stale_artifacts_are_removed_without_touching_foreign_results(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / mpe.NATIVE_REPORT_FILENAME).write_text("stale")
    for stamp in ("2000-01-01T00-00-00.000000", "2001-01-01T00-00-00.000000"):
        (output_dir / f"results_minimax_vendor_{stamp}.json").write_text("stale")
    foreign = output_dir / "results_kimi_vendor_keep.json"
    foreign.write_text("keep")

    assert _run(output_dir)
    assert len(list(output_dir.glob(mpe.COMPATIBILITY_GLOB))) == 1
    assert _native(output_dir)["completed"] is True
    assert foreign.read_text() == "keep"


def test_integration_error_cli_is_dependency_free_and_writes_both_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "jsonschema" or name.startswith("jsonschema."):
            raise ImportError("jsonschema setup failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert (
        mpe.main(
            [
                "--model",
                "MiniMax-M3",
                "--output-dir",
                str(output_dir),
                "--integration-error",
                "dependency installation failed",
            ]
        )
        == 0
    )

    native = _native(output_dir)
    assert native["completed"] is False
    assert len(native["results"]) == 3
    assert native["integration_error"] == {
        "type": "RuntimeError",
        "message": "dependency installation failed",
    }
    compatibility = _compatibility(output_dir)
    assert _score(output_dir) == 0.0
    assert compatibility["n-samples"][mpe.TASK_NAME] == {
        "original": 3,
        "effective": 0,
    }
    assert compatibility["integration_error"]["message"] == (
        "dependency installation failed"
    )


def test_invalid_runtime_input_writes_zero_score_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    called = False

    def http_post(**request: Any) -> Any:
        nonlocal called
        called = True
        return _response_for(request["payload"])

    assert not mpe.run_evaluation(
        base_url="provider-without-a-scheme",
        api_key="secret",
        model="MiniMax-M3",
        output_dir=output_dir,
        http_post=http_post,
    )
    assert called is False
    assert _native(output_dir)["integration_error"]["type"] == "ValueError"
    assert _score(output_dir) == 0.0
    assert _compatibility(output_dir)["n-samples"][mpe.TASK_NAME]["effective"] == 0


@pytest.mark.parametrize(
    "field",
    ["indices", "ref", "license", "rows", "prompt", "tool_schema"],
)
def test_fixture_rejects_unpinned_or_incomplete_input(
    tmp_path: Path, field: str
) -> None:
    fixture = json.loads(mpe.DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8"))
    if field == "indices":
        fixture[field] = [0, 71]
    elif field == "ref":
        fixture[field] = "main"
    elif field == "license":
        fixture[field] = "MIT License"
    elif field == "rows":
        fixture[field] = fixture[field][:-1]
    elif field == "prompt":
        fixture["rows"][0]["messages"][0]["content"] = "changed"
    else:
        fixture["rows"][1]["tools"][0]["function"]["parameters"]["type"] = "array"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(ValueError):
        mpe.load_fixture(path)


def test_cli_validates_required_url_and_positive_bounds(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        mpe.parse_args(["--model", "MiniMax-M3", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        mpe.parse_args(
            [
                "--model",
                "MiniMax-M3",
                "--output-dir",
                str(tmp_path),
                "--base-url",
                "https://provider.example/v1",
                "--timeout-seconds",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        mpe.parse_args(
            [
                "--model",
                "MiniMax-M3",
                "--output-dir",
                str(tmp_path),
                "--base-url",
                "https://provider.example/v1",
                "--request-timeout-seconds",
                "nan",
            ]
        )
