#!/usr/bin/env python3
"""Run the pinned three-case MiniMax M3 provider compatibility smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_NAME = "minimax_m3_smoke"
NATIVE_REPORT_FILENAME = "minimax_vendor_report.json"
COMPATIBILITY_GLOB = "results_minimax_vendor_*.json"
DEFAULT_FIXTURE_PATH = Path(__file__).with_name("minimax_m3_smoke.json")
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
DEFAULT_TIMEOUT_SECONDS = 900.0
M3_DEFAULT_MAX_TOKENS = 40960
RESULT_FORMAT = "inferencex-eval-v1"
ADAPTER_NAME = "minimax-provider-verifier"
EXPECTED_INDICES = (0, 71, 101)
UPSTREAM_REF = "85bf180e54e2ab0b31595cfdc697116c4760876d"
UPSTREAM_SOURCE = (
    "https://raw.githubusercontent.com/MiniMax-AI/MiniMax-Provider-Verifier/"
    f"{UPSTREAM_REF}/sample.jsonl"
)
EXPECTED_LICENSE_SHA256 = (
    "aa7cec386fcb5e555aba0e8b1c31307940af41967708c9bc0f78b4e02e235dd5"
)
EXPECTED_CASE_SHA256 = {
    0: "655d3135fc553b08c376f363165699548396428136fd9536345cf37b564b357a",
    71: "10272004ae08f4a7d08d2306404f6cbb7bbfa794230e1082a235ded036d550ed",
    101: "10c3c2bf8d4e43d520de8ef3955cda1dcdd852c9c904decbc7d8cd040431afd4",
}
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_ATTEMPTS = 2

HttpPost = Callable[..., Any]
Clock = Callable[[], float]


class TransportError(OSError):
    """An HTTP transport failure that may be retried once."""


class SuiteTimeoutError(TimeoutError):
    """The global MiniMax smoke deadline was exhausted."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Keep bearer credentials on the configured endpoint."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_RejectRedirects())


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _positive_float(value: str) -> float:
    try:
        return _positive_number(float(value), "value")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc


def _validate_messages(value: Any, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    for index, message in enumerate(value):
        item = _mapping(message, f"{name}[{index}]")
        if not isinstance(item.get("role"), str) or not isinstance(
            item.get("content"), str
        ):
            raise TypeError(f"{name}[{index}] must contain string role and content")


def _validate_tools(value: Any, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    for index, tool in enumerate(value):
        item = _mapping(tool, f"{name}[{index}]")
        function = _mapping(item.get("function"), f"{name}[{index}].function")
        if item.get("type") != "function" or not isinstance(function.get("name"), str):
            raise ValueError(f"{name}[{index}] must define a named function")
        parameters = _mapping(
            function.get("parameters"), f"{name}[{index}].function.parameters"
        )
        _mapping(
            parameters.get("properties"),
            f"{name}[{index}].function.parameters.properties",
        )


def load_fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate the exact pinned three-case fixture."""
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "fixture")
    if root.get("source") != UPSTREAM_SOURCE or root.get("ref") != UPSTREAM_REF:
        raise ValueError("fixture source or ref does not match the pinned upstream")
    if root.get("indices") != list(EXPECTED_INDICES):
        raise ValueError("fixture indices must be exactly [0, 71, 101]")
    license_text = root.get("license")
    if (
        not isinstance(license_text, str)
        or hashlib.sha256(license_text.encode()).hexdigest() != EXPECTED_LICENSE_SHA256
    ):
        raise ValueError("fixture must preserve the complete upstream MIT notice")

    raw_rows = root.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(EXPECTED_INDICES):
        raise ValueError("fixture must contain exactly three rows")

    rows: list[dict[str, Any]] = []
    expected_checks = {
        0: ["contains_russian_characters_unicode"],
        71: [],
        101: ["scenario_check"],
    }
    for position, raw_row in enumerate(raw_rows):
        row = dict(_mapping(raw_row, f"fixture.rows[{position}]"))
        data_index = row.get("data_index")
        if data_index != EXPECTED_INDICES[position]:
            raise ValueError("fixture rows must retain upstream order and data_index")
        case_digest = hashlib.sha256(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if case_digest != EXPECTED_CASE_SHA256[data_index]:
            raise ValueError(f"fixture row {data_index} differs from pinned upstream")
        _validate_messages(row.get("messages"), f"fixture.rows[{position}].messages")
        check_types = row.get("check_type", [])
        if check_types != expected_checks[data_index]:
            raise ValueError(f"fixture row {data_index} has unexpected check_type")
        if data_index == 0:
            if "expected_tool_call" in row or "tools" in row:
                raise ValueError("fixture row 0 must remain the language-only case")
        else:
            _validate_tools(row.get("tools"), f"fixture.rows[{position}].tools")
            expected_label = data_index == 71
            if row.get("expected_tool_call") is not expected_label:
                raise ValueError(
                    f"fixture row {data_index} has an invalid expected label"
                )
        rows.append(copy.deepcopy(row))

    return dict(root), rows


def build_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    normalized = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    return f"{normalized}/chat/completions"


def prepare_request(row: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Strip evaluator fields and apply the fixed smoke sampling overrides."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    request = copy.deepcopy(dict(row))
    for field in ("data_index", "check_type", "expected_tool_call", "scenario_check"):
        request.pop(field, None)
    request.update(
        model=model,
        temperature=0,
        top_p=1,
        max_tokens=M3_DEFAULT_MAX_TOKENS,
    )
    return request


def _read_response_body(response: Any, deadline: float) -> bytes:
    chunks: list[bytes] = []
    total = 0
    read_chunk = getattr(response, "read1", response.read)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("chat completion response exceeded its deadline")
        sock = getattr(
            getattr(getattr(response, "fp", None), "raw", None),
            "_sock",
            None,
        )
        if sock is not None:
            sock.settimeout(remaining)
        chunk = read_chunk(64 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise ValueError(
                f"chat completion response exceeds {MAX_RESPONSE_BYTES} bytes"
            )


def _default_http_post(
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=timeout_seconds) as response:
            content = _read_response_body(response, deadline).decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 429 or 500 <= exc.code < 600:
            raise TransportError(f"HTTP {exc.code}: {exc.reason}") from exc
        raise ValueError(
            f"chat completion request failed with HTTP {exc.code}: {exc.reason}"
        ) from exc
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        OSError,
    ) as exc:
        raise TransportError(str(exc)) from exc
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"chat completion response is not valid JSON: {exc}") from exc


def _validate_chat_completion_response(value: Any) -> Mapping[str, Any]:
    response = _mapping(value, "chat completion response")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            "chat completion response must contain a non-empty choices array"
        )
    choice = _mapping(choices[0], "chat completion response.choices[0]")
    if not isinstance(choice.get("finish_reason"), str):
        raise TypeError("chat completion response must contain a finish_reason")
    message = _mapping(
        choice.get("message"),
        "chat completion response.choices[0].message",
    )
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError("chat completion message content must be a string or null")
    tool_calls = message.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise TypeError("chat completion message tool_calls must be an array")
    if choice["finish_reason"] == "tool_calls":
        if not tool_calls:
            raise ValueError(
                "tool_calls finish reason requires at least one message tool call"
            )
    elif tool_calls:
        raise ValueError("message tool calls require a tool_calls finish reason")
    return response


# Adapted verbatim from pinned validator/tool_calls.py.
_COMMON_COMMANDS = [
    "ls ",
    "cat ",
    "git ",
    "npm ",
    "npx ",
    "cd ",
    "cp ",
    "mv ",
    "rm ",
    "mkdir ",
    "chmod ",
    "chown ",
    "find ",
    "grep ",
    "curl ",
    "wget ",
    "pip ",
]


def _is_shell_c_invocation(cmd: list[Any]) -> bool:
    if not cmd or len(cmd) < 3:
        return False
    shell = cmd[0]
    if shell not in (
        "bash",
        "sh",
        "zsh",
        "/bin/bash",
        "/bin/sh",
        "/bin/zsh",
        "/usr/bin/bash",
        "/usr/bin/sh",
        "/usr/bin/zsh",
    ):
        return False
    for arg in cmd[1:]:
        if arg in ("-c", "-lc"):
            return True
        if arg in ("-l", "--login"):
            continue
        break
    return False


def is_valid_array_command(cmd: Any) -> bool:
    if not isinstance(cmd, list) or len(cmd) == 0:
        return False
    if _is_shell_c_invocation(cmd):
        return True
    for elem in cmd:
        if not isinstance(elem, str):
            return False
        if " " in elem:
            for prefix in _COMMON_COMMANDS:
                if elem.startswith(prefix):
                    return False
    return not (len(cmd) == 1 and " " in cmd[0])


def validate_tool_call(tool_call: Any, tools: list[dict[str, Any]]) -> bool:
    """Apply pinned JSON Schema and array-command validation lazily."""
    try:
        # Lazy by design: --integration-error must work if dependency setup failed.
        from jsonschema import ValidationError, validate
    except ImportError:
        return False

    try:
        call = _mapping(tool_call, "tool_call")
        function = _mapping(call["function"], "tool_call.function")
        tool_name = function["name"]
        schema = next(
            (
                tool["function"]["parameters"]
                for tool in tools
                if tool["function"]["name"] == tool_name
            ),
            None,
        )
        if not schema:
            return False
        args = function["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        validate(instance=args, schema=schema)
        for param_name, param_schema in schema.get("properties", {}).items():
            if (
                param_name == "command"
                and param_schema.get("type") == "array"
                and param_schema.get("items", {}).get("type") == "string"
            ):
                cmd_value = args.get(param_name)
                if cmd_value is not None and not is_valid_array_command(cmd_value):
                    return False
        return True
    except (json.JSONDecodeError, ValidationError):
        return False
    except Exception:  # noqa: BLE001 - upstream data can fail in arbitrary shapes
        return False


def validate_tool_calls(
    request: dict[str, Any], response: Any, status: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool_calls_finish_reason": None,
        "tool_calls_valid": None,
        "tool_calls_count": 0,
    }
    if status != "success" or not response or "choices" not in response:
        return result
    choice = response["choices"][0] if response["choices"] else {}
    finish_reason = choice.get("finish_reason")
    result["tool_calls_finish_reason"] = finish_reason
    if finish_reason == "tool_calls":
        tools = request.get("tools", [])
        tool_calls = choice.get("message", {}).get("tool_calls", [])
        result["tool_calls_count"] = len(tool_calls)
        if tool_calls:
            result["tool_calls_valid"] = all(
                validate_tool_call(tool_call, tools) for tool_call in tool_calls
            )
        else:
            result["tool_calls_valid"] = False
    return result


# Adapted verbatim from pinned validator/russian_characters.py.
def not_contains_russian_characters_unicode(text: str) -> bool:
    for char in text:
        char_code = ord(char)
        if 0x0400 <= char_code <= 0x04FF:
            return False
    return True


def validate_language(status: str, resp_content: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "language_following_checked": False,
        "language_following_valid": None,
    }
    if status != "success" or not resp_content:
        return result
    result["language_following_checked"] = True
    result["language_following_valid"] = not_contains_russian_characters_unicode(
        resp_content
    )
    return result


# Adapted verbatim from pinned validator/scenario_check.py.
def _extract_expected_order(request: dict[str, Any]) -> list[str] | None:
    tools = request.get("tools")
    if not tools or not isinstance(tools, list):
        return None
    params = tools[0].get("function", {}).get("parameters", {})
    if not params:
        return None
    if "properties" in params:
        return list(params["properties"].keys())
    schema_keywords = {
        "type",
        "description",
        "required",
        "additionalProperties",
        "$schema",
        "items",
        "enum",
        "default",
    }
    keys = [key for key in params if key not in schema_keywords]
    return keys if keys else None


def _get_visible_content(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_actual_order(text: str, expected: list[str]) -> list[str]:
    positions = []
    for param in expected:
        index = text.find(param)
        if index != -1:
            positions.append((index, param))
    positions.sort(key=lambda item: item[0])
    return [param for _, param in positions]


def validate_scenario(
    request: dict[str, Any], status: str, resp_content: Any
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scenario_check_checked": False,
        "scenario_check_valid": None,
        "scenario_check_detail": None,
    }
    if status != "success" or not resp_content:
        return result
    expected_order = _extract_expected_order(request)
    if not expected_order:
        return result
    visible = _get_visible_content(resp_content)
    actual_order = _extract_actual_order(visible, expected_order)
    result["scenario_check_checked"] = True
    result["scenario_check_valid"] = (
        len(actual_order) >= 2 and actual_order == expected_order[: len(actual_order)]
    )
    result["scenario_check_detail"] = {
        "expected": expected_order,
        "actual": actual_order,
    }
    return result


# Adapted verbatim from pinned verify.py::_is_error_only_reasoning_response.
def _is_error_only_reasoning_response(response: Any) -> bool:
    try:
        if not response or "choices" not in response or not response["choices"]:
            return False
        message = response["choices"][0].get("message") or {}
        reasoning = message.get("reasoning") or ""
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            has_tool_calls = len(tool_calls) > 0
        else:
            has_tool_calls = bool(tool_calls)
        return bool(reasoning) and (not content) and (not has_tool_calls)
    except Exception:  # noqa: BLE001 - mirrors the pinned upstream guard
        return False


def _choice_fields(response: Any) -> tuple[Any, Any]:
    if not isinstance(response, Mapping):
        return None, None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None, None
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    return choice.get("finish_reason"), content


def _error_dict(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _evaluate_case(
    *,
    row: dict[str, Any],
    model: str,
    endpoint: str,
    api_key: str,
    request_timeout_seconds: float,
    deadline: float,
    http_post: HttpPost,
    clock: Clock,
) -> dict[str, Any]:
    prepared = prepare_request(row, model)
    started = clock()
    response: Any = None
    status = "failed"
    attempts = 0
    request_error: BaseException | None = None
    suite_timed_out = False

    for attempt in range(MAX_ATTEMPTS):
        remaining = deadline - clock()
        if remaining <= 0:
            request_error = SuiteTimeoutError("global suite timeout exceeded")
            suite_timed_out = True
            break
        attempts += 1
        try:
            raw_response = http_post(
                url=endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                payload=prepared,
                timeout_seconds=min(request_timeout_seconds, remaining),
            )
            if deadline - clock() <= 0:
                request_error = SuiteTimeoutError("global suite timeout exceeded")
                response = None
                suite_timed_out = True
                break
            response = copy.deepcopy(
                dict(_validate_chat_completion_response(raw_response))
            )
            status = "success"
            request_error = None
            break
        except (TransportError, TimeoutError, OSError) as exc:
            if deadline - clock() <= 0:
                request_error = SuiteTimeoutError("global suite timeout exceeded")
                suite_timed_out = True
                break
            request_error = exc
            if attempt + 1 < MAX_ATTEMPTS:
                continue
            break
        except Exception as exc:  # noqa: BLE001 - preserve per-request diagnostics
            request_error = exc
            if deadline - clock() <= 0:
                request_error = SuiteTimeoutError("global suite timeout exceeded")
                suite_timed_out = True
            break

    finish_reason, resp_content = _choice_fields(response)
    result: dict[str, Any] = {
        "data_index": row["data_index"],
        "status": status,
        "attempts": attempts,
        "duration_ms": round(max(0.0, clock() - started) * 1000, 3),
        "expected_tool_call": row.get("expected_tool_call"),
        "finish_reason": finish_reason,
        "response": response
        if response is not None
        else {"error": _error_dict(request_error or RuntimeError("request failed"))},
        "error_only_reasoning_checked": 1,
        "error_only_reasoning": _is_error_only_reasoning_response(response),
    }

    check_types = row.get("check_type", [])
    try:
        if check_types:
            if "contains_russian_characters_unicode" in check_types:
                result.update(validate_language(status, resp_content))
            if "scenario_check" in check_types:
                result.update(validate_scenario(prepared, status, resp_content))
        else:
            result.update(validate_tool_calls(prepared, response, status))
    except Exception as exc:  # noqa: BLE001 - validators must not abort the report
        result["validator_error"] = _error_dict(exc)

    failures: list[str] = []
    if status != "success":
        failures.append("query_failed")
    if result["error_only_reasoning"]:
        failures.append("error_only_reasoning")
    expected_tool_call = row.get("expected_tool_call")
    if isinstance(expected_tool_call, bool):
        expected_finish_reason = "tool_calls" if expected_tool_call else "stop"
        actual_tool_call = finish_reason == "tool_calls"
        if finish_reason != expected_finish_reason:
            failures.append("tool_call_trigger")
        if (
            expected_tool_call
            and actual_tool_call
            and result.get("tool_calls_valid") is not True
        ):
            failures.append("tool_call_schema")
    if (
        "contains_russian_characters_unicode" in check_types
        and result.get("language_following_valid") is not True
    ):
        failures.append("language_following")
    if (
        "scenario_check" in check_types
        and result.get("scenario_check_valid") is not True
    ):
        failures.append("scenario_check")
    if "validator_error" in result:
        failures.append("validator_error")

    result["case_passed"] = not failures
    result["failures"] = failures
    result["suite_timed_out"] = suite_timed_out
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _summarize(
    results: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float]]:
    total = len(results)
    success_count = sum(result.get("status") == "success" for result in results)
    passed_count = sum(result.get("case_passed") is True for result in results)

    labeled = [
        result
        for result in results
        if result.get("expected_tool_call") is True
        or result.get("expected_tool_call") is False
    ]
    true_positive = sum(
        result["expected_tool_call"] is True
        and result.get("finish_reason") == "tool_calls"
        for result in labeled
    )
    false_negative = sum(
        result["expected_tool_call"] is True
        and result.get("finish_reason") != "tool_calls"
        for result in labeled
    )
    false_positive = sum(
        result["expected_tool_call"] is False
        and result.get("finish_reason") == "tool_calls"
        for result in labeled
    )
    expected_tool_finish_stop = sum(
        result["expected_tool_call"] is True and result.get("finish_reason") == "stop"
        for result in labeled
    )
    expected_stop_finish_stop = sum(
        result["expected_tool_call"] is False and result.get("finish_reason") == "stop"
        for result in labeled
    )
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    trigger_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    schema_successes = sum(
        result.get("expected_tool_call") is True
        and result.get("finish_reason") == "tool_calls"
        and result.get("tool_calls_valid") is True
        for result in labeled
    )

    language_checked = sum(
        result.get("language_following_checked") is True for result in results
    )
    language_valid = sum(
        result.get("language_following_valid") is True for result in results
    )
    scenario_checked = sum(
        result.get("scenario_check_checked") is True for result in results
    )
    scenario_valid = sum(
        result.get("scenario_check_valid") is True for result in results
    )
    reasoning_errors = sum(
        result.get("error_only_reasoning") is True for result in results
    )

    metrics = {
        "Query-Success-Rate": _ratio(success_count, total),
        "ToolCalls-Trigger-Similarity": trigger_f1,
        "ToolCalls-Schema-Accuracy": _ratio(schema_successes, true_positive),
        "Error-Only-Reasoning-Rate": _ratio(reasoning_errors, total),
        "Language-Following-Success-Rate": _ratio(language_valid, language_checked),
        "Scenario-Check-Pass-Rate": _ratio(scenario_valid, scenario_checked),
    }
    summary: dict[str, Any] = {
        "total": total,
        "passed_count": passed_count,
        "failed_count": total - passed_count,
        "success_count": success_count,
        "failure_count": total - success_count,
        "tool_calls_finish_tool_calls": true_positive,
        "tool_calls_finish_stop": expected_tool_finish_stop,
        "stop_finish_tool_calls": false_positive,
        "stop_finish_stop": expected_stop_finish_stop,
        "expected_tool_call_total_count": len(labeled),
        "tool_calls_successful_count": schema_successes,
        "tool_calls_schema_validation_error_count": true_positive - schema_successes,
        "error_only_reasoning_checked_count": total,
        "error_only_reasoning_count": reasoning_errors,
        "language_following_checked_count": language_checked,
        "language_following_valid_count": language_valid,
        "language_following_invalid_count": language_checked - language_valid,
        "scenario_check_checked_count": scenario_checked,
        "scenario_check_valid_count": scenario_valid,
        "scenario_check_invalid_count": scenario_checked - scenario_valid,
        "overall_compatibility_score": _ratio(passed_count, len(EXPECTED_INDICES)),
    }
    return summary, metrics


def _native_report(
    *,
    model: str,
    endpoint: str | None,
    fixture_metadata: Mapping[str, Any] | None,
    results: list[dict[str, Any]],
    completed: bool,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    summary, metrics = _summarize(results)
    report: dict[str, Any] = {
        "verifier": ADAPTER_NAME,
        "task": TASK_NAME,
        "model": model,
        "endpoint": endpoint,
        "completed": completed,
        "threshold": 1.0,
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "max_tokens": M3_DEFAULT_MAX_TOKENS,
        },
        "source": {
            "url": (fixture_metadata or {}).get("source", UPSTREAM_SOURCE),
            "ref": (fixture_metadata or {}).get("ref", UPSTREAM_REF),
            "indices": list(EXPECTED_INDICES),
        },
        "summary": summary,
        "metrics": metrics,
        "results": results,
    }
    if integration_error is not None:
        report["integration_error"] = _error_dict(integration_error)
    return report


def _compatibility_result(
    model: str,
    score: float,
    *,
    n_samples: int,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": ADAPTER_NAME,
        "model_name": model,
        "results": {
            TASK_NAME: {
                "exact_match,strict-match": score,
                "exact_match_stderr,strict-match": 0.0,
            }
        },
        "configs": {
            TASK_NAME: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            }
        },
        "n-samples": {
            TASK_NAME: {
                "original": len(EXPECTED_INDICES),
                "effective": n_samples,
            }
        },
    }
    if integration_error is not None:
        result["integration_error"] = _error_dict(integration_error)
    return result


def prepare_compatibility_path(output_dir: Path) -> Path:
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        stale_path.unlink()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    return output_dir / f"results_minimax_vendor_{timestamp}.json"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _integration_results(exc: BaseException) -> list[dict[str, Any]]:
    return [
        {
            "data_index": data_index,
            "status": "failed",
            "attempts": 0,
            "expected_tool_call": True
            if data_index == 71
            else False
            if data_index == 101
            else None,
            "finish_reason": None,
            "response": {"error": _error_dict(exc)},
            "error_only_reasoning_checked": 1,
            "error_only_reasoning": False,
            "case_passed": False,
            "failures": ["integration_error"],
            "suite_timed_out": isinstance(exc, SuiteTimeoutError),
        }
        for data_index in EXPECTED_INDICES
    ]


def _failed_case_result(row: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "data_index": row["data_index"],
        "status": "failed",
        "attempts": 0,
        "expected_tool_call": row.get("expected_tool_call"),
        "finish_reason": None,
        "response": {"error": _error_dict(exc)},
        "error_only_reasoning_checked": 1,
        "error_only_reasoning": False,
        "case_passed": False,
        "failures": ["adapter_error"],
        "suite_timed_out": isinstance(exc, SuiteTimeoutError),
    }


def publish_integration_error(
    *, output_dir: Path, model: str, error: BaseException
) -> None:
    """Publish both required zero-score artifacts without loading jsonschema."""
    output_dir.mkdir(parents=True, exist_ok=True)
    native_path = output_dir / NATIVE_REPORT_FILENAME
    native_path.unlink(missing_ok=True)
    compatibility_path = prepare_compatibility_path(output_dir)
    _write_json(
        native_path,
        _native_report(
            model=model,
            endpoint=None,
            fixture_metadata=None,
            results=_integration_results(error),
            completed=False,
            integration_error=error,
        ),
    )
    _write_json(
        compatibility_path,
        _compatibility_result(model, 0.0, n_samples=0, integration_error=error),
    )


def run_evaluation(
    *,
    base_url: str,
    api_key: str,
    model: str,
    output_dir: Path,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    http_post: HttpPost = _default_http_post,
    clock: Clock = time.monotonic,
) -> bool:
    """Run all three cases sequentially and always publish both artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    native_path = output_dir / NATIVE_REPORT_FILENAME
    native_path.unlink(missing_ok=True)
    compatibility_path = prepare_compatibility_path(output_dir)

    try:
        request_timeout = _positive_number(
            request_timeout_seconds, "request_timeout_seconds"
        )
        suite_timeout = _positive_number(timeout_seconds, "timeout_seconds")
        if not callable(http_post) or not callable(clock):
            raise TypeError("http_post and clock must be callable")
        deadline = clock() + suite_timeout
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string")
        endpoint = build_endpoint(base_url)
        fixture_metadata, rows = load_fixture(fixture_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        _write_json(
            native_path,
            _native_report(
                model=model,
                endpoint=None,
                fixture_metadata=None,
                results=_integration_results(exc),
                completed=False,
                integration_error=exc,
            ),
        )
        _write_json(
            compatibility_path,
            _compatibility_result(model, 0.0, n_samples=0, integration_error=exc),
        )
        return False

    results: list[dict[str, Any]] = []
    for row in rows:
        try:
            result = _evaluate_case(
                row=row,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                request_timeout_seconds=request_timeout,
                deadline=deadline,
                http_post=http_post,
                clock=clock,
            )
        except Exception as exc:  # noqa: BLE001 - continue and report every case
            result = _failed_case_result(row, exc)
        results.append(result)

    timed_out = any(result["suite_timed_out"] for result in results)
    integration_error: BaseException | None = None
    completed = not timed_out
    if timed_out:
        integration_error = SuiteTimeoutError("global suite timeout exceeded")
    native = _native_report(
        model=model,
        endpoint=endpoint,
        fixture_metadata=fixture_metadata,
        results=results,
        completed=completed,
        integration_error=integration_error,
    )
    passed_count = native["summary"]["passed_count"]
    effective = sum(not result["suite_timed_out"] for result in results)
    score = passed_count / len(EXPECTED_INDICES) if completed else 0.0
    compatibility = _compatibility_result(
        model,
        score,
        n_samples=effective,
        integration_error=integration_error,
    )
    _write_json(native_path, native)
    _write_json(compatibility_path, compatibility)
    return completed and passed_count == len(EXPECTED_INDICES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned three-case MiniMax M3 provider smoke."
    )
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--timeout-seconds", type=_positive_float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--integration-error")
    args = parser.parse_args(argv)
    if args.integration_error is None and args.base_url is None:
        parser.error("--base-url required unless --integration-error is provided")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.integration_error is not None:
        publish_integration_error(
            output_dir=args.output_dir,
            model=args.model,
            error=RuntimeError(args.integration_error),
        )
        return 0
    passed = run_evaluation(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        fixture_path=args.fixture,
        request_timeout_seconds=args.request_timeout_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
