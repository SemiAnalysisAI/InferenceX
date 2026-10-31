import json
from pathlib import Path

import pytest

from infx.serving.greenspec import (
    ALLOWLIST,
    MAX_BYTES,
    META_KEYS,
    SERVER_LOG_STUB,
    ConsoleFilter,
    GreenSpecError,
    Settings,
    fix_required,
    fix_tools,
    guard,
    label,
    normalize,
    payload,
    select,
    settings,
    write_selection,
)


def _row(qid: str, category: str, bucket: str, lite: bool, **body) -> dict:
    return {
        "question_id": qid, "category": category, "bucket": bucket, "lite": lite, "isl": 100, "osl": 10,
        "thinking_src": "none", "recorded": {"temperature": 0.6, "max_completion_tokens": 4096},
        "messages": [{"role": "user", "content": f"q {qid}"}], **body,
    }


ROWS = [
    _row("m/aaaa", "agentic_tools", "agentic_tools__0-4k", True, tools=[{"type": "function"}]),
    _row("m/bbbb", "agentic_tools", "agentic_tools__4k-16k", False),
    _row("m/cccc", "llm_task", "llm_task__16k+", True),
]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "greenspec.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    return path


@pytest.mark.parametrize("chosen, expected", [
    (Settings(), ["m/aaaa", "m/bbbb", "m/cccc"]),
    (Settings(lite=True), ["m/aaaa", "m/cccc"]),
    (Settings(subset="agentic_tools"), ["m/aaaa", "m/bbbb"]),
    (Settings(subset="agentic_tools", lite=True), ["m/aaaa"]),
    (Settings(bucket="llm_task__16k+"), ["m/cccc"]),
])
def test_select(dataset, chosen, expected):
    assert [r["question_id"] for r in select(dataset, chosen)] == expected


@pytest.mark.parametrize("overrides, message", [
    ({"subset": "coding"}, "overrides.subset"),
    ({"lite": 1}, "overrides.lite"),
    ({"bucket": "16k"}, "overrides.bucket"),
    ({"mode": "greedy"}, "overrides.mode"),
    ({"max_tokens": 0}, "overrides.max_tokens"),
    ({"requests": True}, "overrides.requests"),
    ({"duration": 180}, "greenspec takes"),
])
def test_bad_overrides(overrides, message):
    with pytest.raises(GreenSpecError, match=message):
        normalize(overrides)


def test_label_names_every_non_default():
    assert label(Settings()) == "greenspec-diverse"
    chosen = normalize({"subset": "agentic_tools", "lite": True, "mode": "recorded", "max_tokens": 4096, "requests": 200})
    assert label(chosen) == "greenspec-agentic_tools-lite-recorded-mt4096-n200"
    assert label(normalize({"bucket": "llm_task__16k+"})) == "greenspec-diverse-llm_task_16kplus"


def test_settings_come_from_the_serving_yaml_of_the_job():
    yaml_text = "name: x\nscenario: greenspec\noverrides: {subset: llm_task, lite: true}\nconc: 4\n"
    assert settings({"SERVING_YAML": yaml_text}) == Settings(subset="llm_task", lite=True)
    assert settings({"SERVING_YAML": "name: x\nscenario: agentx\nconc: 4\n"}) is None
    assert settings({}) is None


def test_payload_target_default_keeps_the_row_and_caps_output():
    body = payload(ROWS[0], "served", Settings())
    assert list(body)[0] == "model" and body["model"] == "served"
    assert not set(META_KEYS) & set(body)
    assert body["max_tokens"] == 8192 and "temperature" not in body
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["tools"] == [{"type": "function"}]


def test_payload_recorded_replays_sampling_and_the_cap_only_lowers():
    low = payload(ROWS[0], "m", Settings(mode="recorded", max_tokens=2048))
    assert low["max_completion_tokens"] == 2048 and "max_tokens" not in low
    high = payload(ROWS[0], "m", Settings(mode="recorded"))
    assert high["max_completion_tokens"] == 4096 and high["temperature"] == 0.6


def test_fix_required_rewrites_the_keyword_at_every_schema_level():
    schema = {
        "type": "object",
        "required": {},
        "properties": {
            "a": {"type": "object", "required": {}, "properties": {"b": {"required": {}}}},
            "c": {"type": "array", "items": {"type": "object", "required": {}}},
            "d": {"anyOf": [{"required": {}}, {"type": "null"}]},
        },
        "$defs": {"e": {"required": {}}},
    }
    fixed = fix_required(schema)
    assert fixed["required"] == []
    assert fixed["properties"]["a"]["required"] == []
    assert fixed["properties"]["a"]["properties"]["b"]["required"] == []
    assert fixed["properties"]["c"]["items"]["required"] == []
    assert fixed["properties"]["d"]["anyOf"][0]["required"] == []
    assert fixed["$defs"]["e"]["required"] == []
    assert schema["required"] == {}  # the input is not mutated


def test_fix_required_leaves_a_property_named_required_and_other_values_alone():
    schema = {
        "type": "object",
        "required": ["x"],
        "properties": {
            "required": {},
            "x": {"type": "string", "enum": [{}], "default": {"required": {}}, "const": {"required": {}}},
        },
    }
    assert fix_required(schema) == schema


def test_fix_tools_only_touches_function_parameters():
    tools = [
        {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "required": {}}}},
        {"type": "function", "function": {"name": "g"}},
        {"type": "function"},
    ]
    fixed = fix_tools(tools)
    assert fixed[0]["function"]["parameters"]["required"] == []
    assert fixed[1:] == tools[1:]
    assert fix_tools(None) is None


def test_payload_sends_valid_tool_schemas():
    row = {**ROWS[0], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"required": {}}}}]}
    assert payload(row, "m", Settings())["tools"][0]["function"]["parameters"] == {"required": []}


def test_write_selection_one_body_per_line(dataset, tmp_path):
    out = tmp_path / "sel" / "selection.jsonl"
    assert write_selection(dataset, Settings(subset="agentic_tools", lite=True), "served", out) == 1
    assert [json.loads(line) for line in out.read_text().splitlines()] == [payload(ROWS[0], "served", Settings())]


def test_an_empty_selection_is_an_error(dataset, tmp_path):
    with pytest.raises(GreenSpecError, match="no row of the release matches"):
        write_selection(dataset, Settings(bucket="llm_task__0-1k"), "m", tmp_path / "out.jsonl")


CONSOLE = (
    "14:27:38.612 INFO     Starting AIPerf System (_single_run.py:64)\n"
    "Profiling:   0% |          | [00:00<?]\r"
    "Profiling:  40% |####      | [00:20<00:30]\r"
    "Profiling:  90% |######### | [00:50<00:05]14:27:46.424 WARNING  OSL mismatch: got 18 tokens\n"
    "14:27:46.500 ERROR    Error in SSE response: ValueError('secret prompt m/aaaa')\n"
    "Traceback (most recent call last):\n"
    "  ValueError: secret prompt m/aaaa\n"
    "\x1b[32mProfiling: 100% |##########| [00:55<00:00]\x1b[0m\r"
    "Processing Records: 40/40 |##########| 100% [00:01<00:00]\n"
    "                     NVIDIA AIPerf | Error Summary                     \n"
    "┏━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓\n"
    "┃ Code ┃        Type ┃ Message                      ┃ Count ┃\n"
    "┡━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩\n"
    "│  400 │ Bad Request │ {\"input_value\": \"secret m/b │ 1,203 │\n"
    "│      │             │ bbb\"}                        │       │\n"
    "│  N/A │         N/A │ timeout                      │     2 │\n"
    "└──────┴─────────────┴──────────────────────────────┴───────┘\n"
    "            NVIDIA AIPerf | LLM Metrics            \n"
    "│ Request Latency (ms) │ 12.00 │\n"
)


def _filtered(text: str, interval: float) -> list[str]:
    out: list[str] = []
    ticks = iter(range(0, 1000, 10))  # each bar line is 10 s after the previous one
    console = ConsoleFilter(out.append, interval, clock=lambda: next(ticks))
    for segment in text.replace("\r", "\n").split("\n"):
        console.feed(segment)
    console.close()
    return out


def test_console_withholds_error_text_and_throttles_bars():
    out = _filtered(CONSOLE, interval=15)
    assert "secret" not in "\n".join(out) and "aaaa" not in "\n".join(out)
    assert out == [
        "14:27:38.612 INFO     Starting AIPerf System (_single_run.py:64)",
        "GreenSpec progress: Profiling (0%) elapsed 00:00 eta ?",
        "GreenSpec progress: Profiling (90%) elapsed 00:50 eta 00:05",
        "14:27:46.424 WARNING  OSL mismatch: got 18 tokens",
        "14:27:46.500 ERROR    <withheld: may echo server error text>",
        "GreenSpec progress: Profiling (100%) elapsed 00:55 eta 00:00",
        "GreenSpec progress: Processing Records 40/40 (100%) elapsed 00:01 eta 00:00",
        "GreenSpec: aiperf error summary withheld (messages can echo requests): 400 x1203, N/A x2",
        "            NVIDIA AIPerf | LLM Metrics",
        "│ Request Latency (ms) │ 12.00 │",
    ]


def test_console_keeps_an_unterminated_error_table_dropped():
    out = _filtered(CONSOLE[: CONSOLE.index("│  N/A")], interval=0)
    assert "secret" not in "\n".join(out)
    assert out[-1] == "GreenSpec: aiperf error summary withheld (messages can echo requests): 400 x1203"


def _result_dir(tmp_path: Path) -> Path:
    result = tmp_path / "results"
    (result / "aiperf_artifacts").mkdir(parents=True)
    for name in ALLOWLIST:
        (result / name).write_text("{}")
    export = {"error_summary": [{"count": 1, "error_details": {"code": 400, "message": "x"}}]}
    (result / "aiperf_artifacts/profile_export_aiperf.json").write_text(json.dumps(export))
    (result / "aiperf_artifacts/inputs.json").write_text("prompts")
    (result / "aiperf_artifacts/profile_export.jsonl").write_text("{}")
    (result / ".dataset_path").write_text("/tmp/greenspec-selection/selection.jsonl")
    (result / "server.log").write_text("engine output")
    (result / "gpu_distribution.json").write_text("{}")
    return result


def test_guard_prunes_redacts_and_keeps_server_logs_on_the_node(dataset, tmp_path):
    result, keep = _result_dir(tmp_path), tmp_path / "kept"
    outside = tmp_path / "outside-server.log"
    outside.write_text("engine output")
    assert guard(dataset, result, [], [outside], keep_log_dir=keep) == []
    assert sorted(p.relative_to(result).as_posix() for p in result.rglob("*") if p.is_file()) == sorted(ALLOWLIST)
    redacted = json.loads((result / "aiperf_artifacts/profile_export_aiperf.json").read_text())
    assert redacted["error_summary"][0]["error_details"] == {"code": 400, "message": "<withheld>"}
    assert (keep / "server.log").read_text() == "engine output"
    assert outside.read_text() == SERVER_LOG_STUB and (keep / "outside-server.log").read_text() == "engine output"


def test_guard_stub_survives_a_server_still_writing(dataset, tmp_path):
    log = tmp_path / "server.log"
    log.write_text("engine output")
    with log.open("r+") as server:  # like `> server.log`: no O_APPEND
        server.seek(0, 2)
        assert guard(dataset, tmp_path / "none", [], [log]) == []
        server.write("late engine output")
    assert log.read_text() == SERVER_LOG_STUB


def test_guard_fails_on_a_row_id_or_a_big_file(dataset, tmp_path):
    result = tmp_path / "results"
    result.mkdir()
    (result / "scenario.yaml").write_text("note: m/bbbb")
    big = tmp_path / "aggregate.json"
    big.write_text("x" * (MAX_BYTES + 1))
    problems = guard(dataset, result, [big], [])
    assert any("question_id" in p for p in problems)
    assert any("bytes" in p for p in problems)


def test_tee_filtered_writes_the_filtered_console_to_stdout_and_the_log(tmp_path, capsys):
    import sys

    from infx.serving.greenspec import tee_filtered

    script = ("import sys; sys.stdout.write("
              "'14:27:46.500 ERROR    Error in SSE response: secret prompt\\nTraceback x\\nplain line\\n'); sys.exit(3)")
    log = tmp_path / "benchmark.log"
    assert tee_filtered([sys.executable, "-c", script], log, {}) == 3
    out = capsys.readouterr().out
    assert "secret" not in out and "secret" not in log.read_text()
    assert log.read_text().splitlines() == ["14:27:46.500 ERROR    <withheld: may echo server error text>"]
