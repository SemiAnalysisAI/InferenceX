import json
from pathlib import Path

import pytest

from infx.serving import client
from infx.serving.speedbench import (
    Settings,
    SpeedBenchError,
    label,
    normalize,
    payload,
    prompt,
    select,
    settings,
    write_selection,
)


def _row(qid: str, category: str, *turns: str) -> dict:
    return {"question_id": qid, "category": category, "messages": [{"role": "user", "content": t} for t in turns]}


ROWS = [
    _row("a" * 32, "coding", "  write fizzbuzz  ", "now in Rust"),
    _row("b" * 32, "coding", "reverse a list"),
    _row("c" * 32, "math", "integrate x"),
    _row("d" * 32, "math", "   "),  # empty first turn: skipped, as the aiperf loader does
    {"question_id": "e" * 32, "category": "math", "messages": []},
]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "qualitative.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    return path


@pytest.mark.parametrize("chosen, expected", [
    (Settings(), ["a" * 32, "b" * 32, "c" * 32]),
    (Settings(category="coding"), ["a" * 32, "b" * 32]),
    (Settings(category="math"), ["c" * 32]),
    (Settings(category="qa"), []),
])
def test_select_skips_rows_without_a_first_turn(dataset, chosen, expected):
    assert [r["question_id"] for r in select(dataset, chosen)] == expected


def test_prompt_is_the_stripped_first_turn_only():
    assert prompt(ROWS[0]) == "write fizzbuzz"
    assert prompt({"messages": None}) == "" and prompt({}) == "" and prompt({"messages": ["x"]}) == ""


def test_payload_defaults_match_the_golden_matrices():
    body = payload(ROWS[0], "m", Settings())
    assert body == {
        "model": "m", "messages": [{"role": "user", "content": "write fizzbuzz"}],
        "max_tokens": 4096, "temperature": 1.0, "top_p": 0.95,
        "stream": True, "stream_options": {"include_usage": True},
    }
    assert "ignore_eos" not in body


def test_payload_carries_the_thinking_switch():
    chosen = normalize({"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 2048, "temperature": 0})
    body = payload(ROWS[1], "m", chosen)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert (body["max_tokens"], body["temperature"]) == (2048, 0.0)


def test_write_selection(dataset, tmp_path):
    out = tmp_path / "sel" / "selection.jsonl"
    assert write_selection(dataset, Settings(category="coding"), "m", out) == 2
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [line["messages"][0]["content"] for line in lines] == ["write fizzbuzz", "reverse a list"]
    with pytest.raises(SpeedBenchError, match="no row of qualitative matches"):
        write_selection(dataset, Settings(category="qa"), "m", out)


@pytest.mark.parametrize("overrides, expected", [
    ({}, "speedbench-qualitative"),
    ({"category": "coding"}, "speedbench-qualitative-coding"),
    ({"config": "throughput_8k", "category": "mixed", "requests": 20}, "speedbench-throughput_8k-mixed-n20"),
    ({"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 1024},
     "speedbench-qualitative-ctk-enable_thinking-false-mt1024"),
    ({"temperature": 0.6, "top_p": 1}, "speedbench-qualitative-t0.6-p1"),
])
def test_label_names_every_non_default_knob(overrides, expected):
    assert label(normalize(overrides)) == expected


def test_dataset_name_is_the_lock_key():
    assert Settings().dataset == "speedbench-qualitative"
    assert Settings(config="throughput_16k").dataset == "speedbench-throughput-16k"


@pytest.mark.parametrize("overrides, message", [
    ({"config": "throughput_4k"}, "overrides.config"),
    ({"category": "mixed"}, "overrides.category"),  # a throughput category on qualitative
    ({"config": "throughput_2k", "category": "coding"}, "overrides.category"),
    ({"max_tokens": 0}, "overrides.max_tokens"),
    ({"requests": True}, "overrides.requests"),
    ({"temperature": -1}, "overrides.temperature"),
    ({"top_p": 1.5}, "overrides.top_p"),
    ({"top_p": True}, "overrides.top_p"),
    ({"chat_template_kwargs": ["x"]}, "overrides.chat_template_kwargs"),
    ({"subset": "diverse"}, "speedbench takes"),
])
def test_bad_overrides(overrides, message):
    with pytest.raises(SpeedBenchError, match=message):
        normalize(overrides)


def test_settings_only_for_a_speedbench_yaml():
    yaml_text = "scenario: speedbench\noverrides:\n  category: math\n"
    assert settings({"SERVING_YAML": yaml_text}) == Settings(category="math")
    assert settings({"SERVING_YAML": "scenario: greenspec\n"}) is None
    assert settings({"SERVING_YAML": ": : ["}) is None and settings({}) is None


def test_selection_family_picks_the_module_of_the_yaml():
    from infx.serving import greenspec, speedbench

    assert client.selection_family({"SERVING_YAML": "scenario: speedbench\n"})[0] is speedbench
    assert client.selection_family({"SERVING_YAML": "scenario: greenspec\n"})[0] is greenspec
    assert client.selection_family({"SERVING_YAML": "scenario: agentx\n"}) is None
