import json

import pytest
import yaml

from infx.bench import speedbench_al
from infx.bench.env import BenchError, InputError
from infx.bench.speedbench_al import levels, read_acceptance, select_category, write_matrix


def test_select_category_preserves_all_matching_prompts(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "category": category,
                    "messages": [{"role": "user", "content": content}],
                }
            )
            for category, content in [("coding", "one"), ("math", "two"), ("coding", "three")]
        )
    )
    destination = tmp_path / "selected.jsonl"
    assert select_category(source, destination, "coding") == 2
    assert [json.loads(row) for row in destination.read_text().splitlines()] == [
        {"category": "coding", "messages": [{"role": "user", "content": "one"}]},
        {"category": "coding", "messages": [{"role": "user", "content": "three"}]},
    ]
    with pytest.raises(BenchError, match="No SPEED-Bench"):
        select_category(source, destination, "missing")


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "instruction"}, {"role": "user", "content": "question"}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": ["multimodal"]}],
    ],
)
def test_select_category_rejects_invalid_first_turn(tmp_path, messages):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"category": "coding", "messages": messages}))
    with pytest.raises(BenchError, match="start with"):
        select_category(source, tmp_path / "selected.jsonl", "coding")


def test_select_category_preserves_multiturn_rows_for_native_vllm_loader(tmp_path):
    source = tmp_path / "source.jsonl"
    row = {
        "category": "coding",
        "messages": [
            {"role": "user", "content": "Write a parser"},
            {"role": "user", "content": "Add error handling"},
        ],
    }
    source.write_text(json.dumps(row))
    destination = tmp_path / "selected.jsonl"
    assert select_category(source, destination, "coding") == 1
    assert json.loads(destination.read_text()) == row


def test_wait_gpu_idle_waits_for_workers(monkeypatch):
    readings = iter(["15000\n16000\n", "1000\n1900\n"])
    monkeypatch.setattr(speedbench_al.subprocess, "check_output", lambda *a, **kw: next(readings))
    waits = []
    monkeypatch.setattr(speedbench_al.time, "sleep", waits.append)
    speedbench_al.wait_gpu_idle()
    assert waits == [3]


@pytest.mark.parametrize("output", ["", "3000\n4000\n"])
def test_wait_gpu_idle_fails_closed(monkeypatch, output):
    monkeypatch.setattr(speedbench_al.subprocess, "check_output", lambda *a, **kw: output)
    clock = iter([0, 121])
    monkeypatch.setattr(speedbench_al.time, "monotonic", lambda: next(clock))
    with pytest.raises(BenchError, match="did not drain"):
        speedbench_al.wait_gpu_idle()


def test_acceptance_and_matrix(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "completed": 20,
                "total_output_tokens": 100,
                "spec_decode_acceptance_length": 2.3456789,
                "spec_decode_num_drafts": 10000000,
                "spec_decode_accepted_tokens": 13456789,
            }
        )
    )
    value = read_acceptance(result, expected=20, mtp=3)
    matrix = tmp_path / "matrix.yaml"
    write_matrix(matrix, "zai-org/GLM-5.3-Flash", {3: value}, tmp_path / "provenance.json")
    assert yaml.safe_load(matrix.read_text()) == {"glm-5.3-flash": {"thinking_on": {3: 2.345679}}}


@pytest.mark.parametrize(
    "updates",
    [
        {"completed": 19},
        {"total_output_tokens": 0},
        {"failed": 1},
        {"spec_decode_acceptance_length": None},
        {"spec_decode_acceptance_length": "N/A"},
        {"spec_decode_acceptance_length": float("nan")},
        {"spec_decode_acceptance_length": float("inf")},
        {"spec_decode_acceptance_length": True},
        {"spec_decode_acceptance_length": 0.99},
        {"spec_decode_acceptance_length": 4.01},
        {"spec_decode_num_drafts": 0},
        {"spec_decode_num_drafts": None},
        {"spec_decode_accepted_tokens": -1},
        {"spec_decode_accepted_tokens": 301},
    ],
)
def test_reject_incomplete_or_invalid_measurements(tmp_path, updates):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "completed": 20,
                "total_output_tokens": 100,
                "spec_decode_acceptance_length": 2.5,
                "spec_decode_num_drafts": 200,
                "spec_decode_accepted_tokens": 300,
            }
            | updates
        )
    )
    with pytest.raises(BenchError):
        read_acceptance(result, expected=20, mtp=3)


@pytest.mark.parametrize("content", ["", "{}\n{}\n"])
def test_reject_missing_or_stale_results(tmp_path, content):
    result = tmp_path / "result.jsonl"
    result.write_text(content)
    with pytest.raises(BenchError, match="Expected one"):
        read_acceptance(result, expected=20, mtp=3)


def test_levels():
    assert levels("1 3 8") == [1, 3, 8]
    for invalid in ("", "1 1", "0", "-1", "a"):
        with pytest.raises(InputError):
            levels(invalid)
