from __future__ import annotations

import argparse
import json

import pytest
import yaml

from infx.bench import speedbench, speedbench_al as al
from infx.bench.env import BenchError, InputError


def test_levels_and_thinking_modes():
    assert al.levels("1 2 4 8") == [1, 2, 4, 8]
    assert al.modes("off on", '{"enable_thinking": true}') == {
        "off": {"enable_thinking": False},
        "on": {"enable_thinking": True},
    }


@pytest.mark.parametrize("value", ["", "0", "-1", "1 1", "x"])
def test_invalid_depths(value):
    with pytest.raises(InputError):
        al.levels(value)


@pytest.mark.parametrize("value", ["", "on on", "true", "on maybe"])
def test_invalid_modes(value):
    with pytest.raises(InputError):
        al.modes(value, '{"enable_thinking": true}')


@pytest.mark.parametrize("kwargs", ["{}", '{"thinking": true}', '{"enable_thinking": false}'])
def test_incorrect_template_switch(kwargs):
    with pytest.raises(InputError):
        al.modes("on", kwargs)


def result():
    return {
        "completed": 80,
        "failed": 0,
        "total_output_tokens": 1000,
        "spec_decode_acceptance_length": 2.5,
        "spec_decode_num_drafts": 100,
        "spec_decode_accepted_tokens": 150,
    }


def test_acceptance_matches_counter_deltas(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result()))
    assert al.read_acceptance(path, expected=80, mtp=2) == 2.5


@pytest.mark.parametrize(
    "patch",
    [
        {"completed": 79},
        {"failed": 1},
        {"total_output_tokens": 0},
        {"spec_decode_acceptance_length": None},
        {"spec_decode_acceptance_length": True},
        {"spec_decode_acceptance_length": float("nan")},
        {"spec_decode_acceptance_length": 4},
        {"spec_decode_num_drafts": 0},
        {"spec_decode_num_drafts": 100.0},
        {"spec_decode_accepted_tokens": 149},
        {"spec_decode_accepted_tokens": -1},
    ],
)
def test_incomplete_or_inconsistent_result_fails(tmp_path, patch):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({**result(), **patch}))
    with pytest.raises(BenchError):
        al.read_acceptance(path, expected=80, mtp=2)


def test_dataset_preserves_exact_prompts(tmp_path, monkeypatch):
    prompts = ["Implement a trie.", "用 Python 排序。"]
    monkeypatch.setattr(speedbench, "PROMPTS_SHA256", speedbench.digest(prompts))
    path = al.write_dataset(tmp_path, prompts)
    assert [json.loads(row) for row in path.read_text().splitlines()] == [
        {"category": "coding", "messages": [{"role": "user", "content": p}]} for p in prompts
    ]
    with pytest.raises(BenchError, match="digest"):
        al.write_dataset(tmp_path, prompts[::-1])


def test_benchmark_command_uses_template_natural_eos_and_all_prompts(tmp_path):
    args = argparse.Namespace(concurrency=64, seed=42)
    sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
    command = al.benchmark_command(
        "/model/revision",
        8888,
        tmp_path / "qualitative.jsonl",
        4096,
        tmp_path / "result.json",
        {"enable_thinking": False},
        sampling,
        args,
    )
    options = dict(zip(command[3::2], command[4::2], strict=False))
    assert options["--num-prompts"] == "-1"
    assert options["--max-concurrency"] == "64"
    assert options["--speed-bench-output-len"] == "4096"
    assert json.loads(options["--chat-template-kwargs"]) == {"enable_thinking": False}
    assert json.loads(options["--extra-body"]) == sampling
    assert "--ignore-eos" not in command
    assert "--skip-chat-template" not in command


def test_matrix_has_distinct_mode_and_checkpoint_identity(tmp_path):
    path = tmp_path / "matrix.yaml"
    al.write_matrix(path, "Qwen/Qwen3.6-35B-A3B-FP8", {"on": {1: 1.8}, "off": {1: 1.7}})
    assert yaml.safe_load(path.read_text()) == {
        "qwen3.6-35b-a3b-fp8": {"thinking_on": {1: 1.8}, "thinking_off": {1: 1.7}}
    }


@pytest.mark.parametrize("fail", [False, True])
def test_only_complete_collection_publishes_matrix(tmp_path, monkeypatch, fail):
    model_root = tmp_path / "models"
    revision = "a" * 40
    model = model_root / revision
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {"quant_method": "fp8"},
                "text_config": {"mtp_num_hidden_layers": 1},
            }
        )
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("stale previous run")
    for key, value in {
        "CATEGORY": "coding",
        "CHAT_TEMPLATE_KWARGS_ON": '{"enable_thinking": true}',
        "IMAGE": "vllm@sha256:test",
        "MODEL": "Qwen/Qwen3.6-35B-A3B-FP8",
        "MODEL_PATH": str(model_root),
        "MODEL_REVISION": revision,
        "MTP_LIST": "1 2",
        "OUT_YAML": str(matrix),
        "PORT": "8888",
        "SPEEDBENCH_OUTPUT_LEN": "4096",
        "THINKING_MODES": "off on",
    }.items():
        monkeypatch.setenv(key, value)
    commands = []
    monkeypatch.setattr(al, "run", lambda command: commands.append(command))
    monkeypatch.setattr(al.importlib.metadata, "version", lambda _: "test-version")
    monkeypatch.setattr(al, "prepare_dataset", lambda path: (path / "qualitative.jsonl", 80))
    calls = []

    def measure(**kwargs):
        calls.append(kwargs)
        assert not matrix.exists()
        if fail and kwargs["mtp"] == 2:
            raise BenchError("server failed")
        return {"on": 1.0 + kwargs["mtp"] / 2, "off": 1.0 + kwargs["mtp"] / 3}

    monkeypatch.setattr(al, "collect_depth", measure)
    args = [
        "--concurrency",
        "64",
        "--seed",
        "42",
        "--spec-method",
        "mtp",
        "--sampling",
        '{"on":{"temperature":1.0,"top_p":0.95},"off":{"temperature":0.7,"top_p":0.8}}',
        "--",
        "--tensor-parallel-size",
        "4",
    ]
    if fail:
        with pytest.raises(BenchError, match="server failed"):
            al.main(args)
        assert not matrix.exists()
    else:
        assert al.main(args) == 0
        measured = yaml.safe_load(matrix.read_text())["qwen3.6-35b-a3b-fp8"]
        assert measured["thinking_on"] == {1: 1.5, 2: 2.0}
        assert measured["thinking_off"] == {1: 1.333333, 2: 1.666667}
    provenance = json.loads(
        (tmp_path / "speedbench_results/speedbench_provenance.json").read_text()
    )
    assert provenance["complete"] is not fail
    assert provenance["inputs"]["MODEL_REVISION"] == revision
    assert provenance["server_args"] == ["--tensor-parallel-size", "4"]
    assert [(call["mtp"], call["count"]) for call in calls] == [(1, 80), (2, 80)]
    assert commands == [
        [
            "hf",
            "download",
            "Qwen/Qwen3.6-35B-A3B-FP8",
            "--revision",
            revision,
            "--local-dir",
            str(model),
        ]
    ]
