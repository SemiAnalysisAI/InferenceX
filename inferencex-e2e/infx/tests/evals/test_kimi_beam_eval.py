"""``kimi_beam_eval.py``: the real adapter with stub harness scripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import infx.evals.kimi_beam_eval as kbe

BASE_URL = "http://127.0.0.1:8000/v1"
GENERATE_STUB = r'''
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
plan = json.loads(os.environ["BEAM_STUB_PLAN"])
record = {"script": "generate", "argv": argv, "cwd": os.getcwd(), "key": os.environ.get("BEAM_API_KEY")}
Path(os.environ["BEAM_STUB_LOG"]).open("a").write(json.dumps(record) + "\n")
with Path(argv[argv.index("--output") + 1]).open("a") as sink:
    for row in plan.get("answers", []):
        sink.write(json.dumps(row) + "\n")
sys.exit(plan.get("generate_rc", 0))
'''
JUDGE_STUB = r'''
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
plan = json.loads(os.environ["BEAM_STUB_PLAN"])
record = {"script": "judge", "argv": argv, "cwd": os.getcwd(), "key": os.environ.get("JUDGE_API_KEY")}
Path(os.environ["BEAM_STUB_LOG"]).open("a").write(json.dumps(record) + "\n")
with Path(argv[argv.index("--output") + 1]).open("a") as sink:
    for row in plan.get("scores", []):
        sink.write(json.dumps(row) + "\n")
sys.exit(plan.get("judge_rc", 0))
'''
ANSWERS = [
    {"chat_id": "c1", "question_type": "abstention", "question_index": 0, "error": None},
    {"chat_id": "c1", "question_type": "summarization", "question_index": 1, "error": None},
]
SCORES = [
    {"chat_id": "c1", "question_type": "abstention", "question_index": 0, "score": 1.0},
    {"chat_id": "c1", "question_type": "summarization", "question_index": 1, "score": 0.5},
]


def _checkout(root: Path, *, pointer: bool = False) -> Path:
    beam = root / "verifier" / "beam"
    (beam / "tokenizer").mkdir(parents=True)
    (beam / "data").mkdir()
    (beam / "beam_generate.py").write_text(GENERATE_STUB)
    (beam / "beam_judge.py").write_text(JUDGE_STUB)
    (beam / "tokenizer" / "tokenizer_config.json").write_text("{}")
    data = b"version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n" if pointer else b"x"
    (beam / "data" / "beam_1m_chats.jsonl.gz").write_bytes(data)
    (beam / "data" / "beam_1m_questions.jsonl").write_bytes(data)
    return beam.parent


def _run(
    tmp_path: Path, plan: dict[str, Any], *, pointer: bool = False, **env: str
) -> tuple[bool, list[dict[str, Any]], Path]:
    verifier = _checkout(tmp_path, pointer=pointer)
    output = tmp_path / "out"
    log = tmp_path / "calls.jsonl"
    environ = {
        **os.environ,
        "BEAM_STUB_PLAN": json.dumps(plan),
        "BEAM_STUB_LOG": str(log),
        "BEAM_API_KEY": "sk-under-test",
        **env,
    }
    completed = kbe.run_evaluation(
        verifier_dir=verifier, base_url=BASE_URL, model="served", output_dir=output, environ=environ
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return completed, calls, output


def _result(output_dir: Path) -> dict[str, Any]:
    [path] = output_dir.glob(kbe.COMPATIBILITY_GLOB)
    assert re.fullmatch(r"results_kimi_beam_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}\.json", path.name)
    return json.loads(path.read_text())


def _report(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / kbe.NATIVE_REPORT_FILENAME).read_text())


def test_generate_command_pins_the_protocol_and_keeps_the_key_off_argv(tmp_path: Path) -> None:
    beam = tmp_path / "beam"

    command = kbe.build_generate_command(
        python="py", beam_dir=beam, base_url=BASE_URL, model="served",
        answers_path=tmp_path / "a.jsonl", concurrency=4, thinking_json='{"thinking":{}}', limit=2,
    )  # fmt: skip

    assert command == [
        "py", str(beam / "beam_generate.py"),
        "--model", "served", "--base-url", BASE_URL, "--output", str(tmp_path / "a.jsonl"),
        "--chats-file", str(beam / "data" / "beam_1m_chats.jsonl.gz"),
        "--questions-file", str(beam / "data" / "beam_1m_questions.jsonl"),
        "--tokenizer", str(beam / "tokenizer"),
        "--concurrency", "4", "--temperature", "1.0", "--top-p", "0.95", "--max-tokens", "32768",
        "--max-context-tokens", "1000000", "--thinking-json", '{"thinking":{}}', "--limit", "2",
    ]  # fmt: skip
    assert "--api-key" not in command
    assert "--limit" not in kbe.build_generate_command(
        python="py", beam_dir=beam, base_url=BASE_URL, model="served",
        answers_path=tmp_path / "a.jsonl", concurrency=16, thinking_json="", limit=None,
    )  # fmt: skip


def test_judge_command_judges_with_the_model_under_test(tmp_path: Path) -> None:
    beam = tmp_path / "beam"

    command = kbe.build_judge_command(
        python="py", beam_dir=beam, base_url=BASE_URL, model="served",
        answers_path=tmp_path / "a.jsonl", scores_path=tmp_path / "s.jsonl",
    )  # fmt: skip

    assert command == [
        "py", str(beam / "beam_judge.py"), "--answers", str(tmp_path / "a.jsonl"),
        "--output", str(tmp_path / "s.jsonl"), "--judge-model", "served",
        "--judge-base-url", BASE_URL, "--judge-temperature", "0.3", "--concurrency", "32",
    ]  # fmt: skip


def test_settings_read_the_optional_inputs_and_default_the_rest() -> None:
    assert kbe.settings({}) == {
        "api_key": "EMPTY",
        "judge_api_key": "EMPTY",
        "limit": None,
        "concurrency": 16,
        "thinking_json": kbe.DEFAULT_THINKING_JSON,
    }
    assert kbe.settings(
        {"BEAM_API_KEY": "k", "EVAL_LIMIT": "3", "BEAM_CONCURRENCY": "2", "BEAM_THINKING_JSON": ""}
    ) == {"api_key": "k", "judge_api_key": "k", "limit": 3, "concurrency": 2, "thinking_json": ""}
    assert kbe.settings({"BEAM_API_KEY": "k", "JUDGE_API_KEY": "j"})["judge_api_key"] == "j"
    with pytest.raises(ValueError, match="EVAL_LIMIT must be a positive integer, got '0'"):
        kbe.settings({"EVAL_LIMIT": "0"})


def test_project_reduces_a_complete_run() -> None:
    native, result, completed = kbe.project("served", ANSWERS, SCORES, expected=2, limit=2)

    assert completed
    assert result["results"][kbe.TASK_NAME]["exact_match,strict-match"] == 0.75
    assert result["n-samples"][kbe.TASK_NAME] == {"original": 700, "effective": 2}
    assert result["limit"] == 2
    assert "integration_error" not in result
    assert native["summary"]["accuracy_by_question_type"] == {"abstention": 1.0, "summarization": 0.5}
    assert native["summary"]["errored"] == 0
    assert native["completed"]


@pytest.mark.parametrize(
    ("answers", "scores", "expected", "message"),
    [
        (ANSWERS, SCORES, 3, "2 of 3 questions answered"),
        ([{**ANSWERS[0], "error": "RuntimeError()"}, ANSWERS[1]], SCORES, 2, "1 questions failed every retry"),
        (ANSWERS, SCORES[:1], 2, "1 of 2 answers judged"),
        (ANSWERS, [SCORES[0], {**SCORES[1], "score": 1.5}], 2, "1 scores are malformed or orphaned"),
        (ANSWERS, [SCORES[0], {**SCORES[1], "question_index": 9}], 2, "1 scores are malformed or orphaned"),
    ],
    ids=["short", "errored", "unjudged", "out-of-range", "orphaned"],
)
def test_project_turns_an_incomplete_run_into_a_zero_score_integration_error(
    answers: list[dict[str, Any]], scores: list[dict[str, Any]], expected: int, message: str
) -> None:
    native, result, completed = kbe.project("served", answers, scores, expected=expected, limit=None)

    assert not completed
    assert message in result["integration_error"]["message"]
    assert result["results"][kbe.TASK_NAME]["exact_match,strict-match"] == 0.0
    assert result["n-samples"][kbe.TASK_NAME]["effective"] == 0
    assert native["integration_error"]["message"] == result["integration_error"]["message"]


def test_run_evaluation_drives_both_scripts_with_keys_in_the_environment(tmp_path: Path) -> None:
    completed, calls, output = _run(
        tmp_path, {"answers": ANSWERS, "scores": SCORES}, EVAL_LIMIT="2", JUDGE_API_KEY="sk-judge"
    )

    assert completed
    generate, judge = calls
    assert generate["key"] == "sk-under-test"
    assert judge["key"] == "sk-judge"
    assert "sk-under-test" not in json.dumps([generate["argv"], judge["argv"]])
    assert generate["cwd"] == judge["cwd"] == str(tmp_path / "verifier" / "beam")
    assert generate["argv"][generate["argv"].index("--limit") + 1] == "2"
    assert judge["argv"][judge["argv"].index("--answers") + 1] == str(output / kbe.ANSWERS_FILENAME)
    assert _result(output)["results"][kbe.TASK_NAME]["exact_match,strict-match"] == 0.75
    assert _report(output)["limit"] == 2
    [result_path] = output.glob(kbe.COMPATIBILITY_GLOB)
    assert sorted(path.name for path in output.iterdir()) == sorted(
        [kbe.NATIVE_REPORT_FILENAME, kbe.ANSWERS_FILENAME, kbe.SCORES_FILENAME, result_path.name]
    )


def test_run_evaluation_refuses_unresolved_lfs_pointers_before_any_request(tmp_path: Path) -> None:
    completed, calls, output = _run(tmp_path, {"answers": ANSWERS, "scores": SCORES}, pointer=True)

    assert not completed
    assert calls == []
    message = _result(output)["integration_error"]["message"]
    assert message == "beam/data/beam_1m_chats.jsonl.gz is a git-lfs pointer, not the data file"
    assert _report(output)["integration_error"]["message"] == message


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ({"answers": [], "generate_rc": 3}, "beam_generate.py exited with code 3"),
        ({"answers": ANSWERS, "scores": [], "judge_rc": 2}, "beam_judge.py exited with code 2"),
    ],
    ids=["generate-fails", "judge-fails"],
)
def test_run_evaluation_records_a_failed_script_as_an_integration_error(
    tmp_path: Path, plan: dict[str, Any], message: str
) -> None:
    completed, _, output = _run(tmp_path, plan, EVAL_LIMIT="2")

    assert not completed
    assert _result(output)["integration_error"]["message"] == message
    assert _result(output)["n-samples"][kbe.TASK_NAME]["effective"] == 0


def test_parse_args_requires_the_endpoint_and_checkout_unless_recording_a_failure() -> None:
    with pytest.raises(SystemExit) as exited:
        kbe.parse_args(["--model", "m", "--output-dir", "out"])
    assert exited.value.code == 2

    args = kbe.parse_args(["--model", "m", "--output-dir", "out", "--integration-error", "boom"])
    assert args.integration_error == "boom"


def test_cli_integration_error_writes_a_zero_score_artifact_under_the_image_python(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable, "-I", str(Path(kbe.__file__)),
            "--model", "served", "--output-dir", str(output),
            "--integration-error", "Kimi BEAM (1M) dependency installation failed with exit code 12",
        ],  # fmt: skip
        check=False,
    )

    assert completed.returncode == 0
    result = _result(output)
    assert result["integration_error"]["message"].endswith("exit code 12")
    assert result["n-samples"][kbe.TASK_NAME]["effective"] == 0
    assert _report(output)["completed"] is False
