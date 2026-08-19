import builtins
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bfcl_eval as be
import validate_scores as vs


def _compatibility_path(output_dir: Path) -> Path:
    path = output_dir / be.COMPATIBILITY_FILENAME
    assert path.exists()
    return path


def _compatibility(output_dir: Path) -> dict[str, Any]:
    return json.loads(_compatibility_path(output_dir).read_text(encoding="utf-8"))


def _native(output_dir: Path) -> dict[str, Any]:
    return json.loads(
        (output_dir / be.NATIVE_REPORT_FILENAME).read_text(encoding="utf-8")
    )


def _score(output_dir: Path) -> float:
    return _compatibility(output_dir)["results"][be.TASK_NAME]["acc,none"]


def test_thresholds_are_stdlib_readable_without_pyyaml(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
    thresholds = vs.load_config(str(Path(vs.__file__).with_name("thresholds.yaml")))

    assert thresholds["default"]["bfcl_smoke"] == 0.75
    assert thresholds["default"]["bfcl_parallel"] == 0.0


def _write_result(
    project_root: Path,
    category: str,
    rows: list[dict[str, Any]] | None = None,
) -> Path:
    result_path = (
        project_root
        / "result"
        / "model-a"
        / "nested"
        / f"BFCL_v4_{category}_result.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [{"id": be.SMOKE_CASE_IDS[category][0], "result": []}]
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return result_path


def _write_score(
    project_root: Path,
    category: str,
    correct_count: int,
    *,
    header_override: str | None = None,
    record_id: str | None = None,
) -> None:
    score_path = (
        project_root / "score" / "model-a" / "nested" / f"BFCL_v4_{category}_score.json"
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if header_override is not None:
        first_line = header_override
    else:
        first_line = json.dumps(
            {
                "accuracy": float(correct_count),
                "correct_count": correct_count,
                "total_count": 1,
            }
        )
    record_text = ""
    if correct_count == 0 or record_id is not None:
        case_id = be.SMOKE_CASE_IDS[category][0] if record_id is None else record_id
        record_text = json.dumps({"id": case_id}) + "\n"
    score_path.write_text(first_line + "\n" + record_text, encoding="utf-8")
    _write_result(project_root, category)


def _score_runner(
    correct_by_category: dict[str, int] | None = None,
    *,
    missing: str | None = None,
    malformed: str | None = None,
    invocation: dict[str, Any] | None = None,
):
    expected = correct_by_category or {category: 1 for category in be.SMOKE_CASE_IDS}

    def run(**kwargs: Any) -> None:
        if invocation is not None:
            invocation.update(kwargs)
        project_root = kwargs["project_root"]
        for category in be.SMOKE_CASE_IDS:
            if category == missing:
                _write_result(project_root, category)
                continue
            _write_score(
                project_root,
                category,
                expected[category],
                header_override="{not-json" if category == malformed else None,
            )

    return run


def _run(
    tmp_path: Path,
    runner,
    *,
    output_dir: Path | None = None,
) -> tuple[bool, Path, Path]:
    output = output_dir or tmp_path / "output"
    project_root = tmp_path / "bfcl-project"
    passed = be.run_evaluation(
        base_url="http://127.0.0.1:8000/v1/",
        api_key="EMPTY",
        model="model-a",
        output_dir=output,
        bfcl_project_root=project_root,
        upstream_runner=runner,
    )
    return passed, output, project_root


def test_command_defaults_and_required_runtime_inputs(tmp_path: Path) -> None:
    args = be.parse_args(
        [
            "--base-url",
            "http://localhost:8000/v1/",
            "--model",
            "model-a",
            "--output-dir",
            str(tmp_path / "output"),
            "--bfcl-project-root",
            str(tmp_path / "bfcl"),
        ]
    )

    assert args.base_url == "http://localhost:8000/v1"
    assert args.api_key == "EMPTY"
    assert args.num_threads == 4
    assert args.request_timeout_seconds == 180.0
    assert args.integration_error is None

    with pytest.raises(SystemExit):
        be.parse_args(["--model", "model-a", "--output-dir", str(tmp_path / "missing")])


@pytest.mark.parametrize(
    ("flag", "value"),
    (("--num-threads", "0"), ("--request-timeout-seconds", "nan")),
)
def test_cli_rejects_invalid_positive_values(
    tmp_path: Path, flag: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        be.parse_args(
            [
                "--base-url",
                "http://localhost/v1",
                "--model",
                "model-a",
                "--output-dir",
                str(tmp_path / "output"),
                "--bfcl-project-root",
                str(tmp_path / "bfcl"),
                flag,
                value,
            ]
        )


def test_cli_rejects_chat_completions_endpoint_instead_of_api_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        be.parse_args(
            [
                "--base-url",
                "http://localhost/v1/chat/completions",
                "--model",
                "model-a",
                "--output-dir",
                str(tmp_path / "output"),
                "--bfcl-project-root",
                str(tmp_path / "bfcl"),
            ]
        )


def test_perfect_score_projects_pinned_ids_and_upstream_headers(
    tmp_path: Path,
) -> None:
    invocation: dict[str, Any] = {}
    passed, output_dir, project_root = _run(
        tmp_path, _score_runner(invocation=invocation)
    )

    assert passed
    assert invocation == {
        "model": "model-a",
        "project_root": project_root,
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "EMPTY",
        "num_threads": 4,
        "request_timeout_seconds": 180.0,
    }
    assert json.loads(
        (project_root / "test_case_ids_to_generate.json").read_text(encoding="utf-8")
    ) == {
        "simple_python": ["simple_python_141"],
        "multiple": ["multiple_38"],
        "parallel": ["parallel_1"],
        "irrelevance": ["irrelevance_0"],
    }

    compatibility = _compatibility(output_dir)
    native = _native(output_dir)
    assert compatibility["result_format"] == "inferencex-eval-v1"
    expected_tasks = [
        "bfcl_smoke",
        "bfcl_simple_python",
        "bfcl_multiple",
        "bfcl_parallel",
        "bfcl_irrelevance",
    ]
    assert list(compatibility["results"]) == expected_tasks
    assert all(
        compatibility["results"][task_name] == {"acc,none": 1.0, "acc_stderr,none": 0.0}
        for task_name in expected_tasks
    )
    assert all(
        compatibility["configs"][task_name]
        == {
            "metric_list": [{"metric": "acc"}],
            "filter_list": [{"name": "none"}],
        }
        for task_name in expected_tasks
    )
    assert compatibility["n-samples"][be.TASK_NAME] == {
        "original": 4,
        "effective": 4,
    }
    assert all(
        compatibility["n-samples"][f"bfcl_{category}"]
        == {"original": 1, "effective": 1}
        for category in be.SMOKE_CASE_IDS
    )
    assert compatibility["bfcl"]["source"]["package_version"] == "2026.3.23"
    assert compatibility["bfcl"]["source"]["wheel_sha256"] == (
        "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"
    )
    assert compatibility["bfcl"]["source"]["source_revision"] == (
        "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
    )
    assert [entry["category"] for entry in compatibility["bfcl"]["categories"]] == [
        "simple_python",
        "multiple",
        "parallel",
        "irrelevance",
    ]
    assert all(
        entry["score_header"] == {"accuracy": 1.0, "correct_count": 1, "total_count": 1}
        for entry in compatibility["bfcl"]["categories"]
    )
    assert all(
        entry["score_records"] == [] for entry in compatibility["bfcl"]["categories"]
    )
    assert native["completed"] is True
    assert native["passed"] is True
    assert native["summary"] == {
        "accuracy": 1.0,
        "correct_count": 4,
        "total_count": 4,
        "expected_count": 4,
    }
    assert native["bfcl"] == compatibility["bfcl"]
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "bfcl_report.json",
        "results_bfcl.json",
    ]


def test_weighted_score_failure_is_complete_and_left_to_threshold_validator(
    tmp_path: Path,
) -> None:
    completed, output_dir, _ = _run(
        tmp_path,
        _score_runner(
            {
                "simple_python": 1,
                "multiple": 0,
                "parallel": 0,
                "irrelevance": 1,
            }
        ),
    )

    assert completed
    assert _score(output_dir) == 0.5
    compatibility = _compatibility(output_dir)
    assert compatibility["results"]["bfcl_multiple"]["acc,none"] == 0.0
    assert compatibility["results"]["bfcl_parallel"]["acc,none"] == 0.0
    assert compatibility["n-samples"][be.TASK_NAME]["effective"] == 4
    assert "integration_error" not in compatibility
    native = _native(output_dir)
    assert native["completed"] is True
    assert native["passed"] is False
    assert native["threshold"] == 0.75
    assert native["summary"]["correct_count"] == 2


def test_stale_compatibility_outputs_are_removed_without_touching_foreign_results(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stale = output_dir / "results_bfcl_2000-01-01T00-00-00.000000.json"
    stale.write_text("{}", encoding="utf-8")
    stale_native = output_dir / be.NATIVE_REPORT_FILENAME
    stale_native.write_text("{}", encoding="utf-8")
    foreign = output_dir / "results_other_eval.json"
    foreign.write_text("{}", encoding="utf-8")

    passed, _, _ = _run(tmp_path, _score_runner(), output_dir=output_dir)

    assert passed
    assert not stale.exists()
    assert foreign.exists()
    assert len(list(output_dir.glob(be.COMPATIBILITY_GLOB))) == 1
    assert _native(output_dir)["summary"]["accuracy"] == 1.0


def test_swallowed_inference_error_is_an_integration_failure(
    tmp_path: Path,
) -> None:
    base_runner = _score_runner()

    def runner(**kwargs: Any) -> None:
        base_runner(**kwargs)
        _write_result(
            kwargs["project_root"],
            "parallel",
            [
                {
                    "id": "parallel_1",
                    "result": "Error during inference: request timed out",
                    "traceback": "TimeoutError: request timed out",
                }
            ],
        )

    completed, output_dir, _ = _run(tmp_path, runner)

    assert not completed
    assert _score(output_dir) == 0.0
    assert _compatibility(output_dir)["integration_error"] == {
        "type": "RuntimeError",
        "message": "parallel result parallel_1 contains an inference error",
    }


@pytest.mark.parametrize(
    ("mode", "expected_message"),
    (
        (
            "missing",
            "parallel result ids [] do not match expected ids ['parallel_1']",
        ),
        ("duplicate", "parallel result file contains duplicate id parallel_1"),
    ),
)
def test_missing_or_duplicate_generated_id_is_an_integration_failure(
    tmp_path: Path,
    mode: str,
    expected_message: str,
) -> None:
    base_runner = _score_runner()

    def runner(**kwargs: Any) -> None:
        base_runner(**kwargs)
        row = {"id": "parallel_1", "result": []}
        _write_result(
            kwargs["project_root"],
            "parallel",
            [] if mode == "missing" else [row, row],
        )

    completed, output_dir, _ = _run(tmp_path, runner)

    assert not completed
    assert _score(output_dir) == 0.0
    assert _compatibility(output_dir)["integration_error"] == {
        "type": "ValueError",
        "message": expected_message,
    }


def test_missing_category_is_an_integration_failure(tmp_path: Path) -> None:
    passed, output_dir, _ = _run(tmp_path, _score_runner(missing="parallel"))

    assert not passed
    compatibility = _compatibility(output_dir)
    assert _score(output_dir) == 0.0
    assert compatibility["n-samples"][be.TASK_NAME]["effective"] == 0
    assert compatibility["integration_error"]["type"] == "ValueError"
    assert "parallel score file" in compatibility["integration_error"]["message"]
    assert _native(output_dir)["completed"] is False


def test_malformed_score_header_is_an_integration_failure(tmp_path: Path) -> None:
    passed, output_dir, _ = _run(tmp_path, _score_runner(malformed="multiple"))

    assert not passed
    compatibility = _compatibility(output_dir)
    assert _score(output_dir) == 0.0
    assert compatibility["integration_error"] == {
        "type": "ValueError",
        "message": "multiple score header is malformed JSON",
    }


def test_unexpected_score_record_id_is_an_integration_failure(
    tmp_path: Path,
) -> None:
    def runner(**kwargs: Any) -> None:
        project_root = kwargs["project_root"]
        for category in be.SMOKE_CASE_IDS:
            _write_score(
                project_root,
                category,
                0 if category == "parallel" else 1,
                record_id="parallel_0" if category == "parallel" else None,
            )

    completed, output_dir, _ = _run(tmp_path, runner)

    assert not completed
    compatibility = _compatibility(output_dir)
    assert _score(output_dir) == 0.0
    assert compatibility["integration_error"] == {
        "type": "ValueError",
        "message": "parallel score file contains unexpected id parallel_0",
    }


def test_incomplete_score_header_is_an_integration_failure(tmp_path: Path) -> None:
    def runner(**kwargs: Any) -> None:
        project_root = kwargs["project_root"]
        for category in be.SMOKE_CASE_IDS:
            if category == "irrelevance":
                _write_score(
                    project_root,
                    category,
                    1,
                    header_override=json.dumps({"accuracy": 1.0, "correct_count": 1}),
                )
            else:
                _write_score(project_root, category, 1)

    passed, output_dir, _ = _run(tmp_path, runner)

    assert not passed
    assert _compatibility(output_dir)["integration_error"]["message"] == (
        "irrelevance score header missing: total_count"
    )


def test_upstream_exception_publishes_zero_score_reports(tmp_path: Path) -> None:
    def fail(**kwargs: Any) -> None:
        raise RuntimeError("BFCL generation failed")

    passed, output_dir, project_root = _run(tmp_path, fail)

    assert not passed
    assert (project_root / "test_case_ids_to_generate.json").exists()
    assert _score(output_dir) == 0.0
    compatibility = _compatibility(output_dir)
    assert compatibility["integration_error"] == {
        "type": "RuntimeError",
        "message": "BFCL generation failed",
    }
    assert compatibility["bfcl"]["source"]["case_ids"] == {
        "simple_python": ["simple_python_141"],
        "multiple": ["multiple_38"],
        "parallel": ["parallel_1"],
        "irrelevance": ["irrelevance_0"],
    }
    assert _native(output_dir)["summary"]["total_count"] == 0


def test_integration_error_cli_is_stdlib_only_and_returns_nonzero(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    script = Path(be.__file__).resolve()

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            "--model",
            "model-a",
            "--output-dir",
            str(output_dir),
            "--integration-error",
            "pinned wheel installation failed",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert _score(output_dir) == 0.0
    compatibility = _compatibility(output_dir)
    assert compatibility["n-samples"][be.TASK_NAME] == {
        "original": 4,
        "effective": 0,
    }
    assert compatibility["integration_error"] == {
        "type": "RuntimeError",
        "message": "pinned wheel installation failed",
    }
    native = _native(output_dir)
    assert native["completed"] is False
    assert native["integration_error"] == compatibility["integration_error"]
