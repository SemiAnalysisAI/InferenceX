import builtins
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bfcl_adapter as be
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


def _category_score(
    category: str,
    total_count: int,
    correct_count: int,
) -> be.CategoryScore:
    case_ids = tuple(f"{category}_{index}" for index in range(total_count))
    return be.CategoryScore(
        category=category,
        case_ids=case_ids,
        score_file=f"score/model-a/BFCL_v4_{category}_score.json",
        header={
            "accuracy": correct_count / total_count,
            "correct_count": correct_count,
            "total_count": total_count,
        },
        records=tuple({"id": case_id} for case_id in case_ids[correct_count:]),
        accuracy=correct_count / total_count,
        correct_count=correct_count,
        total_count=total_count,
    )


def test_thresholds_are_stdlib_readable_without_pyyaml(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
    thresholds = vs.load_config(str(Path(vs.__file__).with_name("thresholds.yaml")))

    assert thresholds["default"]["bfcl_smoke"] == 0.0
    assert thresholds["default"]["bfcl_parallel"] == 0.0


def test_score_validator_uses_declared_bfcl_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = {
        "results": {"bfcl_smoke": {"acc,none": 0.8, "acc_stderr,none": 0.1}},
        "configs": {
            "bfcl_smoke": {"metric_list": [{"metric": "acc", "aggregation": "mean"}]}
        },
        "n-samples": {"bfcl_smoke": {"original": 4, "effective": 4}},
    }
    (tmp_path / "results_bfcl.json").write_text(json.dumps(result))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["validate_scores.py"])

    assert vs.main() == 0


def test_adapter_module_does_not_collide_with_upstream_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = object()
    child = object()
    monkeypatch.setitem(sys.modules, "bfcl_eval", upstream)
    monkeypatch.setitem(sys.modules, "bfcl_eval.constants", child)

    be._clear_upstream_modules()

    assert be.__name__ == "bfcl_adapter"
    assert "bfcl_eval" not in sys.modules
    assert "bfcl_eval.constants" not in sys.modules


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
    assert args.integration_error is None
    assert args.suite == be.TASK_NAME
    with pytest.raises(SystemExit):
        be.parse_args(["--model", "model-a", "--output-dir", str(tmp_path / "missing")])


def test_cli_selects_suite_defaults_and_rejects_unknown_suite(tmp_path: Path) -> None:
    args = be.parse_args(
        [
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "model-a",
            "--output-dir",
            str(tmp_path / "output"),
            "--bfcl-project-root",
            str(tmp_path / "bfcl"),
            "--suite",
            "bfcl_vllm_kimi",
        ]
    )

    assert args.suite == "bfcl_vllm_kimi"
    assert args.num_threads == 16

    with pytest.raises(SystemExit):
        be.parse_args(
            [
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "model-a",
                "--output-dir",
                str(tmp_path / "output"),
                "--bfcl-project-root",
                str(tmp_path / "bfcl"),
                "--suite",
                "bfcl_unknown",
            ]
        )


def test_cli_forwards_selected_suite_on_normal_path(
    tmp_path: Path, monkeypatch
) -> None:
    invocation: dict[str, Any] = {}

    def run_evaluation(**kwargs: Any) -> bool:
        invocation.update(kwargs)
        return True

    monkeypatch.setattr(be, "run_evaluation", run_evaluation)

    return_code = be.main(
        [
            "--base-url",
            "http://localhost:8000/v1",
            "--model",
            "model-a",
            "--output-dir",
            str(tmp_path / "output"),
            "--bfcl-project-root",
            str(tmp_path / "bfcl"),
            "--suite",
            "bfcl_vllm_kimi",
        ]
    )

    assert return_code == 0
    assert invocation["suite"] is be.KIMI_SUITE
    assert invocation["num_threads"] == 16


@pytest.mark.parametrize(("flag", "value"), (("--num-threads", "0"),))
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


@pytest.mark.parametrize(
    "base_url",
    (
        "http://localhost/v1/chat/completions",
        "http://localhost/v1?mode=test",
        "http://localhost/v1#fragment",
    ),
)
def test_cli_rejects_invalid_api_root(tmp_path: Path, base_url: str) -> None:
    with pytest.raises(SystemExit):
        be.parse_args(
            [
                "--base-url",
                base_url,
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
    }
    assert json.loads(
        (project_root / "test_case_ids_to_generate.json").read_text(encoding="utf-8")
    ) == {
        "simple_python": ["simple_python_141"],
        "multiple": ["multiple_38"],
        "parallel": ["parallel_1"],
        "irrelevance": ["irrelevance_0"],
    }
    assert (
        (project_root / be.UPSTREAM_LICENSE_FILENAME)
        .read_text(encoding="utf-8")
        .startswith("                                 Apache License")
    )
    attribution = json.loads(
        (project_root / be.UPSTREAM_ATTRIBUTION_FILENAME).read_text(encoding="utf-8")
    )
    assert attribution == {
        "artifact": "BFCL-generated evaluation results",
        "upstream": {
            "package": "bfcl-eval",
            "package_version": "2026.3.23",
            "wheel_sha256": be.BFCL_WHEEL_SHA256,
            "repository": "https://github.com/ShishirPatil/gorilla",
            "source_revision": be.SOURCE_REVISION,
            "vllm_integration_revision": be.VLLM_INTEGRATION_REF,
            "license": "Apache-2.0",
            "license_url": (
                "https://github.com/ShishirPatil/gorilla/blob/"
                f"{be.SOURCE_REVISION}/LICENSE"
            ),
            "license_file": be.UPSTREAM_LICENSE_FILENAME,
        },
        "modifications": (
            "InferenceX selected deterministic case subsets and projected upstream "
            "scores; this archive does not modify upstream BFCL source."
        ),
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


def test_weighted_score_is_complete_and_diagnostic(
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
    assert native["passed"] is True
    assert native["threshold"] == 0.0
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


def test_full_suite_sets_project_root_before_dataset_import(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "output"
    project_root = tmp_path / "project"
    observed_roots: list[str | None] = []
    monkeypatch.setenv("BFCL_PROJECT_ROOT", "stale-root")

    def stop_after_observing_root(suite: be.SuiteSpec):
        observed_roots.append(os.environ.get("BFCL_PROJECT_ROOT"))
        raise RuntimeError("stop after project-root check")

    monkeypatch.setattr(be, "_build_suite_case_ids", stop_after_observing_root)

    passed = be.run_evaluation(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model="model-a",
        output_dir=output_dir,
        bfcl_project_root=project_root,
        suite=be.MINIMAX_SUITE,
    )

    assert passed is False
    assert observed_roots == [str(project_root)]


@pytest.mark.parametrize("suite", (be.MINIMAX_SUITE, be.KIMI_SUITE))
def test_full_suite_ids_use_exact_sorted_leaf_allocations(
    monkeypatch,
    suite: be.SuiteSpec,
) -> None:
    multi_turn_leaves = (
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    )
    upstream_multi_turn_leaves = (
        "multi_turn_base",
        "multi_turn_long_context",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
    )
    available_counts = {
        "simple_python": 400,
        "multiple": 200,
        "parallel": 200,
        "parallel_multiple": 200,
        **{leaf: 75 for leaf in multi_turn_leaves},
    }

    def load_dataset_entry(category: str) -> list[dict[str, str]]:
        return [
            {"id": f"{category}_{index:03d}"}
            for index in reversed(range(available_counts[category]))
        ]

    def parse_test_category_argument(categories: list[str]) -> list[str]:
        assert len(categories) == 1
        return (
            list(upstream_multi_turn_leaves)
            if categories[0] == "multi_turn"
            else categories
        )

    def load_helpers():
        return (
            load_dataset_entry,
            parse_test_category_argument,
            lambda entry: entry["id"],
        )

    monkeypatch.setattr(be, "_load_dataset_helpers", load_helpers)

    selected = be._build_suite_case_ids(suite)

    assert (
        tuple((category, len(case_ids)) for category, case_ids in selected.items())
        == suite.expected_leaf_counts
    )
    assert sum(map(len, selected.values())) == suite.expected_sample_count
    assert all(list(case_ids) == sorted(case_ids) for case_ids in selected.values())
    if suite is be.KIMI_SUITE:
        assert {leaf: len(selected[leaf]) for leaf in multi_turn_leaves} == {
            leaf: 60 for leaf in multi_turn_leaves
        }
        assert all(selected[leaf][-1].endswith("_059") for leaf in multi_turn_leaves)


def test_mixed_category_diagnostics_project_each_failure_id() -> None:
    score = be.CategoryScore(
        category="parallel",
        case_ids=("parallel_0", "parallel_1", "parallel_2"),
        score_file="score/model-a/BFCL_v4_parallel_score.json",
        header={"accuracy": 2 / 3, "correct_count": 2, "total_count": 3},
        records=({"id": "parallel_1", "error": "wrong call"},),
        accuracy=2 / 3,
        correct_count=2,
        total_count=3,
    )

    assert score.as_dict()["case_scores"] == [
        {"id": "parallel_0", "score": 1.0, "correct": True},
        {"id": "parallel_1", "score": 0.0, "correct": False},
        {"id": "parallel_2", "score": 1.0, "correct": True},
    ]


def test_kimi_projects_namespaced_leaf_and_weighted_aggregate_scores() -> None:
    assert be.KIMI_SUITE.expected_sample_count == 1240
    selected = {
        category: tuple(f"{category}_{index}" for index in range(total_count))
        for category, total_count in be.KIMI_SUITE.expected_leaf_counts
    }
    correct_counts = {
        "simple_python": 200,
        "multiple": 100,
        "parallel": 50,
        "parallel_multiple": 200,
        "multi_turn_base": 60,
        "multi_turn_miss_func": 30,
        "multi_turn_miss_param": 0,
        "multi_turn_long_context": 15,
    }
    scores = [
        _category_score(category, total_count, correct_counts[category])
        for category, total_count in be.KIMI_SUITE.expected_leaf_counts
    ]

    compatibility = be._compatibility_result(
        suite=be.KIMI_SUITE,
        case_ids_by_category=selected,
        model="model-a",
        scores=scores,
    )

    assert compatibility["results"]["bfcl_vllm_kimi"]["acc,none"] == 655 / 1240
    assert (
        compatibility["results"]["bfcl_vllm_kimi_multi_turn"]["acc,none"] == 105 / 240
    )
    assert compatibility["n-samples"]["bfcl_vllm_kimi_multi_turn"] == {
        "original": 240,
        "effective": 240,
    }
    assert all(
        f"bfcl_vllm_kimi_{category}" in compatibility["results"]
        for category in selected
    )
    assert "bfcl_simple_python" not in compatibility["results"]
    assert "bfcl_multi_turn" not in compatibility["results"]


def test_selected_suite_integration_error_preserves_suite_identity(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"

    return_code = be.main(
        [
            "--model",
            "model-a",
            "--output-dir",
            str(output_dir),
            "--suite",
            "bfcl_vllm_minimax_m3",
            "--integration-error",
            "pinned wheel installation failed",
        ]
    )

    assert return_code == 1
    compatibility = _compatibility(output_dir)
    native = _native(output_dir)
    assert native["task"] == "bfcl_vllm_minimax_m3"
    assert native["summary"]["expected_count"] == 1000
    assert native["sampling"] == {
        "temperature": 0.001,
        "num_threads": 8,
    }
    assert list(compatibility["results"]) == [
        "bfcl_vllm_minimax_m3",
        "bfcl_vllm_minimax_m3_simple_python",
        "bfcl_vllm_minimax_m3_multiple",
        "bfcl_vllm_minimax_m3_parallel",
        "bfcl_vllm_minimax_m3_parallel_multiple",
    ]
    assert compatibility["n-samples"]["bfcl_vllm_minimax_m3"] == {
        "original": 1000,
        "effective": 0,
    }
    assert native["integration_error"] == compatibility["integration_error"]


def test_score_total_must_match_every_selected_id(tmp_path: Path) -> None:
    project_root = tmp_path / "bfcl"
    score_path = project_root / "score" / "model-a" / "BFCL_v4_simple_python_score.json"
    score_path.parent.mkdir(parents=True)
    score_path.write_text(
        json.dumps({"accuracy": 1.0, "correct_count": 1, "total_count": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="simple_python evaluated 1 cases; expected 2",
    ):
        be._collect_scores(
            project_root,
            {"simple_python": ("simple_python_0", "simple_python_1")},
        )


def test_full_suite_handler_bounds_openai_requests() -> None:
    class StockOpenAICompletionsHandler:
        def _build_client_kwargs(self) -> dict[str, Any]:
            return {"api_key": "stock-key"}

    handler = be._bounded_openai_handler(StockOpenAICompletionsHandler)

    assert issubclass(handler, StockOpenAICompletionsHandler)
    assert handler()._build_client_kwargs() == {
        "api_key": "stock-key",
        "timeout": 180,
        "max_retries": 2,
    }


def test_kimi_suite_caps_multi_turn_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    constants = ModuleType("bfcl_eval.constants")
    constants.__path__ = []
    prompts = ModuleType("bfcl_eval.constants.default_prompts")
    prompts.MAXIMUM_STEP_LIMIT = 20
    constants.default_prompts = prompts
    monkeypatch.setitem(sys.modules, "bfcl_eval.constants", constants)
    monkeypatch.setitem(
        sys.modules,
        "bfcl_eval.constants.default_prompts",
        prompts,
    )

    be._apply_suite_runtime_limits(be.MINIMAX_SUITE)
    assert prompts.MAXIMUM_STEP_LIMIT == 20

    be._apply_suite_runtime_limits(be.KIMI_SUITE)
    assert prompts.MAXIMUM_STEP_LIMIT == 10


def test_upstream_registration_uses_exact_stock_openai_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "bfcl"
    be._write_id_map(project_root, be.SMOKE_CASE_IDS)
    model_config_mapping: dict[str, Any] = {}

    class ModelConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class OpenAICompletionsHandler:
        pass

    def generate(**_: Any) -> None:
        result_dir = project_root / "result" / "model-a"
        result_dir.mkdir(parents=True)
        for category, case_ids in be.SMOKE_CASE_IDS.items():
            (result_dir / f"BFCL_v4_{category}_result.json").write_text(
                "".join(
                    json.dumps({"id": case_id, "result": []}) + "\n"
                    for case_id in case_ids
                ),
                encoding="utf-8",
            )

    def evaluate(**_: Any) -> None:
        pass

    modules = {
        "bfcl_eval": ModuleType("bfcl_eval"),
        "bfcl_eval.constants": ModuleType("bfcl_eval.constants"),
        "bfcl_eval.constants.model_config": ModuleType(
            "bfcl_eval.constants.model_config"
        ),
        "bfcl_eval.model_handler": ModuleType("bfcl_eval.model_handler"),
        "bfcl_eval.model_handler.api_inference": ModuleType(
            "bfcl_eval.model_handler.api_inference"
        ),
        "bfcl_eval.model_handler.api_inference.openai_completion": ModuleType(
            "bfcl_eval.model_handler.api_inference.openai_completion"
        ),
        "bfcl_eval.__main__": ModuleType("bfcl_eval.__main__"),
    }
    modules[
        "bfcl_eval.constants.model_config"
    ].MODEL_CONFIG_MAPPING = model_config_mapping
    modules["bfcl_eval.constants.model_config"].ModelConfig = ModelConfig
    modules[
        "bfcl_eval.model_handler.api_inference.openai_completion"
    ].OpenAICompletionsHandler = OpenAICompletionsHandler
    modules["bfcl_eval.__main__"].generate = generate
    modules["bfcl_eval.__main__"].evaluate = evaluate
    for name, module in modules.items():
        if name in {
            "bfcl_eval",
            "bfcl_eval.constants",
            "bfcl_eval.model_handler",
            "bfcl_eval.model_handler.api_inference",
        }:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(be, "_function_defaults", lambda _: {})

    be._run_upstream(
        model="model-a",
        project_root=project_root,
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        num_threads=4,
    )

    assert model_config_mapping["model-a"].model_handler is OpenAICompletionsHandler
