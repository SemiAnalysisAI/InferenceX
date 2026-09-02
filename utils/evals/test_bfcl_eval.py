import builtins
import json
import os
import subprocess
import sys
from dataclasses import replace
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


def test_thresholds_are_stdlib_readable_without_pyyaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def import_without_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)
    config = {"default": {"task": 0.25}, "models": {"model-a": {"task": 0.75}}}
    path = tmp_path / "thresholds.yaml"
    path.write_text(json.dumps(config))

    assert vs.load_config(str(path)) == config


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
    (tmp_path / "thresholds.json").write_text(json.dumps({"bfcl_smoke": 0.5}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["validate_scores.py", "--thresholds", "thresholds.json"]
    )

    assert vs.main() == 0


def test_adapter_module_does_not_collide_with_upstream_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = object()
    child = object()
    monkeypatch.setitem(sys.modules, "bfcl_eval", upstream)
    monkeypatch.setitem(sys.modules, "bfcl_eval.constants", child)

    be._clear_upstream_modules()

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


def test_perfect_score_projects_upstream_headers_and_compatibility_metrics(
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
    assert (
        (project_root / be.UPSTREAM_LICENSE_FILENAME)
        .read_bytes()
        == (Path(be.__file__).resolve().parents[2] / "LICENSE").read_bytes()
    )

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


def test_suite_selection_sorts_before_limiting_and_distributes_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = be.SuiteSpec(
        name="synthetic",
        generation_categories=("group", "standalone"),
        expected_leaf_counts=(("a", 2), ("b", 3), ("standalone", 1)),
        category_limits=(("group", 5),),
        temperature=0.0,
        default_num_threads=1,
        threshold=0.0,
    )
    datasets = {
        "a": [{"id": "a_2"}, {"id": "a_0"}, {"id": "a_1"}],
        "b": [{"id": "b_3"}, {"id": "b_1"}, {"id": "b_0"}, {"id": "b_2"}],
        "standalone": [{"id": "standalone_0"}],
    }
    monkeypatch.setattr(
        be,
        "_load_dataset_helpers",
        lambda: (
            datasets.__getitem__,
            lambda categories: ["b", "a"] if categories == ["group"] else categories,
            lambda entry: entry["id"],
        ),
    )

    selected = be._build_suite_case_ids(suite)

    assert list(selected.items()) == [
        ("a", ("a_0", "a_1")),
        ("b", ("b_0", "b_1", "b_2")),
        ("standalone", ("standalone_0",)),
    ]


@pytest.mark.parametrize(
    ("ids", "error"),
    [(("case_0", "case_0"), "duplicate id"), (("case_0",), "selected leaf counts")],
)
def test_suite_selection_rejects_duplicate_or_missing_cases(
    monkeypatch: pytest.MonkeyPatch, ids: tuple[str, ...], error: str,
) -> None:
    suite = be.SuiteSpec(
        name="synthetic",
        generation_categories=("leaf",),
        expected_leaf_counts=(("leaf", 2),),
        temperature=0.0,
        default_num_threads=1,
        threshold=0.0,
    )
    monkeypatch.setattr(
        be,
        "_load_dataset_helpers",
        lambda: (
            lambda _: [{"id": case_id} for case_id in ids],
            lambda categories: categories,
            lambda entry: entry["id"],
        ),
    )

    with pytest.raises(ValueError, match=error):
        be._build_suite_case_ids(suite)


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
    counts = {
        "simple_python": (3, 1),
        "multiple": (1, 1),
        "parallel": (2, 0),
        "parallel_multiple": (4, 3),
        "multi_turn_base": (1, 1),
        "multi_turn_miss_func": (2, 1),
        "multi_turn_miss_param": (3, 0),
        "multi_turn_long_context": (4, 2),
    }
    selected = {
        category: tuple(f"{category}_{index}" for index in range(total_count))
        for category, (total_count, _) in counts.items()
    }
    scores = [
        _category_score(category, total_count, correct_count)
        for category, (total_count, correct_count) in counts.items()
    ]

    compatibility = be._compatibility_result(
        suite=be.KIMI_SUITE,
        case_ids_by_category=selected,
        model="model-a",
        scores=scores,
    )

    assert compatibility["results"]["bfcl_vllm_kimi"]["acc,none"] == 0.45
    assert (
        compatibility["results"]["bfcl_vllm_kimi_multi_turn"]["acc,none"] == 0.4
    )
    assert compatibility["n-samples"]["bfcl_vllm_kimi"]["effective"] == 20
    assert compatibility["n-samples"]["bfcl_vllm_kimi_multi_turn"]["effective"] == 10
    assert all(
        f"bfcl_vllm_kimi_{category}" in compatibility["results"]
        for category in selected
    )
    assert "bfcl_simple_python" not in compatibility["results"]
    assert "bfcl_multi_turn" not in compatibility["results"]


def test_selected_suite_integration_error_preserves_suite_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    suite = replace(
        be.MINIMAX_SUITE,
        name="custom_suite",
        generation_categories=("left", "right"),
        expected_leaf_counts=(("left", 2), ("right", 1)),
        temperature=0.25,
        default_num_threads=7,
    )
    monkeypatch.setattr(be, "SUITE_SPECS", {suite.name: suite})

    return_code = be.main(
        [
            "--model",
            "model-a",
            "--output-dir",
            str(output_dir),
            "--suite",
            "custom_suite",
            "--integration-error",
            "pinned wheel installation failed",
        ]
    )

    assert return_code == 1
    compatibility = _compatibility(output_dir)
    native = _native(output_dir)
    assert native["task"] == "custom_suite"
    assert native["summary"]["expected_count"] == 3
    assert native["sampling"] == {
        "temperature": 0.25,
        "num_threads": 7,
    }
    assert list(compatibility["results"]) == [
        "custom_suite",
        "custom_suite_left",
        "custom_suite_right",
    ]
    assert compatibility["n-samples"]["custom_suite"] == {
        "original": 3,
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
