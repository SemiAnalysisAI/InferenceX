#!/usr/bin/env python3
"""Run pinned BFCL V4 OpenAI chat-completions suites."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

TASK_NAME = "bfcl_smoke"
NATIVE_REPORT_FILENAME = "bfcl_report.json"
COMPATIBILITY_FILENAME = "results_bfcl.json"
COMPATIBILITY_GLOB = "results_bfcl*.json"
RESULT_FORMAT = "inferencex-eval-v1"
ADAPTER_NAME = "bfcl-v4-openai-completions"
DEFAULT_NUM_THREADS = 4
REQUIRED_SCORE = 0.0
FULL_SUITE_REQUEST_TIMEOUT_SECONDS = 180
FULL_SUITE_REQUEST_MAX_RETRIES = 2
KIMI_MAXIMUM_STEP_LIMIT = 10

BFCL_PACKAGE = "bfcl-eval"
BFCL_PACKAGE_VERSION = "2026.3.23"
BFCL_WHEEL_SHA256 = "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"
UPSTREAM_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
UPSTREAM_SOURCE = "https://pypi.org/project/bfcl-eval/2026.3.23/"
UPSTREAM_REF = f"{BFCL_PACKAGE}=={BFCL_PACKAGE_VERSION}"
SOURCE_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
VLLM_INTEGRATION_REF = "7ecb11405df86b202f4c5cca322bd133052fee82"
UPSTREAM_LICENSE = "Apache-2.0"
UPSTREAM_LICENSE_URL = f"{UPSTREAM_REPOSITORY}/blob/{SOURCE_REVISION}/LICENSE"
UPSTREAM_LICENSE_FILENAME = "BFCL_LICENSE.apache-2.0.txt"
UPSTREAM_ATTRIBUTION_FILENAME = "BFCL_ATTRIBUTION.json"

# Dict insertion order is intentional: smoke reports and the upstream run-ID
# file remain byte-for-byte stable.
SMOKE_CASE_IDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "simple_python": ("simple_python_141",),
        "multiple": ("multiple_38",),
        "parallel": ("parallel_1",),
        "irrelevance": ("irrelevance_0",),
    }
)
EXPECTED_SAMPLE_COUNT = sum(len(case_ids) for case_ids in SMOKE_CASE_IDS.values())


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    generation_categories: tuple[str, ...]
    expected_leaf_counts: tuple[tuple[str, int], ...]
    temperature: float
    default_num_threads: int
    threshold: float
    category_limits: tuple[tuple[str, int], ...] = ()

    @property
    def leaf_categories(self) -> tuple[str, ...]:
        return tuple(category for category, _ in self.expected_leaf_counts)

    @property
    def expected_sample_count(self) -> int:
        return sum(count for _, count in self.expected_leaf_counts)

    def projected_task(self, category: str) -> str:
        if self.name == TASK_NAME:
            return f"bfcl_{category}"
        return f"{self.name}_{category}"


SMOKE_SUITE = SuiteSpec(
    name=TASK_NAME,
    generation_categories=tuple(SMOKE_CASE_IDS),
    expected_leaf_counts=tuple(
        (category, len(case_ids)) for category, case_ids in SMOKE_CASE_IDS.items()
    ),
    temperature=0.0,
    default_num_threads=DEFAULT_NUM_THREADS,
    threshold=REQUIRED_SCORE,
)
MINIMAX_SUITE = SuiteSpec(
    name="bfcl_vllm_minimax_m3",
    generation_categories=(
        "simple_python",
        "multiple",
        "parallel",
        "parallel_multiple",
    ),
    expected_leaf_counts=(
        ("simple_python", 400),
        ("multiple", 200),
        ("parallel", 200),
        ("parallel_multiple", 200),
    ),
    temperature=0.001,
    default_num_threads=8,
    threshold=0.0,
)
KIMI_SUITE = SuiteSpec(
    name="bfcl_vllm_kimi",
    generation_categories=(
        "simple_python",
        "multiple",
        "parallel",
        "parallel_multiple",
        "multi_turn",
    ),
    expected_leaf_counts=(
        ("simple_python", 400),
        ("multiple", 200),
        ("parallel", 200),
        ("parallel_multiple", 200),
        ("multi_turn_base", 60),
        ("multi_turn_miss_func", 60),
        ("multi_turn_miss_param", 60),
        ("multi_turn_long_context", 60),
    ),
    temperature=0.001,
    default_num_threads=16,
    threshold=0.0,
    category_limits=(("multi_turn", 240),),
)
SUITE_SPECS: Mapping[str, SuiteSpec] = MappingProxyType(
    {
        suite.name: suite
        for suite in (
            SMOKE_SUITE,
            MINIMAX_SUITE,
            KIMI_SUITE,
        )
    }
)


class UpstreamRunner(Protocol):
    """Injectable boundary around the optional BFCL installation."""

    def __call__(
        self,
        *,
        model: str,
        project_root: Path,
        base_url: str,
        api_key: str,
        num_threads: int,
    ) -> None: ...


@dataclass(frozen=True)
class CategoryScore:
    category: str
    case_ids: tuple[str, ...]
    score_file: str
    header: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    accuracy: float
    correct_count: int
    total_count: int

    def as_dict(self) -> dict[str, Any]:
        failure_ids = {record["id"] for record in self.records}
        return {
            "category": self.category,
            "case_ids": list(self.case_ids),
            "score_file": self.score_file,
            "score_header": dict(self.header),
            "score_records": [dict(record) for record in self.records],
            "case_scores": [
                {
                    "id": case_id,
                    "score": 0.0 if case_id in failure_ids else 1.0,
                    "correct": case_id not in failure_ids,
                }
                for case_id in self.case_ids
            ],
        }


def _nonempty_string(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be a non-empty string")
    return value.strip()


def _absolute_http_url(value: str) -> str:
    normalized = _nonempty_string(value).rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "must be an absolute HTTP(S) URL without query or fragment"
        )
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        raise argparse.ArgumentTypeError(
            "must be an API root URL; the OpenAI client appends /chat/completions"
        )
    return normalized


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _source_details(
    suite: SuiteSpec, case_ids_by_category: Mapping[str, tuple[str, ...]]
) -> dict[str, Any]:
    return {
        "url": UPSTREAM_SOURCE,
        "repository": UPSTREAM_REPOSITORY,
        "ref": UPSTREAM_REF,
        "package": BFCL_PACKAGE,
        "package_version": BFCL_PACKAGE_VERSION,
        "wheel_sha256": BFCL_WHEEL_SHA256,
        "source_revision": SOURCE_REVISION,
        "vllm_integration_ref": VLLM_INTEGRATION_REF,
        "case_ids": {
            category: list(case_ids)
            for category, case_ids in case_ids_by_category.items()
        },
    }


def _error_dict(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _expected_category_details(
    case_ids_by_category: Mapping[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    return [
        {
            "category": category,
            "case_ids": list(case_ids),
            "score_file": None,
            "score_header": None,
            "score_records": [],
            "case_scores": [
                {"id": case_id, "score": 0.0, "correct": False} for case_id in case_ids
            ],
        }
        for category, case_ids in case_ids_by_category.items()
    ]


def _diagnostics(
    suite: SuiteSpec,
    case_ids_by_category: Mapping[str, tuple[str, ...]],
    scores: Sequence[CategoryScore] | None = None,
) -> dict[str, Any]:
    return {
        "source": _source_details(suite, case_ids_by_category),
        "categories": (
            [score.as_dict() for score in scores]
            if scores is not None
            else _expected_category_details(case_ids_by_category)
        ),
    }


def _native_report(
    *,
    suite: SuiteSpec,
    case_ids_by_category: Mapping[str, tuple[str, ...]],
    model: str,
    base_url: str | None,
    num_threads: int,
    scores: Sequence[CategoryScore] | None,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    correct_count = sum(score.correct_count for score in scores or ())
    total_count = sum(score.total_count for score in scores or ())
    accuracy = correct_count / total_count if total_count else 0.0
    report: dict[str, Any] = {
        "verifier": ADAPTER_NAME,
        "task": suite.name,
        "model": model,
        "endpoint": base_url,
        "completed": integration_error is None,
        "passed": integration_error is None and accuracy >= suite.threshold,
        "threshold": suite.threshold,
        "sampling": {
            "temperature": suite.temperature,
            "num_threads": num_threads,
        },
        "summary": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "expected_count": suite.expected_sample_count,
        },
        "bfcl": _diagnostics(suite, case_ids_by_category, scores),
    }
    if integration_error is not None:
        report["integration_error"] = _error_dict(integration_error)
    return report


def _compatibility_result(
    *,
    suite: SuiteSpec,
    case_ids_by_category: Mapping[str, tuple[str, ...]],
    model: str,
    scores: Sequence[CategoryScore] | None,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    score_by_category = {score.category: score for score in scores or ()}
    total_count = sum(score.total_count for score in scores or ())
    correct_count = sum(score.correct_count for score in scores or ())
    accuracy = correct_count / total_count if total_count else 0.0
    task_scores = {suite.name: accuracy}
    task_samples = {
        suite.name: {
            "original": suite.expected_sample_count,
            "effective": total_count,
        }
    }
    for category, expected_count in suite.expected_leaf_counts:
        category_score = score_by_category.get(category)
        task_name = suite.projected_task(category)
        task_scores[task_name] = (
            category_score.accuracy if category_score is not None else 0.0
        )
        task_samples[task_name] = {
            "original": expected_count,
            "effective": (
                category_score.total_count if category_score is not None else 0
            ),
        }

    if suite is KIMI_SUITE:
        multi_turn_categories = suite.leaf_categories[-4:]
        multi_turn_scores = [
            score_by_category[category]
            for category in multi_turn_categories
            if category in score_by_category
        ]
        multi_turn_total = sum(score.total_count for score in multi_turn_scores)
        multi_turn_correct = sum(score.correct_count for score in multi_turn_scores)
        multi_turn_accuracy = (
            multi_turn_correct / multi_turn_total if multi_turn_total else 0.0
        )
        aggregate_task = suite.projected_task("multi_turn")
        task_scores[aggregate_task] = multi_turn_accuracy
        task_samples[aggregate_task] = {
            "original": 240,
            "effective": multi_turn_total,
        }

    results = {
        task_name: {
            "acc,none": task_score,
            "acc_stderr,none": 0.0,
        }
        for task_name, task_score in task_scores.items()
    }
    configs = {
        task_name: {
            "metric_list": [{"metric": "acc"}],
            "filter_list": [{"name": "none"}],
        }
        for task_name in task_scores
    }
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": ADAPTER_NAME,
        "model_name": model,
        "results": results,
        "configs": configs,
        "n-samples": task_samples,
        "bfcl": _diagnostics(suite, case_ids_by_category, scores),
    }
    if integration_error is not None:
        result["integration_error"] = _error_dict(integration_error)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_upstream_attribution(project_root: Path) -> None:
    """Keep BFCL provenance and its Apache license with archived outputs."""
    project_root.mkdir(parents=True, exist_ok=True)
    repository_license = Path(__file__).resolve().parents[2] / "LICENSE"
    if not repository_license.is_file():
        raise FileNotFoundError(f"Apache license file not found: {repository_license}")
    (project_root / UPSTREAM_LICENSE_FILENAME).write_bytes(
        repository_license.read_bytes()
    )
    _write_json(
        project_root / UPSTREAM_ATTRIBUTION_FILENAME,
        {
            "artifact": "BFCL-generated evaluation results",
            "upstream": {
                "package": BFCL_PACKAGE,
                "package_version": BFCL_PACKAGE_VERSION,
                "wheel_sha256": BFCL_WHEEL_SHA256,
                "repository": UPSTREAM_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "vllm_integration_revision": VLLM_INTEGRATION_REF,
                "license": UPSTREAM_LICENSE,
                "license_url": UPSTREAM_LICENSE_URL,
                "license_file": UPSTREAM_LICENSE_FILENAME,
            },
            "modifications": (
                "InferenceX selected deterministic case subsets and projected "
                "upstream scores; this archive does not modify upstream BFCL source."
            ),
        },
    )


def _prepare_output_paths(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    native_path = output_dir / NATIVE_REPORT_FILENAME
    native_path.unlink(missing_ok=True)
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        if stale_path.is_file() or stale_path.is_symlink():
            stale_path.unlink()
    compatibility_path = output_dir / COMPATIBILITY_FILENAME
    return native_path, compatibility_path


def _clear_upstream_modules() -> None:
    """Reload BFCL's import-time paths and limits for each adapter invocation."""
    for module_name in tuple(sys.modules):
        if module_name == "bfcl_eval" or module_name.startswith("bfcl_eval."):
            sys.modules.pop(module_name, None)


def _apply_suite_runtime_limits(suite: SuiteSpec) -> None:
    """Apply pinned BFCL limits before importing its model handlers."""
    if suite is KIMI_SUITE:
        from bfcl_eval.constants import default_prompts as bfcl_prompts

        bfcl_prompts.MAXIMUM_STEP_LIMIT = KIMI_MAXIMUM_STEP_LIMIT


def _bounded_openai_handler(stock_handler: type[Any]) -> type[Any]:
    """Retain BFCL's handler while bounding its OpenAI transport."""

    class BoundedOpenAICompletionsHandler(stock_handler):
        def _build_client_kwargs(self) -> dict[str, Any]:
            kwargs = super()._build_client_kwargs()
            kwargs.update(
                timeout=FULL_SUITE_REQUEST_TIMEOUT_SECONDS,
                max_retries=FULL_SUITE_REQUEST_MAX_RETRIES,
            )
            return kwargs

    return BoundedOpenAICompletionsHandler




def _write_id_map(
    project_root: Path, case_ids_by_category: Mapping[str, tuple[str, ...]]
) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        project_root / "test_case_ids_to_generate.json",
        {
            category: list(case_ids)
            for category, case_ids in case_ids_by_category.items()
        },
    )


def _function_defaults(function: Callable[..., Any]) -> dict[str, Any]:
    """Resolve Typer OptionInfo defaults before directly calling a command function."""
    import typer  # Lazy: only the real BFCL path needs this third-party dependency.

    defaults: dict[str, Any] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue
        default = parameter.default
        if isinstance(default, typer.models.OptionInfo):
            default = default.default
        defaults[name] = default
    return defaults


def _load_dataset_helpers() -> tuple[
    Callable[[str], Any],
    Callable[[list[str]], Any],
    Callable[[Any], Any],
]:
    """Import BFCL's pinned dataset helpers only when a full suite is selected."""
    from bfcl_eval.utils import (
        load_dataset_entry,
        parse_test_category_argument,
        sort_key,
    )

    return load_dataset_entry, parse_test_category_argument, sort_key


def _build_suite_case_ids(
    suite: SuiteSpec,
) -> dict[str, tuple[str, ...]]:
    if suite is SMOKE_SUITE:
        return dict(SMOKE_CASE_IDS)

    load_dataset_entry, parse_test_category_argument, sort_key = _load_dataset_helpers()
    category_limits = dict(suite.category_limits)
    selected_by_leaf: dict[str, tuple[str, ...]] = {}
    seen_ids: set[str] = set()
    for category in suite.generation_categories:
        leaf_categories = list(parse_test_category_argument([category]))
        by_leaf = {
            leaf: sorted(load_dataset_entry(leaf), key=sort_key)
            for leaf in leaf_categories
        }
        limit = category_limits.get(category)
        if limit is None:
            quotas = [len(by_leaf[leaf]) for leaf in leaf_categories]
        else:
            base, extra = divmod(limit, len(leaf_categories))
            quotas = [base + (index < extra) for index in range(len(leaf_categories))]
        for leaf, quota in zip(leaf_categories, quotas, strict=True):
            entries = by_leaf[leaf][:quota]
            ids: list[str] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"{leaf} dataset entry must be a mapping")
                case_id = entry.get("id")
                if not isinstance(case_id, str) or not case_id:
                    raise ValueError(
                        f"{leaf} dataset entry must contain a non-empty string id"
                    )
                if case_id in seen_ids:
                    raise ValueError(f"BFCL dataset contains duplicate id {case_id}")
                seen_ids.add(case_id)
                ids.append(case_id)
            selected_by_leaf[leaf] = tuple(ids)

    actual_counts = {
        category: len(case_ids) for category, case_ids in selected_by_leaf.items()
    }
    expected_counts = dict(suite.expected_leaf_counts)
    if actual_counts != expected_counts:
        raise ValueError(
            f"{suite.name} selected leaf counts {actual_counts!r}; "
            f"expected {expected_counts!r}"
        )
    return {
        category: selected_by_leaf[category]
        for category, _ in suite.expected_leaf_counts
    }


def _read_selected_suite(
    project_root: Path,
) -> tuple[SuiteSpec, dict[str, tuple[str, ...]]]:
    raw = json.loads(
        (project_root / "test_case_ids_to_generate.json").read_text(encoding="utf-8")
    )
    if not isinstance(raw, Mapping):
        raise ValueError("BFCL test-case ID map must be a JSON object")
    case_ids_by_category: dict[str, tuple[str, ...]] = {}
    for category, case_ids in raw.items():
        if not isinstance(category, str) or not isinstance(case_ids, list):
            raise ValueError("BFCL test-case ID map has an invalid category entry")
        if not all(isinstance(case_id, str) and case_id for case_id in case_ids):
            raise ValueError(f"{category} test-case IDs must be non-empty strings")
        case_ids_by_category[category] = tuple(case_ids)

    shape = tuple(
        (category, len(case_ids)) for category, case_ids in case_ids_by_category.items()
    )
    for suite in SUITE_SPECS.values():
        if shape != suite.expected_leaf_counts:
            continue
        if suite is SMOKE_SUITE and case_ids_by_category != dict(SMOKE_CASE_IDS):
            continue
        return suite, case_ids_by_category
    raise ValueError(f"test-case ID map does not match a supported suite: {shape!r}")


def _run_upstream(
    *,
    model: str,
    project_root: Path,
    base_url: str,
    api_key: str,
    num_threads: int,
) -> None:
    """Lazily load and invoke the pinned BFCL API against an existing server."""
    suite, case_ids_by_category = _read_selected_suite(project_root)
    _apply_suite_runtime_limits(suite)
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key

    import bfcl_eval.constants.model_config as bfcl_model_config
    from bfcl_eval.__main__ import evaluate, generate
    from bfcl_eval.constants.model_config import ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    handler = (
        OpenAICompletionsHandler
        if suite is SMOKE_SUITE
        else _bounded_openai_handler(OpenAICompletionsHandler)
    )

    bfcl_model_config.MODEL_CONFIG_MAPPING[model] = ModelConfig(
        model_name=model,
        display_name=f"{model} (FC) (InferenceX)",
        url="",
        org="",
        license="unknown",
        model_handler=handler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )

    categories = list(suite.generation_categories)
    generation_kwargs = _function_defaults(generate)
    generation_kwargs.update(
        model=[model],
        test_category=categories,
        temperature=suite.temperature,
        num_threads=num_threads,
        skip_server_setup=True,
        run_ids=True,
        allow_overwrite=True,
    )
    generate(**generation_kwargs)
    _validate_generated_results(project_root, case_ids_by_category)

    evaluation_kwargs = _function_defaults(evaluate)
    evaluation_kwargs.update(
        model=[model],
        test_category=categories,
        partial_eval=True,
    )
    evaluate(**evaluation_kwargs)


def _validate_generated_results(
    project_root: Path,
    case_ids_by_category: Mapping[str, tuple[str, ...]] = SMOKE_CASE_IDS,
) -> None:
    for category, case_ids in case_ids_by_category.items():
        matches = sorted(project_root.glob(f"result/**/BFCL_v4_{category}_result.json"))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {category} result file, found {len(matches)}"
            )
        result_ids: list[str] = []
        with matches[0].open(encoding="utf-8") as result_file:
            for line_number, result_line in enumerate(result_file, start=1):
                try:
                    record = json.loads(result_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{category} result record on line {line_number} "
                        "is malformed JSON"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"{category} result record on line {line_number} "
                        "must be a JSON object"
                    )
                case_id = record.get("id")
                if not isinstance(case_id, str) or not case_id:
                    raise ValueError(
                        f"{category} result record on line {line_number} "
                        "must contain a non-empty string id"
                    )
                if case_id not in case_ids:
                    raise ValueError(
                        f"{category} result file contains unexpected id {case_id}"
                    )
                if case_id in result_ids:
                    raise ValueError(
                        f"{category} result file contains duplicate id {case_id}"
                    )
                if "traceback" in record or (
                    isinstance(record.get("result"), str)
                    and record["result"].startswith("Error during inference:")
                ):
                    raise RuntimeError(
                        f"{category} result {case_id} contains an inference error"
                    )
                result_ids.append(case_id)
        if result_ids != list(case_ids):
            raise ValueError(
                f"{category} result ids {result_ids!r} "
                f"do not match expected ids {list(case_ids)!r}"
            )


def _validate_header(
    *, category: str, header: Any, expected_count: int
) -> tuple[float, int, int]:
    if not isinstance(header, Mapping):
        raise ValueError(f"{category} score header must be a JSON object")
    missing = {"accuracy", "correct_count", "total_count"} - header.keys()
    if missing:
        raise ValueError(
            f"{category} score header missing: {', '.join(sorted(missing))}"
        )

    correct_count = header["correct_count"]
    total_count = header["total_count"]
    accuracy = header["accuracy"]
    if isinstance(correct_count, bool) or not isinstance(correct_count, int):
        raise ValueError(f"{category} correct_count must be an integer")
    if isinstance(total_count, bool) or not isinstance(total_count, int):
        raise ValueError(f"{category} total_count must be an integer")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
    ):
        raise ValueError(f"{category} accuracy must be a finite number")
    if total_count != expected_count:
        raise ValueError(
            f"{category} evaluated {total_count} cases; expected {expected_count}"
        )
    if correct_count < 0 or correct_count > total_count:
        raise ValueError(f"{category} correct_count is outside [0, total_count]")

    computed_accuracy = correct_count / total_count
    if not math.isclose(float(accuracy), computed_accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{category} accuracy is inconsistent with its count fields")
    return float(accuracy), correct_count, total_count


def _collect_scores(
    project_root: Path,
    case_ids_by_category: Mapping[str, tuple[str, ...]] = SMOKE_CASE_IDS,
) -> list[CategoryScore]:
    scores: list[CategoryScore] = []
    for category, case_ids in case_ids_by_category.items():
        matches = sorted(project_root.glob(f"score/**/BFCL_v4_{category}_score.json"))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {category} score file, found {len(matches)}"
            )
        score_path = matches[0]
        with score_path.open(encoding="utf-8") as score_file:
            first_line = score_file.readline()
            record_lines = list(score_file)
        if not first_line:
            raise ValueError(f"{category} score file has no header")
        try:
            header = json.loads(first_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{category} score header is malformed JSON") from exc
        accuracy, correct_count, total_count = _validate_header(
            category=category,
            header=header,
            expected_count=len(case_ids),
        )
        records: list[dict[str, Any]] = []
        record_ids: list[str] = []
        for line_number, record_line in enumerate(record_lines, start=2):
            try:
                record = json.loads(record_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{category} score record on line {line_number} is malformed JSON"
                ) from exc
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"{category} score record on line {line_number} "
                    "must be a JSON object"
                )
            case_id = record.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(
                    f"{category} score record on line {line_number} "
                    "must contain a non-empty string id"
                )
            if case_id not in case_ids:
                raise ValueError(
                    f"{category} score file contains unexpected id {case_id}"
                )
            if case_id in record_ids:
                raise ValueError(
                    f"{category} score file contains duplicate id {case_id}"
                )
            record_ids.append(case_id)
            records.append(dict(record))
        expected_failure_count = total_count - correct_count
        if len(records) != expected_failure_count:
            raise ValueError(
                f"{category} score file contains {len(records)} failure records; "
                f"expected {expected_failure_count}"
            )
        scores.append(
            CategoryScore(
                category=category,
                case_ids=case_ids,
                score_file=score_path.relative_to(project_root).as_posix(),
                header=dict(header),
                records=tuple(records),
                accuracy=accuracy,
                correct_count=correct_count,
                total_count=total_count,
            )
        )
    return scores


def publish_integration_error(
    *,
    output_dir: Path,
    model: str,
    error: BaseException,
    suite: SuiteSpec = SMOKE_SUITE,
) -> None:
    """Publish required zero-score artifacts without importing BFCL or Typer."""
    native_path, compatibility_path = _prepare_output_paths(output_dir)
    case_ids_by_category = (
        dict(SMOKE_CASE_IDS)
        if suite is SMOKE_SUITE
        else {category: () for category in suite.leaf_categories}
    )
    _write_json(
        native_path,
        _native_report(
            suite=suite,
            case_ids_by_category=case_ids_by_category,
            model=model,
            base_url=None,
            num_threads=suite.default_num_threads,
            scores=None,
            integration_error=error,
        ),
    )
    _write_json(
        compatibility_path,
        _compatibility_result(
            suite=suite,
            case_ids_by_category=case_ids_by_category,
            model=model,
            scores=None,
            integration_error=error,
        ),
    )


def run_evaluation(
    *,
    base_url: str,
    api_key: str,
    model: str,
    output_dir: Path,
    bfcl_project_root: Path,
    suite: SuiteSpec = SMOKE_SUITE,
    num_threads: int | None = None,
    upstream_runner: UpstreamRunner = _run_upstream,
) -> bool:
    """Run one immutable BFCL suite and always publish both report formats."""
    native_path, compatibility_path = _prepare_output_paths(output_dir)
    selected_case_ids: dict[str, tuple[str, ...]] = (
        dict(SMOKE_CASE_IDS)
        if suite is SMOKE_SUITE
        else {category: () for category in suite.leaf_categories}
    )
    resolved_num_threads = (
        suite.default_num_threads if num_threads is None else num_threads
    )
    try:
        if suite.name not in SUITE_SPECS or SUITE_SPECS[suite.name] is not suite:
            raise ValueError(f"unsupported BFCL suite: {suite.name}")
        normalized_url = _absolute_http_url(base_url)
        normalized_model = _nonempty_string(model)
        normalized_key = _nonempty_string(api_key)
        if isinstance(resolved_num_threads, bool) or not isinstance(
            resolved_num_threads, int
        ):
            raise ValueError("num_threads must be a positive integer")
        if resolved_num_threads <= 0:
            raise ValueError("num_threads must be a positive integer")
        if not callable(upstream_runner):
            raise TypeError("upstream_runner must be callable")

        os.environ["BFCL_PROJECT_ROOT"] = str(bfcl_project_root)
        _clear_upstream_modules()
        _write_upstream_attribution(bfcl_project_root)
        selected_case_ids = _build_suite_case_ids(suite)
        _write_id_map(bfcl_project_root, selected_case_ids)
        upstream_runner(
            model=normalized_model,
            project_root=bfcl_project_root,
            base_url=normalized_url,
            api_key=normalized_key,
            num_threads=resolved_num_threads,
        )
        _validate_generated_results(bfcl_project_root, selected_case_ids)
        scores = _collect_scores(bfcl_project_root, selected_case_ids)
        if sum(score.total_count for score in scores) != suite.expected_sample_count:
            raise ValueError(
                f"{suite.name} evaluated an unexpected total number of cases"
            )
    except Exception as exc:  # noqa: BLE001 - artifact publication is the boundary
        _write_json(
            native_path,
            _native_report(
                suite=suite,
                case_ids_by_category=selected_case_ids,
                model=model,
                base_url=base_url,
                num_threads=resolved_num_threads,
                scores=None,
                integration_error=exc,
            ),
        )
        _write_json(
            compatibility_path,
            _compatibility_result(
                suite=suite,
                case_ids_by_category=selected_case_ids,
                model=model,
                scores=None,
                integration_error=exc,
            ),
        )
        return False

    native = _native_report(
        suite=suite,
        case_ids_by_category=selected_case_ids,
        model=normalized_model,
        base_url=normalized_url,
        num_threads=resolved_num_threads,
        scores=scores,
    )
    compatibility = _compatibility_result(
        suite=suite,
        case_ids_by_category=selected_case_ids,
        model=normalized_model,
        scores=scores,
    )
    _write_json(native_path, native)
    _write_json(compatibility_path, compatibility)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pinned BFCL V4 OpenAI completions suite."
    )
    parser.add_argument("--base-url", type=_absolute_http_url)
    parser.add_argument("--api-key", type=_nonempty_string, default="EMPTY")
    parser.add_argument("--model", type=_nonempty_string, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bfcl-project-root", type=Path)
    parser.add_argument(
        "--suite",
        choices=tuple(SUITE_SPECS),
        default=TASK_NAME,
    )
    parser.add_argument("--num-threads", type=_positive_int)
    parser.add_argument("--integration-error")
    args = parser.parse_args(argv)
    args.num_threads = (
        SUITE_SPECS[args.suite].default_num_threads
        if args.num_threads is None
        else args.num_threads
    )
    if args.integration_error is None:
        if args.base_url is None:
            parser.error("--base-url required unless --integration-error is provided")
        if args.bfcl_project_root is None:
            parser.error(
                "--bfcl-project-root required unless --integration-error is provided"
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suite = SUITE_SPECS[args.suite]
    if args.integration_error is not None:
        publish_integration_error(
            output_dir=args.output_dir,
            model=args.model,
            error=RuntimeError(args.integration_error),
            suite=suite,
        )
        return 1

    passed = run_evaluation(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        bfcl_project_root=args.bfcl_project_root,
        suite=suite,
        num_threads=args.num_threads,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
