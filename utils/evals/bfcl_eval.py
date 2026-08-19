#!/usr/bin/env python3
"""Run the pinned four-case BFCL V4 OpenAI chat-completions smoke."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Protocol

TASK_NAME = "bfcl_smoke"
NATIVE_REPORT_FILENAME = "bfcl_report.json"
COMPATIBILITY_FILENAME = "results_bfcl.json"
COMPATIBILITY_GLOB = "results_bfcl*.json"
RESULT_FORMAT = "inferencex-eval-v1"
ADAPTER_NAME = "bfcl-v4-openai-completions"
DEFAULT_NUM_THREADS = 4
DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
REQUIRED_SCORE = 0.75

BFCL_PACKAGE = "bfcl-eval"
BFCL_PACKAGE_VERSION = "2026.3.23"
BFCL_WHEEL_SHA256 = "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"
UPSTREAM_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
UPSTREAM_SOURCE = "https://pypi.org/project/bfcl-eval/2026.3.23/"
UPSTREAM_REF = f"{BFCL_PACKAGE}=={BFCL_PACKAGE_VERSION}"
SOURCE_REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
VLLM_INTEGRATION_REF = "7ecb11405df86b202f4c5cca322bd133052fee82"

# Dict insertion order is intentional: reports and the upstream run-ID file are stable.
SMOKE_CASE_IDS: dict[str, tuple[str, ...]] = {
    "simple_python": ("simple_python_141",),
    "multiple": ("multiple_38",),
    "parallel": ("parallel_1",),
    "irrelevance": ("irrelevance_0",),
}
EXPECTED_SAMPLE_COUNT = sum(len(case_ids) for case_ids in SMOKE_CASE_IDS.values())


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
        request_timeout_seconds: float,
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
        return {
            "category": self.category,
            "case_ids": list(self.case_ids),
            "score_file": self.score_file,
            "score_header": dict(self.header),
            "score_records": [dict(record) for record in self.records],
            "case_scores": [
                {
                    "id": case_id,
                    "score": self.accuracy,
                    "correct": self.correct_count == self.total_count,
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
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("must be an absolute HTTP(S) URL")
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


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _source_details() -> dict[str, Any]:
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
            category: list(case_ids) for category, case_ids in SMOKE_CASE_IDS.items()
        },
    }


def _error_dict(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _expected_category_details() -> list[dict[str, Any]]:
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
        for category, case_ids in SMOKE_CASE_IDS.items()
    ]


def _diagnostics(scores: Sequence[CategoryScore] | None = None) -> dict[str, Any]:
    return {
        "source": _source_details(),
        "categories": (
            [score.as_dict() for score in scores]
            if scores is not None
            else _expected_category_details()
        ),
    }


def _native_report(
    *,
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
        "task": TASK_NAME,
        "model": model,
        "endpoint": base_url,
        "completed": integration_error is None,
        "passed": integration_error is None and accuracy >= REQUIRED_SCORE,
        "threshold": REQUIRED_SCORE,
        "sampling": {"temperature": 0.0, "num_threads": num_threads},
        "summary": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "expected_count": EXPECTED_SAMPLE_COUNT,
        },
        "bfcl": _diagnostics(scores),
    }
    if integration_error is not None:
        report["integration_error"] = _error_dict(integration_error)
    return report


def _compatibility_result(
    *,
    model: str,
    scores: Sequence[CategoryScore] | None,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    score_by_category = {score.category: score for score in scores or ()}
    total_count = sum(score.total_count for score in scores or ())
    correct_count = sum(score.correct_count for score in scores or ())
    accuracy = correct_count / total_count if total_count else 0.0
    task_scores = {TASK_NAME: accuracy}
    task_samples = {
        TASK_NAME: {
            "original": EXPECTED_SAMPLE_COUNT,
            "effective": total_count,
        }
    }
    for category, case_ids in SMOKE_CASE_IDS.items():
        category_score = score_by_category.get(category)
        task_name = f"bfcl_{category}"
        task_scores[task_name] = (
            category_score.accuracy if category_score is not None else 0.0
        )
        task_samples[task_name] = {
            "original": len(case_ids),
            "effective": (
                category_score.total_count if category_score is not None else 0
            ),
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
        "bfcl": _diagnostics(scores),
    }
    if integration_error is not None:
        result["integration_error"] = _error_dict(integration_error)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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


def _write_smoke_id_map(project_root: Path) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        project_root / "test_case_ids_to_generate.json",
        {category: list(case_ids) for category, case_ids in SMOKE_CASE_IDS.items()},
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


def _run_upstream(
    *,
    model: str,
    project_root: Path,
    base_url: str,
    api_key: str,
    num_threads: int,
    request_timeout_seconds: float,
) -> None:
    """Lazily load and invoke the pinned BFCL API against an existing server."""
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key

    # The adapter filename intentionally matches the installed package. When the
    # file is executed directly, hide its directory during package resolution.
    adapter_directory = Path(__file__).resolve().parent
    original_sys_path = sys.path[:]
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or os.curdir).resolve() != adapter_directory
    ]
    try:
        import bfcl_eval.constants.model_config as bfcl_model_config
        from bfcl_eval.__main__ import evaluate, generate
        from bfcl_eval.constants.model_config import ModelConfig
        from bfcl_eval.model_handler.api_inference.openai_completion import (
            OpenAICompletionsHandler,
        )
    finally:
        sys.path[:] = original_sys_path
    request_failures: SimpleQueue[Exception] = SimpleQueue()

    class BoundedOpenAICompletionsHandler(OpenAICompletionsHandler):
        def _build_client_kwargs(self) -> dict[str, Any]:
            kwargs = super()._build_client_kwargs()
            kwargs.update(timeout=request_timeout_seconds, max_retries=0)
            return kwargs

        def generate_with_backoff(self, **kwargs: Any) -> tuple[Any, float]:
            # The upstream method has an unbounded RateLimitError retry decorator.
            # The surrounding eval process owns the suite deadline, so issue once.
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                request_failures.put(exc)
                raise
            return response, time.monotonic() - started

    bfcl_model_config.MODEL_CONFIG_MAPPING[model] = ModelConfig(
        model_name=model,
        display_name=f"{model} (FC) (InferenceX)",
        url="",
        org="",
        license="unknown",
        model_handler=BoundedOpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )

    categories = list(SMOKE_CASE_IDS)
    generation_kwargs = _function_defaults(generate)
    generation_kwargs.update(
        model=[model],
        test_category=categories,
        temperature=0.0,
        num_threads=num_threads,
        skip_server_setup=True,
        run_ids=True,
        allow_overwrite=True,
    )
    generate(**generation_kwargs)
    if not request_failures.empty():
        raise request_failures.get()
    _validate_generated_results(project_root)

    evaluation_kwargs = _function_defaults(evaluate)
    evaluation_kwargs.update(
        model=[model],
        test_category=categories,
        partial_eval=True,
    )
    evaluate(**evaluation_kwargs)


def _validate_generated_results(project_root: Path) -> None:
    for category, case_ids in SMOKE_CASE_IDS.items():
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


def _collect_scores(project_root: Path) -> list[CategoryScore]:
    scores: list[CategoryScore] = []
    for category, case_ids in SMOKE_CASE_IDS.items():
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
    *, output_dir: Path, model: str, error: BaseException
) -> None:
    """Publish required zero-score artifacts without importing BFCL or Typer."""
    native_path, compatibility_path = _prepare_output_paths(output_dir)
    _write_json(
        native_path,
        _native_report(
            model=model,
            base_url=None,
            num_threads=DEFAULT_NUM_THREADS,
            scores=None,
            integration_error=error,
        ),
    )
    _write_json(
        compatibility_path,
        _compatibility_result(
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
    num_threads: int = DEFAULT_NUM_THREADS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    upstream_runner: UpstreamRunner = _run_upstream,
) -> bool:
    """Run the exact smoke IDs and always publish native and compatibility reports."""
    native_path, compatibility_path = _prepare_output_paths(output_dir)
    try:
        normalized_url = _absolute_http_url(base_url)
        normalized_model = _nonempty_string(model)
        normalized_key = _nonempty_string(api_key)
        if isinstance(num_threads, bool) or not isinstance(num_threads, int):
            raise ValueError("num_threads must be a positive integer")
        if num_threads <= 0:
            raise ValueError("num_threads must be a positive integer")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or not math.isfinite(float(request_timeout_seconds))
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive and finite")
        if not callable(upstream_runner):
            raise TypeError("upstream_runner must be callable")

        _write_smoke_id_map(bfcl_project_root)
        upstream_runner(
            model=normalized_model,
            project_root=bfcl_project_root,
            base_url=normalized_url,
            api_key=normalized_key,
            num_threads=num_threads,
            request_timeout_seconds=float(request_timeout_seconds),
        )
        _validate_generated_results(bfcl_project_root)
        scores = _collect_scores(bfcl_project_root)
    except Exception as exc:  # noqa: BLE001 - artifact publication is the boundary
        _write_json(
            native_path,
            _native_report(
                model=model,
                base_url=base_url,
                num_threads=num_threads,
                scores=None,
                integration_error=exc,
            ),
        )
        _write_json(
            compatibility_path,
            _compatibility_result(
                model=model,
                scores=None,
                integration_error=exc,
            ),
        )
        return False

    native = _native_report(
        model=normalized_model,
        base_url=normalized_url,
        num_threads=num_threads,
        scores=scores,
    )
    compatibility = _compatibility_result(model=normalized_model, scores=scores)
    _write_json(native_path, native)
    _write_json(compatibility_path, compatibility)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned four-case BFCL V4 OpenAI completions smoke."
    )
    parser.add_argument("--base-url", type=_absolute_http_url)
    parser.add_argument("--api-key", type=_nonempty_string, default="EMPTY")
    parser.add_argument("--model", type=_nonempty_string, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bfcl-project-root", type=Path)
    parser.add_argument(
        "--num-threads", type=_positive_int, default=DEFAULT_NUM_THREADS
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    parser.add_argument("--integration-error")
    args = parser.parse_args(argv)
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
    if args.integration_error is not None:
        publish_integration_error(
            output_dir=args.output_dir,
            model=args.model,
            error=RuntimeError(args.integration_error),
        )
        return 1

    passed = run_evaluation(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        bfcl_project_root=args.bfcl_project_root,
        num_threads=args.num_threads,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
