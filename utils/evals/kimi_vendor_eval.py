#!/usr/bin/env python3
"""Run the stock Kimi Vendor Verifier and project its native report."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_NAME = "kimi_tool_call_schema"
FULL_TASK_NAME = "kimi_tool_call_schema_full"
SUPPORTED_TASK_NAMES = (TASK_NAME, FULL_TASK_NAME)
NATIVE_REPORT_FILENAME = "kimi_vendor_report.json"
COMPATIBILITY_GLOB = "results_kimi_vendor_*.json"
EXPECTED_MODES = {"non-stream", "stream"}
EXPECTED_TOTALS = {TASK_NAME: 2, FULL_TASK_NAME: 408}
FULL_SELECTED_CASES = 204
DEFAULT_TIMEOUT_SECONDS = 900
FULL_TIMEOUT_SECONDS = 7200
FULL_WORKERS = 8
RESULT_FORMAT = "inferencex-eval-v1"
ADAPTER_NAME = "kimi-vendor-verifier"

ENDPOINT_REJECTION_RE = re.compile(
    r"(?im)^(?:E\s+)?AssertionError:.*tool schema rejected:"
)


def prepare_compatibility_path(output_dir: Path) -> Path:
    """Remove stale projections and return a timestamped collector artifact path."""
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        stale_path.unlink()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    return output_dir / f"results_kimi_vendor_{timestamp}.json"


def build_pytest_command(
    *,
    base_url: str,
    api_key: str,
    model: str,
    model_prefix: str = "",
    report_path: Path,
    task_name: str = TASK_NAME,
) -> list[str]:
    """Build the fixed invocation of the pinned upstream verifier."""
    _expected_total(task_name)
    thinking_args = (
        ["--think-mode", "opensource", "--thinking"]
        if model_prefix == "dsv4"
        else ["--think-mode", "none"]
    )
    parallel_args = ["-n", str(FULL_WORKERS)] if task_name == FULL_TASK_NAME else []
    selection_args = (
        ["--selection", "all"]
        if task_name == FULL_TASK_NAME
        else ["--selection", "object", "--max-cases", "1"]
    )
    return [
        sys.executable,
        "-m",
        "pytest",
        "tests/tool_call_json_schema/test_tool_call_json_schema.py",
        *parallel_args,
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--smoke-model",
        model,
        *thinking_args,
        *selection_args,
        "--case-dir",
        "testdata/walle_validator_cases/validator_cases",
        "--max-tokens",
        "2048",
        "--tool-json-report",
        str(report_path),
    ]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _expected_total(task_name: str) -> int:
    try:
        return EXPECTED_TOTALS[task_name]
    except KeyError as exc:
        raise ValueError(f"unsupported Kimi task: {task_name}") from exc


def _endpoint_rejection_messages(report: Any) -> list[str]:
    """Return upstream failures rejected before argument-schema validation."""
    root = _mapping(report, "report")
    results = root.get("results")
    if not isinstance(results, list):
        raise ValueError("report.results must be an array")
    messages: list[str] = []
    for index, result in enumerate(results):
        record = _mapping(result, f"report.results[{index}]")
        message = record.get("message")
        if (
            record.get("status") == "failed"
            and isinstance(message, str)
            and ENDPOINT_REJECTION_RE.search(message)
        ):
            messages.append(message)
    return messages


def _project_report(
    model: str,
    report: Any,
    *,
    task_name: str = TASK_NAME,
) -> tuple[dict[str, Any], bool]:
    expected_total = _expected_total(task_name)
    root = _mapping(report, "report")
    summary = _mapping(root.get("summary"), "report.summary")
    results = root.get("results")
    if not isinstance(results, list):
        raise ValueError("report.results must be an array")

    total = summary.get("total")
    by_status = _mapping(summary.get("by_status"), "report.summary.by_status")
    if not isinstance(total, int) or isinstance(total, bool):
        raise ValueError("report summary contains invalid counts")
    for status, count in by_status.items():
        if (
            status not in {"passed", "failed"}
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError("report summary contains invalid counts")
    if sum(by_status.values()) != total:
        raise ValueError("report summary does not match total")
    passed = by_status.get("passed", 0)
    selected_identities: set[tuple[str, int]] = set()
    if task_name == FULL_TASK_NAME:
        selected_cases = root.get("selected_cases")
        if (
            not isinstance(selected_cases, list)
            or len(selected_cases) != FULL_SELECTED_CASES
        ):
            raise ValueError(
                f"report.selected_cases must contain {FULL_SELECTED_CASES} cases"
            )
        selected_keys: set[tuple[str, int, str]] = set()
        for index, selected_case in enumerate(selected_cases):
            record = _mapping(
                selected_case,
                f"report.selected_cases[{index}]",
            )
            suite = record.get("suite")
            line = record.get("line")
            selection_reason = record.get("selection_reason")
            if (
                not isinstance(suite, str)
                or not suite
                or not isinstance(line, int)
                or isinstance(line, bool)
                or line < 1
                or not isinstance(selection_reason, str)
            ):
                raise ValueError(f"report.selected_cases[{index}] has invalid identity")
            selected_key = (suite, line, selection_reason)
            if selected_key in selected_keys:
                raise ValueError("report.selected_cases contains a duplicate case")
            selected_keys.add(selected_key)
            selected_identities.add((suite, line))

    modes: list[str] = []
    case_modes: dict[tuple[str, int], set[str]] = {}
    result_passes = 0
    for index, result in enumerate(results):
        record = _mapping(result, f"report.results[{index}]")
        mode = record.get("mode")
        status = record.get("status")
        if (
            not isinstance(mode, str)
            or not isinstance(status, str)
            or status not in {"passed", "failed"}
        ):
            raise ValueError(f"report.results[{index}] has invalid mode or status")
        modes.append(mode)
        result_passes += status == "passed"

        if task_name == FULL_TASK_NAME:
            suite = record.get("suite")
            line = record.get("line")
            if (
                not isinstance(suite, str)
                or not suite
                or not isinstance(line, int)
                or isinstance(line, bool)
                or line < 1
            ):
                raise ValueError(
                    f"report.results[{index}] has invalid suite or line identity"
                )
            identity = (suite, line)
            identity_modes = case_modes.setdefault(identity, set())
            if mode in identity_modes:
                raise ValueError(
                    "report contains a duplicate mode for a selected suite and line"
                )
            identity_modes.add(mode)

    if total != len(results) or passed != result_passes:
        raise ValueError("report summary does not match result records")
    if total != expected_total:
        raise ValueError(
            f"report contains {total} records; expected {expected_total} for {task_name}"
        )

    if task_name == TASK_NAME:
        if set(modes) != EXPECTED_MODES or len(modes) != len(set(modes)):
            raise ValueError("report does not contain the expected stream modes")
    elif any(
        modes_for_case != EXPECTED_MODES for modes_for_case in case_modes.values()
    ):
        raise ValueError(
            "report does not contain exactly one of each stream mode "
            "for every selected suite and line"
        )
    if task_name == FULL_TASK_NAME and set(case_modes) != selected_identities:
        raise ValueError(
            "report results do not match the selected suite and line identities"
        )

    score = passed / total
    return (
        _compatibility_result(
            model,
            score,
            task_name=task_name,
            n_samples=total,
        ),
        passed == total,
    )


def _compatibility_result(
    model: str,
    score: float,
    *,
    task_name: str = TASK_NAME,
    n_samples: int,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    expected_total = _expected_total(task_name)
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": ADAPTER_NAME,
        "model_name": model,
        "results": {
            task_name: {
                "exact_match,strict-match": score,
                "exact_match_stderr,strict-match": 0.0,
            }
        },
        "configs": {
            task_name: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            }
        },
        "n-samples": {
            task_name: {
                "original": expected_total,
                "effective": n_samples,
            }
        },
    }
    if integration_error is not None:
        result["integration_error"] = {
            "type": type(integration_error).__name__,
            "message": str(integration_error),
        }
    return result


def _write_native_failure(
    path: Path,
    *,
    model: str,
    task_name: str,
    error: BaseException,
) -> None:
    """Write a native diagnostic envelope when upstream cannot produce one."""
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "task": task_name,
                "completed": False,
                "summary": {
                    "total": 0,
                    "expected_total": _expected_total(task_name),
                    "by_status": {},
                },
                "results": [],
                "integration_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_compatibility(path: Path, result: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run_evaluation(
    *,
    verifier_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    model_prefix: str = "",
    output_dir: Path,
    task_name: str = TASK_NAME,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Run upstream pytest and always attempt to publish a compatibility result."""
    expected_total = _expected_total(task_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    native_report = output_dir / NATIVE_REPORT_FILENAME
    compatibility_path = prepare_compatibility_path(output_dir)
    subprocess_rc: int | None = None
    integration_error: BaseException | None = None
    compatibility = _compatibility_result(
        model,
        0.0,
        task_name=task_name,
        n_samples=0,
    )
    all_passed = False
    completed_successfully = False
    try:
        native_report.unlink(missing_ok=True)
        completed = subprocess.run(
            build_pytest_command(
                base_url=base_url,
                api_key=api_key,
                model=model,
                model_prefix=model_prefix,
                report_path=native_report.resolve(),
                task_name=task_name,
            ),
            cwd=verifier_dir,
            check=False,
            timeout=timeout_seconds,
        )
        subprocess_rc = completed.returncode
        report = json.loads(native_report.read_text(encoding="utf-8"))
        compatibility, all_passed = _project_report(
            model,
            report,
            task_name=task_name,
        )
        endpoint_rejections = _endpoint_rejection_messages(report)
        valid_outcome = (subprocess_rc == 0 and all_passed) or (
            subprocess_rc == 1 and not all_passed
        )
        completed_successfully = subprocess_rc == 0 and all_passed
        if endpoint_rejections:
            integration_error = RuntimeError(
                "upstream verifier reported "
                f"{len(endpoint_rejections)} endpoint request or response failure(s)"
            )
            compatibility = _compatibility_result(
                model,
                0.0,
                task_name=task_name,
                n_samples=0,
                integration_error=integration_error,
            )
            completed_successfully = False
        elif valid_outcome:
            completed_successfully = True
        elif not valid_outcome:
            integration_error = RuntimeError(
                f"upstream verifier exited with code {subprocess_rc}"
            )
            compatibility = _compatibility_result(
                model,
                0.0,
                task_name=task_name,
                n_samples=0,
                integration_error=integration_error,
            )
            completed_successfully = False
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        integration_error = exc
        compatibility = _compatibility_result(
            model,
            0.0,
            task_name=task_name,
            n_samples=0,
            integration_error=exc,
        )
        if not native_report.exists():
            _write_native_failure(
                native_report,
                model=model,
                task_name=task_name,
                error=exc,
            )
    finally:
        try:
            _write_compatibility(compatibility_path, compatibility)
        except OSError as exc:
            if integration_error is not None:
                exc.add_note(f"Earlier integration error: {integration_error}")
            raise

    return completed_successfully and integration_error is None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned stock Kimi Vendor Verifier tool-schema evaluation."
    )
    parser.add_argument("--verifier-dir", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-prefix", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--task-name",
        choices=SUPPORTED_TASK_NAMES,
        default=TASK_NAME,
    )
    parser.add_argument("--timeout-seconds", type=_positive_int)
    parser.add_argument("--integration-error")
    args = parser.parse_args(argv)
    if args.integration_error is None:
        missing = [
            option
            for option, value in (
                ("--verifier-dir", args.verifier_dir),
                ("--base-url", args.base_url),
            )
            if value is None
        ]
        if missing:
            parser.error(
                f"{', '.join(missing)} required unless --integration-error is provided"
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.integration_error is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        error = RuntimeError(args.integration_error)
        _write_native_failure(
            args.output_dir / NATIVE_REPORT_FILENAME,
            model=args.model,
            task_name=args.task_name,
            error=error,
        )
        _write_compatibility(
            prepare_compatibility_path(args.output_dir),
            _compatibility_result(
                args.model,
                0.0,
                task_name=args.task_name,
                n_samples=0,
                integration_error=error,
            ),
        )
        return 0
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else (
            FULL_TIMEOUT_SECONDS
            if args.task_name == FULL_TASK_NAME
            else DEFAULT_TIMEOUT_SECONDS
        )
    )
    passed = run_evaluation(
        verifier_dir=args.verifier_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        model_prefix=args.model_prefix,
        output_dir=args.output_dir,
        task_name=args.task_name,
        timeout_seconds=timeout_seconds,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
