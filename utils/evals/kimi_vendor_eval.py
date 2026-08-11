#!/usr/bin/env python3
"""Run the stock Kimi Vendor Verifier and project its native report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_NAME = "kimi_tool_call_schema"
NATIVE_REPORT_FILENAME = "kimi_vendor_report.json"
COMPATIBILITY_GLOB = "results_kimi_vendor_*.json"
EXPECTED_MODES = {"non-stream", "stream"}
DEFAULT_TIMEOUT_SECONDS = 900
RESULT_FORMAT = "inferencex-eval-v1"
ADAPTER_NAME = "kimi-vendor-verifier"


def prepare_compatibility_path(output_dir: Path) -> Path:
    """Remove stale projections and return a timestamped collector artifact path."""
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        stale_path.unlink()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    return output_dir / f"results_kimi_vendor_{timestamp}.json"


def build_pytest_command(
    *, base_url: str, api_key: str, model: str, report_path: Path
) -> list[str]:
    """Build the fixed Phase 1 invocation of the upstream verifier."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "tests/tool_call_json_schema/test_tool_call_json_schema.py",
        "--base-url",
        base_url,
        "--api-key",
        api_key,
        "--smoke-model",
        model,
        "--think-mode",
        "none",
        "--selection",
        "object",
        "--max-cases",
        "1",
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


def _project_report(model: str, report: Any) -> tuple[dict[str, Any], bool]:
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

    modes: list[str] = []
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

    if total != len(results) or passed != result_passes:
        raise ValueError("report summary does not match result records")

    if (
        total != 2
        or len(results) != 2
        or set(modes) != EXPECTED_MODES
        or len(modes) != len(set(modes))
    ):
        raise ValueError("report does not contain the expected stream modes")
    score = passed / 2.0
    return _compatibility_result(model, score, n_samples=2), passed == 2


def _compatibility_result(
    model: str,
    score: float,
    *,
    n_samples: int,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": ADAPTER_NAME,
        "model_name": model,
        "results": {
            TASK_NAME: {
                "exact_match,strict-match": score,
                "exact_match_stderr,strict-match": 0.0,
            }
        },
        "configs": {
            TASK_NAME: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            }
        },
        "n-samples": {
            TASK_NAME: {
                "original": len(EXPECTED_MODES),
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


def _write_compatibility(path: Path, result: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def run_evaluation(
    *,
    verifier_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    output_dir: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> bool:
    """Run upstream pytest and always attempt to publish a compatibility result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    native_report = output_dir / NATIVE_REPORT_FILENAME
    compatibility_path = prepare_compatibility_path(output_dir)
    subprocess_rc: int | None = None
    integration_error: BaseException | None = None
    compatibility = _compatibility_result(model, 0.0, n_samples=0)
    complete_pass = False

    try:
        native_report.unlink(missing_ok=True)
        completed = subprocess.run(
            build_pytest_command(
                base_url=base_url,
                api_key=api_key,
                model=model,
                report_path=native_report.resolve(),
            ),
            cwd=verifier_dir,
            check=False,
            timeout=timeout_seconds,
        )
        subprocess_rc = completed.returncode
        report = json.loads(native_report.read_text(encoding="utf-8"))
        compatibility, complete_pass = _project_report(model, report)
        if subprocess_rc != 0 and complete_pass:
            integration_error = RuntimeError(
                f"upstream verifier exited with code {subprocess_rc}"
            )
            compatibility = _compatibility_result(
                model, 0.0, n_samples=2, integration_error=integration_error
            )
            complete_pass = False
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        integration_error = exc
        compatibility = _compatibility_result(
            model, 0.0, n_samples=0, integration_error=exc
        )
    finally:
        try:
            _write_compatibility(compatibility_path, compatibility)
        except OSError as exc:
            if integration_error is not None:
                exc.add_note(f"Earlier integration error: {integration_error}")
            raise

    return subprocess_rc == 0 and complete_pass and integration_error is None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned stock Kimi Vendor Verifier tool-schema smoke test."
    )
    parser.add_argument("--verifier-dir", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--timeout-seconds", type=_positive_int, default=DEFAULT_TIMEOUT_SECONDS
    )
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
        (args.output_dir / NATIVE_REPORT_FILENAME).unlink(missing_ok=True)
        _write_compatibility(
            prepare_compatibility_path(args.output_dir),
            _compatibility_result(
                args.model,
                0.0,
                n_samples=0,
                integration_error=RuntimeError(args.integration_error),
            ),
        )
        return 0
    passed = run_evaluation(
        verifier_dir=args.verifier_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
