#!/usr/bin/env python3
"""Run a pinned MiniMax M3 smoke subset through the stock provider verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .minimax_m3_full_eval import UPSTREAM_REF, verify_source_tree
else:
    from minimax_m3_full_eval import UPSTREAM_REF, verify_source_tree

TASK_NAME = "minimax_m3_smoke"
NATIVE_REPORT_FILENAME = "minimax_vendor_report.json"
NATIVE_RESULTS_FILENAME = "minimax_vendor_results.jsonl"
COMPATIBILITY_GLOB = "results_minimax_vendor_*.json"
DEFAULT_FIXTURE_PATH = Path(__file__).with_name("minimax_m3_smoke.json")
RESULT_FORMAT = "inferencex-eval-v1"
ADAPTER_NAME = "minimax-provider-verifier"
EXPECTED_INDICES = (71,)
EXPECTED_LICENSE_SHA256 = (
    "aa7cec386fcb5e555aba0e8b1c31307940af41967708c9bc0f78b4e02e235dd5"
)
EXPECTED_CASE_SHA256 = {
    71: "3d51571a1ed7d0bb644c3ae978ef5822b3150479b1e34bbbae7276f671657870",
}
UPSTREAM_SOURCE = (
    "https://raw.githubusercontent.com/MiniMax-AI/MiniMax-Provider-Verifier/"
    f"{UPSTREAM_REF}/sample.jsonl"
)
UPSTREAM_TIMEOUT_SECONDS = 60 * 60

Runner = Callable[..., subprocess.CompletedProcess[Any]]


class SmokeSuiteError(RuntimeError):
    """The stock verifier could not produce one complete smoke result."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def load_fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the exact pinned row without modifying its request fields."""
    root = _mapping(json.loads(path.read_text(encoding="utf-8")), "fixture")
    if root.get("source") != UPSTREAM_SOURCE or root.get("ref") != UPSTREAM_REF:
        raise ValueError("fixture source or ref does not match the pinned upstream")
    if root.get("indices") != list(EXPECTED_INDICES):
        raise ValueError("fixture indices must be exactly [71]")
    license_text = root.get("license")
    if (
        not isinstance(license_text, str)
        or hashlib.sha256(license_text.encode()).hexdigest() != EXPECTED_LICENSE_SHA256
    ):
        raise ValueError("fixture must preserve the complete upstream MIT notice")

    raw_rows = root.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 1:
        raise ValueError("fixture must contain exactly one row")
    row = dict(_mapping(raw_rows[0], "fixture.rows[0]"))
    if "data_index" in row:
        raise ValueError("fixture request must not contain adapter-only data_index")
    digest = hashlib.sha256(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != EXPECTED_CASE_SHA256[EXPECTED_INDICES[0]]:
        raise ValueError("fixture row 71 differs from pinned upstream")
    return dict(root), [row]


def _normalized_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url must be a non-empty string")
    normalized = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "base_url must be an absolute HTTP(S) URL without query or fragment"
        )
    return normalized


def prepare_smoke_input(*, fixture_path: Path, destination: Path) -> None:
    """Write the pinned row as stock verifier JSONL input."""
    _, rows = load_fixture(fixture_path)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_verifier_command(
    *,
    python: Path,
    source_dir: Path,
    sample_path: Path,
    base_url: str,
    model: str,
    output_dir: Path,
) -> list[str]:
    """Build one stock verify.py invocation over the pinned smoke row."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    extra_body = json.dumps(
        {"temperature": 0, "top_p": 1, "max_tokens": 40960},
        separators=(",", ":"),
    )
    return [
        str(python),
        str(source_dir / "verify.py"),
        str(sample_path),
        "--model",
        model,
        "--base-url",
        _normalized_base_url(base_url),
        "--api-key",
        "EMPTY",
        "--concurrency",
        "1",
        "--output",
        str(output_dir / NATIVE_RESULTS_FILENAME),
        "--summary",
        str(output_dir / NATIVE_REPORT_FILENAME),
        "--timeout",
        "600",
        "--retries",
        "3",
        "--extra-body",
        extra_body,
    ]


def _error_dict(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _rate(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise SmokeSuiteError(f"native summary {name} must be a finite rate")
    return float(value)


def _nonnegative_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SmokeSuiteError(f"native summary {name} must be a non-negative integer")
    return value


def _compatibility_path(output_dir: Path) -> Path:
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        stale_path.unlink()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    return output_dir / f"results_minimax_vendor_{timestamp}.json"


def _compatibility_result(
    *,
    model: str,
    score: float,
    effective: int,
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
                "native_metrics": [
                    "tool_calls_match_rate",
                    "tool_calls_schema_validation_error_count",
                    "tool_calls_total_count",
                    "error_only_reasoning_rate",
                ],
            }
        },
        "n-samples": {
            TASK_NAME: {"original": 1, "effective": effective},
        },
        "source": {
            "repository": "MiniMax-AI/MiniMax-Provider-Verifier",
            "ref": UPSTREAM_REF,
            "indices": list(EXPECTED_INDICES),
        },
    }
    if integration_error is not None:
        result["integration_error"] = _error_dict(integration_error)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def project_native_artifacts(*, output_dir: Path, model: str) -> Path:
    """Validate stock outputs and project only their published metrics."""
    report = _mapping(
        json.loads((output_dir / NATIVE_REPORT_FILENAME).read_text(encoding="utf-8")),
        "native summary",
    )
    result_lines = (
        (output_dir / NATIVE_RESULTS_FILENAME).read_text(encoding="utf-8").splitlines()
    )
    if len(result_lines) != 1 or not result_lines[0].strip():
        raise SmokeSuiteError("native results must contain exactly one row")
    result = _mapping(json.loads(result_lines[0]), "native result")
    if result.get("data_index") != 1:
        raise SmokeSuiteError("native result must identify the one-row smoke input")
    if result.get("status") != "success":
        raise SmokeSuiteError("native verifier reported a request failure")
    if report.get("model") != model:
        raise SmokeSuiteError("native summary model does not match the requested model")
    if report.get("success_count") != 1 or report.get("failure_count") != 0:
        raise SmokeSuiteError("native summary does not describe one successful request")

    match_rate = _rate(report.get("tool_calls_match_rate"), "tool_calls_match_rate")
    schema_errors = _nonnegative_count(
        report.get("tool_calls_schema_validation_error_count"),
        "tool_calls_schema_validation_error_count",
    )
    tool_call_total = _nonnegative_count(
        report.get("tool_calls_total_count"), "tool_calls_total_count"
    )
    if schema_errors > tool_call_total:
        raise SmokeSuiteError(
            "native summary tool-call schema counts are inconsistent"
        )
    schema_rate = (
        0.0 if tool_call_total == 0 else 1.0 - (schema_errors / tool_call_total)
    )
    reasoning_error_rate = _rate(
        report.get("error_only_reasoning_rate"), "error_only_reasoning_rate"
    )
    score = min(match_rate, schema_rate, 1.0 - reasoning_error_rate)
    compatibility_path = _compatibility_path(output_dir)
    _write_json(
        compatibility_path,
        _compatibility_result(model=model, score=score, effective=1),
    )
    return compatibility_path


def publish_failure(*, output_dir: Path, model: str, error: BaseException) -> Path:
    """Publish integration metadata without rewriting stock artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    native_report_path = output_dir / NATIVE_REPORT_FILENAME
    if not native_report_path.exists():
        _write_json(
            native_report_path,
            {
                "verifier": ADAPTER_NAME,
                "task": TASK_NAME,
                "model": model,
                "completed": False,
                "source": {"ref": UPSTREAM_REF, "indices": list(EXPECTED_INDICES)},
                "integration_error": _error_dict(error),
            },
        )
    compatibility_path = _compatibility_path(output_dir)
    _write_json(
        compatibility_path,
        _compatibility_result(
            model=model,
            score=0.0,
            effective=0,
            integration_error=error,
        ),
    )
    return compatibility_path


def run_evaluation(
    *,
    python: Path,
    source_dir: Path,
    dependency_dir: Path,
    base_url: str,
    model: str,
    output_dir: Path,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    runner: Runner = subprocess.run,
) -> bool:
    """Run the stock upstream verifier once, then project its native metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (NATIVE_REPORT_FILENAME, NATIVE_RESULTS_FILENAME):
        (output_dir / filename).unlink(missing_ok=True)
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        stale_path.unlink()
    smoke_input = output_dir / "minimax_vendor_smoke_input.jsonl"
    smoke_input.unlink(missing_ok=True)
    try:
        verify_source_tree(source_dir)
        prepare_smoke_input(fixture_path=fixture_path, destination=smoke_input)
        command = build_verifier_command(
            python=python,
            source_dir=source_dir,
            sample_path=smoke_input,
            base_url=base_url,
            model=model,
            output_dir=output_dir,
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(source_dir), str(dependency_dir))
        )
        environment["PYTHONNOUSERSITE"] = "1"
        completed = runner(
            command,
            env=environment,
            timeout=UPSTREAM_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            raise SmokeSuiteError(
                f"pinned upstream verifier exited with code {completed.returncode}"
            )
        project_native_artifacts(output_dir=output_dir, model=model)
    except (
        OSError,
        ValueError,
        SmokeSuiteError,
        subprocess.TimeoutExpired,
    ) as exc:
        publish_failure(output_dir=output_dir, model=model, error=exc)
        return False
    finally:
        smoke_input.unlink(missing_ok=True)
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned MiniMax M3 smoke through stock verify.py."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--python", required=True, type=Path)
    run.add_argument("--source-dir", required=True, type=Path)
    run.add_argument("--dependency-dir", required=True, type=Path)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)

    failure = subparsers.add_parser("failure")
    failure.add_argument("--model", required=True)
    failure.add_argument("--output-dir", required=True, type=Path)
    failure.add_argument("--message", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "failure":
        publish_failure(
            output_dir=args.output_dir,
            model=args.model,
            error=SmokeSuiteError(args.message),
        )
        return 0
    completed = run_evaluation(
        python=args.python,
        source_dir=args.source_dir,
        dependency_dir=args.dependency_dir,
        base_url=args.base_url,
        model=args.model,
        output_dir=args.output_dir,
        fixture_path=args.fixture,
    )
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
