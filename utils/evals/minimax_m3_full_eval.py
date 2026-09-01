#!/usr/bin/env python3
"""Run and project the pinned full MiniMax M3 provider verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

TASK_NAME = "minimax_m3_full"
RESULT_FORMAT = "inferencex-eval-v1"
ADAPTER_NAME = "minimax-provider-verifier"
NATIVE_REPORT_FILENAME = "minimax_vendor_report.json"
NATIVE_RESULTS_FILENAME = "minimax_vendor_results.jsonl"
COMPATIBILITY_GLOB = "results_minimax_vendor_*.json"
EXPECTED_RESULT_COUNT = 102
UPSTREAM_REF = "c899f95e17bfc4a338ddd4cb1638279125885e55"
UPSTREAM_BASE_URL = (
    "https://raw.githubusercontent.com/MiniMax-AI/MiniMax-Provider-Verifier/"
    f"{UPSTREAM_REF}"
)
EXPECTED_SAMPLE_SHA256 = (
    "3ead102af0f888acc95867b3a9916942524b02f4f64931f020a1bfb4fee9aae2"
)
REQUIRED_SOURCE_SHA256 = {
    "verify.py": "6bc00948d9be06189f31c5a53bb7929b15555402f0b3495609d26b468090ee4a",
    "sample.jsonl": EXPECTED_SAMPLE_SHA256,
    "validator/__init__.py": "955a5ee77b72fb1d128f5d1ab6c65072e54b8cf527916e70a69e251811087e7b",
    "validator/base.py": "00f1776d4b4d4200e4ce865f044ef9cdd7f375ca3b752a8eefba4ab375f7e03d",
    "validator/tool_calls.py": "eb6a91a704a3e1706a1fc0f0f4233f4dd08ecb38b7f41fd19abaf5e981d0406b",
    "validator/russian_characters.py": "09b429f45b43b34c241d5b54401aab6c4c55de228d87c882bb7f8858c9856c0c",
    "validator/repeat_ngram.py": "72948cd9501e8daeb49a2c9fac4421a182191fc4251846713021c27d1a5fa315",
    "validator/scenario_check.py": "51d691a1595f6fa3193f4f624a98581a9795323e77439df7a2fd5d83950c5ba1",
}
MAX_SOURCE_BYTES = 16 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY_SECONDS = 3
UPSTREAM_TIMEOUT_SECONDS = 7 * 60 * 60

Runner = Callable[..., subprocess.CompletedProcess[Any]]
Fetcher = Callable[[str], bytes]


class FullSuiteError(RuntimeError):
    """The full verifier could not produce one complete diagnostic run."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_RejectRedirects())


def source_url(relative_path: str) -> str:
    """Return the immutable URL for one allowlisted required source file."""
    if relative_path not in REQUIRED_SOURCE_SHA256:
        raise ValueError(f"unapproved upstream source path: {relative_path!r}")
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe upstream source path: {relative_path!r}")
    return f"{UPSTREAM_BASE_URL}/{relative_path}"


def verify_source_content(relative_path: str, content: bytes) -> None:
    """Verify one allowlisted file against its pinned SHA256 identity."""
    if relative_path not in REQUIRED_SOURCE_SHA256:
        raise ValueError(f"unapproved upstream source path: {relative_path!r}")
    if not isinstance(content, bytes):
        raise TypeError("source content must be bytes")
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(f"pinned source {relative_path} exceeds the size limit")
    actual = hashlib.sha256(content).hexdigest()
    expected = REQUIRED_SOURCE_SHA256[relative_path]
    if actual != expected:
        raise ValueError(
            f"pinned source {relative_path} SHA256 mismatch: "
            f"expected {expected}, got {actual}"
        )


def _fetch_source(relative_path: str) -> bytes:
    url = source_url(relative_path)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/octet-stream"},
        method="GET",
    )
    last_error: BaseException | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with _NO_REDIRECT_OPENER.open(
                request, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                status = getattr(response, "status", None)
                if status != 200 or response.geturl() != url:
                    raise FullSuiteError(
                        f"unexpected response for pinned source {relative_path}: "
                        f"status={status!r}, url={response.geturl()!r}"
                    )
                declared_size = response.headers.get("Content-Length")
                if (
                    declared_size is not None
                    and int(declared_size) > MAX_SOURCE_BYTES
                ):
                    raise FullSuiteError(
                        f"pinned source {relative_path} exceeds the size limit"
                    )
                content = response.read(MAX_SOURCE_BYTES + 1)
        except (
            FullSuiteError,
            OSError,
            ValueError,
            urllib.error.URLError,
        ) as exc:
            last_error = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)
                continue
            break
        verify_source_content(relative_path, content)
        return content
    raise FullSuiteError(
        f"failed to download pinned source {relative_path} after "
        f"{DOWNLOAD_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def _validate_sample(content: bytes) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("pinned sample.jsonl is not UTF-8") from exc
    lines = text.splitlines()
    if len(lines) != EXPECTED_RESULT_COUNT or any(not line.strip() for line in lines):
        raise ValueError(
            f"pinned sample.jsonl must contain exactly {EXPECTED_RESULT_COUNT} rows"
        )
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"pinned sample.jsonl row {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"pinned sample.jsonl row {line_number} must be an object"
            )


def verify_source_tree(source_dir: Path) -> None:
    """Verify every required file in an already prepared source directory."""
    root = source_dir.resolve()
    for relative_path in REQUIRED_SOURCE_SHA256:
        path = source_dir / relative_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required pinned source is missing: {relative_path}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"required pinned source escapes runtime: {relative_path}")
        content = path.read_bytes()
        verify_source_content(relative_path, content)
        if relative_path == "sample.jsonl":
            _validate_sample(content)


def prepare_source_tree(source_dir: Path, fetcher: Fetcher = _fetch_source) -> None:
    """Download only the allowlisted verifier files and verify the complete tree."""
    source_dir.mkdir(parents=True, exist_ok=False)
    for relative_path in REQUIRED_SOURCE_SHA256:
        content = fetcher(relative_path)
        verify_source_content(relative_path, content)
        destination = source_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
    verify_source_tree(source_dir)


def build_verifier_command(
    *,
    python: Path,
    source_dir: Path,
    base_url: str,
    model: str,
    output_dir: Path,
) -> list[str]:
    """Build the single pinned-upstream invocation for all 102 rows."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    normalized_base_url = base_url.strip().rstrip("/")
    parsed_base_url = urllib.parse.urlsplit(normalized_base_url)
    if (
        parsed_base_url.scheme not in {"http", "https"}
        or not parsed_base_url.netloc
        or parsed_base_url.query
        or parsed_base_url.fragment
    ):
        raise ValueError(
            "base_url must be an absolute HTTP(S) URL without query or fragment"
        )
    extra_body = json.dumps(
        {"temperature": 0, "top_p": 1, "max_tokens": 40960},
        separators=(",", ":"),
    )
    return [
        str(python),
        str(source_dir / "verify.py"),
        str(source_dir / "sample.jsonl"),
        "--model",
        model,
        "--base-url",
        normalized_base_url,
        "--api-key",
        "EMPTY",
        "--concurrency",
        "5",
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
                "native_metric": "tool_calls_match_rate",
                "diagnostic_threshold": 0.0,
            }
        },
        "n-samples": {
            TASK_NAME: {
                "original": EXPECTED_RESULT_COUNT,
                "effective": effective,
            }
        },
        "source": {
            "repository": "MiniMax-AI/MiniMax-Provider-Verifier",
            "ref": UPSTREAM_REF,
            "sample_sha256": EXPECTED_SAMPLE_SHA256,
        },
    }
    if integration_error is not None:
        result["integration_error"] = _error_dict(integration_error)
    return result


def _compatibility_path(output_dir: Path) -> Path:
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        stale_path.unlink()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")
    return output_dir / f"results_minimax_vendor_full_{timestamp}.json"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_native_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullSuiteError(f"native summary is unavailable or invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise FullSuiteError("native summary must be a JSON object")
    return value


def _read_native_results(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FullSuiteError(f"native results are unavailable: {exc}") from exc
    if len(lines) != EXPECTED_RESULT_COUNT or any(not line.strip() for line in lines):
        raise FullSuiteError(
            f"native results must contain exactly {EXPECTED_RESULT_COUNT} rows, "
            f"found {len(lines)}"
        )
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FullSuiteError(
                f"native result row {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise FullSuiteError(
                f"native result row {line_number} must be a JSON object"
            )
        results.append(value)
    return results


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FullSuiteError(f"native summary {name} must be a non-negative integer")
    return value


def _rate(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise FullSuiteError(f"native summary {name} must be a finite rate")
    return float(value)


def project_native_artifacts(*, output_dir: Path, model: str) -> Path:
    """Validate a complete native run and emit its compatibility projection."""
    report = _read_native_report(output_dir / NATIVE_REPORT_FILENAME)
    results = _read_native_results(output_dir / NATIVE_RESULTS_FILENAME)
    indices = [row.get("data_index") for row in results]
    if indices != list(range(1, EXPECTED_RESULT_COUNT + 1)):
        raise FullSuiteError("native results must retain ordered data_index values 1..102")
    failed_indices = [
        row["data_index"] for row in results if row.get("status") != "success"
    ]
    if failed_indices:
        raise FullSuiteError(
            "native verifier has transport failures at data_index "
            + ", ".join(str(index) for index in failed_indices)
        )
    success_count = _count(report.get("success_count"), "success_count")
    failure_count = _count(report.get("failure_count"), "failure_count")
    if success_count != EXPECTED_RESULT_COUNT or failure_count != 0:
        raise FullSuiteError(
            "native summary is incomplete: "
            f"success_count={success_count}, failure_count={failure_count}"
        )
    if report.get("model") != model:
        raise FullSuiteError("native summary model does not match the requested model")
    score = _rate(report.get("tool_calls_match_rate"), "tool_calls_match_rate")
    compatibility_path = _compatibility_path(output_dir)
    _write_json(
        compatibility_path,
        _compatibility_result(
            model=model,
            score=score,
            effective=EXPECTED_RESULT_COUNT,
        ),
    )
    return compatibility_path


def publish_failure(*, output_dir: Path, model: str, error: BaseException) -> Path:
    """Publish canonical failure metadata while retaining partial result rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    native_report_path = output_dir / NATIVE_REPORT_FILENAME
    native_results_path = output_dir / NATIVE_RESULTS_FILENAME
    _write_json(
        native_report_path,
        {
            "verifier": ADAPTER_NAME,
            "task": TASK_NAME,
            "model": model,
            "completed": False,
            "threshold": 0.0,
            "source": {
                "ref": UPSTREAM_REF,
                "sample_sha256": EXPECTED_SAMPLE_SHA256,
            },
            "success_count": 0,
            "failure_count": EXPECTED_RESULT_COUNT,
            "tool_calls_match_rate": 0.0,
            "integration_error": _error_dict(error),
        },
    )
    if not native_results_path.exists():
        native_results_path.write_text(
            json.dumps(
                {
                    "status": "integration_error",
                    "model": model,
                    "error": _error_dict(error),
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
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


def run_full_suite(
    *,
    python: Path,
    source_dir: Path,
    dependency_dir: Path,
    base_url: str,
    model: str,
    output_dir: Path,
    runner: Runner = subprocess.run,
) -> bool:
    """Run upstream once, then classify transport versus diagnostic failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in (NATIVE_REPORT_FILENAME, NATIVE_RESULTS_FILENAME):
        (output_dir / filename).unlink(missing_ok=True)
    for stale_path in output_dir.glob(COMPATIBILITY_GLOB):
        stale_path.unlink()
    try:
        verify_source_tree(source_dir)
        command = build_verifier_command(
            python=python,
            source_dir=source_dir,
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
            raise FullSuiteError(
                f"pinned upstream verifier exited with code {completed.returncode}"
            )
        project_native_artifacts(output_dir=output_dir, model=model)
    except (OSError, ValueError, FullSuiteError, subprocess.TimeoutExpired) as exc:
        publish_failure(output_dir=output_dir, model=model, error=exc)
        return False
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or project the pinned full MiniMax M3 verifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-source")
    prepare.add_argument("--source-dir", required=True, type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--python", required=True, type=Path)
    run.add_argument("--source-dir", required=True, type=Path)
    run.add_argument("--dependency-dir", required=True, type=Path)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--output-dir", required=True, type=Path)

    project = subparsers.add_parser("project")
    project.add_argument("--model", required=True)
    project.add_argument("--output-dir", required=True, type=Path)

    failure = subparsers.add_parser("failure")
    failure.add_argument("--model", required=True)
    failure.add_argument("--output-dir", required=True, type=Path)
    failure.add_argument("--message", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare-source":
        try:
            prepare_source_tree(args.source_dir)
        except (OSError, ValueError, FullSuiteError) as exc:
            print(f"ERROR: {exc}", file=os.sys.stderr)
            return 1
        return 0
    if args.command == "failure":
        (args.output_dir / NATIVE_RESULTS_FILENAME).unlink(missing_ok=True)
        publish_failure(
            output_dir=args.output_dir,
            model=args.model,
            error=FullSuiteError(args.message),
        )
        return 0
    if args.command == "project":
        try:
            project_native_artifacts(output_dir=args.output_dir, model=args.model)
        except (OSError, ValueError, FullSuiteError) as exc:
            publish_failure(output_dir=args.output_dir, model=args.model, error=exc)
            print(f"ERROR: {exc}", file=os.sys.stderr)
            return 1
        return 0
    completed = run_full_suite(
        python=args.python,
        source_dir=args.source_dir,
        dependency_dir=args.dependency_dir,
        base_url=args.base_url,
        model=args.model,
        output_dir=args.output_dir,
    )
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
