#!/usr/bin/env python3
"""Run the stock Kimi Vendor Verifier and project its native report."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
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
DIAGNOSTIC_REPORT_FILENAME = "eval_kimi_tool_call_diagnostic.json"
DEFAULT_DIAGNOSTIC_TEMPERATURE = 0.0
DEFAULT_DIAGNOSTIC_SEED = 1
DEFAULT_DIAGNOSTIC_SEQUENCE = (
    "unary",
    "unary",
    "stream",
    "stream",
    "stream",
    "unary",
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert OpenAI response models into JSON-compatible evidence."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _response_ids(value: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(value, Mapping):
        response_id = value.get("id")
        if isinstance(response_id, str) and response_id not in ids:
            ids.append(response_id)
        for item in value.values():
            for nested_id in _response_ids(item):
                if nested_id not in ids:
                    ids.append(nested_id)
    elif isinstance(value, list):
        for item in value:
            for nested_id in _response_ids(item):
                if nested_id not in ids:
                    ids.append(nested_id)
    return ids


class _CapturingStream:
    """Record stream chunks while preserving the upstream iterator contract."""

    def __init__(self, stream: Any, record: dict[str, Any]) -> None:
        self._stream = stream
        self._record = record

    def __iter__(self):
        try:
            for chunk in self._stream:
                raw_chunk = _jsonable(chunk)
                self._record["raw_chunks"].append(raw_chunk)
                for response_id in _response_ids(raw_chunk):
                    if response_id not in self._record["response_ids"]:
                        self._record["response_ids"].append(response_id)
                yield chunk
        except Exception as exc:
            self._record["transport_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise


class _CapturingCompletions:
    """Delegate OpenAI requests while retaining exact payloads and raw replies."""

    def __init__(
        self,
        completions: Any,
        record: dict[str, Any],
        request_overrides: Mapping[str, Any],
    ) -> None:
        self._completions = completions
        self._record = record
        self._request_overrides = request_overrides

    def create(self, **request: Any) -> Any:
        request.update(self._request_overrides)
        self._record["request_payload"] = _jsonable(request)
        self._record["request_started_at"] = _utc_timestamp()
        try:
            response = self._completions.create(**request)
        except Exception as exc:
            self._record["response_received_at"] = _utc_timestamp()
            self._record["transport_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        self._record["response_received_at"] = _utc_timestamp()
        if request.get("stream"):
            self._record["raw_chunks"] = []
            return _CapturingStream(response, self._record)
        raw_response = _jsonable(response)
        self._record["raw_response"] = raw_response
        self._record["response_ids"] = _response_ids(raw_response)
        return response


class _CapturingClient:
    """Expose the OpenAI chat interface expected by the upstream helper."""

    def __init__(
        self,
        client: Any,
        record: dict[str, Any],
        request_overrides: Mapping[str, Any],
    ) -> None:
        chat_type = type("_CapturingChat", (), {})
        self.chat = chat_type()
        self.chat.completions = _CapturingCompletions(
            client.chat.completions, record, request_overrides
        )


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version}
    for distribution in ("openai", "httpx", "jsonschema"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions

def _capture_version_endpoint(base_url: str) -> dict[str, Any]:
    version_url = f"{base_url.removesuffix('/v1').rstrip('/')}/version"
    started_at = _utc_timestamp()
    try:
        httpx = importlib.import_module("httpx")
        response = httpx.get(version_url, timeout=5.0)
        safe_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower()
            in {
                "content-type",
                "date",
                "server",
                "x-request-id",
                "x-sglang-version",
            }
        }
        return {
            "url": version_url,
            "started_at": started_at,
            "completed_at": _utc_timestamp(),
            "status_code": response.status_code,
            "headers": safe_headers,
            "body": response.text,
        }
    except Exception as exc:
        return {
            "url": version_url,
            "started_at": started_at,
            "completed_at": _utc_timestamp(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def run_diagnostic_sequence(
    *,
    verifier_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    output_dir: Path,
    temperature: float = DEFAULT_DIAGNOSTIC_TEMPERATURE,
    seed: int = DEFAULT_DIAGNOSTIC_SEED,
    sequence: Sequence[str] = DEFAULT_DIAGNOSTIC_SEQUENCE,
) -> None:
    """Run ordered stock-helper requests and preserve client-visible evidence."""
    report_path = output_dir / DIAGNOSTIC_REPORT_FILENAME
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "started_at": _utc_timestamp(),
        "controls": {
            "temperature": temperature,
            "seed": seed,
            "sequence": list(sequence),
        },
        "runtime_versions": _runtime_versions(),
        "version_endpoint": _capture_version_endpoint(base_url),
        "results": records,
    }
    sys.path.insert(0, str(verifier_dir))
    client: Any = None
    try:
        validator = importlib.import_module("tests.tool_call_json_schema.validator")
        report["parser"] = {
            "module": validator.__name__,
            "file": str(Path(validator.__file__).resolve()),
        }
        cases = validator.load_cases(
            verifier_dir / "testdata" / "walle_validator_cases" / "validator_cases"
        )
        selected = validator.select_cases(
            cases,
            selection="object",
            requested_cases=set(),
            max_cases=1,
        )
        if len(selected) != 1:
            raise ValueError(
                f"diagnostic expected one selected case, found {len(selected)}"
            )
        case, schema, selection_reason = selected[0]
        report["case"] = {
            "suite": case.suite,
            "line": case.line,
            "selection_reason": selection_reason,
            "schema": schema,
        }
        client = validator.make_client(base_url, api_key, 120)
        mode_occurrences = {"unary": 0, "stream": 0}
        for index, mode in enumerate(sequence, start=1):
            mode_occurrences[mode] += 1
            record: dict[str, Any] = {
                "index": index,
                "mode": mode,
                "mode_occurrence": mode_occurrences[mode],
                "temperature_state": (
                    "cold" if mode_occurrences[mode] == 1 else "warm"
                ),
                "sampling_controls": {
                    "temperature": temperature,
                    "seed": seed,
                },
                "started_at": _utc_timestamp(),
                "response_ids": [],
            }
            records.append(record)
            try:
                response = validator.send_tool_schema(
                    _CapturingClient(
                        client,
                        record,
                        {"temperature": temperature, "seed": seed},
                    ),
                    model,
                    schema,
                    2048,
                    False,
                    "none",
                    stream=mode == "stream",
                )
                valid, validation_message = validator.validate_arguments(
                    schema, response.arguments
                )
                record["parser_output"] = {
                    "accepted": response.accepted,
                    "message": response.message,
                    "arguments": response.arguments,
                    "arguments_valid": valid,
                    "validation_message": validation_message,
                    "http_status": response.http_status,
                    "error_type": response.error_type,
                }
            except Exception as exc:
                record["diagnostic_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            finally:
                record["completed_at"] = _utc_timestamp()
    except Exception as exc:
        report["setup_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                report["client_close_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        report["completed_at"] = _utc_timestamp()
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        sys.path.pop(0)


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
    diagnostic: bool = False,
    diagnostic_temperature: float = DEFAULT_DIAGNOSTIC_TEMPERATURE,
    diagnostic_seed: int = DEFAULT_DIAGNOSTIC_SEED,
    diagnostic_sequence: Sequence[str] = DEFAULT_DIAGNOSTIC_SEQUENCE,
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
        if diagnostic:
            try:
                run_diagnostic_sequence(
                    verifier_dir=verifier_dir,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    output_dir=output_dir,
                    temperature=diagnostic_temperature,
                    seed=diagnostic_seed,
                    sequence=diagnostic_sequence,
                )
            except Exception as exc:
                print(
                    f"WARNING: Kimi diagnostic collection failed: {exc}",
                    file=sys.stderr,
                )
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

def _diagnostic_sequence(value: str) -> tuple[str, ...]:
    sequence = tuple(item.strip() for item in value.split(",") if item.strip())
    if not sequence or any(mode not in {"unary", "stream"} for mode in sequence):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated sequence of unary and stream"
        )
    return sequence


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
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run ordered raw request diagnostics before the unchanged verifier",
    )
    parser.add_argument(
        "--diagnostic-temperature",
        type=float,
        default=DEFAULT_DIAGNOSTIC_TEMPERATURE,
    )
    parser.add_argument(
        "--diagnostic-seed",
        type=int,
        default=DEFAULT_DIAGNOSTIC_SEED,
    )
    parser.add_argument(
        "--diagnostic-sequence",
        type=_diagnostic_sequence,
        default=DEFAULT_DIAGNOSTIC_SEQUENCE,
    )
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
        diagnostic=args.diagnostic,
        diagnostic_temperature=args.diagnostic_temperature,
        diagnostic_seed=args.diagnostic_seed,
        diagnostic_sequence=args.diagnostic_sequence,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
