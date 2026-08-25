"""Adapt AIPerf profiling artifacts to the strict power-window contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils.aggregate_power import (
    POWER_METRIC_SCHEMA_VERSION,
    _empty_integration,
    _patch_power_result,
    _validation_payload,
    _write_json_atomic,
)
from utils.aggregate_power import run as run_power
from utils.aggregate_power_multinode import _ALL_POWER_METRIC_KEYS
from utils.aggregate_power_multinode import run as run_multinode_power

from .process_agentic_result import _resolve_artifact_dir
from .request_metrics import extract_per_record_ints, load_aggregate, load_records

_UTC_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MULTINODE_WINDOW_STEM_RE = re.compile(r"^agentic_power_concurrency_([1-9][0-9]*)$")
_FORMAL_WINDOW_ENV = (
    "SRT_MEASUREMENT_WINDOW_DIR",
    "SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE",
    "SRT_MEASUREMENT_WINDOW_CONCURRENCIES",
    "SRT_MEASUREMENT_WINDOW_RESULT_ROOT",
)


def _captured_timezone(result_dir: Path) -> tuple[timezone | None, str | None]:
    """Load the launch-time UTC offset captured beside AgentX telemetry."""
    offset_path = result_dir / "agentic_power_timezone_offset.txt"
    if not offset_path.is_file():
        return None, "profile_timezone_offset_missing"
    try:
        raw_offset = offset_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None, "profile_timezone_offset_invalid"
    match = _UTC_OFFSET_RE.fullmatch(raw_offset)
    if match is None:
        return None, "profile_timezone_offset_invalid"
    hours, minutes = int(match.group(2)), int(match.group(3))
    if hours > 23 or minutes > 59:
        return None, "profile_timezone_offset_invalid"
    direction = 1 if match.group(1) == "+" else -1
    return timezone(direction * timedelta(hours=hours, minutes=minutes)), None


def _parse_profile_timestamp(value: Any, *, fallback_tz: timezone | None) -> float | None:
    """Parse a timezone-aware ISO timestamp or Unix epoch seconds."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        timestamp = float(value)
        return timestamp if math.isfinite(timestamp) else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        if fallback_tz is None:
            return None
        parsed = parsed.replace(tzinfo=fallback_tz)
    return parsed.astimezone(timezone.utc).timestamp()


def build_power_window(result_dir: Path) -> tuple[dict[str, int | float] | None, list[str]]:
    """Build a strict benchmark window from successful profiling requests."""
    artifact_dir = _resolve_artifact_dir(result_dir)
    aggregate_path = artifact_dir / "profile_export_aiperf.json"
    records_path = artifact_dir / "profile_export.jsonl"
    if not aggregate_path.is_file() or not records_path.is_file():
        return None, ["profile_artifacts_missing"]

    try:
        aggregate = load_aggregate(aggregate_path)
        records = load_records(records_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None, ["profile_artifacts_invalid"]

    raw_start = aggregate.get("start_time")
    raw_end = aggregate.get("end_time")
    if raw_start is None or raw_end is None:
        return None, ["profile_window_missing"]
    parsed_datetimes: list[datetime] = []
    for value in (raw_start, raw_end):
        if isinstance(value, str):
            try:
                parsed_datetimes.append(
                    datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
                )
            except ValueError:
                pass
    needs_captured_timezone = any(value.tzinfo is None for value in parsed_datetimes)
    fallback_tz = None
    if needs_captured_timezone:
        fallback_tz, timezone_reason = _captured_timezone(result_dir)
        if timezone_reason is not None:
            return None, [timezone_reason]

    start = _parse_profile_timestamp(raw_start, fallback_tz=fallback_tz)
    end = _parse_profile_timestamp(raw_end, fallback_tz=fallback_tz)
    if start is None or end is None or end <= start:
        return None, ["profile_window_invalid"]

    completed = len(records)
    if completed <= 0:
        return None, ["successful_request_count_invalid"]

    input_tokens = extract_per_record_ints(records, "input_sequence_length")
    output_tokens = extract_per_record_ints(records, "output_sequence_length")
    if (
        len(input_tokens) != completed
        or len(output_tokens) != completed
        or any(value < 0 for value in input_tokens + output_tokens)
        or sum(input_tokens) <= 0
        or sum(output_tokens) <= 0
    ):
        return None, ["incomplete_token_accounting"]

    return (
        {
            "benchmark_start_time_unix": start,
            "benchmark_end_time_unix": end,
            "duration": end - start,
            "completed": completed,
            "total_input_tokens": sum(input_tokens),
            "total_output_tokens": sum(output_tokens),
        },
        [],
    )


def _record_adapter_failure(
    *,
    result_dir: Path,
    agg_result: Path,
    expected_num_gpus: int | None,
    reasons: list[str],
) -> None:
    """Write the same invalid aggregate and audit artifacts as aggregate_power."""
    csv_path = result_dir / "gpu_metrics.csv"
    window_path = result_dir / "agentic_power_window.json"
    validation_path = result_dir / "power_validation.json"
    integration = _empty_integration(
        expected_num_gpus=expected_num_gpus,
        reasons=reasons,
    )
    _patch_power_result(agg_result, power_valid=False, metrics={})
    payload = _validation_payload(
        csv_path=csv_path,
        bench_result=window_path,
        benchmark=None,
        integration=integration,
        power_valid=False,
        reasons=reasons,
        metrics={},
        accumulator_check=None,
    )
    payload["window_source"] = "aiperf_profile_lifecycle"
    _write_json_atomic(validation_path, payload)


def run_agentic_power(
    *,
    result_dir: Path,
    agg_result: Path,
    expected_num_gpus: int | None,
    require_power: bool = False,
) -> int:
    """Validate AgentX power telemetry, failing only in strict mode."""
    window, reasons = build_power_window(result_dir)
    if window is None:
        try:
            _record_adapter_failure(
                result_dir=result_dir,
                agg_result=agg_result,
                expected_num_gpus=expected_num_gpus,
                reasons=reasons,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(
                f"[agentx_power] Failed to record adapter failure: {exc}",
                file=sys.stderr,
            )
        print(
            f"[agentx_power] Power-window adaptation failed: {', '.join(reasons)}",
            file=sys.stderr,
        )
        return 1 if require_power else 0

    window_path = result_dir / "agentic_power_window.json"
    try:
        _write_json_atomic(window_path, window)
    except OSError:
        reasons = ["power_window_unwritable"]
        try:
            _record_adapter_failure(
                result_dir=result_dir,
                agg_result=agg_result,
                expected_num_gpus=expected_num_gpus,
                reasons=reasons,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(
                f"[agentx_power] Failed to record adapter failure: {exc}",
                file=sys.stderr,
            )
        print(
            f"[agentx_power] Power-window adaptation failed: {', '.join(reasons)}",
            file=sys.stderr,
        )
        return 1 if require_power else 0
    return run_power(
        result_dir / "gpu_metrics.csv",
        window_path,
        agg_result,
        expected_num_gpus=expected_num_gpus,
        validation_result=result_dir / "power_validation.json",
        require_power=require_power,
    )


def _fail_multinode_adapter(message: str, *, require_power: bool) -> int:
    print(f"[agentx_power] {message}", file=sys.stderr)
    return 1 if require_power else 0


def _positive_concurrencies(raw: str) -> list[int] | None:
    tokens = raw.split()
    if not tokens or any(not token.isdecimal() for token in tokens):
        return None
    values = [int(token) for token in tokens]
    if any(value <= 0 for value in values) or len(set(values)) != len(values):
        return None
    return values


def _multinode_window_contract(
    *,
    result_dir: Path,
    concurrency: int,
) -> tuple[Path, Path, Path] | None:
    """Resolve and validate the formal custom-benchmark window contract."""
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        return None
    values = {name: os.environ.get(name, "") for name in _FORMAL_WINDOW_ENV}
    if any(not value for value in values.values()):
        return None
    if values["SRT_MEASUREMENT_WINDOW_BENCHMARK_TYPE"] != "custom":
        return None
    measured = _positive_concurrencies(
        values["SRT_MEASUREMENT_WINDOW_CONCURRENCIES"]
    )
    if measured is None or concurrency not in measured:
        return None

    window_dir = Path(values["SRT_MEASUREMENT_WINDOW_DIR"])
    result_root = Path(values["SRT_MEASUREMENT_WINDOW_RESULT_ROOT"])
    if (
        not window_dir.is_absolute()
        or not result_root.is_absolute()
        or not result_dir.is_absolute()
        or not window_dir.is_dir()
        or not result_root.is_dir()
        or not result_dir.is_dir()
    ):
        return None
    try:
        window_dir.resolve().relative_to(result_root.resolve())
        relative_result_dir = result_dir.resolve().relative_to(result_root.resolve())
    except (OSError, ValueError):
        return None

    stem = f"agentic_power_concurrency_{concurrency}"
    formal_result = result_dir / f"{stem}.json"
    formal_window = window_dir / f"{stem}.json"
    result_path = relative_result_dir / formal_result.name
    return formal_result, formal_window, result_path


def _window_payload(
    *,
    result_path: Path,
    concurrency: int,
    status: str,
    start: float,
    end: float | None,
    duration: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_type": "custom",
        "result_path": result_path.as_posix(),
        "concurrency": concurrency,
        "benchmark_start_time_unix": start,
        "benchmark_end_time_unix": end,
        "duration": duration,
        "clock_source": "head_node_unix_clock",
        "status": status,
        "reason": None,
    }


def write_multinode_power_window(
    *,
    result_dir: Path,
    concurrency: int,
    state: str,
    require_power: bool = False,
) -> int:
    """Publish one AgentX custom-benchmark formal window on the head clock."""
    contract = _multinode_window_contract(
        result_dir=result_dir,
        concurrency=concurrency,
    )
    if contract is None:
        return _fail_multinode_adapter(
            "Invalid formal measurement-window contract for multinode AgentX",
            require_power=require_power,
        )
    formal_result, formal_window, result_path = contract

    if state == "running":
        payload = _window_payload(
            result_path=result_path,
            concurrency=concurrency,
            status="running",
            start=time.time(),
            end=None,
            duration=None,
        )
        try:
            _write_json_atomic(formal_window, payload)
        except OSError as exc:
            return _fail_multinode_adapter(
                f"Failed to write formal measurement-window contract: {exc}",
                require_power=require_power,
            )
        return 0

    if state != "completed":
        return _fail_multinode_adapter(
            f"Invalid formal measurement-window state: {state}",
            require_power=require_power,
        )

    boundary, reasons = build_power_window(result_dir)
    if boundary is None:
        return _fail_multinode_adapter(
            "Failed to complete formal measurement-window contract: "
            + ", ".join(reasons),
            require_power=require_power,
        )
    formal_result_payload = {"max_concurrency": concurrency, **boundary}
    completed = _window_payload(
        result_path=result_path,
        concurrency=concurrency,
        status="completed",
        start=float(boundary["benchmark_start_time_unix"]),
        end=float(boundary["benchmark_end_time_unix"]),
        duration=float(boundary["duration"]),
    )
    try:
        # The collector validates the result referenced by a completed window.
        # Publish that result first so it can never observe a completed window
        # that points to a missing or partially written result.
        _write_json_atomic(formal_result, formal_result_payload)
        _write_json_atomic(formal_window, completed)
    except OSError as exc:
        return _fail_multinode_adapter(
            f"Failed to write formal measurement-window contract: {exc}",
            require_power=require_power,
        )
    return 0


def _record_multinode_adapter_failure(
    *,
    agg_result: Path,
    validation_result: Path,
    reasons: list[str],
) -> None:
    aggregate = json.loads(agg_result.read_text(encoding="utf-8"))
    if not isinstance(aggregate, dict):
        raise ValueError("AgentX aggregate must be a JSON object")
    for key in _ALL_POWER_METRIC_KEYS:
        aggregate.pop(key, None)
    aggregate["power_metric_schema_version"] = POWER_METRIC_SCHEMA_VERSION
    aggregate["power_valid"] = 0
    aggregate.pop("power_invalid_reasons", None)
    _write_json_atomic(agg_result, aggregate)
    _write_json_atomic(
        validation_result,
        {
            "power_valid": False,
            "reasons": reasons,
            "window_source": "aiperf_multinode_custom_benchmark",
        },
    )


def _formal_result_for_directory(result_dir: Path) -> Path | None:
    candidates = [
        path
        for path in result_dir.glob("agentic_power_concurrency_*.json")
        if _MULTINODE_WINDOW_STEM_RE.fullmatch(path.stem)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _gpu_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def run_multinode_agentic_power(
    *,
    result_dir: Path,
    agg_result: Path,
    power_dir: Path,
    logs_root: Path,
    expected_producer_sha: str,
    require_power: bool = False,
) -> int:
    """Join one AgentX aggregate to the finalized central multinode package."""
    validation_result = result_dir / "power_validation.json"
    reasons: list[str] = []
    try:
        aggregate = json.loads(agg_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        aggregate = None
        reasons.append("agentic_aggregate_invalid")
    if not isinstance(aggregate, dict):
        if not reasons:
            reasons.append("agentic_aggregate_invalid")
        aggregate = {}

    prefill_gpus = _gpu_count(aggregate.get("num_prefill_gpu"))
    decode_gpus = _gpu_count(aggregate.get("num_decode_gpu"))
    disagg = aggregate.get("disagg")
    if (
        not isinstance(disagg, bool)
        or prefill_gpus is None
        or decode_gpus is None
        or prefill_gpus + decode_gpus <= 0
    ):
        reasons.append("agentic_gpu_topology_invalid")
    bench_result = _formal_result_for_directory(result_dir)
    if bench_result is None:
        reasons.append("formal_benchmark_result_missing")
    try:
        result_dir.resolve().relative_to(logs_root.resolve())
    except (OSError, ValueError):
        reasons.append("agentic_result_outside_logs_root")
    if _COMMIT_SHA_RE.fullmatch(expected_producer_sha) is None:
        reasons.append("expected_producer_sha_invalid")

    if reasons:
        try:
            if agg_result.is_file():
                _record_multinode_adapter_failure(
                    agg_result=agg_result,
                    validation_result=validation_result,
                    reasons=reasons,
                )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(
                f"[agentx_power] Failed to record multinode adapter failure: {exc}",
                file=sys.stderr,
            )
        return _fail_multinode_adapter(
            "Multinode AgentX power adaptation failed: " + ", ".join(reasons),
            require_power=require_power,
        )

    assert prefill_gpus is not None
    assert decode_gpus is not None
    assert isinstance(disagg, bool)
    assert bench_result is not None
    aggregate_gpus = 0
    if not disagg:
        aggregate_gpus = prefill_gpus + decode_gpus
        prefill_gpus = 0
        decode_gpus = 0
    return run_multinode_power(
        power_dir=power_dir,
        bench_result=bench_result,
        agg_result=agg_result,
        prefill_gpus=prefill_gpus,
        decode_gpus=decode_gpus,
        aggregate_gpus=aggregate_gpus,
        expected_producer_sha=expected_producer_sha,
        logs_root=logs_root,
        validation_result=validation_result,
        require_power=require_power,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--agg-result", type=Path)
    parser.add_argument("--expected-num-gpus", type=int)
    parser.add_argument("--write-multinode-window", choices=("running", "completed"))
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--power-dir", type=Path)
    parser.add_argument("--logs-root", type=Path)
    parser.add_argument("--expected-producer-sha")
    parser.add_argument(
        "--require-power",
        action="store_true",
        default=os.environ.get("REQUIRE_POWER", "").lower() in {"1", "true", "yes"},
    )
    args = parser.parse_args()
    if args.write_multinode_window is not None:
        if args.concurrency is None:
            parser.error("--concurrency is required with --write-multinode-window")
        return write_multinode_power_window(
            result_dir=args.result_dir,
            concurrency=args.concurrency,
            state=args.write_multinode_window,
            require_power=args.require_power,
        )
    if args.power_dir is not None:
        if args.agg_result is None or args.logs_root is None or args.expected_producer_sha is None:
            parser.error(
                "--agg-result, --logs-root, and --expected-producer-sha are required with --power-dir"
            )
        return run_multinode_agentic_power(
            result_dir=args.result_dir,
            agg_result=args.agg_result,
            power_dir=args.power_dir,
            logs_root=args.logs_root,
            expected_producer_sha=args.expected_producer_sha,
            require_power=args.require_power,
        )
    if args.agg_result is None:
        parser.error("--agg-result is required for single-node AgentX power")
    return run_agentic_power(
        result_dir=args.result_dir,
        agg_result=args.agg_result,
        expected_num_gpus=args.expected_num_gpus,
        require_power=args.require_power,
    )


if __name__ == "__main__":
    raise SystemExit(main())
