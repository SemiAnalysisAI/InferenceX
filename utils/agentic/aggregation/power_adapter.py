"""Adapt AIPerf profiling artifacts to the strict power-window contract."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils.aggregate_power import (
    _empty_integration,
    _patch_power_result,
    _validation_payload,
    _write_json_atomic,
)
from utils.aggregate_power import run as run_power

from .process_agentic_result import _resolve_artifact_dir
from .request_metrics import extract_per_record_ints, load_aggregate, load_records

_UTC_OFFSET_RE = re.compile(r"^([+-])(\d{2}):?(\d{2})$")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--agg-result", type=Path, required=True)
    parser.add_argument("--expected-num-gpus", type=int)
    parser.add_argument(
        "--require-power",
        action="store_true",
        default=os.environ.get("REQUIRE_POWER", "").lower() in {"1", "true", "yes"},
    )
    args = parser.parse_args()
    return run_agentic_power(
        result_dir=args.result_dir,
        agg_result=args.agg_result,
        expected_num_gpus=args.expected_num_gpus,
        require_power=args.require_power,
    )


if __name__ == "__main__":
    raise SystemExit(main())
