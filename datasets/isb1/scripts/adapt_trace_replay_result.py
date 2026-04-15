#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pick(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            value = _to_float(row.get(key))
            if value is not None:
                return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt kv-cache trace replay CSV output into ISB1 replay JSON schema"
    )
    parser.add_argument("--input-dir", default="/workspace", help="Directory containing trace replay outputs")
    parser.add_argument(
        "--detailed-csv",
        default="detailed_results.csv",
        help="Detailed replay CSV filename (inside --input-dir)",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path (used as supplemental source if present)",
    )
    parser.add_argument("--output-json", required=True, help="Output adapted replay JSON path")
    parser.add_argument("--model-id", default="", help="Model ID for output metadata")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Max concurrency used")
    parser.add_argument("--request-mode", default="multi-turn", help="Request mode metadata")
    parser.add_argument(
        "--benchmark-certification-status",
        default="dataset_replay_verified",
        help="Benchmark certification status to stamp in selection",
    )
    parser.add_argument(
        "--support-status",
        default="reviewed_preview",
        help="Support status to stamp in selection",
    )
    parser.add_argument(
        "--result-stem",
        default="",
        help="Optional result stem to infer total wall time from /workspace/<stem>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    detailed_csv_path = input_dir / args.detailed_csv
    output_path = Path(args.output_json)

    if not detailed_csv_path.exists():
        raise SystemExit(f"Missing detailed CSV: {detailed_csv_path}")

    rows = _read_csv_rows(detailed_csv_path)
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    output_tokens: list[float] = []
    prompt_tokens: list[float] = []
    session_ids: set[str] = set()

    for row in rows:
        ttft = _pick(row, "ttft_ms", "ttft", "time_to_first_token_ms")
        if ttft is not None:
            ttft_ms.append(ttft)

        tpot = _pick(row, "tpot_ms", "tpot", "time_per_output_token_ms")
        if tpot is not None:
            tpot_ms.append(tpot)

        out_tok = _pick(row, "output_tokens", "generated_tokens", "completion_tokens")
        if out_tok is not None:
            output_tokens.append(out_tok)

        in_tok = _pick(row, "input_tokens", "prompt_tokens", "content_token_count")
        if in_tok is not None:
            prompt_tokens.append(in_tok)

        for key in ("session_id", "session", "conversation_id"):
            sid = row.get(key)
            if sid:
                session_ids.add(str(sid))
                break

    completed_sessions = len(session_ids) if session_ids else len(rows)
    total_sessions = completed_sessions

    total_output_tokens = sum(output_tokens)
    total_prompt_tokens = sum(prompt_tokens)
    total_token_count = total_output_tokens + total_prompt_tokens

    total_wall_time_s = 0.0
    if args.result_stem:
        maybe_summary = input_dir / f"{args.result_stem}.json"
        if maybe_summary.exists():
            try:
                summary = json.loads(maybe_summary.read_text(encoding="utf-8"))
                total_wall_time_s = float(
                    _to_float(summary.get("test_duration_seconds"))
                    or _to_float(summary.get("duration_s"))
                    or _to_float(summary.get("total_duration_s"))
                    or 0.0
                )
            except Exception:
                total_wall_time_s = 0.0

    if total_wall_time_s <= 0 and args.summary_json:
        summary_path = Path(args.summary_json)
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                total_wall_time_s = float(
                    _to_float(summary.get("test_duration_seconds"))
                    or _to_float(summary.get("duration_s"))
                    or _to_float(summary.get("total_duration_s"))
                    or 0.0
                )
            except Exception:
                total_wall_time_s = 0.0

    if total_wall_time_s <= 0:
        total_wall_time_s = 1.0

    aggregate_metrics = {
        "total_token_throughput_tps": total_token_count / total_wall_time_s,
        "output_throughput_tps": total_output_tokens / total_wall_time_s,
        "mean_ttft_ms": mean(ttft_ms) if ttft_ms else 0.0,
        "median_ttft_ms": _percentile(ttft_ms, 0.50),
        "p99_ttft_ms": _percentile(ttft_ms, 0.99),
        "mean_tpot_ms": mean(tpot_ms) if tpot_ms else 0.0,
        "median_tpot_ms": _percentile(tpot_ms, 0.50),
        "p99_tpot_ms": _percentile(tpot_ms, 0.99),
        "completed_sessions": completed_sessions,
        "total_sessions": total_sessions,
        "session_throughput_sps": completed_sessions / total_wall_time_s,
        "total_wall_time_s": total_wall_time_s,
    }

    adapted = {
        "model_id": args.model_id,
        "max_concurrency": args.max_concurrency,
        "request_mode": args.request_mode,
        "harness_request_mode": "auto",
        "aggregate_metrics": aggregate_metrics,
        "selection": {
            "support_statuses": [args.support_status],
            "benchmark_certification_statuses": [args.benchmark_certification_status],
        },
        "server_metrics_summary": {
            "observability_status": "unavailable",
            "gpu_cache_metric_name": None,
            "cpu_cache_metric_name": None,
            "gpu_cache_usage_peak": 0.0,
            "cpu_cache_usage_peak": 0.0,
            "preemption_count": 0,
            "kv_offload_observed": False,
            "cpu_cache_metric_available": False,
        },
        "depth_telemetry": {
            "total_actual_input_tokens": int(total_prompt_tokens),
            "max_actual_context_len_per_turn": int(max(prompt_tokens) if prompt_tokens else 0),
        },
        "num_sessions": total_sessions,
        "max_turns": None,
        "per_turn_metrics": {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(adapted, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote adapted replay JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
