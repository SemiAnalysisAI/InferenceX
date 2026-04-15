#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ISL/OSL/turn distributions for ISB1 exports or kv-cache traces")
    parser.add_argument("--export-file", default=None, help="ISB1 export JSON file")
    parser.add_argument("--trace-dir", default=None, help="kv-cache-tester trace directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    return parser.parse_args()


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


def _histogram(values: list[int], bins: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        placed = False
        prev = 0
        for bound in bins:
            if value <= bound:
                key = f"{prev + 1}-{bound}"
                counts[key] = counts.get(key, 0) + 1
                placed = True
                break
            prev = bound
        if not placed:
            key = f">{bins[-1]}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _extract_isb1(export_payload: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    isl: list[int] = []
    osl: list[int] = []
    turns_per_session: list[int] = []

    for cell in export_payload.get("exports", []):
        session = cell.get("session") or {}
        turns = session.get("turns") or []
        turns_per_session.append(len(turns))
        for turn in turns:
            input_tokens = (
                turn.get("actual_input_tokens")
                or turn.get("content_token_count")
                or turn.get("prompt_tokens")
                or turn.get("input_tokens")
                or 0
            )
            output_tokens = (
                turn.get("expected_output_tokens")
                or turn.get("target_output_tokens")
                or turn.get("output_tokens")
                or 0
            )
            try:
                isl.append(int(input_tokens))
            except Exception:
                isl.append(0)
            try:
                osl.append(int(output_tokens))
            except Exception:
                osl.append(0)

    return isl, osl, turns_per_session


def _extract_trace_dir(trace_dir: Path) -> tuple[list[int], list[int], list[int]]:
    isl: list[int] = []
    osl: list[int] = []
    turns_per_session: list[int] = []

    files = list(sorted(trace_dir.glob("*.json")))
    if not files:
        raise SystemExit(f"No JSON traces found in {trace_dir}")

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sessions = payload.get("sessions") or []
        for session in sessions:
            turns = session.get("turns") or []
            turns_per_session.append(len(turns))
            for turn in turns:
                isl.append(int(turn.get("content_token_count", 0) or 0))
                osl.append(int(turn.get("target_output_tokens", 0) or 0))

    return isl, osl, turns_per_session


def build_report(isl: list[int], osl: list[int], turns_per_session: list[int], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "num_sessions": len(turns_per_session),
        "num_turns": len(isl),
        "isl": {
            "p50": _percentile([float(x) for x in isl], 0.50),
            "p95": _percentile([float(x) for x in isl], 0.95),
            "max": max(isl) if isl else 0,
            "histogram": _histogram(isl, [1024, 4096, 8192, 16384, 32768, 65536]),
        },
        "osl": {
            "p50": _percentile([float(x) for x in osl], 0.50),
            "p95": _percentile([float(x) for x in osl], 0.95),
            "max": max(osl) if osl else 0,
            "histogram": _histogram(osl, [64, 128, 256, 512, 1024, 2048, 4096]),
        },
        "turns_per_session": {
            "p50": _percentile([float(x) for x in turns_per_session], 0.50),
            "p95": _percentile([float(x) for x in turns_per_session], 0.95),
            "max": max(turns_per_session) if turns_per_session else 0,
            "histogram": _histogram(turns_per_session, [2, 4, 8, 16, 32]),
        },
    }


def main() -> int:
    args = parse_args()
    if bool(args.export_file) == bool(args.trace_dir):
        raise SystemExit("Provide exactly one of --export-file or --trace-dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.export_file:
        export_path = Path(args.export_file)
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        isl, osl, turns_per_session = _extract_isb1(payload)
        report = build_report(isl, osl, turns_per_session, source=str(export_path))
    else:
        trace_dir = Path(args.trace_dir)
        isl, osl, turns_per_session = _extract_trace_dir(trace_dir)
        report = build_report(isl, osl, turns_per_session, source=str(trace_dir))

    output_path = output_dir / "distribution_report.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
