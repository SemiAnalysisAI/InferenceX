"""Aggregate measured GPU power from a vendor SMI CSV into the agg result JSON.

Reads a GPU-metrics CSV produced by `start_gpu_monitor` (nvidia-smi or amd-smi),
filters samples to the benchmark load window using start/end Unix timestamps
written by benchmark_serving.py, and patches two keys into the aggregated
result JSON consumed by InferenceX-app's ETL:

  - avg_power_w:               mean per-GPU power draw (W) during the load window
  - joules_per_output_token:   (avg_power_w * num_gpus * duration_s) / total_output_tokens

The ETL (`packages/db/src/etl/benchmark-mapper.ts`) auto-captures any numeric
field in the agg JSON into the `metrics` JSONB column, so no schema migration
is required.

Vendor schema detection is regex-based: any timestamp-like column + any column
whose name contains "power" (excluding "limit"/"cap"/"max") is picked up.
NVIDIA emits "power.draw [W]"; AMD's amd-smi varies by version. Both are
handled.

This script is best-effort. Missing or malformed CSV exits 0 without patching
so a monitoring hiccup never breaks the benchmark upload.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


_POWER_COL_RE = re.compile(r"power", re.IGNORECASE)
_POWER_EXCLUDE_RE = re.compile(r"limit|cap|max|min", re.IGNORECASE)
_TIMESTAMP_COL_RE = re.compile(r"time", re.IGNORECASE)
_GPU_INDEX_COL_RE = re.compile(r"^(index|gpu|gpu_id|gpu_index|card|device)$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_timestamp(value: str) -> float | None:
    """Best-effort timestamp parse to Unix epoch seconds (local wall clock).

    Handles the formats observed in practice:
      - nvidia-smi: "2025/01/15 12:34:56.789" (local time, no TZ)
      - amd-smi:    ISO 8601 "2025-01-15T12:34:56.789" or epoch seconds
      - Plain numeric epoch (int or float, s or ms)
    """
    value = value.strip()
    if not value:
        return None
    # Plain epoch number — accept both seconds and milliseconds.
    if _NUMBER_RE.fullmatch(value):
        n = float(value)
        return n / 1000.0 if n > 1e12 else n
    # nvidia-smi: "YYYY/MM/DD HH:MM:SS.ffffff"
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            pass
    # ISO 8601 (amd-smi variants). fromisoformat tolerates 'T' or space separator
    # in Python 3.11+; older versions need 'T'.
    iso_value = value.replace(" ", "T", 1) if " " in value and "T" not in value else value
    try:
        dt = datetime.fromisoformat(iso_value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # Treat naive timestamps as local time (matches nvidia-smi convention).
        return dt.timestamp()
    return dt.astimezone(timezone.utc).timestamp()


def _parse_power(value: str) -> float | None:
    """Extract the first numeric value from a power cell.

    nvidia-smi formats power as "412.34 W"; some configurations report
    "[N/A]" when power capping is disabled. AMD reports a bare number.
    """
    value = value.strip()
    if not value or value.lower() in {"[n/a]", "n/a", "na"}:
        return None
    m = _NUMBER_RE.search(value)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _detect_columns(header: list[str]) -> tuple[str | None, str | None, str | None]:
    """Return (timestamp_col, power_col, gpu_index_col) from a CSV header.

    Power column: contains "power" and not "limit"/"cap"/"max"/"min".
    Timestamp column: contains "time".
    GPU index column: optional — used to count distinct GPUs per sample.
    """
    timestamp_col = next((c for c in header if _TIMESTAMP_COL_RE.search(c)), None)
    power_col = next(
        (c for c in header if _POWER_COL_RE.search(c) and not _POWER_EXCLUDE_RE.search(c)),
        None,
    )
    gpu_col = next((c for c in header if _GPU_INDEX_COL_RE.match(c.strip())), None)
    return timestamp_col, power_col, gpu_col


def aggregate_power(
    csv_path: Path,
    start_unix: float,
    end_unix: float,
) -> tuple[float, int] | None:
    """Return (per_gpu_avg_power_w, num_gpus) for samples in [start, end].

    Returns None if the CSV is missing, empty, has no detectable power column,
    or no rows fall in the window.
    """
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return None
    if end_unix <= start_unix:
        return None

    try:
        with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            header = [c.strip() for c in (reader.fieldnames or [])]
            reader.fieldnames = header
            timestamp_col, power_col, gpu_col = _detect_columns(header)
            if not timestamp_col or not power_col:
                return None

            # Group power readings by sample timestamp so per-sample total power
            # (sum across GPUs) is computed correctly even if rows are interleaved.
            per_sample_total: dict[float, float] = {}
            per_sample_gpus: dict[float, set[str]] = {}
            gpu_keys: set[str] = set()

            for row in reader:
                ts_raw = (row.get(timestamp_col) or "").strip()
                pw_raw = (row.get(power_col) or "").strip()
                ts = _parse_timestamp(ts_raw)
                pw = _parse_power(pw_raw)
                if ts is None or pw is None:
                    continue
                if ts < start_unix or ts > end_unix:
                    continue
                # Bucket by sample timestamp (rounded to ms to absorb sub-ms drift).
                bucket = round(ts, 3)
                per_sample_total[bucket] = per_sample_total.get(bucket, 0.0) + pw
                gpu_id = (row.get(gpu_col) or "0").strip() if gpu_col else "0"
                per_sample_gpus.setdefault(bucket, set()).add(gpu_id)
                gpu_keys.add(gpu_id)
    except (OSError, csv.Error):
        return None

    if not per_sample_total:
        return None

    # Number of distinct GPUs seen across the window.
    num_gpus = max(len(gpu_keys), 1)
    # Per-sample mean power per GPU = sum across GPUs at that timestamp / GPUs seen at that timestamp.
    per_sample_mean_per_gpu = [
        total / max(len(per_sample_gpus[ts]), 1) for ts, total in per_sample_total.items()
    ]
    return mean(per_sample_mean_per_gpu), num_gpus


def _load_bench_window(bench_result_path: Path) -> tuple[float, float, float, int] | None:
    """Read (start_unix, end_unix, duration_s, total_output_tokens) from the raw bench JSON."""
    try:
        bench = json.loads(bench_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    start = bench.get("benchmark_start_time_unix")
    end = bench.get("benchmark_end_time_unix")
    duration = bench.get("duration")
    total_output = bench.get("total_output_tokens")
    if not all(isinstance(v, (int, float)) for v in (start, end, duration)):
        return None
    if not isinstance(total_output, int) or total_output <= 0:
        return None
    return float(start), float(end), float(duration), int(total_output)


def patch_agg_result(
    agg_path: Path,
    avg_power_w: float,
    joules_per_output_token: float,
) -> None:
    """Read the agg JSON, add the two power keys, and write it back atomically."""
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    data["avg_power_w"] = round(avg_power_w, 3)
    data["joules_per_output_token"] = round(joules_per_output_token, 6)
    tmp_path = agg_path.with_suffix(agg_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(agg_path)


def run(csv_path: Path, bench_result: Path, agg_result: Path) -> int:
    window = _load_bench_window(bench_result)
    if window is None:
        print(
            f"[aggregate_power] No bench window in {bench_result} — skipping power aggregation",
            file=sys.stderr,
        )
        return 0
    start, end, duration, total_output = window

    result = aggregate_power(csv_path, start, end)
    if result is None:
        print(
            f"[aggregate_power] No usable power samples in {csv_path} for "
            f"window [{start}, {end}] — skipping",
            file=sys.stderr,
        )
        return 0
    avg_power_w, num_gpus = result

    # Joules consumed by the system during the bench window / output tokens.
    joules_per_output_token = (avg_power_w * num_gpus * duration) / total_output

    if not agg_result.is_file():
        print(
            f"[aggregate_power] Agg result {agg_result} missing — cannot patch",
            file=sys.stderr,
        )
        return 0

    try:
        patch_agg_result(agg_result, avg_power_w, joules_per_output_token)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[aggregate_power] Failed to patch {agg_result}: {exc}", file=sys.stderr)
        return 0

    print(
        f"[aggregate_power] avg_power_w={avg_power_w:.2f} (per GPU, n={num_gpus}) "
        f"joules_per_output_token={joules_per_output_token:.4f} "
        f"duration={duration:.1f}s output_tokens={total_output} -> {agg_result}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/workspace/gpu_metrics.csv"),
        help="Path to gpu_metrics.csv from start_gpu_monitor (default: /workspace/gpu_metrics.csv)",
    )
    parser.add_argument(
        "--bench-result",
        type=Path,
        required=True,
        help="Path to the raw benchmark_serving.py result JSON (provides bench window + token counts)",
    )
    parser.add_argument(
        "--agg-result",
        type=Path,
        required=True,
        help="Path to the agg_<run>.json output of process_result.py (will be patched in place)",
    )
    args = parser.parse_args()
    return run(args.csv, args.bench_result, args.agg_result)


if __name__ == "__main__":
    sys.exit(main())
