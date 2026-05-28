"""Aggregate measured GPU telemetry (power, temp, utilization, memory) from a
vendor SMI CSV into the agg result JSON.

Reads a GPU-metrics CSV produced by `start_gpu_monitor` (nvidia-smi or amd-smi)
or by srt-slurm's per-node perfmon (multinode), filters samples to the benchmark
load window using start/end Unix timestamps written by benchmark_serving.py, and
patches the aggregated result JSON with cluster-wide and per-worker telemetry
consumed by InferenceX-app's ETL.

Cluster-wide fields (always written when any power data exists):
  - avg_power_w:               mean per-GPU power draw (W) during the load window
  - joules_per_output_token:   energy / total_output_tokens. CLUSTER-WIDE
                               (total_system_energy) on single-node / non-disagg;
                               OVERRIDDEN to per-stage decode_energy for disagg
                               (see below).
  - joules_per_total_token:    total_system_energy / (input + output) tokens
                               (cluster-wide; always — overall efficiency number)
  - avg_temp_c:                mean per-GPU temperature (Celsius), when the
                               CSV exposes a temperature column
  - peak_temp_c:               max instantaneous per-GPU temperature in window
  - avg_util_pct:              mean per-GPU GPU-utilization percent
  - avg_mem_used_mb:           mean per-GPU memory used (MiB/MB)

For disaggregated multinode runs (DISAGG=true) where filenames carry the perfmon
role/index encoding AND both prefill+decode workers are present, the per-token
energy metrics use PER-STAGE attribution — each token type is divided by only the
GPUs of the stage that produces it (the standard disagg-serving convention):

  - joules_per_input_token:    prefill_energy / total_input_tokens — input tokens
                               are processed by the prefill GPUs only.
  - joules_per_output_token:   decode_energy / total_output_tokens — output tokens
                               are produced by the decode GPUs only. (For
                               single-node / non-disagg this stays the cluster-wide
                               total_system_energy / output_tokens.)
  - prefill_avg_power_w:       per-GPU mean power across prefill workers
  - decode_avg_power_w:        per-GPU mean power across decode workers

Per-worker breakdown (multinode only — single-node has no role concept), emitted
under the `workers` key to match InferenceX-app's BenchmarkRow.workers shape:
  - workers: list of {role, worker_idx, hosts[], num_gpus, avg_power_w,
                       avg_temp_c?, peak_temp_c?, avg_util_pct?, avg_mem_used_mb?}
             where role is "prefill", "decode", "agg", or "frontend".

Both multinode paths encode the worker role and index in the perfmon CSV
filename: `perf_samples_<role>_w<worker_idx>_<host>.csv` — NVIDIA via the
srt-slurm fork's benchmark_stage._start_perf_monitor, AMD via start_perf_monitor
in benchmarks/benchmark_lib.sh (each SGLang/vLLM disagg node starts its own
amd-smi monitor). Filenames that don't match this pattern (e.g. single-node
`gpu_metrics.csv`) fall back to a single cluster-wide bucket.

Multinode: accepts multiple CSV paths (one per worker node). GPU indices are
namespaced by source CSV stem to avoid the same-index collision across nodes —
e.g. 8 nodes each reporting indices 0..3 would otherwise be miscounted as 4
total GPUs instead of 32.

Vendor schema detection is regex-based:
  - Power: timestamp + column whose name contains "power" (excluding
    "limit"/"cap"/"max"/"min"). NVIDIA: "power.draw [W]". AMD: "socket_power".
    srt-slurm: "power_w".
  - Temperature: column name contains "temp". NVIDIA: "temperature.gpu". AMD:
    "temperature". srt-slurm: "temp_c". Unit: Celsius.
  - Utilization: column name starts with "utilization" or contains "util".
    NVIDIA: "utilization.gpu". srt-slurm: "util_pct". Unit: percent.
  - Memory: column name contains "mem" but not "total"/"clock"/"util" — so
    "memory.total", "clocks.current.memory" (a frequency), and
    "utilization.memory" (a percent) are all rejected; only memory *used* is
    picked. NVIDIA: "memory.used [MiB]". srt-slurm: "mem_used_mb". Unit: MiB/MB.

Power is required for aggregation to fire; the other metrics degrade gracefully
when their columns are absent (those fields are simply omitted from the output).

This script is best-effort. Missing or malformed CSV exits 0 without patching
so a monitoring hiccup never breaks the benchmark upload.
"""

from __future__ import annotations

import argparse
import csv
import glob as glob_module
import json
import re
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


_POWER_COL_RE = re.compile(r"power", re.IGNORECASE)
_POWER_EXCLUDE_RE = re.compile(r"limit|cap|max|min", re.IGNORECASE)
_TEMP_COL_RE = re.compile(r"temp", re.IGNORECASE)
_UTIL_COL_RE = re.compile(r"^utilization|util", re.IGNORECASE)
_MEM_COL_RE = re.compile(r"mem", re.IGNORECASE)
# Exclude "total" (memory.total), "clock" (clocks.current.memory — a frequency,
# not memory used), and "util" (utilization.memory — a percent). nvidia-smi's
# query emits clocks.current.memory BEFORE any used-memory column, so without
# these excludes _MEM_COL_RE would grab the memory *clock* (~2500 MHz) as
# avg_mem_used_mb.
_MEM_EXCLUDE_RE = re.compile(r"total|clock|util", re.IGNORECASE)
_TIMESTAMP_COL_RE = re.compile(r"time", re.IGNORECASE)
_GPU_INDEX_COL_RE = re.compile(r"^(index|gpu|gpu_id|gpu_index|card|device)$", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# srt-slurm perfmon filename: perf_samples_<role>_w<worker_idx>_<host>.csv
# Roles: prefill, decode, agg, frontend (see srt-slurm benchmark_stage._label).
# Host may contain hyphens and digits; greedy `.+` is fine because the `_w<idx>_`
# anchor is unambiguous.
_PERFMON_LABEL_RE = re.compile(
    r"^perf_samples_(?P<role>prefill|decode|agg|frontend)_w(?P<idx>\d+)_(?P<host>.+)$"
)

# Metric names recognized in the multi-metric row dicts. Power is special-cased
# as required; others are best-effort.
_METRICS_AVG = ("power", "temp", "util", "mem")  # mean across samples
_METRICS_MAX = ("temp",)  # additionally compute peak (max raw)


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


def _parse_numeric_cell(value: str) -> float | None:
    """Extract the first numeric value from a cell.

    Vendors decorate values with units ("412.34 W", "65 C", "85 %", "1024 MiB")
    or report "[N/A]" when a sensor is unavailable. We strip and pull the first
    signed-decimal token; returns None for empty / NA / non-numeric cells.
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


# Back-compat shim — some external callers may have imported _parse_power.
_parse_power = _parse_numeric_cell


def _detect_columns(header: list[str]) -> tuple[str | None, str | None, str | None]:
    """Return (timestamp_col, power_col, gpu_index_col) from a CSV header.

    Power column: contains "power" and not "limit"/"cap"/"max"/"min".
    Timestamp column: contains "time".
    GPU index column: optional — used to count distinct GPUs per sample.

    Kept for back-compat with tests that imported _detect_columns directly;
    new code uses _detect_all_columns to also pick up temp/util/mem.
    """
    timestamp_col = next((c for c in header if _TIMESTAMP_COL_RE.search(c)), None)
    power_col = next(
        (c for c in header if _POWER_COL_RE.search(c) and not _POWER_EXCLUDE_RE.search(c)),
        None,
    )
    gpu_col = next((c for c in header if _GPU_INDEX_COL_RE.match(c.strip())), None)
    return timestamp_col, power_col, gpu_col


def _detect_all_columns(header: list[str]) -> dict[str, str | None]:
    """Return a mapping of role -> column name for every metric we know about.

    Roles: timestamp, gpu, power, temp, util, mem. Missing roles map to None.

    The detection is greedy + first-match: with a vendor like NVIDIA whose
    header lists `utilization.gpu` followed by `utilization.memory`, the
    util slot picks the first; that's fine — we only need ONE util column and
    `utilization.gpu` is the canonical one. Memory excludes "total" so
    `memory.used` wins over `memory.total`.
    """
    timestamp_col = next((c for c in header if _TIMESTAMP_COL_RE.search(c)), None)
    power_col = next(
        (c for c in header if _POWER_COL_RE.search(c) and not _POWER_EXCLUDE_RE.search(c)),
        None,
    )
    temp_col = next((c for c in header if _TEMP_COL_RE.search(c)), None)
    util_col = next((c for c in header if _UTIL_COL_RE.search(c)), None)
    mem_col = next(
        (c for c in header if _MEM_COL_RE.search(c) and not _MEM_EXCLUDE_RE.search(c)),
        None,
    )
    gpu_col = next((c for c in header if _GPU_INDEX_COL_RE.match(c.strip())), None)
    return {
        "timestamp": timestamp_col,
        "gpu": gpu_col,
        "power": power_col,
        "temp": temp_col,
        "util": util_col,
        "mem": mem_col,
    }


def _parse_perfmon_label(path: Path) -> tuple[str, int, str] | None:
    """Extract (role, worker_idx, host) from a srt-slurm perfmon CSV filename.

    Returns None for filenames not matching the perfmon pattern (e.g.
    single-node `gpu_metrics.csv`). Used to group node-level CSVs by the
    worker(s) running on each node.
    """
    m = _PERFMON_LABEL_RE.match(path.stem)
    if not m:
        return None
    return m.group("role"), int(m.group("idx")), m.group("host")


def _read_samples(
    path: Path, start_unix: float, end_unix: float
) -> tuple[list[tuple[float, str | None, dict[str, float]]], bool] | None:
    """Read one CSV → list of (timestamp_bucket, gpu_id, {metric: value}) in window.

    Returns (rows, saw_gpu_col) on success, None if the file is unreadable /
    missing the required power column. Empty rows list is valid (file readable
    but no samples landed in the window).

    Each row's metric dict carries whichever of power/temp/util/mem the CSV
    exposed (power is always present — rows lacking it are skipped). Missing
    metric columns simply don't appear in the dict; callers gracefully degrade.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            header = [c.strip() for c in (reader.fieldnames or [])]
            reader.fieldnames = header
            cols = _detect_all_columns(header)
            timestamp_col = cols["timestamp"]
            power_col = cols["power"]
            if not timestamp_col or not power_col:
                return None
            gpu_col = cols["gpu"]
            # Map metric name -> CSV column. Power is required (we just
            # checked); temp/util/mem are optional.
            metric_cols: dict[str, str] = {"power": power_col}
            for metric in ("temp", "util", "mem"):
                col = cols[metric]
                if col is not None:
                    metric_cols[metric] = col
            rows: list[tuple[float, str | None, dict[str, float]]] = []
            for row in reader:
                ts = _parse_timestamp((row.get(timestamp_col) or "").strip())
                if ts is None:
                    continue
                if ts < start_unix or ts > end_unix:
                    continue
                # Power must parse; rows with [N/A] or empty power are useless
                # for aggregation (same behavior as before the multi-metric
                # extension).
                pw = _parse_numeric_cell((row.get(power_col) or "").strip())
                if pw is None:
                    continue
                values: dict[str, float] = {"power": pw}
                for metric, col in metric_cols.items():
                    if metric == "power":
                        continue
                    v = _parse_numeric_cell((row.get(col) or "").strip())
                    if v is not None:
                        values[metric] = v
                gpu_id = (row.get(gpu_col) or "").strip() if gpu_col else None
                rows.append((round(ts, 3), gpu_id or None, values))
            return rows, gpu_col is not None
    except (OSError, csv.Error):
        return None


def _aggregate_rows(
    sources: list[tuple[Path, list[tuple[float, str | None, dict[str, float]]], bool]],
    *,
    namespace: bool,
) -> dict | None:
    """Merge rows across CSVs into a metric-dict + num_gpus.

    `sources` is a list of (path, rows, saw_gpu_col) for the CSVs to roll up
    together. Rows are bucketed by ms-rounded timestamp so nodes with sub-ms
    clock drift land in the same bucket. GPU indices are namespaced by the
    source path's stem when `namespace=True` (multi-source case) to keep
    same-local-index across nodes from collapsing.

    Returns a dict with at minimum {"power": float, "num_gpus": int}. Each
    additional metric (temp/util/mem) is included only when at least one
    source emitted it. peak_temp is the global max across the window
    (instantaneous, not per-bucket-mean).
    """
    # Per-bucket totals keyed by metric name. Bucket = ms-rounded timestamp.
    per_sample_total: dict[str, dict[float, float]] = {m: {} for m in _METRICS_AVG}
    per_sample_count: dict[str, dict[float, int]] = {m: {} for m in _METRICS_AVG}
    per_sample_row_count: dict[float, int] = {}  # for no-gpu-col GPU inference
    per_sample_gpus: dict[float, set[str]] = {}
    gpu_keys: set[str] = set()
    saw_gpu_col_any = False
    saw_metric: dict[str, bool] = {m: False for m in _METRICS_AVG}
    peak_per_metric: dict[str, float] = {}

    for path, rows, saw_gpu_col in sources:
        if saw_gpu_col:
            saw_gpu_col_any = True
        for bucket, gpu_id, values in rows:
            per_sample_row_count[bucket] = per_sample_row_count.get(bucket, 0) + 1
            for metric, v in values.items():
                if metric not in per_sample_total:
                    continue
                per_sample_total[metric][bucket] = (
                    per_sample_total[metric].get(bucket, 0.0) + v
                )
                per_sample_count[metric][bucket] = (
                    per_sample_count[metric].get(bucket, 0) + 1
                )
                saw_metric[metric] = True
                if metric in _METRICS_MAX:
                    cur = peak_per_metric.get(metric)
                    peak_per_metric[metric] = v if cur is None else max(cur, v)
            if gpu_id is not None:
                ns_id = f"{path.stem}:{gpu_id}" if namespace else gpu_id
                per_sample_gpus.setdefault(bucket, set()).add(ns_id)
                gpu_keys.add(ns_id)

    if not per_sample_total["power"]:
        return None

    # GPU count:
    # - If any path exposed a GPU column, trust distinct (namespaced) GPU IDs.
    # - Otherwise, infer from row count (one row per GPU per sample, summed
    #   across all paths' rows that fell into the same timestamp bucket).
    if saw_gpu_col_any and gpu_keys:
        num_gpus = len(gpu_keys)
    else:
        num_gpus = max(per_sample_row_count.values())

    def _avg_per_gpu(metric: str) -> float | None:
        if not saw_metric.get(metric):
            return None
        totals = per_sample_total[metric]
        if not totals:
            return None
        if saw_gpu_col_any and gpu_keys:
            # bucket mean = sum / distinct GPU count in that bucket
            per_sample_mean = [
                total / max(len(per_sample_gpus.get(ts, ())), 1)
                for ts, total in totals.items()
            ]
        else:
            # bucket mean = sum / row count in that bucket (= GPU count when
            # one row per GPU per sample, the universal vendor convention)
            per_sample_mean = [
                total / per_sample_count[metric][ts] for ts, total in totals.items()
            ]
        return mean(per_sample_mean) if per_sample_mean else None

    result: dict = {"num_gpus": num_gpus, "power": _avg_per_gpu("power")}
    for metric in ("temp", "util", "mem"):
        avg = _avg_per_gpu(metric)
        if avg is not None:
            result[metric] = avg
    # Peak (max raw value, not per-bucket-mean): meaningful for temperature
    # where the worst-case GPU's hottest sample is the thermal-headroom signal.
    if "temp" in peak_per_metric:
        result["peak_temp"] = peak_per_metric["temp"]
    return result


def aggregate_power(
    csv_path: Path | Iterable[Path],
    start_unix: float,
    end_unix: float,
) -> tuple[float, int] | None:
    """Return (per_gpu_avg_power_w, num_gpus) for samples in [start, end].

    Backward-compatible wrapper around aggregate_metrics that returns just the
    legacy (avg_power_w, num_gpus) tuple for callers (and tests) that don't
    need temperature/util/memory.
    """
    res = aggregate_metrics(csv_path, start_unix, end_unix)
    if res is None:
        return None
    return res["power"], res["num_gpus"]


def aggregate_metrics(
    csv_path: Path | Iterable[Path],
    start_unix: float,
    end_unix: float,
) -> dict | None:
    """Return a dict of cluster-wide per-GPU metrics for samples in [start, end].

    Accepts either a single Path (single-node case) or an iterable of Paths
    (multinode case: one CSV per worker node, all written by srt-slurm's
    perfmon). For multi-path inputs, GPU indices are namespaced by source
    CSV stem so the distinct-id count reflects the true total — each node
    independently reports indices 0..N, and without namespacing the union
    would collapse to a single node's worth.

    Returns None if no CSVs are usable, none have a detectable power column,
    or no rows fall in the window across all paths.

    Result keys: num_gpus, power (always when not None); temp, util, mem,
    peak_temp (only when the corresponding column existed in at least one CSV).
    """
    paths = [csv_path] if isinstance(csv_path, Path) else list(csv_path)
    if not paths or end_unix <= start_unix:
        return None

    sources: list[tuple[Path, list[tuple[float, str | None, dict[str, float]]], bool]] = []
    for path in paths:
        read = _read_samples(path, start_unix, end_unix)
        if read is None:
            continue
        rows, saw_gpu_col = read
        sources.append((path, rows, saw_gpu_col))
    if not sources:
        return None

    return _aggregate_rows(sources, namespace=len(paths) > 1)


def aggregate_power_by_worker(
    csv_paths: Iterable[Path],
    start_unix: float,
    end_unix: float,
) -> list[dict] | None:
    """Group CSVs by (role, worker_idx) and return per-worker telemetry rollups.

    Each entry: {role, worker_idx, hosts: sorted list, num_gpus, avg_power_w,
                  avg_temp_c?, peak_temp_c?, avg_util_pct?, avg_mem_used_mb?}.
    The optional fields appear only when the CSVs for that worker carried
    temperature / utilization / memory columns.

    Returns None if no CSVs have parseable filenames OR no labeled CSV yields
    usable samples. Unlabeled CSVs in the input are silently skipped — they
    can't be attributed to a worker.

    Hosts are listed because a single worker can span multiple nodes (e.g.
    a 16-GPU decode worker over 4 nodes, all labeled decode_w0_<host>).
    Multiple node-CSVs sharing the same (role, worker_idx) collapse into one
    worker entry whose num_gpus is the sum across nodes.
    """
    paths = list(csv_paths)
    if not paths or end_unix <= start_unix:
        return None

    # Group paths by (role, worker_idx); discard unlabeled.
    by_worker: dict[tuple[str, int], list[Path]] = {}
    hosts_by_worker: dict[tuple[str, int], set[str]] = {}
    for p in paths:
        label = _parse_perfmon_label(p)
        if label is None:
            continue
        role, worker_idx, host = label
        key = (role, worker_idx)
        by_worker.setdefault(key, []).append(p)
        hosts_by_worker.setdefault(key, set()).add(host)
    if not by_worker:
        return None

    out: list[dict] = []
    for (role, worker_idx), worker_paths in by_worker.items():
        sources: list[tuple[Path, list[tuple[float, str | None, dict[str, float]]], bool]] = []
        for path in worker_paths:
            read = _read_samples(path, start_unix, end_unix)
            if read is None:
                continue
            rows, saw_gpu_col = read
            sources.append((path, rows, saw_gpu_col))
        if not sources:
            continue
        # Namespace across paths within a worker too — a 16-GPU decode worker
        # spans 4 nodes, each reporting local indices 0..3.
        result = _aggregate_rows(sources, namespace=len(sources) > 1)
        if result is None:
            continue
        entry: dict = {
            "role": role,
            "worker_idx": worker_idx,
            "hosts": sorted(hosts_by_worker[(role, worker_idx)]),
            "num_gpus": result["num_gpus"],
            "avg_power_w": round(result["power"], 3),
        }
        if "temp" in result:
            entry["avg_temp_c"] = round(result["temp"], 3)
        if "peak_temp" in result:
            entry["peak_temp_c"] = round(result["peak_temp"], 3)
        if "util" in result:
            entry["avg_util_pct"] = round(result["util"], 3)
        if "mem" in result:
            entry["avg_mem_used_mb"] = round(result["mem"], 3)
        out.append(entry)
    if not out:
        return None
    # Stable order: role (prefill < decode < agg < frontend), then worker_idx.
    role_order = {"prefill": 0, "decode": 1, "agg": 2, "frontend": 3}
    out.sort(key=lambda w: (role_order.get(w["role"], 99), w["worker_idx"]))
    return out


def _load_bench_window(
    bench_result_path: Path,
) -> tuple[float, float, float, int, int] | None:
    """Read (start_unix, end_unix, duration_s, total_output_tokens, total_input_tokens)
    from the raw bench JSON. Returns None if a window cannot be resolved.

    Window resolution order, tried in turn:
      1. benchmark_start_time_unix + benchmark_end_time_unix (our benchmark_serving.py
         writes both — single-node, brackets the actual load window exactly).
      2. date + duration (srt-slurm sa-bench writes "YYYYMMDD-HHMMSS" UTC as the
         result write time — multinode; treat as bench end and subtract duration
         for start. Overshoots by post-bench JSON serialization, typically <5s).
      3. file mtime + duration (last resort if `date` is absent or unparseable —
         same end-of-bench proxy as #2 via the result file's mtime).

    total_input_tokens defaults to 0 if absent (older bench JSONs may not have it);
    this only degrades joules_per_total_token to equal joules_per_output_token in
    that case, never breaks the rest of the aggregation.
    """
    try:
        bench = json.loads(bench_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    duration = bench.get("duration")
    total_output = bench.get("total_output_tokens")
    total_input = bench.get("total_input_tokens", 0)
    if not isinstance(duration, (int, float)):
        return None
    if not isinstance(total_output, int) or total_output <= 0:
        return None
    if not isinstance(total_input, int) or total_input < 0:
        total_input = 0

    # Tier 1: explicit Unix timestamps (single-node bench_serving.py).
    start = bench.get("benchmark_start_time_unix")
    end = bench.get("benchmark_end_time_unix")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return float(start), float(end), float(duration), int(total_output), int(total_input)

    # Tier 2: parse `date` field (srt-slurm sa-bench multinode). On observed
    # runs the string matches file mtime to the second, confirming it's the
    # JSON write time.
    date_str = bench.get("date")
    if isinstance(date_str, str):
        try:
            end_dt = datetime.strptime(date_str, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
            end_unix = end_dt.timestamp()
            return (
                float(end_unix - duration),
                float(end_unix),
                float(duration),
                int(total_output),
                int(total_input),
            )
        except ValueError:
            pass

    # Tier 3: file mtime as last-resort bench-end proxy.
    try:
        end_unix = bench_result_path.stat().st_mtime
    except OSError:
        return None
    return (
        float(end_unix - duration),
        float(end_unix),
        float(duration),
        int(total_output),
        int(total_input),
    )


def patch_agg_result(
    agg_path: Path,
    avg_power_w: float,
    joules_per_output_token: float,
    joules_per_total_token: float,
    joules_per_input_token: float | None = None,
    prefill_avg_power_w: float | None = None,
    decode_avg_power_w: float | None = None,
    avg_temp_c: float | None = None,
    peak_temp_c: float | None = None,
    avg_util_pct: float | None = None,
    avg_mem_used_mb: float | None = None,
    workers: list[dict] | None = None,
) -> None:
    """Read the agg JSON, add the telemetry keys, and write it back atomically.

    All optional fields (anything except avg_power_w / joules_per_output_token /
    joules_per_total_token) are omitted from the JSON when None — keeps the
    pre-disagg / single-node agg JSONs from gaining meaningless null fields, and
    keeps non-power-instrumented runs (e.g. no temp sensor) from emitting nulls.
    """
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    data["avg_power_w"] = round(avg_power_w, 3)
    data["joules_per_output_token"] = round(joules_per_output_token, 6)
    data["joules_per_total_token"] = round(joules_per_total_token, 6)
    if joules_per_input_token is not None:
        data["joules_per_input_token"] = round(joules_per_input_token, 6)
    if prefill_avg_power_w is not None:
        data["prefill_avg_power_w"] = round(prefill_avg_power_w, 3)
    if decode_avg_power_w is not None:
        data["decode_avg_power_w"] = round(decode_avg_power_w, 3)
    if avg_temp_c is not None:
        data["avg_temp_c"] = round(avg_temp_c, 3)
    if peak_temp_c is not None:
        data["peak_temp_c"] = round(peak_temp_c, 3)
    if avg_util_pct is not None:
        data["avg_util_pct"] = round(avg_util_pct, 3)
    if avg_mem_used_mb is not None:
        data["avg_mem_used_mb"] = round(avg_mem_used_mb, 3)
    if workers is not None:
        data["workers"] = workers
    tmp_path = agg_path.with_suffix(agg_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(agg_path)


def _disagg_stage_rollup(
    workers: list[dict], duration: float
) -> dict | None:
    """Roll up per-worker entries into per-stage energy + per-GPU mean power.

    Returns a dict with keys:
      - prefill_energy_j, decode_energy_j: sum of (avg_power_w * num_gpus *
        duration) across workers in each role
      - prefill_avg_power_w, decode_avg_power_w: per-GPU mean power weighted
        by num_gpus (matches the cluster avg_power_w semantics, but scoped to
        each role)

    Returns None if either stage is absent — without both stages we can't do
    per-stage attribution and the caller should fall back to total-energy math.
    """
    prefill_energy = 0.0
    decode_energy = 0.0
    prefill_gpus = 0
    decode_gpus = 0
    prefill_pw_x_gpus = 0.0
    decode_pw_x_gpus = 0.0
    has_prefill = False
    has_decode = False
    for w in workers:
        e = w["avg_power_w"] * w["num_gpus"] * duration
        if w["role"] == "prefill":
            prefill_energy += e
            prefill_gpus += w["num_gpus"]
            prefill_pw_x_gpus += w["avg_power_w"] * w["num_gpus"]
            has_prefill = True
        elif w["role"] == "decode":
            decode_energy += e
            decode_gpus += w["num_gpus"]
            decode_pw_x_gpus += w["avg_power_w"] * w["num_gpus"]
            has_decode = True
        # "frontend" / "agg" / unknown roles deliberately excluded — they
        # don't belong to either stage's per-token cost or per-stage power.
    if not (has_prefill and has_decode):
        return None
    return {
        "prefill_energy_j": prefill_energy,
        "decode_energy_j": decode_energy,
        "prefill_avg_power_w": prefill_pw_x_gpus / prefill_gpus if prefill_gpus else None,
        "decode_avg_power_w": decode_pw_x_gpus / decode_gpus if decode_gpus else None,
    }


def run(
    csv_path: Path | Iterable[Path],
    bench_result: Path,
    agg_result: Path,
    *,
    disagg: bool = False,
) -> int:
    window = _load_bench_window(bench_result)
    if window is None:
        print(
            f"[aggregate_power] No bench window in {bench_result} — skipping power aggregation",
            file=sys.stderr,
        )
        return 0
    start, end, duration, total_output, total_input = window

    paths = [csv_path] if isinstance(csv_path, Path) else list(csv_path)
    cluster = aggregate_metrics(paths, start, end)
    if cluster is None:
        label = str(paths[0]) if len(paths) == 1 else f"{len(paths)} CSVs"
        print(
            f"[aggregate_power] No usable power samples in {label} for "
            f"window [{start}, {end}] — skipping",
            file=sys.stderr,
        )
        return 0
    avg_power_w = cluster["power"]
    num_gpus = cluster["num_gpus"]
    avg_temp_c = cluster.get("temp")
    peak_temp_c = cluster.get("peak_temp")
    avg_util_pct = cluster.get("util")
    avg_mem_used_mb = cluster.get("mem")

    # Per-worker rollup is best-effort: only emitted when CSV filenames carry
    # the perfmon role/index encoding. Single-node `gpu_metrics.csv` won't
    # parse, so aggregate_power_by_worker returns None and the field is omitted.
    workers = aggregate_power_by_worker(paths, start, end)

    # Per-token energy attribution.
    #   - joules_per_total_token stays CLUSTER-WIDE on every topology
    #     (total_system_energy / all tokens) — the overall efficiency number.
    #   - For disagg with BOTH stages present, joules_per_output_token and
    #     joules_per_input_token use PER-STAGE energy: output tokens are produced
    #     by the decode GPUs (decode_energy / output), input tokens by the
    #     prefill GPUs (prefill_energy / input). This is the standard per-stage
    #     attribution requested for disagg serving.
    #   - Single-node / non-disagg / single-stage fall back to the cluster-wide
    #     output ratio so the field is always populated.
    total_system_energy_j = avg_power_w * num_gpus * duration
    total_tokens = total_output + total_input
    joules_per_output_token = total_system_energy_j / total_output  # cluster fallback
    joules_per_total_token = (
        total_system_energy_j / total_tokens if total_tokens > 0 else joules_per_output_token
    )

    joules_per_input_token: float | None = None
    prefill_avg_power_w: float | None = None
    decode_avg_power_w: float | None = None

    if disagg and workers is not None:
        stage = _disagg_stage_rollup(workers, duration)
        if stage is not None:
            # Per-stage attribution: decode GPUs produce output tokens, prefill
            # GPUs process input tokens. Strictly more accurate than total-energy
            # ratios when prefill/decode have different per-GPU power profiles
            # (typical: prefill is compute-bound and draws more than memory-bound
            # decode). joules_per_output_token is OVERRIDDEN to the decode-only
            # value here (symmetric with the prefill-only joules_per_input_token).
            prefill_avg_power_w = stage["prefill_avg_power_w"]
            decode_avg_power_w = stage["decode_avg_power_w"]
            joules_per_output_token = stage["decode_energy_j"] / total_output
            joules_per_input_token = (
                stage["prefill_energy_j"] / total_input if total_input > 0 else None
            )

    if not agg_result.is_file():
        print(
            f"[aggregate_power] Agg result {agg_result} missing — cannot patch",
            file=sys.stderr,
        )
        return 0

    try:
        patch_agg_result(
            agg_result,
            avg_power_w,
            joules_per_output_token,
            joules_per_total_token,
            joules_per_input_token=joules_per_input_token,
            prefill_avg_power_w=prefill_avg_power_w,
            decode_avg_power_w=decode_avg_power_w,
            avg_temp_c=avg_temp_c,
            peak_temp_c=peak_temp_c,
            avg_util_pct=avg_util_pct,
            avg_mem_used_mb=avg_mem_used_mb,
            workers=workers,
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[aggregate_power] Failed to patch {agg_result}: {exc}", file=sys.stderr)
        return 0

    worker_summary = (
        f"workers={len(workers)}" if workers else "workers=cluster-only"
    )
    jpit_summary = (
        f"joules_per_input_token={joules_per_input_token:.4f} "
        if joules_per_input_token is not None
        else ""
    )
    print(
        f"[aggregate_power] avg_power_w={avg_power_w:.2f} (per GPU, n={num_gpus}) "
        f"joules_per_output_token={joules_per_output_token:.4f} "
        f"{jpit_summary}"
        f"joules_per_total_token={joules_per_total_token:.4f} "
        f"duration={duration:.1f}s output_tokens={total_output} input_tokens={total_input} "
        f"{worker_summary} -> {agg_result}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Single gpu_metrics.csv from start_gpu_monitor (single-node). "
        "Falls back to /workspace/gpu_metrics.csv when neither --csv nor --csv-glob is set.",
    )
    source.add_argument(
        "--csv-glob",
        type=str,
        default=None,
        help="Shell glob expanding to per-node perf_samples_*.csv files (multinode, "
        "written by srt-slurm's perfmon). GPU indices are namespaced by source CSV stem.",
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
    parser.add_argument(
        "--disagg",
        action="store_true",
        help="Treat as disaggregated inference: emit prefill_avg_power_w / "
        "decode_avg_power_w, and use PER-STAGE energy attribution for "
        "joules_per_input_token (prefill energy / input tokens) and "
        "joules_per_output_token (decode energy / output tokens). "
        "joules_per_total_token stays cluster-wide. Requires CSV filenames to "
        "carry the perfmon role/index encoding.",
    )
    args = parser.parse_args()

    if args.csv_glob:
        paths = sorted(Path(p) for p in glob_module.glob(args.csv_glob))
        if not paths:
            print(
                f"[aggregate_power] No CSVs matched glob {args.csv_glob!r} — skipping",
                file=sys.stderr,
            )
            return 0
        return run(paths, args.bench_result, args.agg_result, disagg=args.disagg)
    return run(
        args.csv or Path("/workspace/gpu_metrics.csv"),
        args.bench_result,
        args.agg_result,
        disagg=args.disagg,
    )


if __name__ == "__main__":
    sys.exit(main())
