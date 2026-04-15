#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "datasets/isb1/results/isb1_results.db"
TABLE_NAME = "benchmark_runs"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  id INTEGER PRIMARY KEY,
  run_id TEXT,
  timestamp TEXT,
  gpu_type TEXT,
  model TEXT,
  engine TEXT,
  context_band TEXT,
  workload_type TEXT,
  max_model_len INTEGER,
  tp INTEGER,
  vllm_cpu_offload_gb REAL,
  vllm_swap_space_gb REAL,
  sglang_mem_fraction REAL,
  sglang_chunked_prefill INTEGER,
  ttft_p50_ms REAL,
  ttft_p99_ms REAL,
  tpot_p50_ms REAL,
  tpot_p99_ms REAL,
  throughput_tok_s REAL,
  total_sessions INTEGER,
  completed_sessions INTEGER,
  total_turns INTEGER,
  completed_turns INTEGER,
  preemption_count INTEGER,
  gpu_mem_peak_gb REAL,
  gpu_mem_avg_gb REAL,
  gpu_util_avg_pct REAL,
  kv_cache_usage_pct REAL,
  server_startup_s REAL,
  benchmark_duration_s REAL,
  campaign_class TEXT,
  trace_source TEXT,
  total_actual_input_tokens INTEGER,
  max_actual_context_len INTEGER,
  depth_coverage_ratio REAL,
  depth_coverage_class TEXT,
  producer_estimated_kv_bytes_peak INTEGER,
  producer_expected_offload_mode TEXT,
  offload_mode_match INTEGER,
  offload_mode TEXT,
  kv_cache_dtype TEXT,
  disable_prefix_caching INTEGER,
  cpu_cache_usage_peak_pct REAL,
  raw_result_json TEXT,
  status TEXT,
  error_message TEXT
)
"""

INSERT_COLUMNS = [
    "run_id",
    "timestamp",
    "gpu_type",
    "model",
    "engine",
    "context_band",
    "workload_type",
    "max_model_len",
    "tp",
    "vllm_cpu_offload_gb",
    "vllm_swap_space_gb",
    "sglang_mem_fraction",
    "sglang_chunked_prefill",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "tpot_p99_ms",
    "throughput_tok_s",
    "total_sessions",
    "completed_sessions",
    "total_turns",
    "completed_turns",
    "preemption_count",
    "gpu_mem_peak_gb",
    "gpu_mem_avg_gb",
    "gpu_util_avg_pct",
    "kv_cache_usage_pct",
    "server_startup_s",
    "benchmark_duration_s",
    "campaign_class",
    "trace_source",
    "total_actual_input_tokens",
    "max_actual_context_len",
    "depth_coverage_ratio",
    "depth_coverage_class",
    "producer_estimated_kv_bytes_peak",
    "producer_expected_offload_mode",
    "offload_mode_match",
    "offload_mode",
    "kv_cache_dtype",
    "disable_prefix_caching",
    "cpu_cache_usage_peak_pct",
    "raw_result_json",
    "status",
    "error_message",
]

GROUPABLE_COLUMNS = {
    "gpu_type",
    "model",
    "engine",
    "context_band",
    "workload_type",
    "status",
    "tp",
    "max_model_len",
    "depth_coverage_class",
    "offload_mode",
    "campaign_class",
    "trace_source",
}

DEFAULT_QUERY_COLUMNS = [
    "timestamp",
    "gpu_type",
    "model",
    "engine",
    "context_band",
    "workload_type",
    "status",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "throughput_tok_s",
    "gpu_mem_peak_gb",
    "gpu_util_avg_pct",
    "preemption_count",
    "depth_coverage_ratio",
    "max_actual_context_len",
    "depth_coverage_class",
    "run_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Store and analyze ISB1 benchmark runs in SQLite.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Read a processed ISB1 JSON file and insert a benchmark run.")
    ingest.add_argument("json_file", help="Path to utils/process_result_isb1.py output JSON.")
    ingest.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite DB path.")
    ingest.add_argument("--gpu-type", required=True, choices=["h100", "h200", "b200"])
    ingest.add_argument("--model", required=True, choices=["qwen3.5", "gptoss", "dsr1"])
    ingest.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    ingest.add_argument("--context-band", required=True, choices=["8k", "32k", "64k", "131k", "500k", "1m"])
    ingest.add_argument("--workload-type", choices=["chat", "code"], help="Workload type (chat or code)")
    ingest.add_argument("--run-id", help="Optional run UUID. Generated if omitted.")
    ingest.add_argument("--timestamp", help="Optional ISO-8601 timestamp. Uses current UTC time if omitted.")
    ingest.add_argument("--max-model-len", type=int)
    ingest.add_argument("--tp", type=int)
    ingest.add_argument("--vllm-cpu-offload-gb", type=float)
    ingest.add_argument("--vllm-swap-space-gb", type=float)
    ingest.add_argument("--sglang-mem-fraction", type=float)
    ingest.add_argument("--sglang-chunked-prefill", type=int)
    ingest.add_argument("--ttft-p50-ms", type=float)
    ingest.add_argument("--ttft-p99-ms", type=float)
    ingest.add_argument("--tpot-p50-ms", type=float)
    ingest.add_argument("--tpot-p99-ms", type=float)
    ingest.add_argument("--throughput-tok-s", type=float)
    ingest.add_argument("--total-sessions", type=int)
    ingest.add_argument("--completed-sessions", type=int)
    ingest.add_argument("--total-turns", type=int)
    ingest.add_argument("--completed-turns", type=int)
    ingest.add_argument("--preemption-count", type=int)
    ingest.add_argument("--gpu-mem-peak-gb", type=float)
    ingest.add_argument("--gpu-mem-avg-gb", type=float)
    ingest.add_argument("--gpu-util-avg-pct", type=float)
    ingest.add_argument("--kv-cache-usage-pct", type=float)
    ingest.add_argument("--server-startup-s", type=float)
    ingest.add_argument("--benchmark-duration-s", type=float)
    ingest.add_argument("--campaign-class")
    ingest.add_argument("--trace-source", choices=["isb1", "kv_cache_tester", "aiperf"])
    ingest.add_argument("--offload-mode", choices=["on", "off", "noprefix", "legacy"])
    ingest.add_argument("--kv-cache-dtype", choices=["auto", "fp8"])
    ingest.add_argument("--disable-prefix-caching", type=int, choices=[0, 1])
    ingest.add_argument("--gpu-profile-csv", help="Optional GPU profile CSV path to stash in raw_result_json metadata.")
    ingest.add_argument("--status", default="success", choices=["success", "failed", "timeout"])
    ingest.add_argument("--error-message")

    query = subparsers.add_parser("query", help="Print runs or an aggregated grouped view.")
    query.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite DB path.")
    query.add_argument("--group-by", help="Comma-separated columns to group by, for example gpu_type,context_band.")

    export_csv = subparsers.add_parser("export-csv", help="Export all benchmark rows to CSV.")
    export_csv.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite DB path.")
    export_csv.add_argument("--output", help="Destination CSV path. Defaults to stdout.")

    summary = subparsers.add_parser("summary", help="Print a concise findings summary.")
    summary.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite DB path.")

    return parser.parse_args()


_MIGRATIONS = [
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN total_actual_input_tokens INTEGER",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN max_actual_context_len INTEGER",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN depth_coverage_ratio REAL",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN depth_coverage_class TEXT",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN producer_estimated_kv_bytes_peak INTEGER",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN producer_expected_offload_mode TEXT",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN offload_mode_match INTEGER",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN offload_mode TEXT",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN kv_cache_dtype TEXT",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN disable_prefix_caching INTEGER",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN cpu_cache_usage_peak_pct REAL",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN workload_type TEXT",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN campaign_class TEXT",
    f"ALTER TABLE {TABLE_NAME} ADD COLUMN trace_source TEXT",
]


def ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA_SQL)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_run_id ON {TABLE_NAME}(run_id)")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_grouping "
        f"ON {TABLE_NAME}(gpu_type, model, engine, context_band, status)"
    )
    # Idempotent migrations for existing databases
    for migration_sql in _MIGRATIONS:
        try:
            conn.execute(migration_sql)
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_db(conn)
    return conn


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def seconds_to_ms(value: Any) -> float | None:
    parsed = to_float(value)
    return None if parsed is None else parsed * 1000.0


def choose(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def load_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected a JSON object in {path}")
    return payload


def derive_total_turns(payload: dict[str, Any], total_sessions: int | None) -> int | None:
    max_turns = to_int(payload.get("max_turns"))
    if max_turns is not None and total_sessions is not None:
        return max_turns * total_sessions
    per_turn_metrics = payload.get("per_turn_metrics") or {}
    if isinstance(per_turn_metrics, dict) and total_sessions is not None:
        return len(per_turn_metrics) * total_sessions
    return None


def derive_completed_turns(payload: dict[str, Any]) -> int | None:
    per_turn_metrics = payload.get("per_turn_metrics") or {}
    if not isinstance(per_turn_metrics, dict):
        return None
    completed = 0
    saw_value = False
    for turn_metrics in per_turn_metrics.values():
        if not isinstance(turn_metrics, dict):
            continue
        value = to_int(turn_metrics.get("completed"))
        if value is None:
            continue
        completed += value
        saw_value = True
    return completed if saw_value else None


def build_raw_payload(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    enriched = dict(payload)
    metadata = {
        "source_json": str(Path(args.json_file).resolve()),
        "db_path": str(Path(args.db_path).resolve()),
    }
    if args.gpu_profile_csv:
        metadata["gpu_profile_csv"] = str(Path(args.gpu_profile_csv).resolve())
    if args.status != "success":
        metadata["status_override"] = args.status
    if args.error_message:
        metadata["error_message"] = args.error_message
    enriched["_isb1_results_db"] = metadata
    return enriched


def insert_run(args: argparse.Namespace) -> None:
    payload = load_payload(args.json_file)
    aggregate = payload.get("aggregate_metrics") or {}
    runtime_overrides = payload.get("runtime_overrides") or {}
    server_metrics_summary = payload.get("server_metrics_summary") or {}

    total_sessions = to_int(choose(args.total_sessions, payload.get("total_sessions"), aggregate.get("total_sessions")))
    completed_sessions = to_int(
        choose(args.completed_sessions, payload.get("completed_sessions"), aggregate.get("completed_sessions"))
    )

    gpu_cache_peak = to_float(server_metrics_summary.get("gpu_cache_usage_peak"))
    if gpu_cache_peak is None:
        gpu_cache_peak = to_float(payload.get("peak_gpu_cache_usage"))

    row = {
        "run_id": args.run_id or str(uuid.uuid4()),
        "timestamp": args.timestamp or utc_now_iso(),
        "gpu_type": args.gpu_type,
        "model": args.model,
        "engine": args.engine,
        "context_band": args.context_band,
        "workload_type": choose(
            getattr(args, 'workload_type', None),
            payload.get("benchmark_surface"),
        ),
        "max_model_len": to_int(choose(args.max_model_len, payload.get("max_model_len"))),
        "tp": to_int(choose(args.tp, payload.get("tp"))),
        "vllm_cpu_offload_gb": to_float(
            choose(
                args.vllm_cpu_offload_gb,
                runtime_overrides.get("vllm_cpu_offload_gb"),
                payload.get("vllm_cpu_offload_gb"),
            )
        ),
        "vllm_swap_space_gb": to_float(
            choose(
                args.vllm_swap_space_gb,
                runtime_overrides.get("vllm_swap_space_gb"),
                payload.get("vllm_swap_space_gb"),
            )
        ),
        "sglang_mem_fraction": to_float(
            choose(
                args.sglang_mem_fraction,
                runtime_overrides.get("sglang_mem_fraction_override"),
                payload.get("sglang_mem_fraction_override"),
            )
        ),
        "sglang_chunked_prefill": to_int(
            choose(
                args.sglang_chunked_prefill,
                runtime_overrides.get("sglang_chunked_prefill_override"),
                payload.get("sglang_chunked_prefill_override"),
            )
        ),
        "ttft_p50_ms": to_float(
            choose(args.ttft_p50_ms, aggregate.get("median_ttft_ms"), seconds_to_ms(payload.get("median_ttft")))
        ),
        "ttft_p99_ms": to_float(
            choose(args.ttft_p99_ms, aggregate.get("p99_ttft_ms"), seconds_to_ms(payload.get("p99_ttft")))
        ),
        "tpot_p50_ms": to_float(
            choose(args.tpot_p50_ms, aggregate.get("median_tpot_ms"), seconds_to_ms(payload.get("median_tpot")))
        ),
        "tpot_p99_ms": to_float(
            choose(args.tpot_p99_ms, aggregate.get("p99_tpot_ms"), seconds_to_ms(payload.get("p99_tpot")))
        ),
        "throughput_tok_s": to_float(
            choose(args.throughput_tok_s, aggregate.get("total_token_throughput_tps"), payload.get("throughput_tok_s"))
        ),
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "total_turns": to_int(choose(args.total_turns, derive_total_turns(payload, total_sessions))),
        "completed_turns": to_int(choose(args.completed_turns, derive_completed_turns(payload))),
        "preemption_count": to_int(choose(args.preemption_count, payload.get("preemption_count"))),
        "gpu_mem_peak_gb": to_float(choose(args.gpu_mem_peak_gb, payload.get("gpu_mem_peak_gb"))),
        "gpu_mem_avg_gb": to_float(choose(args.gpu_mem_avg_gb, payload.get("gpu_mem_avg_gb"))),
        "gpu_util_avg_pct": to_float(choose(args.gpu_util_avg_pct, payload.get("gpu_util_avg_pct"))),
        "kv_cache_usage_pct": to_float(
            choose(args.kv_cache_usage_pct, payload.get("kv_cache_usage_pct"), gpu_cache_peak * 100.0 if gpu_cache_peak is not None else None)
        ),
        "server_startup_s": to_float(choose(args.server_startup_s, payload.get("server_startup_s"))),
        "benchmark_duration_s": to_float(
            choose(args.benchmark_duration_s, payload.get("benchmark_duration_s"), aggregate.get("total_wall_time_s"))
        ),
        "campaign_class": choose(
            getattr(args, 'campaign_class', None),
            payload.get("campaign_class"),
        ),
        "trace_source": choose(
            getattr(args, 'trace_source', None),
            payload.get("trace_source"),
        ),
        "total_actual_input_tokens": to_int(
            (payload.get("depth_telemetry") or {}).get("total_actual_input_tokens")
            or payload.get("total_actual_input_tokens")
        ),
        "max_actual_context_len": to_int(
            (payload.get("depth_telemetry") or {}).get("max_actual_context_len_per_turn")
            or payload.get("max_actual_context_len_per_turn")
        ),
        "depth_coverage_ratio": to_float(payload.get("depth_coverage_ratio")),
        "depth_coverage_class": payload.get("depth_coverage_class"),
        "producer_estimated_kv_bytes_peak": to_int(payload.get("producer_estimated_kv_bytes_peak")),
        "producer_expected_offload_mode": payload.get("producer_expected_offload_mode"),
        "offload_mode_match": (
            1 if payload.get("producer_expectation_validation", {}).get("offload_mode_match") is True
            else 0 if payload.get("producer_expectation_validation", {}).get("offload_mode_match") is False
            else None
        ),
        "offload_mode": choose(getattr(args, 'offload_mode', None), payload.get("offload_mode")),
        "kv_cache_dtype": choose(getattr(args, 'kv_cache_dtype', None), payload.get("kv_cache_dtype")),
        "disable_prefix_caching": to_int(
            choose(
                getattr(args, 'disable_prefix_caching', None),
                payload.get("disable_prefix_caching"),
            )
        ),
        "cpu_cache_usage_peak_pct": to_float(
            payload.get("peak_cpu_cache_usage", 0.0) * 100.0
            if payload.get("peak_cpu_cache_usage") is not None else None
        ),
        "raw_result_json": json.dumps(build_raw_payload(payload, args), sort_keys=True),
        "status": args.status,
        "error_message": choose(args.error_message, payload.get("error_message")),
    }

    conn = connect_db(args.db_path)
    placeholders = ", ".join("?" for _ in INSERT_COLUMNS)
    sql = f"INSERT INTO {TABLE_NAME} ({', '.join(INSERT_COLUMNS)}) VALUES ({placeholders})"
    conn.execute(sql, [row[column] for column in INSERT_COLUMNS])
    conn.commit()
    conn.close()

    print(
        f"Inserted run {row['run_id']} into {Path(args.db_path)} "
        f"({row['gpu_type']} {row['model']} {row['engine']} {row['context_band']}, status={row['status']})."
    )


def fetch_rows(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    return list(conn.execute(sql, params))


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    normalized_rows = [[stringify(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in normalized_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(row: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

    divider = "-+-".join("-" * width for width in widths)
    lines = [fmt_row(headers), divider]
    for row in normalized_rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def print_query(args: argparse.Namespace) -> None:
    conn = connect_db(args.db_path)

    if args.group_by:
        group_columns = [column.strip() for column in args.group_by.split(",") if column.strip()]
        if not group_columns:
            raise SystemExit("--group-by requires at least one column")
        invalid = [column for column in group_columns if column not in GROUPABLE_COLUMNS]
        if invalid:
            raise SystemExit(
                f"Unsupported --group-by columns: {', '.join(invalid)}. "
                f"Allowed: {', '.join(sorted(GROUPABLE_COLUMNS))}"
            )

        select_prefix = ", ".join(group_columns)
        sql = f"""
            SELECT
              {select_prefix},
              COUNT(*) AS runs,
              SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_runs,
              SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS non_success_runs,
              ROUND(AVG(ttft_p50_ms), 2) AS avg_ttft_p50_ms,
              ROUND(AVG(throughput_tok_s), 2) AS avg_throughput_tok_s,
              ROUND(MAX(gpu_mem_peak_gb), 2) AS max_gpu_mem_peak_gb,
              SUM(CASE WHEN COALESCE(preemption_count, 0) > 0 THEN 1 ELSE 0 END) AS preemption_runs
            FROM {TABLE_NAME}
            GROUP BY {select_prefix}
            ORDER BY {select_prefix}
        """
        rows = fetch_rows(conn, sql)
        headers = group_columns + [
            "runs",
            "success_runs",
            "non_success_runs",
            "avg_ttft_p50_ms",
            "avg_throughput_tok_s",
            "max_gpu_mem_peak_gb",
            "preemption_runs",
        ]
        print(render_table(headers, ([row[header] for header in headers] for row in rows)))
    else:
        sql = f"SELECT {', '.join(DEFAULT_QUERY_COLUMNS)} FROM {TABLE_NAME} ORDER BY id DESC"
        rows = fetch_rows(conn, sql)
        print(render_table(DEFAULT_QUERY_COLUMNS, ([row[column] for column in DEFAULT_QUERY_COLUMNS] for row in rows)))

    conn.close()


def export_csv_rows(args: argparse.Namespace) -> None:
    conn = connect_db(args.db_path)
    rows = fetch_rows(conn, f"SELECT * FROM {TABLE_NAME} ORDER BY id ASC")
    headers = [description[0] for description in conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 0").description]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", newline="")
    else:
        handle = sys.stdout

    try:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[header] for header in headers])
    finally:
        if args.output:
            handle.close()
            print(f"Exported {len(rows)} rows to {args.output}")

    conn.close()


def print_summary(args: argparse.Namespace) -> None:
    conn = connect_db(args.db_path)
    total_runs = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    if total_runs == 0:
        print(f"No runs found in {args.db_path}")
        conn.close()
        return

    status_rows = fetch_rows(conn, f"SELECT status, COUNT(*) AS count FROM {TABLE_NAME} GROUP BY status ORDER BY status")
    preemption_rows = fetch_rows(
        conn,
        f"""
        SELECT gpu_type, model, engine, context_band, preemption_count, status
        FROM {TABLE_NAME}
        WHERE COALESCE(preemption_count, 0) > 0
        ORDER BY preemption_count DESC, id DESC
        LIMIT 10
        """,
    )
    highest_memory_rows = fetch_rows(
        conn,
        f"""
        SELECT gpu_type, model, engine, context_band, gpu_mem_peak_gb, kv_cache_usage_pct, status
        FROM {TABLE_NAME}
        WHERE gpu_mem_peak_gb IS NOT NULL
        ORDER BY gpu_mem_peak_gb DESC, id DESC
        LIMIT 5
        """,
    )
    slowest_ttft_rows = fetch_rows(
        conn,
        f"""
        SELECT gpu_type, model, engine, context_band, ttft_p50_ms, ttft_p99_ms, status
        FROM {TABLE_NAME}
        WHERE ttft_p50_ms IS NOT NULL
        ORDER BY ttft_p50_ms DESC, id DESC
        LIMIT 5
        """,
    )
    highest_kv_rows = fetch_rows(
        conn,
        f"""
        SELECT gpu_type, model, engine, context_band, kv_cache_usage_pct, gpu_mem_peak_gb, status
        FROM {TABLE_NAME}
        WHERE kv_cache_usage_pct IS NOT NULL
        ORDER BY kv_cache_usage_pct DESC, id DESC
        LIMIT 5
        """,
    )
    long_context_rollup = fetch_rows(
        conn,
        f"""
        SELECT
          context_band,
          COUNT(*) AS runs,
          SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_runs,
          ROUND(AVG(ttft_p50_ms), 2) AS avg_ttft_p50_ms,
          ROUND(MAX(gpu_mem_peak_gb), 2) AS max_gpu_mem_peak_gb,
          SUM(CASE WHEN COALESCE(preemption_count, 0) > 0 THEN 1 ELSE 0 END) AS preemption_runs
        FROM {TABLE_NAME}
        WHERE context_band IN ('131k', '500k', '1m')
        GROUP BY context_band
        ORDER BY CASE context_band WHEN '131k' THEN 1 WHEN '500k' THEN 2 WHEN '1m' THEN 3 ELSE 99 END
        """,
    )

    print(f"ISB1 results summary ({args.db_path})")
    print(f"Total runs: {total_runs}")
    print(render_table(["status", "count"], ([row["status"], row["count"]] for row in status_rows)))
    print()

    if long_context_rollup:
        print("Long-context rollup")
        print(
            render_table(
                ["context_band", "runs", "success_runs", "avg_ttft_p50_ms", "max_gpu_mem_peak_gb", "preemption_runs"],
                (
                    [
                        row["context_band"],
                        row["runs"],
                        row["success_runs"],
                        row["avg_ttft_p50_ms"],
                        row["max_gpu_mem_peak_gb"],
                        row["preemption_runs"],
                    ]
                    for row in long_context_rollup
                ),
            )
        )
        print()

    # Depth coverage rollup
    depth_coverage_rows = fetch_rows(
        conn,
        f"""
        SELECT
          context_band,
          COUNT(*) AS runs,
          ROUND(AVG(depth_coverage_ratio), 4) AS avg_depth_coverage,
          MAX(max_actual_context_len) AS max_actual_ctx,
          SUM(CASE WHEN depth_coverage_class = 'configuration_only' THEN 1 ELSE 0 END) AS config_only_runs,
          SUM(CASE WHEN depth_coverage_class = 'full' THEN 1 ELSE 0 END) AS full_depth_runs
        FROM {TABLE_NAME}
        WHERE context_band IN ('131k', '500k', '1m')
          AND depth_coverage_ratio IS NOT NULL
        GROUP BY context_band
        ORDER BY CASE context_band WHEN '131k' THEN 1 WHEN '500k' THEN 2 WHEN '1m' THEN 3 ELSE 99 END
        """,
    )
    if depth_coverage_rows:
        print("Depth coverage (actual vs configured)")
        print(
            render_table(
                ["context_band", "runs", "avg_depth_coverage", "max_actual_ctx", "config_only_runs", "full_depth_runs"],
                (
                    [
                        row["context_band"],
                        row["runs"],
                        row["avg_depth_coverage"],
                        row["max_actual_ctx"],
                        row["config_only_runs"],
                        row["full_depth_runs"],
                    ]
                    for row in depth_coverage_rows
                ),
            )
        )
        print()

    if preemption_rows:
        print("Runs with preemptions")
        print(
            render_table(
                ["gpu_type", "model", "engine", "context_band", "preemption_count", "status"],
                (
                    [
                        row["gpu_type"],
                        row["model"],
                        row["engine"],
                        row["context_band"],
                        row["preemption_count"],
                        row["status"],
                    ]
                    for row in preemption_rows
                ),
            )
        )
        print()
    else:
        print("Runs with preemptions: none")
        print()

    if highest_memory_rows:
        print("Highest peak GPU memory")
        print(
            render_table(
                ["gpu_type", "model", "engine", "context_band", "gpu_mem_peak_gb", "kv_cache_usage_pct", "status"],
                (
                    [
                        row["gpu_type"],
                        row["model"],
                        row["engine"],
                        row["context_band"],
                        row["gpu_mem_peak_gb"],
                        row["kv_cache_usage_pct"],
                        row["status"],
                    ]
                    for row in highest_memory_rows
                ),
            )
        )
        print()

    if slowest_ttft_rows:
        print("Slowest TTFT p50 runs")
        print(
            render_table(
                ["gpu_type", "model", "engine", "context_band", "ttft_p50_ms", "ttft_p99_ms", "status"],
                (
                    [
                        row["gpu_type"],
                        row["model"],
                        row["engine"],
                        row["context_band"],
                        row["ttft_p50_ms"],
                        row["ttft_p99_ms"],
                        row["status"],
                    ]
                    for row in slowest_ttft_rows
                ),
            )
        )
        print()

    if highest_kv_rows:
        print("Highest KV-cache usage")
        print(
            render_table(
                ["gpu_type", "model", "engine", "context_band", "kv_cache_usage_pct", "gpu_mem_peak_gb", "status"],
                (
                    [
                        row["gpu_type"],
                        row["model"],
                        row["engine"],
                        row["context_band"],
                        row["kv_cache_usage_pct"],
                        row["gpu_mem_peak_gb"],
                        row["status"],
                    ]
                    for row in highest_kv_rows
                ),
            )
        )

    conn.close()


def main() -> int:
    args = parse_args()
    if args.command == "ingest":
        insert_run(args)
    elif args.command == "query":
        print_query(args)
    elif args.command == "export-csv":
        export_csv_rows(args)
    elif args.command == "summary":
        print_summary(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
