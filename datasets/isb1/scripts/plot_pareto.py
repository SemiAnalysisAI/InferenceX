#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Pareto frontier for KV sweep throughput vs p99 TTFT")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (benchmark_runs)")
    parser.add_argument("--json-dir", default=None, help="Directory containing sweep summary JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory for pareto outputs")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows_from_db(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT offload_mode, ttft_p99_ms, throughput_tok_s, max_concurrency, raw_result_json
        FROM benchmark_runs
        WHERE offload_mode IS NOT NULL
          AND ttft_p99_ms IS NOT NULL
          AND throughput_tok_s IS NOT NULL
        ORDER BY id ASC
        """
    ).fetchall()
    conn.close()

    normalized: list[dict[str, Any]] = []
    for row in rows:
        concurrency = row["max_concurrency"]
        if concurrency in (None, "") and row["raw_result_json"]:
            try:
                payload = json.loads(row["raw_result_json"])
                concurrency = payload.get("conc") or payload.get("max_concurrency")
            except Exception:
                pass
        normalized.append(
            {
                "offload_mode": row["offload_mode"],
                "concurrency": int(concurrency) if concurrency not in (None, "") else None,
                "throughput_tok_s": _to_float(row["throughput_tok_s"]),
                "ttft_p99_ms": _to_float(row["ttft_p99_ms"]),
                "source": "db",
            }
        )
    return normalized


def load_rows_from_json_dir(json_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if isinstance(payload, dict) and isinstance(payload.get("summary"), list):
            for row in payload["summary"]:
                rows.append(
                    {
                        "offload_mode": row.get("offload_mode"),
                        "concurrency": row.get("concurrency"),
                        "throughput_tok_s": _to_float(row.get("throughput_tok_s")),
                        "ttft_p99_ms": _to_float(row.get("ttft_p99_ms")),
                        "source": str(path.name),
                    }
                )
        elif isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    rows.append(
                        {
                            "offload_mode": row.get("offload_mode"),
                            "concurrency": row.get("concurrency"),
                            "throughput_tok_s": _to_float(row.get("throughput_tok_s")),
                            "ttft_p99_ms": _to_float(row.get("ttft_p99_ms")),
                            "source": str(path.name),
                        }
                    )
    return rows


def compute_pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [p for p in points if p["throughput_tok_s"] is not None and p["ttft_p99_ms"] is not None]
    if not valid:
        return []

    # maximize throughput, minimize ttft_p99_ms
    sorted_points = sorted(valid, key=lambda p: (p["throughput_tok_s"], -p["ttft_p99_ms"]), reverse=True)
    frontier: list[dict[str, Any]] = []
    best_latency = float("inf")
    for point in sorted_points:
        latency = point["ttft_p99_ms"]
        if latency <= best_latency:
            frontier.append(point)
            best_latency = latency
    return sorted(frontier, key=lambda p: (p["throughput_tok_s"], p["ttft_p99_ms"]))


def write_csv(path: Path, rows: list[dict[str, Any]], frontier_keys: set[tuple[str, int | None, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["offload_mode", "concurrency", "throughput_tok_s", "ttft_p99_ms", "is_frontier", "source"])
        for row in rows:
            key = (row.get("offload_mode") or "", row.get("concurrency"), row.get("throughput_tok_s") or 0.0, row.get("ttft_p99_ms") or 0.0)
            writer.writerow([
                row.get("offload_mode"),
                row.get("concurrency"),
                row.get("throughput_tok_s"),
                row.get("ttft_p99_ms"),
                key in frontier_keys,
                row.get("source"),
            ])


def maybe_write_plot(output_path: Path, grouped_frontiers: dict[str, list[dict[str, Any]]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    plt.figure(figsize=(10, 6))
    for mode, frontier in sorted(grouped_frontiers.items()):
        x = [p["throughput_tok_s"] for p in frontier]
        y = [p["ttft_p99_ms"] for p in frontier]
        if not x:
            continue
        plt.plot(x, y, marker="o", label=mode)
    plt.xlabel("Throughput (tokens/sec)")
    plt.ylabel("p99 TTFT (ms)")
    plt.title("Pareto Frontier by Offload Mode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.db_path and not args.json_dir:
        raise SystemExit("Provide --db-path or --json-dir")

    rows: list[dict[str, Any]] = []
    if args.db_path:
        rows.extend(load_rows_from_db(Path(args.db_path)))
    if args.json_dir:
        rows.extend(load_rows_from_json_dir(Path(args.json_dir)))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mode = row.get("offload_mode")
        if not mode:
            continue
        grouped.setdefault(mode, []).append(row)

    grouped_frontiers: dict[str, list[dict[str, Any]]] = {}
    for mode, points in grouped.items():
        grouped_frontiers[mode] = compute_pareto_frontier(points)

    frontier_keys: set[tuple[str, int | None, float, float]] = set()
    for mode, frontier in grouped_frontiers.items():
        for point in frontier:
            frontier_keys.add((mode, point.get("concurrency"), point.get("throughput_tok_s") or 0.0, point.get("ttft_p99_ms") or 0.0))

    csv_path = output_dir / "pareto_data.csv"
    write_csv(csv_path, rows, frontier_keys)

    summary = {
        "total_points": len(rows),
        "offload_modes": sorted(grouped.keys()),
        "frontier": {mode: frontier for mode, frontier in grouped_frontiers.items()},
    }
    summary_path = output_dir / "pareto_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    plot_written = maybe_write_plot(output_dir / "pareto_frontier.png", grouped_frontiers)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    if plot_written:
        print(f"Wrote: {output_dir / 'pareto_frontier.png'}")
    else:
        print("Skipped pareto_frontier.png (matplotlib unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
