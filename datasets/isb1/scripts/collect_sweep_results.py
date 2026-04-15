#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sweep results from DB or agg_*.json directory")
    parser.add_argument("--db-path", default=None, help="SQLite DB path")
    parser.add_argument("--json-dir", default=None, help="Directory containing agg_*.json files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--cliff-ttft-ms", type=float, default=5000.0, help="TTFT p99 threshold for capacity cliff")
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def collect_from_db(db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT offload_mode, throughput_tok_s, ttft_p99_ms, max_concurrency, raw_result_json
        FROM benchmark_runs
        WHERE offload_mode IS NOT NULL
        ORDER BY id ASC
        """
    ).fetchall()
    conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        concurrency = row["max_concurrency"]
        if concurrency in (None, "") and row["raw_result_json"]:
            try:
                payload = json.loads(row["raw_result_json"])
                concurrency = payload.get("conc") or payload.get("max_concurrency")
            except Exception:
                pass
        out.append(
            {
                "offload_mode": row["offload_mode"],
                "concurrency": _to_int(concurrency),
                "throughput_tok_s": _to_float(row["throughput_tok_s"]),
                "ttft_p99_ms": _to_float(row["ttft_p99_ms"]),
                "source": "db",
            }
        )
    return out


def collect_from_json_dir(json_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(json_dir.glob("agg_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "offload_mode": payload.get("offload_mode"),
                "concurrency": _to_int(payload.get("conc") or payload.get("max_concurrency")),
                "throughput_tok_s": _to_float(payload.get("throughput_tok_s") or payload.get("tput_per_gpu")),
                "ttft_p99_ms": _to_float(payload.get("ttft_p99_ms") or payload.get("p99_ttft_ms")),
                "source": str(path.name),
            }
        )
    return rows


def compute_capacity_cliff(rows: list[dict[str, Any]], threshold_ms: float) -> dict[str, Any]:
    cliff: dict[str, Any] = {}
    for mode in sorted({row.get("offload_mode") for row in rows if row.get("offload_mode")}):
        mode_rows = sorted(
            [r for r in rows if r.get("offload_mode") == mode and r.get("concurrency") is not None],
            key=lambda r: r["concurrency"],
        )
        cliff_row = None
        for row in mode_rows:
            if (row.get("ttft_p99_ms") or 0.0) > threshold_ms:
                cliff_row = row
                break
        cliff[str(mode)] = cliff_row
    return cliff


def compute_offload_benefit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_conc: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        conc = row.get("concurrency")
        mode = row.get("offload_mode")
        if conc is None or mode is None:
            continue
        by_conc.setdefault(int(conc), {})[str(mode)] = row

    deltas: list[dict[str, Any]] = []
    for conc in sorted(by_conc):
        modes = by_conc[conc]
        on = modes.get("on")
        off = modes.get("off")
        if not on or not off:
            continue
        on_tput = on.get("throughput_tok_s") or 0.0
        off_tput = off.get("throughput_tok_s") or 0.0
        deltas.append(
            {
                "concurrency": conc,
                "throughput_on": on_tput,
                "throughput_off": off_tput,
                "offload_benefit_delta_tps": on_tput - off_tput,
            }
        )
    return deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["offload_mode", "concurrency", "throughput_tok_s", "ttft_p99_ms", "source"])
        for row in rows:
            writer.writerow([
                row.get("offload_mode"),
                row.get("concurrency"),
                row.get("throughput_tok_s"),
                row.get("ttft_p99_ms"),
                row.get("source"),
            ])


def main() -> int:
    args = parse_args()
    if not args.db_path and not args.json_dir:
        raise SystemExit("Provide --db-path or --json-dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if args.db_path:
        rows.extend(collect_from_db(Path(args.db_path)))
    if args.json_dir:
        rows.extend(collect_from_json_dir(Path(args.json_dir)))

    summary = {
        "num_rows": len(rows),
        "capacity_cliff": compute_capacity_cliff(rows, args.cliff_ttft_ms),
        "offload_benefit": compute_offload_benefit(rows),
        "rows": rows,
    }

    json_path = output_dir / "sweep_aggregate.json"
    csv_path = output_dir / "sweep_aggregate.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(csv_path, rows)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
