#!/usr/bin/env python3
"""Generate metrics_plots.png from aiperf agentic-replay artifacts.

Reads $RESULT_DIR/trace_replay/profile_export.jsonl (per-record stream)
and $RESULT_DIR/trace_replay/server_metrics_export.json (Prometheus
scrape) and emits $RESULT_DIR/metrics_plots.png — a multi-panel figure
matching the kv-cache-tester output shape.

Panels (3x2 layout):
1. TTFT vs request time (scatter + rolling avg)
2. End-to-end latency vs request time (scatter + rolling avg)
3. Inter-token latency vs request time (avg per request, scatter +
   rolling)
4. Input + output sequence length distributions
5. Server GPU prefix-cache hit rate over time (from server_metrics
   timeslices when present, else final aggregate as a flat line)
6. Server KV-cache usage % over time

The script is best-effort — runs to completion even when a backing data
source is missing (writes an empty-ish panel and continues). Exit code
is non-zero only on a fully unrecoverable error (e.g. missing JSONL).

Usage:
    python3 generate_aiperf_plots.py <result_dir>

The launcher invokes it after write_agentic_result_json so the PNG
lands next to server.log / benchmark.log / agg_*.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "ERROR: matplotlib not installed; cannot generate plots", file=sys.stderr
    )
    sys.exit(1)


def load_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                continue
            records.append(obj)
    return records


def load_server_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def metric_value(record: dict, key: str) -> float | None:
    m = record.get("metrics", {}).get(key)
    if m is None:
        return None
    v = m.get("value") if isinstance(m, dict) else m
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rolling_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return values
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def scatter_with_rolling(
    ax,
    times_s: list[float],
    values: list[float],
    *,
    color: str,
    ylabel: str,
    title: str,
) -> None:
    if not values:
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return
    ax.scatter(times_s, values, alpha=0.3, s=8, c=color)
    window = min(50, max(1, len(values) // 10))
    if window > 1:
        rolling = rolling_average(values, window)
        ax.plot(times_s, rolling, "r-", linewidth=1.5, label=f"Rolling avg (n={window})")
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def histogram_panel(
    ax,
    isls: list[float],
    osls: list[float],
) -> None:
    if not isls and not osls:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Sequence length distributions")
        return
    bins = 40
    if isls:
        ax.hist(isls, bins=bins, alpha=0.55, label=f"ISL (n={len(isls)})", color="C0")
    if osls:
        ax.hist(osls, bins=bins, alpha=0.55, label=f"OSL (n={len(osls)})", color="C1")
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Requests")
    ax.set_title("Input / Output Sequence Length Distributions")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def server_metric_timeseries(
    server_metrics: dict, name: str
) -> tuple[list[float], list[float]]:
    """Return (times_s, values) for a server metric's timeslices.

    Returns ([], []) when the metric or timeslices are missing.
    """
    metrics = server_metrics.get("metrics") or {}
    entry = metrics.get(name)
    if not isinstance(entry, dict):
        return [], []
    series_list = entry.get("series") or []
    if not isinstance(series_list, list):
        return [], []

    summary = server_metrics.get("summary") or {}
    info = (summary.get("endpoint_info") or {}).values()
    first_ns_options = [v.get("first_update_ns") for v in info if isinstance(v, dict)]
    first_ns = min((x for x in first_ns_options if x is not None), default=None)

    times: list[float] = []
    values: list[float] = []
    for s in series_list:
        slices = s.get("timeslices") or []
        for ts in slices:
            start = ts.get("start_ns")
            if start is None:
                continue
            t = (start - first_ns) / 1e9 if first_ns is not None else len(times)
            for k in ("rate", "avg", "max", "total"):
                if k in ts and ts[k] is not None:
                    try:
                        values.append(float(ts[k]))
                        times.append(t)
                        break
                    except (TypeError, ValueError):
                        continue
    if not values:
        # Fall back to scalar aggregate (flat line across the run).
        for s in series_list:
            stats = s.get("stats") or {}
            for k in ("avg", "max", "total"):
                if k in stats and stats[k] is not None:
                    return [], [float(stats[k])]
    return times, values


def cache_hit_rate_panel(ax, server_metrics: dict) -> None:
    hits_t, hits_v = server_metric_timeseries(server_metrics, "vllm:prefix_cache_hits")
    queries_t, queries_v = server_metric_timeseries(
        server_metrics, "vllm:prefix_cache_queries"
    )
    if hits_t and queries_t and len(hits_t) == len(queries_t):
        rates = [
            (h / q * 100.0) if q > 0 else 0.0 for h, q in zip(hits_v, queries_v)
        ]
        ax.plot(hits_t, rates, "g-", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GPU Cache Hit Rate (%)")
        ax.set_title("Server Prefix-Cache Hit Rate Over Time")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        return
    if hits_v and queries_v:
        # Aggregate-only fallback.
        rate = (hits_v[0] / queries_v[0] * 100.0) if queries_v[0] > 0 else 0.0
        ax.axhline(y=rate, color="g", linewidth=2)
        ax.text(
            0.5,
            0.85,
            f"Final: {rate:.1f}% (timeslices unavailable)",
            ha="center",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("GPU Cache Hit Rate (%)")
        ax.set_title("Server Prefix-Cache Hit Rate")
        return
    ax.text(0.5, 0.5, "no server cache metrics", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Server Prefix-Cache Hit Rate")


def kv_usage_panel(ax, server_metrics: dict) -> None:
    times, values = server_metric_timeseries(
        server_metrics, "vllm:kv_cache_usage_perc"
    )
    if values:
        if times:
            ax.plot(times, [v * 100 if v <= 1.0 else v for v in values], "b-", linewidth=1.5)
            ax.set_xlabel("Time (s)")
        else:
            v = values[0] * 100 if values[0] <= 1.0 else values[0]
            ax.axhline(y=v, color="b", linewidth=2)
            ax.text(
                0.5,
                0.85,
                f"Avg: {v:.1f}% (timeslices unavailable)",
                ha="center",
                transform=ax.transAxes,
                fontsize=9,
            )
        ax.set_ylabel("KV Cache Usage (%)")
        ax.set_title("vLLM KV Cache Usage")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        return
    ax.text(0.5, 0.5, "no KV usage metric", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("vLLM KV Cache Usage")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate metrics_plots.png from aiperf artifacts"
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Result dir containing trace_replay/ subdirectory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <result_dir>/metrics_plots.png)",
    )
    args = parser.parse_args(argv)

    artifact = args.result_dir / "trace_replay"
    jsonl_path = artifact / "profile_export.jsonl"
    server_metrics_path = artifact / "server_metrics_export.json"

    if not jsonl_path.exists():
        # Per-run subdir layout (--num-profile-runs > 1).
        for child in sorted(artifact.iterdir()) if artifact.is_dir() else []:
            if child.is_dir() and (child / "profile_export.jsonl").is_file():
                jsonl_path = child / "profile_export.jsonl"
                server_metrics_path = child / "server_metrics_export.json"
                break

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return 1

    records = load_jsonl_records(jsonl_path)
    server_metrics = load_server_metrics(server_metrics_path)

    if not records:
        print("WARNING: no successful records to plot", file=sys.stderr)

    starts_ns = [
        int(r["metadata"]["request_start_ns"])
        for r in records
        if r.get("metadata", {}).get("request_start_ns")
    ]
    first_start = min(starts_ns) if starts_ns else 0
    times_s = [(s - first_start) / 1e9 for s in starts_ns]

    ttfts_ms: list[float] = []
    e2es_ms: list[float] = []
    itls_ms: list[float] = []
    isls: list[float] = []
    osls: list[float] = []
    for r in records:
        ttft = metric_value(r, "time_to_first_token")
        e2e = metric_value(r, "request_latency")
        itl = metric_value(r, "inter_token_latency")
        isl = metric_value(r, "input_sequence_length")
        osl = metric_value(r, "output_sequence_length")
        ttfts_ms.append(ttft if ttft is not None else float("nan"))
        e2es_ms.append(e2e if e2e is not None else float("nan"))
        itls_ms.append(itl if itl is not None else float("nan"))
        if isl is not None:
            isls.append(isl)
        if osl is not None:
            osls.append(osl)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"AIPerf agentic-replay metrics — {len(records)} requests",
        fontsize=13,
    )

    ttft_clean = [v for v in zip(times_s, ttfts_ms) if v[1] == v[1]]
    e2e_clean = [v for v in zip(times_s, e2es_ms) if v[1] == v[1]]
    itl_clean = [v for v in zip(times_s, itls_ms) if v[1] == v[1]]

    if ttft_clean:
        t, v = zip(*ttft_clean)
        scatter_with_rolling(
            axes[0, 0],
            list(t),
            list(v),
            color="C0",
            ylabel="TTFT (ms)",
            title="Time to First Token vs Time",
        )
    if e2e_clean:
        t, v = zip(*e2e_clean)
        scatter_with_rolling(
            axes[0, 1],
            list(t),
            list(v),
            color="C2",
            ylabel="E2E Latency (ms)",
            title="End-to-End Latency vs Time",
        )
    if itl_clean:
        t, v = zip(*itl_clean)
        scatter_with_rolling(
            axes[1, 0],
            list(t),
            list(v),
            color="C4",
            ylabel="ITL (ms / token)",
            title="Inter-Token Latency vs Time",
        )

    histogram_panel(axes[1, 1], isls, osls)
    cache_hit_rate_panel(axes[2, 0], server_metrics)
    kv_usage_panel(axes[2, 1], server_metrics)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = args.output or (args.result_dir / "metrics_plots.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")
    if records:
        print(
            f"  Records: {len(records)} | "
            f"TTFT median {statistics.median([v for v in ttfts_ms if v == v]):.0f}ms | "
            f"E2E median {statistics.median([v for v in e2es_ms if v == v]):.0f}ms"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
