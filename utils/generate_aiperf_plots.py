#!/usr/bin/env python3
"""Generate metrics_plots.png matching kv-cache-tester's 6x2 layout.

Reads aiperf's per-record JSONL + server-metrics JSON (with timeslices
enabled via ``--slice-duration``) and emits a PNG with the same panels
the legacy kv-cache-tester pipeline produced. The launchers feed this
$RESULT_DIR after each run so downstream tooling and humans see the
same visual.

Layout (6 rows x 2 cols, suptitle "vLLM Server Metrics During Benchmark"):
    (0,0) KV Cache Utilization Over Time (HBM + External)
    (0,1) Request Queue Depth (running / waiting / total)
    (1,0) Prefix Cache Hit Rate Per Interval (GPU / External / Combined)
    (1,1) Throughput (Total & Decode) with running average
    (2,0) KV Offload Transfer Rate (GPU↔CPU MB/s)
    (2,1) Cumulative Prefill Token Source Breakdown (stackplot)
    (3,0) KV Offload GPU→CPU (Cumulative GB)
    (3,1) KV Offload CPU→GPU (Cumulative GB)
    (4,0) TTFT vs Time (scatter + rolling avg)
    (4,1) Request Latency vs Time (scatter + rolling avg)
    (5,0) Interactivity 1/TPOT vs Time (scatter + rolling avg)
    (5,1) Preemptions Over Time (rate + cumulative)

Time-series data comes from server_metrics_export.json's per-series
``timeslices`` array (populated when ``--slice-duration`` is set on the
aiperf CLI). Per-record TTFT / Latency / ITL come from
profile_export.jsonl. Panels with no data still render so the output
shape is constant across run configs.

Usage:
    python3 generate_aiperf_plots.py <result_dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from enum import Enum
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed; cannot generate plots", file=sys.stderr)
    sys.exit(1)


# ---- Loaders --------------------------------------------------------------


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


# ---- Server-metrics helpers ----------------------------------------------


def first_update_ns(server_metrics: dict) -> int | None:
    summary = server_metrics.get("summary") or {}
    info = (summary.get("endpoint_info") or {}).values()
    candidates = [
        v.get("first_update_ns")
        for v in info
        if isinstance(v, dict) and v.get("first_update_ns") is not None
    ]
    return min(candidates) if candidates else None


class Metric(str, Enum):
    """Engine-neutral metric identifiers the panels are written against.

    Panels reference these semantic names only; the per-engine mapping in
    :data:`ENGINE_METRICS` translates them to whatever each inference engine
    actually exports. This keeps the plotting code free of any single engine's
    naming so adding a new backend is a localized change.
    """

    KV_CACHE_USAGE = "kv_cache_usage"
    CPU_KV_CACHE_USAGE = "cpu_kv_cache_usage"
    NUM_RUNNING = "num_running"
    NUM_WAITING = "num_waiting"
    GENERATION_TOKENS = "generation_tokens"
    PROMPT_TOKENS = "prompt_tokens"
    NUM_PREEMPTIONS = "num_preemptions"
    PREFIX_CACHE_HITS = "prefix_cache_hits"
    PREFIX_CACHE_QUERIES = "prefix_cache_queries"
    EXT_PREFIX_CACHE_HITS = "external_prefix_cache_hits"
    EXT_PREFIX_CACHE_QUERIES = "external_prefix_cache_queries"
    # A ready-made prefix-cache hit-rate gauge (fraction), for engines that
    # expose it directly instead of raw hit/query counters.
    PREFIX_CACHE_HIT_RATE = "prefix_cache_hit_rate"
    PROMPT_TOKENS_BY_SOURCE = "prompt_tokens_by_source"
    KV_OFFLOAD_G2C = "kv_offload_bytes_gpu_to_cpu"
    KV_OFFLOAD_C2G = "kv_offload_bytes_cpu_to_gpu"


# A metric source is either the concrete series name an engine exports, or an
# adapter that synthesizes the series when the engine has no 1:1 equivalent
# (e.g. a ratio of two counters). Adapters take ``(server_metrics, t0_ns)`` plus
# the same keyword args as :func:`aggregate_timeseries` and return (times, values).
MetricSource = "str | Callable[..., tuple[list[float], list[float]]]"


def _sum_by_start(
    server_metrics: dict, name: str, value_key_priority
) -> dict[int, float]:
    """Sum a metric's per-slice value across all its series, keyed by start_ns."""
    out: dict[int, float] = defaultdict(float)
    entry = (server_metrics.get("metrics") or {}).get(name)
    if not isinstance(entry, dict):
        return out
    for s in entry.get("series") or []:
        for ts in s.get("timeslices") or []:
            start = ts.get("start_ns")
            if start is None:
                continue
            for k in value_key_priority:
                v = ts.get(k)
                if v is not None:
                    try:
                        out[int(start)] += float(v)
                        break
                    except (TypeError, ValueError):
                        continue
    return out


def _ratio_adapter(used_name: str, total_name: str):
    """Adapter yielding per-timestamp ``used/total`` (fraction) across series.

    Synthesizes a usage-fraction series from two token-count gauges the way
    vLLM exposes it as a single ``*_usage_perc`` gauge. Panels normalize the
    0..1 fraction to a percentage themselves.
    """

    def _adapter(
        server_metrics: dict, t0_ns: int | None, *, aggregator=None,
        value_key_priority=("avg", "max"),
    ) -> tuple[list[float], list[float]]:
        if t0_ns is None:
            return [], []
        used = _sum_by_start(server_metrics, used_name, value_key_priority)
        total = _sum_by_start(server_metrics, total_name, value_key_priority)
        times: list[float] = []
        values: list[float] = []
        for start in sorted(used):
            tot = total.get(start)
            if not tot:
                continue
            times.append((start - t0_ns) / 1e9)
            values.append(used[start] / tot)
        return times, values

    return _adapter


# Per-engine metric maps: semantic name -> the concrete series name that engine
# emits (or an adapter for composite metrics). A missing key means the engine
# does not export that metric, so the corresponding panel/series simply stays
# blank. Adding support for a new engine (e.g. TensorRT-LLM) is a single new
# table here — no panel code changes, and each engine's coverage (and gaps) is
# visible at a glance.
#
# NOTE: sglang mappings use only upstream (main-branch) metric names; the
# fine-grained per-tier prefix-cache counters (cache_hit_tokens_l1/l2/l3,
# cache_miss_tokens) live in an unmerged fork and are intentionally not relied
# on, so the GPU/External hit-rate split and the prefill-source breakdown stay
# blank for sglang.
ENGINE_METRICS: dict[str, dict[Metric, MetricSource]] = {
    "vllm": {
        Metric.KV_CACHE_USAGE: "vllm:kv_cache_usage_perc",
        Metric.CPU_KV_CACHE_USAGE: "vllm:cpu_kv_cache_usage_perc",
        Metric.NUM_RUNNING: "vllm:num_requests_running",
        Metric.NUM_WAITING: "vllm:num_requests_waiting",
        Metric.GENERATION_TOKENS: "vllm:generation_tokens",
        Metric.PROMPT_TOKENS: "vllm:prompt_tokens",
        Metric.NUM_PREEMPTIONS: "vllm:num_preemptions",
        Metric.PREFIX_CACHE_HITS: "vllm:prefix_cache_hits",
        Metric.PREFIX_CACHE_QUERIES: "vllm:prefix_cache_queries",
        Metric.EXT_PREFIX_CACHE_HITS: "vllm:external_prefix_cache_hits",
        Metric.EXT_PREFIX_CACHE_QUERIES: "vllm:external_prefix_cache_queries",
        Metric.PROMPT_TOKENS_BY_SOURCE: "vllm:prompt_tokens_by_source",
        Metric.KV_OFFLOAD_G2C: "vllm:kv_offload_bytes_gpu_to_cpu",
        Metric.KV_OFFLOAD_C2G: "vllm:kv_offload_bytes_cpu_to_gpu",
    },
    "sglang": {
        Metric.KV_CACHE_USAGE: "sglang:token_usage",
        # Host (CPU) KV offload pool occupancy, synthesized from the HiCache
        # used/total token gauges — the analogue of vLLM cpu_kv_cache_usage_perc.
        Metric.CPU_KV_CACHE_USAGE: _ratio_adapter(
            "sglang:hicache_host_used_tokens", "sglang:hicache_host_total_tokens"
        ),
        Metric.NUM_RUNNING: "sglang:num_running_reqs",
        Metric.NUM_WAITING: "sglang:num_queue_reqs",
        Metric.GENERATION_TOKENS: "sglang:generation_tokens",
        Metric.PROMPT_TOKENS: "sglang:prompt_tokens",
        Metric.NUM_PREEMPTIONS: "sglang:num_retracted_reqs",
        # Ready-made aggregate hit-rate gauge (upstream). The per-tier
        # GPU/External split needs fork-only counters, so it is left out.
        Metric.PREFIX_CACHE_HIT_RATE: "sglang:cache_hit_rate",
        # HiCache offload traffic (token counts, not bytes — sglang upstream
        # exposes no per-token KV byte size). G2C = backup to host, C2G =
        # load back from host. Offload panels render these in tokens.
        Metric.KV_OFFLOAD_G2C: "sglang:backuped_tokens",
        Metric.KV_OFFLOAD_C2G: "sglang:load_back_tokens",
    },
}

_ENGINE_CACHE_KEY = "_resolved_engine"


def detect_engine(server_metrics: dict) -> str:
    """Infer the inference engine from the namespace prefix of exported metrics.

    Result is cached on ``server_metrics`` so detection runs once per export.
    Falls back to ``"vllm"`` when no known namespace is found.
    """
    cached = server_metrics.get(_ENGINE_CACHE_KEY)
    if cached:
        return cached
    metrics = server_metrics.get("metrics") or {}
    prefixes = {k.split(":", 1)[0] for k in metrics if ":" in k}
    engine = next((e for e in ENGINE_METRICS if e in prefixes), "vllm")
    server_metrics[_ENGINE_CACHE_KEY] = engine
    return engine


def resolve_metric(server_metrics: dict, metric: Metric):
    """Map an engine-neutral :class:`Metric` to this export's source.

    Returns a concrete metric name (str), an adapter (callable), or ``None``
    when the detected engine does not export that metric.
    """
    return ENGINE_METRICS.get(detect_engine(server_metrics), {}).get(metric)


def metric_entry(server_metrics: dict, metric: Metric) -> dict | None:
    target = resolve_metric(server_metrics, metric)
    if not isinstance(target, str):
        return None
    entry = (server_metrics.get("metrics") or {}).get(target)
    return entry if isinstance(entry, dict) else None


def all_series(entry: dict | None) -> list[dict]:
    if entry is None:
        return []
    s = entry.get("series") or []
    return s if isinstance(s, list) else []


def series_with_label(
    entry: dict | None, label_key: str, label_value: str
) -> dict | None:
    """Pick the series whose labels[label_key] matches label_value."""
    for s in all_series(entry):
        labels = s.get("labels") or {}
        if labels.get(label_key) == label_value:
            return s
    return None


def timeseries_from_series(
    series: dict | None, t0_ns: int | None, value_key_priority=("avg", "rate", "total", "max")
) -> tuple[list[float], list[float]]:
    """Extract (relative-time-s, value) pairs from a series' timeslices."""
    if series is None or t0_ns is None:
        return [], []
    slices = series.get("timeslices") or []
    times: list[float] = []
    values: list[float] = []
    for ts in slices:
        start = ts.get("start_ns")
        if start is None:
            continue
        for k in value_key_priority:
            if k in ts and ts[k] is not None:
                try:
                    values.append(float(ts[k]))
                    times.append((start - t0_ns) / 1e9)
                    break
                except (TypeError, ValueError):
                    continue
    return times, values


def _aggregate_named(
    server_metrics: dict, name: str, t0_ns: int | None,
    *,
    aggregator,
    value_key_priority,
) -> tuple[list[float], list[float]]:
    """Aggregate timeslices across every series of a concrete metric name."""
    entry = (server_metrics.get("metrics") or {}).get(name)
    if not isinstance(entry, dict) or t0_ns is None:
        return [], []
    bucket: dict[int, list[float]] = defaultdict(list)
    for s in all_series(entry):
        for ts in s.get("timeslices") or []:
            start = ts.get("start_ns")
            if start is None:
                continue
            for k in value_key_priority:
                if k in ts and ts[k] is not None:
                    try:
                        bucket[int(start)].append(float(ts[k]))
                        break
                    except (TypeError, ValueError):
                        continue
    if not bucket:
        return [], []
    times: list[float] = []
    values: list[float] = []
    for start_ns in sorted(bucket):
        times.append((start_ns - t0_ns) / 1e9)
        values.append(aggregator(bucket[start_ns]))
    return times, values


def aggregate_timeseries(
    server_metrics: dict, metric: Metric, t0_ns: int | None,
    *,
    aggregator=sum,
    value_key_priority=("avg", "rate", "total", "max"),
) -> tuple[list[float], list[float]]:
    """Resolve a semantic metric and aggregate its timeslices (sums by default).

    Dispatches to an adapter when the engine maps the metric to a callable.
    """
    target = resolve_metric(server_metrics, metric)
    if target is None or t0_ns is None:
        return [], []
    if callable(target):
        return target(
            server_metrics, t0_ns,
            aggregator=aggregator, value_key_priority=value_key_priority,
        )
    return _aggregate_named(
        server_metrics, target, t0_ns,
        aggregator=aggregator, value_key_priority=value_key_priority,
    )


def rolling_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return list(values)
    out: list[float] = []
    for i in range(len(values)):
        chunk = values[max(0, i - window) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def rolling_window(n: int, max_window: int = 50) -> int:
    if n <= 10:
        return 1
    return min(max_window, max(1, n // 10))


# ---- Panels --------------------------------------------------------------


def panel_kv_cache_usage(ax, server_metrics: dict, t0_ns: int | None) -> None:
    times, values = aggregate_timeseries(
        server_metrics, Metric.KV_CACHE_USAGE, t0_ns, aggregator=max
    )
    cpu_times, cpu_values = aggregate_timeseries(
        server_metrics, Metric.CPU_KV_CACHE_USAGE, t0_ns, aggregator=max
    )

    def _norm(v: float) -> float:
        return v * 100.0 if 0 <= v <= 1.0 else v

    if values:
        gpu_pct = [min(_norm(v), 100.0) for v in values]
        ax.scatter(times, gpu_pct, alpha=0.15, s=2, c="blue")
        win = rolling_window(len(gpu_pct))
        if win > 1:
            ax.plot(
                times,
                rolling_average(gpu_pct, win),
                "b-",
                linewidth=2,
                label=f"GPU (avg n={win})",
            )
        else:
            ax.plot(times, gpu_pct, "b-", linewidth=2, label="GPU")
    if cpu_values:
        cpu_pct = [_norm(v) for v in cpu_values]
        ax.plot(cpu_times, cpu_pct, "r--", linewidth=1.5, label="External")
    if values or cpu_values:
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("KV Cache Usage (%)")
    ax.set_title("KV Cache Utilization Over Time")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)


def panel_queue_depth(ax, server_metrics: dict, t0_ns: int | None) -> None:
    rt, rv = aggregate_timeseries(
        server_metrics, Metric.NUM_RUNNING, t0_ns, aggregator=max
    )
    wt, wv = aggregate_timeseries(
        server_metrics, Metric.NUM_WAITING, t0_ns, aggregator=max
    )
    if rt:
        win = rolling_window(len(rv))
        running = rolling_average(rv, win) if win > 1 else rv
        ax.plot(rt, running, "g-", label=f"Running (avg n={win})", linewidth=1.5)
    if wt:
        win = rolling_window(len(wv))
        waiting = rolling_average(wv, win) if win > 1 else wv
        ax.plot(wt, waiting, "r-", label=f"Waiting (avg n={win})", linewidth=1.5)
    if rt and wt and len(rt) == len(wt):
        total = [r + w for r, w in zip(rv, wv)]
        win = rolling_window(len(total))
        smoothed = rolling_average(total, win) if win > 1 else total
        ax.plot(rt, smoothed, "b-", label=f"Total (avg n={win})", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests")
    ax.set_title("Request Queue Depth")
    if rt or wt:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _hit_rate_intervals(
    server_metrics: dict,
    hits_metric: Metric,
    queries_metric: Metric,
    t0_ns: int | None,
) -> tuple[list[float], list[float]]:
    """Compute per-interval hit rates from cumulative counters' deltas."""
    ht, hv = aggregate_timeseries(
        server_metrics, hits_metric, t0_ns, value_key_priority=("total",)
    )
    qt, qv = aggregate_timeseries(
        server_metrics, queries_metric, t0_ns, value_key_priority=("total",)
    )
    if not ht or not qt or len(ht) != len(qt):
        return [], []
    times: list[float] = []
    rates: list[float] = []
    last = 0.0
    for i in range(len(ht)):
        dh = hv[i]
        dq = qv[i]
        if dq > 0:
            last = 100.0 * dh / dq
        rates.append(last)
        times.append(ht[i])
    return times, rates


def panel_prefix_cache_hit_rate(ax, server_metrics: dict, t0_ns: int | None) -> None:
    gpu_t, gpu_r = _hit_rate_intervals(
        server_metrics,
        Metric.PREFIX_CACHE_HITS,
        Metric.PREFIX_CACHE_QUERIES,
        t0_ns,
    )
    # Some engines expose a ready-made prefix-cache hit-rate gauge (0..1
    # fraction) rather than hit/query counters — use it when the counters
    # are absent.
    if not gpu_t:
        gt, gv = aggregate_timeseries(
            server_metrics,
            Metric.PREFIX_CACHE_HIT_RATE,
            t0_ns,
            aggregator=max,
            value_key_priority=("avg", "max"),
        )
        if gt:
            gpu_t = gt
            gpu_r = [v * 100.0 if 0.0 <= v <= 1.0 else v for v in gv]
    ext_t, ext_r = _hit_rate_intervals(
        server_metrics,
        Metric.EXT_PREFIX_CACHE_HITS,
        Metric.EXT_PREFIX_CACHE_QUERIES,
        t0_ns,
    )
    if gpu_t:
        ax.scatter(gpu_t, gpu_r, alpha=0.3, s=5, c="purple", label="GPU (HBM)")
        win = rolling_window(len(gpu_r))
        if win > 1:
            ax.plot(
                gpu_t,
                rolling_average(gpu_r, win),
                "purple",
                linewidth=1.5,
                label=f"GPU avg (n={win})",
            )
    has_ext = bool(ext_t and any(r > 0 for r in ext_r))
    if has_ext:
        ax.scatter(ext_t, ext_r, alpha=0.3, s=5, c="orange", label="External")
        win = rolling_window(len(ext_r))
        if win > 1:
            ax.plot(
                ext_t,
                rolling_average(ext_r, win),
                "orange",
                linewidth=1.5,
                label=f"External avg (n={win})",
            )
        # Combined (only meaningful when external exists).
        if gpu_t and len(gpu_t) == len(ext_t):
            combined = [
                (g + e) / 2.0 if (g or e) else 0.0 for g, e in zip(gpu_r, ext_r)
            ]
            ax.scatter(gpu_t, combined, alpha=0.2, s=3, c="green", label="Combined")
            win = rolling_window(len(combined))
            if win > 1:
                ax.plot(
                    gpu_t,
                    rolling_average(combined, win),
                    "green",
                    linewidth=2,
                    label=f"Combined avg (n={win})",
                )
    if gpu_t or has_ext:
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Prefix Cache Hit Rate Per Interval (tokens hit / tokens queried)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)


def panel_throughput(ax, server_metrics: dict, t0_ns: int | None) -> None:
    gen_t, gen_v = aggregate_timeseries(
        server_metrics, Metric.GENERATION_TOKENS, t0_ns, value_key_priority=("rate",)
    )
    prompt_t, prompt_v = aggregate_timeseries(
        server_metrics, Metric.PROMPT_TOKENS, t0_ns, value_key_priority=("rate",)
    )
    if gen_t and prompt_t and len(gen_t) == len(prompt_t):
        total = [g + p for g, p in zip(gen_v, prompt_v)]
        win = rolling_window(len(total))
        if win > 1:
            ax.plot(
                gen_t,
                rolling_average(total, win),
                "steelblue",
                linewidth=1.5,
                label=f"Total (avg n={win})",
            )
            ax.plot(
                gen_t,
                rolling_average(gen_v, win),
                "orange",
                linewidth=1.5,
                label=f"Decode (avg n={win})",
            )
        else:
            ax.plot(gen_t, total, "steelblue", linewidth=1, alpha=0.8, label="Total")
            ax.plot(gen_t, gen_v, "orange", linewidth=1, alpha=0.8, label="Decode")
        # Cumulative running average: cumsum tokens / elapsed.
        if gen_t:
            cumulative_total = []
            t0 = gen_t[0]
            running = 0.0
            for i, t in enumerate(gen_t):
                # rate = tokens/s in that window; multiply by window width.
                width = (gen_t[i] - gen_t[i - 1]) if i > 0 else 0.0
                running += total[i] * width
                elapsed = t - t0 if t > t0 else 1e-9
                cumulative_total.append(running / elapsed if elapsed > 0 else 0.0)
            ax.plot(gen_t, cumulative_total, "red", linewidth=2, label="Total Running Avg")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput (Total & Decode)")
    ax.grid(True, alpha=0.3)


# KV offload counters are bytes on vLLM but token counts on sglang (which
# exposes no per-token KV byte size upstream). The offload panels pick scale +
# axis label from the engine so each renders in its native unit.
_OFFLOAD_UNITS: dict[str, dict] = {
    "vllm": {
        "rate_div": 1e6, "rate_label": "Transfer Rate (MB/s)",
        "cum_div": 1e9, "cum_label": "Cumulative Transfer (GB)",
    },
    "sglang": {
        "rate_div": 1.0, "rate_label": "Transfer Rate (tokens/s)",
        "cum_div": 1e6, "cum_label": "Cumulative Transfer (M tokens)",
    },
}


def offload_unit(server_metrics: dict) -> dict:
    return _OFFLOAD_UNITS.get(detect_engine(server_metrics), _OFFLOAD_UNITS["vllm"])


def panel_kv_offload_transfer_rate(
    ax, server_metrics: dict, t0_ns: int | None
) -> None:
    unit = offload_unit(server_metrics)
    div = unit["rate_div"]
    g2c_t, g2c_v = aggregate_timeseries(
        server_metrics,
        Metric.KV_OFFLOAD_G2C,
        t0_ns,
        value_key_priority=("rate",),
    )
    c2g_t, c2g_v = aggregate_timeseries(
        server_metrics,
        Metric.KV_OFFLOAD_C2G,
        t0_ns,
        value_key_priority=("rate",),
    )
    has_data = (g2c_t and any(v > 0 for v in g2c_v)) or (
        c2g_t and any(v > 0 for v in c2g_v)
    )
    if has_data:
        if g2c_t:
            scaled = [v / div for v in g2c_v]
            ax.scatter(g2c_t, scaled, alpha=0.15, s=3, c="blue")
            win = rolling_window(len(scaled))
            if win > 1:
                ax.plot(
                    g2c_t,
                    rolling_average(scaled, win),
                    "b-",
                    linewidth=1.5,
                    label=f"GPU→CPU (avg n={win})",
                )
            else:
                ax.plot(g2c_t, scaled, "b-", linewidth=1, alpha=0.8, label="GPU→CPU")
        if c2g_t:
            scaled = [v / div for v in c2g_v]
            ax.scatter(c2g_t, scaled, alpha=0.15, s=3, c="red")
            win = rolling_window(len(scaled))
            if win > 1:
                ax.plot(
                    c2g_t,
                    rolling_average(scaled, win),
                    "r-",
                    linewidth=1.5,
                    label=f"CPU→GPU (avg n={win})",
                )
            else:
                ax.plot(c2g_t, scaled, "r-", linewidth=1, alpha=0.8, label="CPU→GPU")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(unit["rate_label"])
    ax.set_title("KV Offload Transfer Rate")
    ax.grid(True, alpha=0.3)


def _prompt_token_source_series(
    server_metrics: dict, source_label: str, t0_ns: int | None
) -> tuple[list[float], list[float]]:
    """prompt_tokens_by_source has labels {source: local_compute|local_cache_hit|external_kv_transfer}."""
    entry = metric_entry(server_metrics, Metric.PROMPT_TOKENS_BY_SOURCE)
    s = series_with_label(entry, "source", source_label)
    return timeseries_from_series(s, t0_ns, value_key_priority=("total",))


def panel_prefill_source_breakdown(
    ax, server_metrics: dict, t0_ns: int | None
) -> None:
    c_t, c_v = _prompt_token_source_series(server_metrics, "local_compute", t0_ns)
    h_t, h_v = _prompt_token_source_series(server_metrics, "local_cache_hit", t0_ns)
    e_t, e_v = _prompt_token_source_series(
        server_metrics, "external_kv_transfer", t0_ns
    )
    # Align timestamps: use the union of all sample timestamps.
    if not (c_t or h_t or e_t):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("% of Prefill Tokens")
        ax.set_title("Cumulative Prefill Token Source Breakdown")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        return
    # Build per-timestamp cumulative values; counters are already cumulative
    # totals from the scrape (rate=delta over slice, but ``total`` here is
    # the slice total — accumulate ourselves).
    samples = sorted(set(c_t) | set(h_t) | set(e_t))

    def _cum_at(times: list[float], values: list[float]) -> dict:
        d: dict[float, float] = {}
        running = 0.0
        for t, v in zip(times, values):
            running += v
            d[t] = running
        # Forward-fill for missing samples.
        out: dict[float, float] = {}
        last = 0.0
        for t in samples:
            if t in d:
                last = d[t]
            out[t] = last
        return out

    cum_c = _cum_at(c_t, c_v)
    cum_h = _cum_at(h_t, h_v)
    cum_e = _cum_at(e_t, e_v)
    pct_c: list[float] = []
    pct_h: list[float] = []
    pct_e: list[float] = []
    for t in samples:
        c = cum_c[t]
        h = cum_h[t]
        e = cum_e[t]
        total = c + h + e
        if total > 0:
            pct_c.append(100.0 * c / total)
            pct_h.append(100.0 * h / total)
            pct_e.append(100.0 * e / total)
        else:
            pct_c.append(0.0)
            pct_h.append(0.0)
            pct_e.append(0.0)
    ax.stackplot(
        samples,
        pct_c,
        pct_h,
        pct_e,
        labels=["Prefill", "HBM Cache Hit", "Offload Cache Hit"],
        colors=["coral", "steelblue", "mediumseagreen"],
        alpha=0.8,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("% of Prefill Tokens")
    ax.set_title("Cumulative Prefill Token Source Breakdown")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)


def panel_kv_offload_cumulative(
    ax,
    server_metrics: dict,
    metric: Metric,
    title: str,
    color: str,
    t0_ns: int | None,
) -> None:
    unit = offload_unit(server_metrics)
    times, values = aggregate_timeseries(
        server_metrics, metric, t0_ns, value_key_priority=("total",)
    )
    if times and any(v > 0 for v in values):
        cumulative: list[float] = []
        running = 0.0
        for v in values:
            running += v
            cumulative.append(running / unit["cum_div"])
        ax.plot(times, cumulative, f"{color}-", linewidth=1.5)
        ax.fill_between(times, cumulative, alpha=0.2, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(unit["cum_label"])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def panel_per_record_metric(
    ax,
    request_times_s: list[float],
    values: list[float],
    *,
    color: str,
    ylabel: str,
    title: str,
) -> None:
    if not values:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return
    ax.scatter(request_times_s, values, alpha=0.3, s=5, c=color)
    win = rolling_window(len(values))
    if win > 1:
        ax.plot(
            request_times_s,
            rolling_average(values, win),
            "r-",
            linewidth=1.5,
            label=f"Rolling avg (n={win})",
        )
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def panel_preemptions(ax, server_metrics: dict, t0_ns: int | None) -> None:
    times, values = aggregate_timeseries(
        server_metrics, Metric.NUM_PREEMPTIONS, t0_ns, value_key_priority=("total",)
    )
    if not times:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Preemptions/sec")
        ax.set_title("Preemptions Over Time")
        ax.grid(True, alpha=0.3)
        return
    # ``total`` is the per-slice delta; convert to rate by dividing by slice
    # width (assume uniform: median diff between consecutive starts).
    if len(times) >= 2:
        diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
        slice_w = max(1e-9, statistics.median(diffs))
    else:
        slice_w = 1.0
    rates = [v / slice_w for v in values]
    if any(r > 0 for r in rates):
        ax.scatter(times, rates, alpha=0.15, s=3, c="red")
        win = rolling_window(len(rates), max_window=30)
        if win > 1:
            ax.plot(
                times,
                rolling_average(rates, win),
                "r-",
                linewidth=1.5,
                label=f"Rolling avg (n={win})",
            )
        # Cumulative on twin axis.
        cumulative: list[float] = []
        running = 0.0
        for v in values:
            running += v
            cumulative.append(running)
        ax2 = ax.twinx()
        ax2.plot(times, cumulative, "b--", linewidth=1, alpha=0.5, label="Cumulative")
        ax2.set_ylabel("Cumulative Preemptions", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Preemptions/sec", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.set_title("Preemptions Over Time")
    ax.grid(True, alpha=0.3)


# ---- Main ----------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate metrics_plots.png from aiperf artifacts (kv-cache-tester layout)"
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

    # benchmark_lib.sh writes aiperf output to <result_dir>/aiperf_artifacts/
    # (--output-artifact-dir). Older runs used trace_replay/, kept as fallback.
    artifact = args.result_dir / "aiperf_artifacts"
    if not (artifact / "profile_export.jsonl").exists():
        legacy = args.result_dir / "trace_replay"
        if (legacy / "profile_export.jsonl").exists():
            artifact = legacy
    jsonl_path = artifact / "profile_export.jsonl"
    server_metrics_path = artifact / "server_metrics_export.json"

    if not jsonl_path.exists() and artifact.is_dir():
        for child in sorted(artifact.iterdir()):
            if child.is_dir() and (child / "profile_export.jsonl").is_file():
                jsonl_path = child / "profile_export.jsonl"
                server_metrics_path = child / "server_metrics_export.json"
                break

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return 1

    records = load_jsonl_records(jsonl_path)
    server_metrics = load_server_metrics(server_metrics_path)
    t0_ns = first_update_ns(server_metrics)

    starts_ns = [
        int(r["metadata"]["request_start_ns"])
        for r in records
        if r.get("metadata", {}).get("request_start_ns")
    ]
    first_record_start = min(starts_ns) if starts_ns else 0
    request_times_s = [(s - first_record_start) / 1e9 for s in starts_ns]

    ttfts_ms: list[float] = []
    e2es_ms: list[float] = []
    interactivities: list[float] = []
    for r in records:
        ttft = metric_value(r, "time_to_first_token")
        e2e = metric_value(r, "request_latency")
        itl = metric_value(r, "inter_token_latency")
        ttfts_ms.append(ttft if ttft is not None else 0.0)
        e2es_ms.append(e2e if e2e is not None else 0.0)
        # Interactivity: tokens/sec from per-token latency (ms).
        interactivities.append(1000.0 / itl if itl and itl > 0 else 0.0)

    fig, axes = plt.subplots(6, 2, figsize=(14, 24))
    fig.suptitle("LLM Server Metrics During Benchmark", fontsize=14)

    panel_kv_cache_usage(axes[0, 0], server_metrics, t0_ns)
    panel_queue_depth(axes[0, 1], server_metrics, t0_ns)
    panel_prefix_cache_hit_rate(axes[1, 0], server_metrics, t0_ns)
    panel_throughput(axes[1, 1], server_metrics, t0_ns)
    panel_kv_offload_transfer_rate(axes[2, 0], server_metrics, t0_ns)
    panel_prefill_source_breakdown(axes[2, 1], server_metrics, t0_ns)
    panel_kv_offload_cumulative(
        axes[3, 0],
        server_metrics,
        Metric.KV_OFFLOAD_G2C,
        "KV Offload: GPU → CPU (Cumulative)",
        "b",
        t0_ns,
    )
    panel_kv_offload_cumulative(
        axes[3, 1],
        server_metrics,
        Metric.KV_OFFLOAD_C2G,
        "KV Offload: CPU → GPU (Cumulative)",
        "r",
        t0_ns,
    )
    panel_per_record_metric(
        axes[4, 0],
        request_times_s,
        ttfts_ms,
        color="blue",
        ylabel="TTFT (ms)",
        title="Time to First Token vs Time",
    )
    panel_per_record_metric(
        axes[4, 1],
        request_times_s,
        e2es_ms,
        color="green",
        ylabel="Latency (ms)",
        title="Request Latency vs Time",
    )
    panel_per_record_metric(
        axes[5, 0],
        request_times_s,
        interactivities,
        color="purple",
        ylabel="Interactivity (tokens/sec)",
        title="Decode Speed (1/TPOT) vs Time",
    )
    panel_preemptions(axes[5, 1], server_metrics, t0_ns)

    plt.tight_layout()
    out_path = args.output or (args.result_dir / "metrics_plots.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    if records:
        ttft_clean = [v for v in ttfts_ms if v > 0]
        e2e_clean = [v for v in e2es_ms if v > 0]
        if ttft_clean and e2e_clean:
            print(
                f"  Records: {len(records)} | "
                f"TTFT median {statistics.median(ttft_clean):.0f}ms | "
                f"E2E median {statistics.median(e2e_clean):.0f}ms"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
