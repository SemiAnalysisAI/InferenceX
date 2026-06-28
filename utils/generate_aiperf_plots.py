#!/usr/bin/env python3
"""Generate metrics_plots.png matching kv-cache-tester's 6x2 layout.

Reads aiperf's per-record JSONL + server-metrics JSON (with timeslices
enabled via ``--slice-duration``) and emits a PNG with the same panels
the legacy kv-cache-tester pipeline produced. The launchers feed this
$RESULT_DIR after each run so downstream tooling and humans see the
same visual.

Backend-aware: server-metric panels resolve both the vLLM (``vllm:``) and
SGLang (``sglang:``) Prometheus names, mirroring aiperf's own accumulator
mapping. The suptitle reflects the detected backend. Panels with no
backend equivalent (KV-offload bytes, prompt-source 3-way split for SGLang,
etc.) render empty rather than erroring.

Layout (6 rows x 2 cols, suptitle "<Backend> Server Metrics During Benchmark"):
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


def metric_entry(server_metrics: dict, name: str) -> dict | None:
    metrics = server_metrics.get("metrics") or {}
    entry = metrics.get(name)
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


def aggregate_timeseries(
    server_metrics: dict, name: str, t0_ns: int | None,
    *,
    aggregator=sum,
    value_key_priority=("avg", "rate", "total", "max"),
) -> tuple[list[float], list[float]]:
    """Aggregate timeslices across every series of a metric (sums by default)."""
    entry = metric_entry(server_metrics, name)
    if entry is None or t0_ns is None:
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


def detect_backend(server_metrics: dict) -> str | None:
    """Infer the inference backend from the exported metric-name prefixes.

    aiperf keys ``server_metrics["metrics"]`` by the raw Prometheus family
    name, so the prefix (``vllm:`` vs ``sglang:``) identifies the server.
    """
    metrics = server_metrics.get("metrics") or {}
    n_sgl = sum(1 for k in metrics if k.startswith("sglang:"))
    n_vllm = sum(1 for k in metrics if k.startswith("vllm:"))
    if not n_sgl and not n_vllm:
        return None
    return "sglang" if n_sgl >= n_vllm and n_sgl else "vllm"


def aggregate_timeseries_any(
    server_metrics: dict, names, t0_ns: int | None, **kwargs
) -> tuple[list[float], list[float]]:
    """aggregate_timeseries over the first candidate name that yields data.

    Lets a panel accept both the vLLM and SGLang spelling of a metric without
    the caller knowing which backend produced the export.
    """
    for name in names:
        times, values = aggregate_timeseries(server_metrics, name, t0_ns, **kwargs)
        if times:
            return times, values
    return [], []


def _sglang_host_usage_pct(
    server_metrics: dict, t0_ns: int | None
) -> tuple[list[float], list[float]]:
    """SGLang HiCache CPU (L2) tier fill = host_used / host_total, as a fraction.

    SGLang has no ``cpu_cache_usage_perc`` gauge; it exposes the host pool's
    used/total token counts instead (HiCache-enabled runs only).
    """
    ut, uv = aggregate_timeseries(
        server_metrics, "sglang:hicache_host_used_tokens", t0_ns, aggregator=max
    )
    _, tv = aggregate_timeseries(
        server_metrics, "sglang:hicache_host_total_tokens", t0_ns, aggregator=max
    )
    if not ut:
        return [], []
    total_cap = max(tv) if tv else 0.0
    if total_cap <= 0:
        return [], []
    return ut, [u / total_cap for u in uv]


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
    # GPU HBM KV usage: vLLM gauge (v1/v0 spellings) or SGLang token_usage.
    times, values = aggregate_timeseries_any(
        server_metrics,
        ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage"),
        t0_ns,
        aggregator=max,
    )
    # CPU/external tier: vLLM offload gauge, else SGLang HiCache host ratio.
    cpu_times, cpu_values = aggregate_timeseries_any(
        server_metrics,
        ("vllm:cpu_kv_cache_usage_perc", "vllm:cpu_cache_usage_perc"),
        t0_ns,
        aggregator=max,
    )
    if not cpu_values:
        cpu_times, cpu_values = _sglang_host_usage_pct(server_metrics, t0_ns)

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
    rt, rv = aggregate_timeseries_any(
        server_metrics,
        ("vllm:num_requests_running", "sglang:num_running_reqs"),
        t0_ns,
        aggregator=max,
    )
    wt, wv = aggregate_timeseries_any(
        server_metrics,
        ("vllm:num_requests_waiting", "sglang:num_queue_reqs"),
        t0_ns,
        aggregator=max,
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
    hits_name: str,
    queries_name: str,
    t0_ns: int | None,
) -> tuple[list[float], list[float]]:
    """Compute per-interval hit rates from cumulative counters' deltas."""
    ht, hv = aggregate_timeseries(
        server_metrics, hits_name, t0_ns, value_key_priority=("total",)
    )
    qt, qv = aggregate_timeseries(
        server_metrics, queries_name, t0_ns, value_key_priority=("total",)
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
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_queries",
        t0_ns,
    )
    ext_t, ext_r = _hit_rate_intervals(
        server_metrics,
        "vllm:external_prefix_cache_hits",
        "vllm:external_prefix_cache_queries",
        t0_ns,
    )
    # SGLang folds radix (HBM) + HiCache (host) hits into one counter pair;
    # there is no GPU/external split. Prefer the cumulative counter deltas
    # (cached_tokens / prompt_tokens) over the per-batch cache_hit_rate gauge,
    # which reads 0 between requests during low-concurrency agentic replay.
    sgl_t, sgl_r = ([], [])
    if not gpu_t and not ext_t:
        sgl_t, sgl_r = _hit_rate_intervals(
            server_metrics, "sglang:cached_tokens", "sglang:prompt_tokens", t0_ns
        )
        if not sgl_t:
            gt, gv = aggregate_timeseries(
                server_metrics, "sglang:cache_hit_rate", t0_ns, aggregator=max
            )
            if gt:
                sgl_t = gt
                sgl_r = [v * 100.0 if 0 <= v <= 1.0 else v for v in gv]
    if sgl_t:
        ax.scatter(sgl_t, sgl_r, alpha=0.3, s=5, c="green", label="SGLang (radix+host)")
        win = rolling_window(len(sgl_r))
        if win > 1:
            ax.plot(
                sgl_t,
                rolling_average(sgl_r, win),
                "green",
                linewidth=2,
                label=f"SGLang avg (n={win})",
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
    if gpu_t or has_ext or sgl_t:
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Prefix Cache Hit Rate Per Interval (tokens hit / tokens queried)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)


def panel_throughput(ax, server_metrics: dict, t0_ns: int | None) -> None:
    gen_t, gen_v = aggregate_timeseries_any(
        server_metrics,
        ("vllm:generation_tokens", "sglang:generation_tokens"),
        t0_ns,
        value_key_priority=("rate",),
    )
    prompt_t, prompt_v = aggregate_timeseries_any(
        server_metrics,
        ("vllm:prompt_tokens", "sglang:prompt_tokens"),
        t0_ns,
        value_key_priority=("rate",),
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


def _offload_direction(
    server_metrics: dict, t0_ns: int | None, direction: str
) -> tuple[list[float], list[float], list[float], str | None]:
    """Per-direction KV-offload series, backend-aware.

    ``direction`` is ``"gpu_to_cpu"`` or ``"cpu_to_gpu"``. Returns
    ``(times, rates, cumulative, unit)`` where ``unit`` is ``"bytes"`` (vLLM)
    or ``"tok"`` (SGLang); ``([], [], [], None)`` when neither is present.

    - vLLM: ``vllm:kv_offload_bytes_<dir>`` counter (bytes; per-slice total +
      rate).
    - SGLang exposes no offload byte counter; the equivalents are in tokens:
        * cpu_to_gpu -> ``sglang:load_back_tokens`` (host-tier prefix loaded
          back into GPU HBM).
        * gpu_to_cpu -> positive deltas of ``sglang:hicache_host_used_tokens``
          (net writes into the host pool; approximate — eviction lowers
          occupancy and is not separately re-counted).
    """
    name = f"vllm:kv_offload_bytes_{direction}"
    t, totals = aggregate_timeseries(
        server_metrics, name, t0_ns, value_key_priority=("total",)
    )
    if t and any(v > 0 for v in totals):
        _, rates = aggregate_timeseries(
            server_metrics, name, t0_ns, value_key_priority=("rate",)
        )
        cum, run = [], 0.0
        for v in totals:
            run += v
            cum.append(run)
        return t, (rates or totals), cum, "bytes"

    if direction == "cpu_to_gpu":
        t, totals = aggregate_timeseries(
            server_metrics,
            "sglang:load_back_tokens",
            t0_ns,
            value_key_priority=("total",),
        )
        if not t:
            return [], [], [], None
        _, rates = aggregate_timeseries(
            server_metrics,
            "sglang:load_back_tokens",
            t0_ns,
            value_key_priority=("rate",),
        )
        cum, run = [], 0.0
        for v in totals:
            run += v
            cum.append(run)
        return t, (rates or totals), cum, "tok"

    # gpu_to_cpu (SGLang): net writes inferred from host-pool occupancy growth.
    t, occ = aggregate_timeseries(
        server_metrics, "sglang:hicache_host_used_tokens", t0_ns, aggregator=max
    )
    if not t:
        return [], [], [], None
    rates: list[float] = []
    cum: list[float] = []
    run = 0.0
    prev: float | None = None
    prev_t: float | None = None
    for tt, v in zip(t, occ):
        if prev is None:
            rates.append(0.0)
            cum.append(0.0)
        else:
            delta = max(v - prev, 0.0)
            width = max(tt - prev_t, 1e-9)
            rates.append(delta / width)
            run += delta
            cum.append(run)
        prev, prev_t = v, tt
    return t, rates, cum, "tok"


def panel_kv_offload_transfer_rate(
    ax, server_metrics: dict, t0_ns: int | None
) -> None:
    g2c_t, g2c_r, _, g2c_unit = _offload_direction(server_metrics, t0_ns, "gpu_to_cpu")
    c2g_t, c2g_r, _, c2g_unit = _offload_direction(server_metrics, t0_ns, "cpu_to_gpu")
    unit = g2c_unit or c2g_unit

    def _scale(vs: list[float]) -> list[float]:
        return [v / 1e6 for v in vs] if unit == "bytes" else list(vs)

    has_data = (g2c_t and any(v > 0 for v in g2c_r)) or (
        c2g_t and any(v > 0 for v in c2g_r)
    )
    if has_data:
        if g2c_t:
            vs = _scale(g2c_r)
            ax.scatter(g2c_t, vs, alpha=0.15, s=3, c="blue")
            win = rolling_window(len(vs))
            if win > 1:
                ax.plot(
                    g2c_t,
                    rolling_average(vs, win),
                    "b-",
                    linewidth=1.5,
                    label=f"GPU→CPU (avg n={win})",
                )
            else:
                ax.plot(g2c_t, vs, "b-", linewidth=1, alpha=0.8, label="GPU→CPU")
        if c2g_t:
            vs = _scale(c2g_r)
            ax.scatter(c2g_t, vs, alpha=0.15, s=3, c="red")
            win = rolling_window(len(vs))
            if win > 1:
                ax.plot(
                    c2g_t,
                    rolling_average(vs, win),
                    "r-",
                    linewidth=1.5,
                    label=f"CPU→GPU (avg n={win})",
                )
            else:
                ax.plot(c2g_t, vs, "r-", linewidth=1, alpha=0.8, label="CPU→GPU")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Transfer Rate (tokens/s)" if unit == "tok" else "Transfer Rate (MB/s)")
    title = "KV Offload Transfer Rate"
    if unit == "tok":
        title += " (HiCache, tokens)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _prompt_token_source_series(
    server_metrics: dict, source_label: str, t0_ns: int | None
) -> tuple[list[float], list[float]]:
    """vllm:prompt_tokens_by_source has labels {source: local_compute|local_cache_hit|external_kv_transfer}."""
    entry = metric_entry(server_metrics, "vllm:prompt_tokens_by_source")
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
    # SGLang has no prompt_tokens_by_source breakdown. Derive a 2-way split
    # from the counter deltas: cache hit = cached_tokens, computed =
    # prompt_tokens - cached_tokens (radix + HiCache host folded together).
    sglang_mode = False
    if not (c_t or h_t or e_t):
        pt, pv = aggregate_timeseries(
            server_metrics, "sglang:prompt_tokens", t0_ns, value_key_priority=("total",)
        )
        ct, ctv = aggregate_timeseries(
            server_metrics, "sglang:cached_tokens", t0_ns, value_key_priority=("total",)
        )
        if pt:
            sglang_mode = True
            cached_at = dict(zip(ct, ctv))
            h_t, h_v, c_t, c_v = [], [], [], []
            for t, p in zip(pt, pv):
                cached = cached_at.get(t, 0.0)
                h_t.append(t)
                h_v.append(cached)
                c_t.append(t)
                c_v.append(max(p - cached, 0.0))
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
    labels = (
        ["Computed", "Cache Hit (radix+host)", "External"]
        if sglang_mode
        else ["Prefill", "HBM Cache Hit", "Offload Cache Hit"]
    )
    ax.stackplot(
        samples,
        pct_c,
        pct_h,
        pct_e,
        labels=labels,
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
    direction: str,
    title: str,
    color: str,
    t0_ns: int | None,
) -> None:
    times, _, cum, unit = _offload_direction(server_metrics, t0_ns, direction)
    if unit == "tok":
        ylabel, scale = "Cumulative (M tokens)", 1e6
    else:
        ylabel, scale = "Cumulative Transfer (GB)", 1e9
    if times and any(v > 0 for v in cum):
        yvals = [v / scale for v in cum]
        ax.plot(times, yvals, f"{color}-", linewidth=1.5)
        ax.fill_between(times, yvals, alpha=0.2, color=color)
    # SGLang's GPU→CPU is inferred from host-pool occupancy growth, not a
    # dedicated write counter — flag it as approximate.
    if unit == "tok" and direction == "gpu_to_cpu":
        title += " (approx)"
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
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
    # SGLang exposes the same concept as `num_retracted_reqs_total` (counter;
    # parser strips `_total`).
    times, values = aggregate_timeseries_any(
        server_metrics,
        ("vllm:num_preemptions", "sglang:num_retracted_reqs"),
        t0_ns,
        value_key_priority=("total",),
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

    backend = detect_backend(server_metrics)
    backend_label = {"sglang": "SGLang", "vllm": "vLLM"}.get(backend, "Server")
    fig, axes = plt.subplots(6, 2, figsize=(14, 24))
    fig.suptitle(f"{backend_label} Server Metrics During Benchmark", fontsize=14)

    panel_kv_cache_usage(axes[0, 0], server_metrics, t0_ns)
    panel_queue_depth(axes[0, 1], server_metrics, t0_ns)
    panel_prefix_cache_hit_rate(axes[1, 0], server_metrics, t0_ns)
    panel_throughput(axes[1, 1], server_metrics, t0_ns)
    panel_kv_offload_transfer_rate(axes[2, 0], server_metrics, t0_ns)
    panel_prefill_source_breakdown(axes[2, 1], server_metrics, t0_ns)
    panel_kv_offload_cumulative(
        axes[3, 0],
        server_metrics,
        "gpu_to_cpu",
        "KV Offload: GPU → CPU (Cumulative)",
        "b",
        t0_ns,
    )
    panel_kv_offload_cumulative(
        axes[3, 1],
        server_metrics,
        "cpu_to_gpu",
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
