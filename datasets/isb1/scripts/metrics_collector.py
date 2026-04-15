#!/usr/bin/env python3
"""Prometheus metrics scraper for ISB1 KV stress benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import signal
import statistics
import time
from pathlib import Path
from typing import Dict
from urllib.request import Request, urlopen

PROM_LINE_RE = re.compile(
    r"^\s*([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*$"
)

CANONICAL_METRICS: dict[str, tuple[str, ...]] = {
    # Required vLLM metrics
    "vllm:gpu_cache_usage_perc": (
        "vllm:gpu_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
    ),
    "vllm:cpu_cache_usage_perc": (
        "vllm:cpu_cache_usage_perc",
        "vllm_cpu_cache_usage_perc",
    ),
    "vllm:num_preemptions_total": (
        "vllm:num_preemptions_total",
        "vllm_num_preemptions_total",
    ),
    "vllm:num_requests_running": (
        "vllm:num_requests_running",
        "vllm_num_requests_running",
    ),
    "vllm:num_requests_waiting": (
        "vllm:num_requests_waiting",
        "vllm_num_requests_waiting",
    ),
    "vllm:kv_offload_bytes_gpu_to_cpu": (
        "vllm:kv_offload_bytes_gpu_to_cpu",
        "vllm_kv_offload_bytes_gpu_to_cpu",
    ),
    "vllm:kv_offload_bytes_cpu_to_gpu": (
        "vllm:kv_offload_bytes_cpu_to_gpu",
        "vllm_kv_offload_bytes_cpu_to_gpu",
    ),
    "vllm:prompt_tokens_total": (
        "vllm:prompt_tokens_total",
        "vllm_prompt_tokens_total",
    ),
    "vllm:generation_tokens_total": (
        "vllm:generation_tokens_total",
        "vllm_generation_tokens_total",
    ),
    # Optional but useful in vLLM
    "vllm:num_requests_swapped": (
        "vllm:num_requests_swapped",
        "vllm_num_requests_swapped",
    ),
    # PR #993 parity metrics (vLLM)
    "vllm:prefix_cache_hit_rate": (
        "vllm:prefix_cache_hit_rate",
        "vllm_prefix_cache_hit_rate",
    ),
    "vllm:cpu_prefix_cache_hit_rate": (
        "vllm:cpu_prefix_cache_hit_rate",
        "vllm_cpu_prefix_cache_hit_rate",
    ),
    "vllm:kv_offload_time_gpu_to_cpu_seconds": (
        "vllm:kv_offload_time_gpu_to_cpu_seconds",
        "vllm_kv_offload_time_gpu_to_cpu_seconds",
    ),
    "vllm:kv_offload_time_cpu_to_gpu_seconds": (
        "vllm:kv_offload_time_cpu_to_gpu_seconds",
        "vllm_kv_offload_time_cpu_to_gpu_seconds",
    ),
    "vllm:prompt_tokens_local_compute": (
        "vllm:prompt_tokens_local_compute",
        "vllm_prompt_tokens_local_compute",
    ),
    "vllm:prompt_tokens_local_cache_hit": (
        "vllm:prompt_tokens_local_cache_hit",
        "vllm_prompt_tokens_local_cache_hit",
    ),
    "vllm:prompt_tokens_external_kv_transfer": (
        "vllm:prompt_tokens_external_kv_transfer",
        "vllm_prompt_tokens_external_kv_transfer",
    ),
    # SGLang equivalents (best-effort)
    "sglang:kv_cache_usage": (
        "sglang:kv_cache_usage",
        "sglang_kv_cache_usage",
        "sglang_kv_cache_utilization",
    ),
    "sglang:cache_hit_rate": (
        "sglang:cache_hit_rate",
        "sglang_cache_hit_rate",
        "sglang_radix_cache_hit_rate",
    ),
    "sglang:num_requests_running": (
        "sglang:num_requests_running",
        "sglang_num_requests_running",
        "sglang_scheduler_num_running_requests",
    ),
    "sglang:num_requests_waiting": (
        "sglang:num_requests_waiting",
        "sglang_num_requests_waiting",
        "sglang_scheduler_num_waiting_requests",
    ),
    "sglang:prompt_tokens_total": (
        "sglang:prompt_tokens_total",
        "sglang_prompt_tokens_total",
        "sglang_num_prompt_tokens_total",
    ),
    "sglang:generation_tokens_total": (
        "sglang:generation_tokens_total",
        "sglang_generation_tokens_total",
        "sglang_num_generation_tokens_total",
    ),
    # PR #993 parity metrics (SGLang)
    "sglang:num_preemptions_total": (
        "sglang:num_preemptions_total",
        "sglang_num_preemptions_total",
    ),
    "sglang:prefix_cache_queries_total": (
        "sglang:prefix_cache_queries_total",
        "sglang_prefix_cache_queries_total",
    ),
}


def _normalize_name(name: str) -> str:
    return name.replace(":", "_")


def parse_prometheus_rows(payload: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROM_LINE_RE.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        try:
            rows.append((name, float(raw_value)))
        except ValueError:
            continue
    return rows


def parse_prometheus_text(payload: str) -> Dict[str, float]:
    samples: Dict[str, float] = {}
    for name, value in parse_prometheus_rows(payload):
        samples[name] = value
    return samples


def map_canonical_metrics(samples: Dict[str, float]) -> Dict[str, float]:
    mapped: Dict[str, float] = {}

    normalized_index: Dict[str, float] = {}
    for key, value in samples.items():
        normalized_index[_normalize_name(key)] = value

    for canonical_name, aliases in CANONICAL_METRICS.items():
        value = None
        for alias in aliases:
            if alias in samples:
                value = samples[alias]
                break
            alias_norm = _normalize_name(alias)
            if alias_norm in normalized_index:
                value = normalized_index[alias_norm]
                break
        if value is not None:
            mapped[canonical_name] = value

    return mapped


def fetch_metrics(metrics_url: str, timeout_s: float = 5.0) -> str:
    request = Request(metrics_url, headers={"Accept": "text/plain"})
    with urlopen(request, timeout=timeout_s) as response:  # nosec B310
        return response.read().decode("utf-8", errors="replace")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _build_summary(metric_values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric_name, values in metric_values.items():
        if not values:
            continue
        summary[metric_name] = {
            "count": float(len(values)),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "p50": _percentile(values, 0.50),
            "p99": _percentile(values, 0.99),
        }
    return summary


async def scrape_loop(
    metrics_url: str,
    output_path: Path,
    interval_s: float,
    duration_s: float,
    wide: bool,
    summary_json_path: Path | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stop_event = asyncio.Event()

    def _request_stop(*_: object) -> None:
        stop_event.set()

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _request_stop)
        loop.add_signal_handler(signal.SIGTERM, _request_stop)
    except NotImplementedError:
        pass

    started_at = time.time()
    metric_values: dict[str, list[float]] = {}

    wide_path = output_path.with_name("kv_metrics_wide.csv")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "metric_name", "metric_value"])

        wide_file = None
        wide_writer = None
        if wide:
            wide_file = wide_path.open("w", newline="", encoding="utf-8")
            wide_writer = csv.writer(wide_file)
            wide_writer.writerow(["timestamp", "metric_name", "metric_value"])

        try:
            while not stop_event.is_set():
                now = time.time()
                if duration_s > 0 and (now - started_at) >= duration_s:
                    break

                try:
                    raw_text = await asyncio.to_thread(fetch_metrics, metrics_url)
                    raw_rows = parse_prometheus_rows(raw_text)
                    samples = parse_prometheus_text(raw_text)
                    mapped = map_canonical_metrics(samples)

                    if wide_writer is not None:
                        for raw_metric_name, raw_metric_value in raw_rows:
                            wide_writer.writerow(
                                [f"{now:.3f}", raw_metric_name, f"{raw_metric_value:.8f}"]
                            )
                        wide_file.flush()

                    for metric_name, metric_value in mapped.items():
                        writer.writerow([f"{now:.3f}", metric_name, f"{metric_value:.8f}"])
                        metric_values.setdefault(metric_name, []).append(metric_value)
                    f.flush()
                except Exception as exc:
                    writer.writerow([f"{now:.3f}", "collector:error", repr(exc)])
                    f.flush()

                await asyncio.sleep(interval_s)
        finally:
            if wide_file is not None:
                wide_file.close()

    if summary_json_path is not None:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(
            json.dumps(_build_summary(metric_values), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Prometheus metrics into CSV")
    parser.add_argument(
        "--metrics-url",
        default="http://0.0.0.0:8888/metrics",
        help="Prometheus endpoint URL",
    )
    parser.add_argument(
        "--output",
        default="kv_metrics.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Scrape interval in seconds",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional max duration in seconds (0 means run until interrupted)",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Also scrape all non-comment Prometheus metric lines into kv_metrics_wide.csv",
    )
    parser.add_argument(
        "--summary-json",
        nargs="?",
        const="kv_metrics_summary.json",
        default=None,
        help="Write per-metric min/max/mean/p50/p99 summary JSON (default: kv_metrics_summary.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_json_path = Path(args.summary_json) if args.summary_json else None
    asyncio.run(
        scrape_loop(
            metrics_url=args.metrics_url,
            output_path=Path(args.output),
            interval_s=max(args.interval, 0.1),
            duration_s=max(args.duration, 0.0),
            wide=args.wide,
            summary_json_path=summary_json_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
