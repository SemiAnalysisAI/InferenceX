#!/usr/bin/env python3
"""Issue the same long prompt twice and report incremental vLLM cache hits."""

from __future__ import annotations

import json
import re
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
METRIC_NAMES = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:gpu_cache_usage_perc",
)


def metrics() -> dict[str, float]:
    text = urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=30).read().decode()
    values: dict[str, float] = {}
    for name in METRIC_NAMES:
        match = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+(\S+)$", text, re.M)
        if match:
            values[name] = float(match.group(1))
    return values


def completion(prompt: str) -> None:
    body = json.dumps(
        {
            "model": "moonshotai/Kimi-K3",
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    if not payload.get("choices"):
        raise RuntimeError(f"completion returned no choices: {payload}")


prompt = "prefix-cache-probe token sequence " * 8192
snapshots = [metrics()]
completion(prompt)
snapshots.append(metrics())
completion(prompt)
snapshots.append(metrics())

for index in (1, 2):
    before, after = snapshots[index - 1], snapshots[index]
    queries = (
        after["vllm:prefix_cache_queries_total"]
        - before["vllm:prefix_cache_queries_total"]
    )
    hits = (
        after["vllm:prefix_cache_hits_total"]
        - before["vllm:prefix_cache_hits_total"]
    )
    rate = hits / queries if queries else 0.0
    print(
        json.dumps(
            {
                "request": index,
                "query_tokens": queries,
                "hit_tokens": hits,
                "hit_rate": rate,
                "gpu_cache_usage": after.get("vllm:gpu_cache_usage_perc"),
            },
            sort_keys=True,
        )
    )
