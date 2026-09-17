#!/usr/bin/env python3
"""Replay nested Weka hash blocks with one output token per request."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
QUERY_METRIC = "vllm:prefix_cache_queries_total"
HIT_METRIC = "vllm:prefix_cache_hits_total"


def metric(name: str) -> float:
    text = urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=30).read().decode()
    match = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+(\S+)$", text, re.M)
    if match is None:
        raise RuntimeError(f"metric not found: {name}")
    return float(match.group(1))


def token_ids(hash_ids: list[int], block_size: int) -> list[int]:
    tokens: list[int] = []
    for hash_id in hash_ids:
        # One deterministic, distinct valid token ID per synthetic hash block.
        token_id = 1000 + (hash_id * 104729) % 120000
        tokens.extend([token_id] * block_size)
    return tokens


def completion(prompt_token_ids: list[int]) -> None:
    body = json.dumps(
        {
            "model": "moonshotai/Kimi-K3",
            "prompt": prompt_token_ids,
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


parser = argparse.ArgumentParser()
parser.add_argument("trace")
args = parser.parse_args()
trace = json.load(open(args.trace))
block_size = int(trace["block_size"])
requests = trace["requests"]

total_queries = 0.0
total_hits = 0.0
previous_hash_ids: list[int] = []

for index, item in enumerate(requests):
    hash_ids = item["hash_ids"]
    before_queries = metric(QUERY_METRIC)
    before_hits = metric(HIT_METRIC)
    completion(token_ids(hash_ids, block_size))
    queries = metric(QUERY_METRIC) - before_queries
    hits = metric(HIT_METRIC) - before_hits

    common_blocks = 0
    for previous, current in zip(previous_hash_ids, hash_ids):
        if previous != current:
            break
        common_blocks += 1

    # The first request is the cold seed. Aggregate only reuse requests.
    if index:
        total_queries += queries
        total_hits += hits
    print(
        json.dumps(
            {
                "request": index,
                "query_tokens": queries,
                "hit_tokens": hits,
                "actual_hit_rate": hits / queries if queries else 0.0,
                "trace_prefix_rate": common_blocks / len(hash_ids),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    previous_hash_ids = hash_ids

print(
    json.dumps(
        {
            "reuse_requests": len(requests) - 1,
            "query_tokens": total_queries,
            "hit_tokens": total_hits,
            "weighted_hit_rate": total_hits / total_queries if total_queries else 0.0,
        },
        sort_keys=True,
    )
)
