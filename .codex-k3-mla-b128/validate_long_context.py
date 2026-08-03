#!/usr/bin/env python3
"""Exercise Kimi-K3 FP8 KV with eight distinct 380K-token requests."""

from __future__ import annotations

import concurrent.futures
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any


BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888"
MODEL = "moonshotai/Kimi-K3"
CONCURRENCY = 8
PROMPT_TOKENS = 380_000
HTTP_TIMEOUT_SECONDS = 900


def post_json(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        return error.code, {"error": body}


long_barrier = threading.Barrier(CONCURRENCY)


def run_long_request(index: int) -> dict[str, Any]:
    # The first token differs so prefix caching cannot collapse the eight
    # contexts into one shared prefix. Token 22931 decodes as "hello".
    prompt = [index + 1] + [22931] * (PROMPT_TOKENS - 1)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 0,
    }
    long_barrier.wait()
    started = time.monotonic()
    try:
        status, response = post_json("/v1/completions", payload)
        choice = (response.get("choices") or [{}])[0]
        usage = response.get("usage") or {}
        return {
            "index": index,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "text": choice.get("text"),
            "finish_reason": choice.get("finish_reason"),
            "error": response.get("error"),
        }
    except Exception as error:  # noqa: BLE001 - preserve every client failure
        return {
            "index": index,
            "status": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "error": repr(error),
        }


chat_barrier = threading.Barrier(CONCURRENCY)


def run_semantic_request(index: int) -> dict[str, Any]:
    marker = f"VALID-{index}"
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Reply with exactly {marker} and nothing else.",
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "seed": 0,
    }
    chat_barrier.wait()
    started = time.monotonic()
    try:
        status, response = post_json("/v1/chat/completions", payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        return {
            "index": index,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "content": content,
            "reasoning_content": reasoning,
            "marker_found": marker in content or marker in reasoning,
            "finish_reason": choice.get("finish_reason"),
            "error": response.get("error"),
        }
    except Exception as error:  # noqa: BLE001 - preserve every client failure
        return {
            "index": index,
            "status": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "marker_found": False,
            "error": repr(error),
        }


def run_concurrently(function: Any) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        results = list(executor.map(function, range(CONCURRENCY)))
    return sorted(results, key=lambda result: result["index"])


long_results = run_concurrently(run_long_request)
long_ok = all(
    result.get("status") == 200
    and result.get("prompt_tokens") == PROMPT_TOKENS
    and result.get("completion_tokens") == 1
    and result.get("text") == "hello"
    and not result.get("error")
    for result in long_results
)

semantic_results = run_concurrently(run_semantic_request) if long_ok else []
semantic_ok = bool(semantic_results) and all(
    result.get("status") == 200
    and result.get("marker_found") is True
    and not result.get("error")
    for result in semantic_results
)

print(
    json.dumps(
        {
            "long_context": long_results,
            "long_context_passed": long_ok,
            "semantic": semantic_results,
            "semantic_passed": semantic_ok,
        },
        indent=2,
        ensure_ascii=False,
    )
)
raise SystemExit(0 if long_ok and semantic_ok else 1)
