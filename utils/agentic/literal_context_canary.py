#!/usr/bin/env python3
"""Exercise a serving endpoint near its advertised one-million-token limit."""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Any

import requests
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse literal-context canary arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=1_040_000)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def chat_token_count(tokenizer: Any, content: str) -> int:
    """Return the exact chat-template prompt length for one user message."""
    messages = [{"role": "user", "content": content}]
    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def build_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    """Build the largest deterministic chat prompt no longer than target_tokens."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    filler = "x\n"
    low = 0
    high = target_tokens
    best_content = ""
    best_count = chat_token_count(tokenizer, best_content)

    while low <= high:
        repeats = (low + high) // 2
        content = filler * repeats
        count = chat_token_count(tokenizer, content)
        if count <= target_tokens:
            best_content = content
            best_count = count
            low = repeats + 1
        else:
            high = repeats - 1

    minimum = target_tokens - max(1024, target_tokens // 1000)
    if best_count < minimum:
        raise RuntimeError(
            f"Could not construct a near-target prompt: {best_count} < {minimum}"
        )
    return best_content, best_count


def send_request(
    *,
    url: str,
    model: str,
    content: str,
    max_tokens: int,
    timeout_seconds: int,
    correlation_id: str,
) -> dict[str, Any]:
    """Send one non-streaming chat request and validate its response envelope."""
    response = requests.post(
        f"{url.rstrip('/')}/v1/chat/completions",
        headers={"X-Correlation-ID": correlation_id},
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    if not response.ok:
        raise RuntimeError(
            f"Canary request failed with HTTP {response.status_code}: "
            f"{response.text[:4096]}"
        )
    payload = response.json()
    if not payload.get("choices"):
        raise RuntimeError(f"Canary response has no choices: {json.dumps(payload)[:4096]}")
    return payload


def validate_usage(payload: dict[str, Any], expected_prompt_tokens: int) -> dict[str, Any]:
    """Require authoritative usage matching the locally rendered prompt."""
    usage = payload.get("usage") or {}
    observed = usage.get("prompt_tokens")
    if observed != expected_prompt_tokens:
        raise RuntimeError(
            "Server prompt length differs from the local chat template: "
            f"observed={observed}, expected={expected_prompt_tokens}"
        )
    choice = payload["choices"][0]
    return {
        "prompt_tokens": observed,
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "response_chars": len((choice.get("message") or {}).get("content") or ""),
    }


def main() -> None:
    """Run a full-length request followed by a long-prefix reuse request."""
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )
    content, prompt_tokens = build_prompt(tokenizer, args.target_prompt_tokens)
    correlation_id = f"literal-1m-canary-{uuid.uuid4()}"
    print(
        json.dumps(
            {
                "event": "literal_1m_canary_start",
                "prompt_tokens": prompt_tokens,
                "max_output_tokens": args.max_output_tokens,
            }
        ),
        flush=True,
    )

    first = send_request(
        url=args.url,
        model=args.model,
        content=content,
        max_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        correlation_id=correlation_id,
    )
    first_summary = validate_usage(first, prompt_tokens)

    second = send_request(
        url=args.url,
        model=args.model,
        content=content,
        max_tokens=16,
        timeout_seconds=args.timeout_seconds,
        correlation_id=correlation_id,
    )
    second_summary = validate_usage(second, prompt_tokens)
    print(
        json.dumps(
            {
                "event": "literal_1m_canary_pass",
                "first": first_summary,
                "prefix_reuse": second_summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
