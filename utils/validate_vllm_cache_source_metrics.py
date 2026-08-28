#!/usr/bin/env python3
"""Exercise and validate vLLM cached-prompt token source metrics.

The target server must run with prefix caching and ``VLLM_SERVER_DEV_MODE=1``.
For an offload tier, use an eager SimpleCPUOffloadConnector so resetting the
local prefix cache leaves a copy available in the connector-managed cache.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


_SAMPLE_RE = re.compile(
    r'^(?P<name>[^\s{]+)(?:\{(?P<labels>.*)\})?\s+(?P<value>[-+0-9.eE]+)$'
)
_CACHED_TOTAL = "vllm:prompt_tokens_cached_total"
_CACHED_BY_SOURCE = "vllm:prompt_tokens_cached_by_source_total"
_PROMPT_BY_SOURCE = "vllm:prompt_tokens_by_source_total"


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    cached_total: float
    cached_by_source: dict[str, float]
    prompt_by_source: dict[str, float]

    def delta(self, previous: "Snapshot") -> "Snapshot":
        return Snapshot(
            cached_total=self.cached_total - previous.cached_total,
            cached_by_source=_dict_delta(self.cached_by_source, previous.cached_by_source),
            prompt_by_source=_dict_delta(self.prompt_by_source, previous.prompt_by_source),
        )


def _dict_delta(current: Mapping[str, float], previous: Mapping[str, float]) -> dict[str, float]:
    return {
        key: current.get(key, 0.0) - previous.get(key, 0.0)
        for key in current.keys() | previous.keys()
        if not math.isclose(current.get(key, 0.0), previous.get(key, 0.0))
    }


def _request(
    base_url: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None or path.startswith("/reset_") else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise ValidationError(f"{request.method} {path} failed: HTTP {error.code}: {detail}") from error


def _json_request(
    base_url: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    value = json.loads(_request(base_url, path, body=body, timeout=timeout))
    if not isinstance(value, dict):
        raise ValidationError(f"{path} returned a non-object JSON value")
    return value


def _parse_labels(raw_labels: str | None) -> dict[str, str]:
    if raw_labels is None:
        return {}

    labels: dict[str, str] = {}
    cursor = 0
    length = len(raw_labels)

    def fail(message: str) -> ValidationError:
        return ValidationError(
            f"invalid Prometheus labels at offset {cursor}: {message}: {raw_labels!r}"
        )

    while cursor < length:
        while cursor < length and raw_labels[cursor] in " \t":
            cursor += 1
        key_start = cursor
        if cursor >= length or not (raw_labels[cursor].isalpha() or raw_labels[cursor] == "_"):
            raise fail("expected label name")
        cursor += 1
        while cursor < length and (
            raw_labels[cursor].isalnum() or raw_labels[cursor] == "_"
        ):
            cursor += 1
        key = raw_labels[key_start:cursor]

        if cursor >= length or raw_labels[cursor] != "=":
            raise fail("expected '='")
        cursor += 1
        if cursor >= length or raw_labels[cursor] != '"':
            raise fail("expected opening quote")
        cursor += 1

        value: list[str] = []
        while cursor < length and raw_labels[cursor] != '"':
            character = raw_labels[cursor]
            cursor += 1
            if character != "\\":
                value.append(character)
                continue
            if cursor >= length:
                raise fail("unterminated escape")
            escaped = raw_labels[cursor]
            cursor += 1
            if escaped == "n":
                value.append("\n")
            elif escaped in {'"', "\\"}:
                value.append(escaped)
            else:
                raise fail(f"unsupported escape {escaped!r}")

        if cursor >= length:
            raise fail("unterminated quoted value")
        cursor += 1
        labels[key] = "".join(value)

        while cursor < length and raw_labels[cursor] in " \t":
            cursor += 1
        if cursor == length:
            break
        if raw_labels[cursor] != ",":
            raise fail("expected ','")
        cursor += 1
        if cursor == length:
            raise fail("trailing ','")

    return labels


def snapshot(base_url: str) -> Snapshot:
    values: dict[str, list[tuple[dict[str, str], float]]] = {}
    text = _request(base_url, "/metrics", timeout=30.0).decode()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name not in {_CACHED_TOTAL, _CACHED_BY_SOURCE, _PROMPT_BY_SOURCE}:
            continue
        values.setdefault(name, []).append(
            (_parse_labels(match.group("labels")), float(match.group("value")))
        )

    def total(name: str) -> float:
        return sum(value for _, value in values.get(name, []))

    def by_source(name: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for labels, value in values.get(name, []):
            source = labels.get("source")
            if source is not None:
                result[source] = result.get(source, 0.0) + value
        return result

    return Snapshot(
        cached_total=total(_CACHED_TOTAL),
        cached_by_source=by_source(_CACHED_BY_SOURCE),
        prompt_by_source=by_source(_PROMPT_BY_SOURCE),
    )


def completion(base_url: str, model: str, token_ids: list[int]) -> dict[str, Any]:
    response = _json_request(
        base_url,
        "/v1/completions",
        body={
            "model": model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "ignore_eos": True,
        },
        timeout=300.0,
    )
    usage = response.get("usage")
    if not isinstance(usage, dict) or usage.get("prompt_tokens") != len(token_ids):
        raise ValidationError(
            f"completion usage did not report the exact {len(token_ids)}-token prompt: {usage!r}"
        )
    return response


def tokenized_text_prompt(base_url: str, model: str, minimum_tokens: int) -> list[int]:
    seed = (
        "A cache validation sentence with ordinary words and punctuation. "
        * (minimum_tokens // 4 + 1)
    )
    response = _json_request(
        base_url,
        "/tokenize",
        body={"model": model, "prompt": seed},
        timeout=120.0,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise ValidationError(f"/tokenize returned invalid tokens: {type(tokens).__name__}")
    if len(tokens) < minimum_tokens:
        raise ValidationError(
            f"tokenizer returned only {len(tokens)} tokens; need {minimum_tokens}"
        )
    return tokens


def wait_for_accounting(
    base_url: str,
    previous: Snapshot,
    prompt_tokens: int,
    timeout: float = 15.0,
) -> Snapshot:
    deadline = time.monotonic() + timeout
    latest = snapshot(base_url)
    while time.monotonic() < deadline:
        logical_delta = sum(latest.delta(previous).prompt_by_source.values())
        if logical_delta >= prompt_tokens:
            return latest
        time.sleep(0.1)
        latest = snapshot(base_url)
    raise ValidationError(
        "metrics logger did not account for the completed request within "
        f"{timeout:g}s; observed logical delta {latest.delta(previous).prompt_by_source}"
    )


def reset_local_cache(base_url: str, attempts: int = 30) -> None:
    for _ in range(attempts):
        response = _json_request(base_url, "/reset_prefix_cache", body={})
        if response.get("success") is True:
            return
        time.sleep(1)
    raise ValidationError("local prefix-cache reset never succeeded")


def drain_async_transfers(base_url: str, model: str, previous: Snapshot) -> Snapshot:
    """Give connector workers time, then run one scheduler step to reap them."""
    time.sleep(2)
    completion(base_url, model, [102])
    wait_for_accounting(base_url, previous, 1)
    time.sleep(1)
    return snapshot(base_url)


def _assert_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=0.001):
        raise ValidationError(f"{message}: expected {expected}, observed {actual}")


def validate_delta(name: str, delta: Snapshot, expected_sources: set[str]) -> None:
    physical_total = sum(delta.cached_by_source.values())
    logical_total = sum(delta.prompt_by_source.values())
    _assert_close(
        physical_total,
        delta.cached_total,
        f"{name}: physical source sum must equal cached-token counter",
    )
    if logical_total:
        cached_logical = (
            delta.prompt_by_source.get("local_cache_hit", 0.0)
            + delta.prompt_by_source.get("external_kv_transfer", 0.0)
        )
        _assert_close(
            cached_logical,
            delta.cached_total,
            f"{name}: logical cache-hit sum must equal cached-token counter",
        )
    observed_sources = {
        source for source, value in delta.cached_by_source.items() if value > 0
    }
    if observed_sources != expected_sources:
        raise ValidationError(
            f"{name}: expected positive sources {sorted(expected_sources)}, "
            f"observed {delta.cached_by_source}"
        )
    print(json.dumps({"case": name, "delta": delta.__dict__}, sort_keys=True))


def run(
    base_url: str,
    expected_tier: str | None,
    short_tokens: int,
    long_tokens: int,
    prompt_mode: str,
) -> None:
    models = _json_request(base_url, "/v1/models").get("data")
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise ValidationError("/v1/models returned no model")
    model = models[0].get("id")
    if not isinstance(model, str):
        raise ValidationError("/v1/models returned an invalid model id")

    # The short prompt is a block-aligned prefix of the long prompt. Direct
    # token IDs give the most deterministic probe; tokenizer mode provides a
    # realistic-input cross-check for hybrid or multimodal model families.
    if prompt_mode == "text":
        token_ids = tokenized_text_prompt(base_url, model, long_tokens)
        long_prompt = token_ids[:long_tokens]
        short_prompt = long_prompt[:short_tokens]
    else:
        short_prompt = [100] * short_tokens
        long_prompt = short_prompt + [101] * (long_tokens - short_tokens)

    baseline = snapshot(base_url)
    completion(base_url, model, long_prompt)
    after_cold = wait_for_accounting(base_url, baseline, len(long_prompt))
    validate_delta("cold_miss", after_cold.delta(baseline), set())

    completion(base_url, model, long_prompt)
    after_device = wait_for_accounting(base_url, after_cold, len(long_prompt))
    validate_delta("exact_device_hit", after_device.delta(after_cold), {"device"})

    if expected_tier is None:
        reset_local_cache(base_url)
        completion(base_url, model, long_prompt)
        after_reset = wait_for_accounting(base_url, after_device, len(long_prompt))
        validate_delta("cold_after_local_reset", after_reset.delta(after_device), set())
        return

    # Keep the connector-managed cache populated while dropping local block
    # references. The first reload must come entirely from the second tier.
    if expected_tier in {"cpu", "disk"}:
        after_device = drain_async_transfers(base_url, model, after_device)
    reset_local_cache(base_url)
    completion(base_url, model, long_prompt)
    after_tier = wait_for_accounting(base_url, after_device, len(long_prompt))
    validate_delta("exact_second_tier_hit", after_tier.delta(after_device), {expected_tier})

    # The reference ExampleConnector addresses whole-request objects rather
    # than individual prefix blocks. It validates generic KV-transfer source
    # attribution, but cannot construct a partial external-prefix hit.
    if expected_tier == "external":
        return

    # Restore only the short prefix to the device, while the connector still
    # retains the full long prompt. A long request should consume both tiers.
    if expected_tier in {"cpu", "disk"}:
        after_tier = drain_async_transfers(base_url, model, after_tier)
    reset_local_cache(base_url)
    completion(base_url, model, short_prompt)
    after_short_tier = wait_for_accounting(base_url, after_tier, len(short_prompt))
    validate_delta("short_second_tier_hit", after_short_tier.delta(after_tier), {expected_tier})

    completion(base_url, model, long_prompt)
    after_mixed = wait_for_accounting(base_url, after_short_tier, len(long_prompt))
    validate_delta(
        "mixed_device_and_second_tier_hit",
        after_mixed.delta(after_short_tier),
        {"device", expected_tier},
    )

    allowed_sources = {"device", expected_tier}
    final_sources = {source for source, value in after_mixed.cached_by_source.items() if value}
    if not final_sources <= allowed_sources:
        raise ValidationError(
            f"unexpected source-label cardinality: {sorted(final_sources - allowed_sources)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--expected-tier",
        choices=("cpu", "disk", "external"),
        help="Omit for a device-only server.",
    )
    # 1088 and 3264 are multiples of both the common 16-token block and the
    # 544-token hybrid page used by Qwen3.5-0.8B.
    parser.add_argument("--short-tokens", type=int, default=1088)
    parser.add_argument("--long-tokens", type=int, default=3264)
    parser.add_argument("--prompt-mode", choices=("token_ids", "text"), default="token_ids")
    args = parser.parse_args()
    if args.short_tokens <= 0 or args.long_tokens <= args.short_tokens:
        parser.error("require 0 < --short-tokens < --long-tokens")
    return args


def main() -> int:
    args = parse_args()
    try:
        run(
            args.base_url,
            args.expected_tier,
            args.short_tokens,
            args.long_tokens,
            args.prompt_mode,
        )
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    print("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
