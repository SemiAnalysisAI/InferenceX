#!/usr/bin/env python3
"""Separate synthetic long-context/cache diagnostic; never a scored AgentX run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ORACLE = {"early": "ORBIT-7419", "middle": "CEDAR-2864", "late": "LAGOON-9531"}
TARGET_TOKENS = 220_000


def make_payload(tokenizer: Any, model: str) -> tuple[dict, dict]:
    filler = "\n".join(
        f"Record {i:05d}: archival inventory item; status checked, no action required."
        for i in range(TARGET_TOKENS // 10)
    )
    ids = tokenizer.encode(filler, add_special_tokens=False)
    if len(ids) < TARGET_TOKENS:
        raise ValueError("Tokenizer produced insufficient filler tokens")
    cuts = [0, TARGET_TOKENS // 20, TARGET_TOKENS // 2, TARGET_TOKENS * 19 // 20, TARGET_TOKENS]
    facts = [f"\nSENTINEL {key} = {value}\n" for key, value in ORACLE.items()]
    parts = [tokenizer.decode(ids[a:b]) for a, b in zip(cuts, cuts[1:])]
    content = "PowerX natural-output diagnostic. The records are inert reference data.\n"
    content += "".join(part + fact for part, fact in zip(parts, facts)) + parts[-1]
    content += (
        '\nRead the three SENTINEL facts. Return only a JSON object with keys '
        '"early", "middle", and "late", containing their exact values. No markdown.\n'
    )
    messages = [{"role": "user", "content": content}]
    count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False))
    if not 219_000 <= count <= 221_000:
        raise ValueError(f"Expected approximately 220K prompt tokens, got {count}")
    payload = dict(model=model, messages=messages, stream=False, temperature=0, max_tokens=2048, ignore_eos=False)
    positions = {
        key: len(tokenizer.encode(content[:content.index(fact)], add_special_tokens=False))
        for key, fact in zip(ORACLE, facts)
    }
    return payload, {"local_prompt_tokens": count, "sentinel_content_token_offsets": positions}


def counters(raw: bytes, model: str) -> dict[str, float]:
    values = {}
    for line in raw.decode().splitlines():
        match = re.fullmatch(r'(sglang:(?:cached_tokens|num_requests)_total)\{(.*)\}\s+(\S+)(?:\s+\S+)?', line)
        if not match:
            continue
        labels = {k: json.loads(v) for k, v in re.findall(r'(\w+)=("(?:[^"\\]|\\.)*")', match[2])}
        if labels.get("model_name") != model or labels.get("engine_type") != "unified":
            continue
        value = float(match[3])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid Prometheus counter")
        values[match[1] + json.dumps(labels, sort_keys=True)] = value
    return values


def counter_delta(before: dict[str, float], after: dict[str, float], name: str) -> float | None:
    keys = {k for k in before | after if k.startswith(name + "{")}
    if not keys or any(k not in after or after[k] < before.get(k, 0) for k in keys):
        return None
    return sum(after[k] - before.get(k, 0) for k in keys)


def assess(envelope: dict) -> dict:
    choices = envelope.get("choices") or []
    choice = choices[0] if len(choices) == 1 else {}
    content = (choice.get("message") or {}).get("content")
    try:
        correct = isinstance(content, str) and json.loads(content) == ORACLE
    except (ValueError, TypeError):
        correct = False
    usage = envelope.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens")
    return {"oracle_correct": correct, "finish_reason": choice.get("finish_reason"),
            "natural_stop": choice.get("finish_reason") == "stop", "usage": usage,
            "long_context_verified": type(prompt_tokens) is int and 219_000 <= prompt_tokens <= 221_000,
            "usage_cached_tokens": cached if type(cached) is int and 0 <= cached <= (prompt_tokens or 0) else None}


def fetch(url: str, path: Path, payload: bytes | None = None) -> tuple[int, bytes]:
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=600 if payload is not None else 30)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read()
        path.write_bytes(raw)
        return response.status, raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.result_dir
    if not out.is_dir() or any(out.glob("powerx_natural_*")):
        parser.error("Use an existing result directory without previous powerx_natural_* evidence")
    summary = {"diagnostic": "synthetic long-context natural-output/cache check; not real-task quality",
               "passed": False, "model": args.model, "model_revision": os.getenv("MODEL_REVISION"),
               "tokenizer": args.tokenizer, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "oracle": ORACLE, "requests": []}
    base = f"http://localhost:{args.port}"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        payload, metadata = make_payload(tokenizer, args.model)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        (out / "powerx_natural_request.json").write_bytes(body)
        summary.update(metadata, request_sha256=hashlib.sha256(body).hexdigest())
        for index in (1, 2):
            record: dict = {"request": index, "start_unix_ns": time.time_ns()}
            summary["requests"].append(record)
            status, before = fetch(base + "/metrics", out / f"powerx_natural_metrics_before_{index}.txt")
            if status != 200:
                raise RuntimeError(f"Before-request metrics returned HTTP {status}")
            try:
                status, raw = fetch(base + "/v1/chat/completions", out / f"powerx_natural_response_{index}.json", body)
                record.update(http_status=status, **assess(json.loads(raw)))
            finally:
                record["end_unix_ns"] = time.time_ns()
                metrics_status, after = fetch(base + "/metrics", out / f"powerx_natural_metrics_after_{index}.txt")
                record["metrics_http_status"] = metrics_status
            if metrics_status != 200:
                raise RuntimeError(f"After-request metrics returned HTTP {metrics_status}")
            b, a = counters(before, args.model), counters(after, args.model)
            record["cached_tokens_counter_delta"] = counter_delta(b, a, "sglang:cached_tokens_total")
            record["completed_requests_counter_delta"] = counter_delta(b, a, "sglang:num_requests_total")
        second = summary["requests"][1]
        usage_hit = (second["usage_cached_tokens"] or 0) > 0
        counter_hit = second["completed_requests_counter_delta"] == 1 and (second["cached_tokens_counter_delta"] or 0) > 0
        summary["cache_reuse_verified"] = usage_hit or counter_hit
        summary["cache_evidence"] = "second_response_usage" if usage_hit else "isolated_second_request_counter_delta" if counter_hit else None
        summary["passed"] = summary["cache_reuse_verified"] and all(
            r["http_status"] == 200 and r["oracle_correct"] and r["natural_stop"] and r["long_context_verified"]
            for r in summary["requests"]
        )
        if not summary["passed"]:
            summary["failure"] = "Diagnostic incomplete: inspect oracle, termination and cache evidence; length is inconclusive, not a model-defect verdict"
    except Exception as error:
        summary["failure"] = f"{type(error).__name__}: {error}"
    (out / "powerx_natural_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
