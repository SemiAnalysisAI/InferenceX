#!/usr/bin/env python3
"""Small OpenAI-compatible client for SpeedBench AL eval load.

This intentionally avoids importing vLLM benchmark code so the eval can run in
TensorRT-LLM and SGLang runtime images.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _load_dsv4_encoder():
    bench_serving_dir = Path(__file__).resolve().parents[1] / "bench_serving"
    sys.path.insert(0, str(bench_serving_dir))
    from encoding_dsv4 import encode_messages  # type: ignore

    return encode_messages


def _load_speedbench_requests(
    dataset_path: Path,
    category: str,
    num_prompts: int,
) -> list[list[dict[str, Any]]]:
    jsonl_path = dataset_path / "qualitative.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"missing SpeedBench JSONL: {jsonl_path}")

    requests: list[list[dict[str, Any]]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if category and row.get("category") != category:
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            requests.append(messages)
            if num_prompts > 0 and len(requests) >= num_prompts:
                break

    if not requests:
        raise ValueError(f"no SpeedBench prompts found for category={category!r}")
    return requests


def _json_post(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = f"HTTP {exc.code} from {url}: {detail[:1000]}"
            last_error = RuntimeError(message)
            if exc.code < 500:
                break
        except URLError as exc:
            last_error = exc
        except TimeoutError as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise last_error


def _chat_payload(
    messages: list[dict[str, Any]],
    model: str,
    output_len: int,
    temperature: float,
    thinking_mode: str,
    thinking_kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": output_len,
        "temperature": temperature,
        "stream": False,
    }
    if thinking_mode == "on" and thinking_kwargs:
        payload["chat_template_kwargs"] = thinking_kwargs
        if "reasoning_effort" in thinking_kwargs:
            payload["reasoning_effort"] = thinking_kwargs["reasoning_effort"]
    return payload


def _completion_payload(
    messages: list[dict[str, Any]],
    model: str,
    output_len: int,
    temperature: float,
    thinking_mode: str,
    thinking_kwargs: dict[str, Any],
    dsv4: bool,
) -> dict[str, Any]:
    if dsv4:
        encode_messages = _load_dsv4_encoder()
        prompt = encode_messages(
            messages,
            thinking_mode="thinking" if thinking_mode == "on" else "chat",
            reasoning_effort=thinking_kwargs.get("reasoning_effort"),
        )
    else:
        first = messages[0]
        prompt = first.get("content", "") if isinstance(first, dict) else str(first)

    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": temperature,
        "stream": False,
    }


def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset_path)
    prompts = _load_speedbench_requests(dataset_path, args.category, args.num_prompts)
    base_url = args.base_url.rstrip("/")
    chat_url = f"{base_url}/v1/chat/completions"
    completions_url = f"{base_url}/v1/completions"
    thinking_kwargs = json.loads(args.thinking_kwargs) if args.thinking_kwargs else {}

    failures = 0
    resolved_endpoint = args.endpoint
    for index, messages in enumerate(prompts, start=1):
        endpoint_attempts = ["chat", "completions"] if resolved_endpoint == "auto" else [resolved_endpoint]
        last_error: Exception | None = None
        success = False
        for endpoint in endpoint_attempts:
            if endpoint == "completions":
                payload = _completion_payload(
                    messages,
                    args.model,
                    args.output_len,
                    args.temperature,
                    args.thinking_mode,
                    thinking_kwargs,
                    args.dsv4,
                )
                url = completions_url
            else:
                payload = _chat_payload(
                    messages,
                    args.model,
                    args.output_len,
                    args.temperature,
                    args.thinking_mode,
                    thinking_kwargs,
                )
                url = chat_url

            try:
                _json_post(url, payload, timeout=args.timeout, retries=args.retries)
            except Exception as exc:
                last_error = exc
                if resolved_endpoint == "auto" and endpoint == "chat":
                    print(
                        "SpeedBench client chat endpoint failed; trying completions "
                        f"fallback: {exc}",
                        file=sys.stderr,
                    )
                    continue
                break
            else:
                if resolved_endpoint == "auto":
                    resolved_endpoint = endpoint
                print(
                    f"SpeedBench client request {index}/{len(prompts)} "
                    f"completed via {endpoint}",
                    flush=True,
                )
                success = True
                break

        if success:
            continue

        if last_error is None:
            last_error = RuntimeError("no SpeedBench endpoint attempts were made")
        failures += 1
        print(
            f"SpeedBench client request {index}/{len(prompts)} failed: {last_error}",
            file=sys.stderr,
        )
        if failures > args.max_failures:
            return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--category", default="coding")
    parser.add_argument("--output-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--thinking-mode", choices=["on", "off"], default="off")
    parser.add_argument("--thinking-kwargs", default="")
    parser.add_argument("--endpoint", choices=["auto", "chat", "completions"], default="auto")
    parser.add_argument("--num-prompts", type=int, default=-1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--dsv4", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
