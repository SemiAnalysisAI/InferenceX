"""Bounded real-verification cross-TP transport diagnostic, not a benchmark."""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def make_messages(index, repetitions):
    code = str(710000 + index)
    text = (
        f"Document cross-tp-{index:03d}. The retrieval code is {code}.\n"
        + "This is neutral padding text for a long document.\n" * repetitions
        + "\nWhat is the retrieval code at the beginning? Reply with only its six digits."
    )
    return [{"role": "user", "content": text}], code


def bounded_prompt(index, token_budget, count_tokens):
    """Find the largest whole-padding prompt under the caller's token budget."""
    low, high = 0, token_budget
    if count_tokens(make_messages(index, low)[0]) > token_budget:
        raise ValueError("Token budget is smaller than the retrieval instruction")
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(make_messages(index, middle)[0]) <= token_budget:
            low = middle
        else:
            high = middle - 1
    messages, code = make_messages(index, low)
    return messages, code, count_tokens(messages)


def request_one(endpoint, model, item, barrier, timeout):
    index, budget, messages, expected, local_tokens = item
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 32,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Dynamo-Session-ID": f"cross-tp-{index:03d}",
        },
        method="POST",
    )
    row = {
        "index": index,
        "token_budget": budget,
        "local_chat_template_tokens": local_tokens,
        "expected": expected,
        "request_sha256": hashlib.sha256(request.data).hexdigest(),
        "transport_ok": False,
        "correct": False,
        "output": "",
        "usage": None,
        "response_ids": [],
        "finish_reason": None,
        "classification": "transport_or_runtime_error",
    }
    barrier.wait()
    started = time.monotonic()
    row["started_unix"] = time.time()
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 180)) as response:
            row["http_status"] = response.status
            done = False
            for raw in response:
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Diagnostic wall-clock deadline exceeded")
                if not raw.startswith(b"data:"):
                    continue
                text = raw[5:].strip()
                if text == b"[DONE]":
                    done = True
                    break
                event = json.loads(text)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("id") and event["id"] not in row["response_ids"]:
                    row["response_ids"].append(event["id"])
                if event.get("usage"):
                    row["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    content = choice.get("delta", {}).get("content") or ""
                    if content and "ttft_seconds" not in row:
                        row["ttft_seconds"] = time.monotonic() - started
                    row["output"] += content
                    if choice.get("finish_reason"):
                        row["finish_reason"] = choice["finish_reason"]
            if not done or row["finish_reason"] is None:
                raise RuntimeError("Stream ended without DONE and finish reason")
        row["transport_ok"] = True
        row["correct"] = row["output"].strip() == expected
        actual_tokens = (row["usage"] or {}).get("prompt_tokens")
        row["actual_prompt_tokens"] = actual_tokens
        row["tokenization_valid"] = (
            isinstance(actual_tokens, int) and abs(actual_tokens - local_tokens) <= 16
        )
        row["classification"] = (
            "tokenization_evidence_mismatch"
            if not row["tokenization_valid"]
            else "pass"
            if row["correct"]
            else "needle_answer_mismatch"
        )
    except Exception as error:
        row["error_type"] = type(error).__name__
        row["error"] = str(error)[:1000]
        if isinstance(error, urllib.error.HTTPError):
            row["http_status"] = error.code
            row["response_error"] = error.read(2000).decode(errors="replace")
    row["elapsed_seconds"] = time.monotonic() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def count_tokens(messages):
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    budgets = [65536] * 12 + [131072] * 12 + [245760] * 8
    items = []
    templates = {}
    for index, budget in enumerate(budgets):
        if budget not in templates:
            templates[budget] = bounded_prompt(0, budget, count_tokens)[0][0]["content"]
        code = str(710000 + index)
        messages = [
            {
                "role": "user",
                "content": templates[budget].replace(
                    "Document cross-tp-000. The retrieval code is 710000.",
                    f"Document cross-tp-{index:03d}. The retrieval code is {code}.",
                    1,
                ),
            }
        ]
        count = count_tokens(messages)
        if count > budget:
            messages, code, count = bounded_prompt(index, budget, count_tokens)
        items.append((index, budget, messages, code, count))
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "purpose": "real-verification transport/needle diagnostic; no performance or power claim",
        "concurrency": len(items),
        "max_output_tokens": 32,
        "enable_thinking": False,
        "budgets": budgets,
        "local_token_counts": [x[-1] for x in items],
        "expected_answers": [x[-2] for x in items],
        "timeout_seconds": args.timeout,
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    barrier = threading.Barrier(len(items))
    rows = []
    with (args.output / "requests.jsonl").open("w") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)) as executor:
            futures = [
                executor.submit(
                    request_one, args.endpoint, args.model, x, barrier, args.timeout
                )
                for x in items
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            k: row.get(k)
                            for k in [
                                "index",
                                "classification",
                                "actual_prompt_tokens",
                                "elapsed_seconds",
                                "error",
                            ]
                        }
                    ),
                    flush=True,
                )
    summary = {
        "requests": len(rows),
        "transport_successes": sum(x["transport_ok"] for x in rows),
        "correct_answers": sum(x["correct"] for x in rows),
        "passed": all(x["classification"] == "pass" for x in rows),
        "classification_counts": {
            k: sum(x["classification"] == k for x in rows)
            for k in sorted({x["classification"] for x in rows})
        },
        "note": "Wrong needle answer alone does not establish a transfer defect. No formal power measurement.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
