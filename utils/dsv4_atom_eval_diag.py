#!/usr/bin/env python3
"""DeepSeek-V4 ATOM concurrency diagnostic probes.

The eval failure seen at high concurrency can come from several places:
batched prefill metadata, decode KV/cache state, long-context sparse
attention/indexer paths, or cross-request leakage. This script issues a small
matrix of deterministic completion requests and writes both per-request rows
and per-case summaries to JSONL so the failure mode is visible from artifacts.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from typing import Any


MARKER_PREFIX = "DSV4MARK"


def _parse_levels(raw: str) -> list[int]:
    levels: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            levels.append(max(1, int(item)))
        except ValueError:
            pass
    levels = sorted(set(levels)) or [1]
    if 1 not in levels:
        levels.insert(0, 1)
    return levels


def _chat_prompt(body: str) -> str:
    tok_bos = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    tok_user = "<\uff5cUser\uff5c>"
    tok_assistant = "<\uff5cAssistant\uff5c>"
    return f"{tok_bos}{tok_user}{body}{tok_assistant}</think>"


def _first_tokenish(text: str) -> str:
    stripped = text.lstrip()
    if not stripped:
        return ""
    return stripped.split(maxsplit=1)[0][:40]


def _extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*\$?(-?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def _marker_for(request_id: int) -> str:
    return f"{MARKER_PREFIX}{request_id:04d}"


def _marker_prompt(prompt_kind: str, pad: str, request_id: int) -> tuple[str, str]:
    marker = _marker_for(request_id)
    body = (
        f"The marker for this request is {marker}.\n"
        f"Output exactly this marker and nothing else: {marker}\n"
        "Answer:"
    )
    if prompt_kind == "long":
        body = pad + "\n\n" + body
    return _chat_prompt(body), marker


def _case_matrix(levels: list[int], isl: int) -> tuple[list[dict[str, Any]], str]:
    math_body = (
        "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast "
        "every morning and bakes muffins for her friends every day with four. "
        "She sells the remainder at the farmers' market daily for $2 per fresh "
        "duck egg. How much in dollars does she make every day at the farmers' "
        "market?\n"
        "End your response with the answer on the last line, formatted as: #### [number]\n"
        "Answer:"
    )
    pad_units = int(os.environ.get("ATOM_DSV4_DIAG_LONG_PAD_UNITS", "0") or "0")
    if pad_units <= 0:
        pad_units = min(max(isl // 16, 64), 800)
    pad = " ".join(
        f"Reference filler sentence {i}: keep this context unchanged."
        for i in range(pad_units)
    )
    short_math_prompt = _chat_prompt(math_body)
    long_math_prompt = _chat_prompt(
        pad
        + "\n\nUse only the final question below; the preceding filler is irrelevant.\n"
        + math_body
    )
    decode_tokens = int(os.environ.get("ATOM_DSV4_DIAG_DECODE_TOKENS", "32"))
    marker_tokens = int(os.environ.get("ATOM_DSV4_DIAG_MARKER_TOKENS", "8"))
    max_level = max(levels)
    cases = [
        {
            "name": "short_identical_1tok",
            "mode": "identical",
            "prompt_kind": "short",
            "prompt": short_math_prompt,
            "max_tokens": 1,
            "levels": levels,
        },
        {
            "name": "short_identical_decode",
            "mode": "identical",
            "prompt_kind": "short",
            "prompt": short_math_prompt,
            "max_tokens": decode_tokens,
            "levels": levels,
        },
        {
            "name": "long_identical_1tok",
            "mode": "identical",
            "prompt_kind": "long",
            "prompt": long_math_prompt,
            "max_tokens": 1,
            "levels": levels,
        },
        {
            "name": "long_identical_decode",
            "mode": "identical",
            "prompt_kind": "long",
            "prompt": long_math_prompt,
            "max_tokens": decode_tokens,
            "levels": levels,
        },
        {
            "name": "short_distinct_marker",
            "mode": "marker",
            "prompt_kind": "short",
            "max_tokens": marker_tokens,
            "levels": [max_level],
        },
        {
            "name": "long_distinct_marker",
            "mode": "marker",
            "prompt_kind": "long",
            "max_tokens": marker_tokens,
            "levels": [max_level],
        },
    ]
    case_filter = os.environ.get("ATOM_DSV4_DIAG_CASES", "").strip()
    if case_filter:
        wanted = {item.strip() for item in case_filter.split(",") if item.strip()}
        cases = [case for case in cases if case["name"] in wanted]
    return cases, pad


def _one_request(
    *,
    case: dict[str, Any],
    level: int,
    request_id: int,
    model: str,
    url: str,
    stop: list[str],
    pad: str,
) -> dict[str, Any]:
    if case["mode"] == "marker":
        prompt, expected_marker = _marker_prompt(case["prompt_kind"], pad, request_id)
    else:
        prompt = case["prompt"]
        expected_marker = None
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": case["max_tokens"],
        "temperature": 0,
        "top_p": 1,
        "stop": stop,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choice = body["choices"][0]
        text = choice.get("text", "")
        markers = sorted(set(re.findall(rf"{MARKER_PREFIX}\d{{4}}", text)))
        wrong_markers = [
            marker
            for marker in markers
            if expected_marker is not None and marker != expected_marker
        ]
        return {
            "kind": "request",
            "case": case["name"],
            "mode": case["mode"],
            "prompt_kind": case["prompt_kind"],
            "max_tokens": case["max_tokens"],
            "level": level,
            "request_id": request_id,
            "ok": True,
            "latency_s": round(time.time() - started, 3),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "length": len(text),
            "has_final_answer": "####" in text,
            "answer": _extract_answer(text),
            "first_tokenish": _first_tokenish(text),
            "finish_reason": choice.get("finish_reason"),
            "expected_marker": expected_marker,
            "contains_expected_marker": (
                expected_marker in text if expected_marker is not None else None
            ),
            "markers_seen": markers,
            "wrong_markers": wrong_markers,
            "text": text,
        }
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            detail = exc.read().decode("utf-8", "replace")
        else:
            detail = ""
        return {
            "kind": "request",
            "case": case["name"],
            "mode": case["mode"],
            "prompt_kind": case["prompt_kind"],
            "max_tokens": case["max_tokens"],
            "level": level,
            "request_id": request_id,
            "ok": False,
            "latency_s": round(time.time() - started, 3),
            "error": repr(exc),
            "detail": detail[:2000],
        }


def _summarize_case(
    case: dict[str, Any],
    level: int,
    rows: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    errors = [row for row in rows if not row.get("ok")]
    hashes = sorted({row.get("sha256") for row in ok_rows})
    firsts = sorted({row.get("first_tokenish") for row in ok_rows})
    latencies = [row["latency_s"] for row in ok_rows if "latency_s" in row]
    summary: dict[str, Any] = {
        "kind": "summary",
        "case": case["name"],
        "mode": case["mode"],
        "prompt_kind": case["prompt_kind"],
        "max_tokens": case["max_tokens"],
        "level": level,
        "ok": len(ok_rows),
        "total": len(rows),
        "errors": len(errors),
        "unique_outputs": len(hashes),
        "unique_first_tokenish": len(firsts),
        "first_tokenish_values": firsts[:12],
        "latency_s_mean": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_s_max": round(max(latencies), 3) if latencies else None,
    }
    if case["mode"] == "identical":
        baseline_hash = baseline.get("sha256") if baseline else None
        baseline_first = baseline.get("first_tokenish") if baseline else None
        summary.update(
            {
                "baseline_sha256": baseline_hash,
                "baseline_first_tokenish": baseline_first,
                "drift_vs_baseline": sum(
                    1
                    for row in ok_rows
                    if baseline_hash is not None and row.get("sha256") != baseline_hash
                ),
                "first_token_drift_vs_baseline": sum(
                    1
                    for row in ok_rows
                    if baseline_first is not None
                    and row.get("first_tokenish") != baseline_first
                ),
                "missing_final": [
                    row["request_id"]
                    for row in ok_rows
                    if case["max_tokens"] > 1 and not row.get("has_final_answer")
                ],
                "answers": sorted(
                    {
                        row.get("answer")
                        for row in ok_rows
                        if row.get("answer") is not None
                    }
                )[:12],
            }
        )
    else:
        missing = [
            row["request_id"]
            for row in ok_rows
            if row.get("contains_expected_marker") is False
        ]
        wrong = [row["request_id"] for row in ok_rows if row.get("wrong_markers")]
        summary.update(
            {
                "missing_expected_marker": missing,
                "wrong_marker_requests": wrong,
                "wrong_markers_seen": sorted(
                    {
                        marker
                        for row in ok_rows
                        for marker in row.get("wrong_markers", [])
                    }
                )[:24],
            }
        )
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    if summary["mode"] == "identical":
        print(
            "[DSv4 diag] "
            f"case={summary['case']} level={summary['level']} "
            f"ok={summary['ok']}/{summary['total']} "
            f"unique_outputs={summary['unique_outputs']} "
            f"first_token_drift={summary['first_token_drift_vs_baseline']} "
            f"drift={summary['drift_vs_baseline']} "
            f"missing_final={summary.get('missing_final', [])[:16]}"
        )
    else:
        print(
            "[DSv4 diag] "
            f"case={summary['case']} level={summary['level']} "
            f"ok={summary['ok']}/{summary['total']} "
            f"missing_marker={summary['missing_expected_marker'][:16]} "
            f"wrong_marker={summary['wrong_marker_requests'][:16]} "
            f"unique_outputs={summary['unique_outputs']}"
        )


def _print_snippets(rows: list[dict[str, Any]]) -> None:
    for row in rows[: min(4, len(rows))]:
        snippet = row.get("text", row.get("error", "")).replace("\n", " ")[:260]
        print(
            "[DSv4 diag] "
            f"  case={row.get('case')} req={row['request_id']} ok={row.get('ok')} "
            f"first={row.get('first_tokenish')!r} len={row.get('length')} "
            f"sha={row.get('sha256', '')[:12]} markers={row.get('markers_seen')} "
            f"snippet={snippet!r}"
        )


def _diagnosis(summaries: list[dict[str, Any]], max_level: int) -> list[str]:
    by_case = {(s["case"], s["level"]): s for s in summaries}
    short_1 = by_case.get(("short_identical_1tok", max_level), {})
    short_decode = by_case.get(("short_identical_decode", max_level), {})
    long_1 = by_case.get(("long_identical_1tok", max_level), {})
    long_decode = by_case.get(("long_identical_decode", max_level), {})
    short_marker = by_case.get(("short_distinct_marker", max_level), {})
    long_marker = by_case.get(("long_distinct_marker", max_level), {})
    notes: list[str] = []
    if short_1.get("first_token_drift_vs_baseline", 0):
        notes.append(
            "short 1-token drift: corruption happens by final prefill logits; "
            "suspect batched prefill metadata, positions/slot mapping, sampler, "
            "or common MHC/FFN path before decode KV growth"
        )
    elif short_decode.get("drift_vs_baseline", 0):
        notes.append(
            "short multi-token drift but 1-token stable: suspect decode KV/cache "
            "state update or per-step scheduling rather than initial prefill"
        )
    if (
        long_1.get("first_token_drift_vs_baseline", 0)
        and not short_1.get("first_token_drift_vs_baseline", 0)
    ):
        notes.append(
            "long-only 1-token drift: suspect long-context DSv4 attention/indexer/"
            "compressor path rather than generic batching"
        )
    if (
        long_decode.get("drift_vs_baseline", 0)
        and not short_decode.get("drift_vs_baseline", 0)
    ):
        notes.append(
            "long-only decode drift: suspect sparse attention/indexer/cache growth "
            "after prefill"
        )
    if short_marker.get("wrong_marker_requests") or long_marker.get(
        "wrong_marker_requests"
    ):
        notes.append(
            "wrong marker copied from another request: direct evidence of "
            "cross-request leakage"
        )
    if short_marker.get("missing_expected_marker") and not short_marker.get(
        "wrong_marker_requests"
    ):
        notes.append(
            "short marker missing without wrong marker: request output is unstable "
            "but not obviously copying another request"
        )
    if long_marker.get("missing_expected_marker") and not long_marker.get(
        "wrong_marker_requests"
    ):
        notes.append(
            "long marker missing without wrong marker: long-context prompt "
            "conditioning or attention may be corrupted"
        )
    if not notes:
        notes.append("diagnostic matrix did not reproduce a clear failure")
    return notes


def main() -> int:
    port = os.environ["DIAG_PORT"]
    model = os.environ["DIAG_MODEL"]
    out_path = os.environ["DIAG_OUT"]
    isl = int(os.environ.get("DIAG_ISL", "8192"))
    levels = _parse_levels(os.environ.get("DIAG_CONC_LIST", "1,2,4,8,16"))
    cases, pad = _case_matrix(levels, isl)
    max_level = max(levels)
    stop = [
        "<\uff5cend\u2581of\u2581sentence\uff5c>",
        "<\uff5cUser\uff5c>",
        "<\uff5cAssistant\uff5c>",
        "</s>",
        "<|im_end|>",
    ]
    url = f"http://127.0.0.1:{port}/v1/completions"
    summaries: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}

    with open(out_path, "w", encoding="utf-8") as out:
        for case in cases:
            print(
                "[DSv4 diag] "
                f"starting case={case['name']} mode={case['mode']} "
                f"prompt={case['prompt_kind']} max_tokens={case['max_tokens']} "
                f"levels={case['levels']}"
            )
            for level in case["levels"]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                    rows = list(
                        pool.map(
                            lambda i: _one_request(
                                case=case,
                                level=level,
                                request_id=i,
                                model=model,
                                url=url,
                                stop=stop,
                                pad=pad,
                            ),
                            range(level),
                        )
                    )
                if case["mode"] == "identical" and level == 1:
                    ok = [row for row in rows if row.get("ok")]
                    if ok:
                        baselines[case["name"]] = ok[0]
                summary = _summarize_case(
                    case, level, rows, baselines.get(case["name"])
                )
                summaries.append(summary)
                _print_summary(summary)
                should_print = (
                    summary["errors"]
                    or summary["unique_outputs"] > 1
                    or summary.get("drift_vs_baseline", 0)
                    or summary.get("first_token_drift_vs_baseline", 0)
                    or summary.get("missing_final")
                    or summary.get("missing_expected_marker")
                    or summary.get("wrong_marker_requests")
                )
                if should_print:
                    _print_snippets(rows)
                out.write(json.dumps(summary, ensure_ascii=True) + "\n")
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=True) + "\n")

        notes = _diagnosis(summaries, max_level)
        final = {"kind": "diagnosis", "max_level": max_level, "notes": notes}
        print("[DSv4 diag] diagnosis=" + " | ".join(notes))
        out.write(json.dumps(final, ensure_ascii=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
