#!/usr/bin/env python3
"""Summarize CollectiveX EP results for logs or a GitHub job summary.

Plain-text mode is also the shard health gate: it fails when no complete EP result
was produced. Markdown mode is reporting-only and always exits successfully.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load_results(results_dir: str, runner: str | None, ts: str | None) -> list[dict]:
    """Load only EP result and failed-case documents from a result directory."""
    docs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("env_"):
            continue
        if runner and not base.startswith(f"{runner}_"):
            continue
        if ts and ts not in base:
            continue
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(doc, dict) and doc.get("family") == "moe":
            docs.append(doc)
    return docs


def _fnum(value, fmt: str) -> str:
    return format(value, fmt) if isinstance(value, (int, float)) else "-"


def _doc_status(doc: dict) -> str:
    return str(doc.get("publication_status") or doc.get("status") or "unknown")


def _execution_valid(doc: dict) -> bool:
    return doc.get("record_type") != "failed-case" and doc.get("status") == "valid"


def _completed(docs: list[dict]) -> list[dict]:
    return sorted(
        (doc for doc in docs if doc.get("record_type") != "failed-case"),
        key=lambda doc: (doc.get("backend", ""), doc.get("phase", ""), doc.get("ep_size", 0)),
    )


def _failed(docs: list[dict]) -> list[dict]:
    return sorted(
        (doc for doc in docs if doc.get("record_type") == "failed-case"),
        key=lambda doc: (doc.get("backend", ""), doc.get("phase", ""), doc.get("attempt_id", "")),
    )


def _shape_label(doc: dict) -> str:
    shape = doc.get("shape") or {}
    return (
        f"H{shape.get('hidden', '?')} top{shape.get('topk', '?')} "
        f"E{shape.get('experts', '?')} {shape.get('dispatch_dtype', '?')} "
        f"{shape.get('routing', '?')}"
    )


def _sweep_table(doc: dict) -> list[str]:
    rows = doc.get("rows") or []
    if not rows:
        return []
    out = [
        (f"\n**`{doc.get('backend')}` · {doc.get('phase')} · ep{doc.get('ep_size')} · "
         f"{_shape_label(doc)}**\n"),
        "| tokens/rank | fan-out | dispatch p50 us | combine p50 us | roundtrip p50 us | tokens/s | recv max | correct |",
        "|--:|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    for row in rows:
        out.append(
            f"| {row.get('tokens_per_rank')} | {_fnum(row.get('fanout_mean'), '.2f')} | "
            f"{_fnum(row.get('dispatch_us_p50'), '.2f')} | "
            f"{_fnum(row.get('combine_us_p50'), '.2f')} | "
            f"{_fnum(row.get('roundtrip_us_p50'), '.2f')} | "
            f"{_fnum(row.get('roundtrip_tokens_per_second'), '.3e')} | "
            f"{row.get('recv_tokens_max', '-')} | {'yes' if row.get('correct') else 'no'} |"
        )
    return out


def render_plain(docs: list[dict]) -> str:
    out = ["CollectiveX EP results", "======================"]
    complete = _completed(docs)
    failed = _failed(docs)
    if complete:
        out.append(
            f"  {'backend':<16}{'phase':<9}{'ep':>3} {'publication':<24}"
            f"{'T*':>5}{'roundtrip p99 us':>19}  correct"
        )
        for doc in complete:
            metrics = doc.get("metrics") or {}
            correctness = doc.get("correctness") or {}
            out.append(
                f"  {str(doc.get('backend', '')):<16}{str(doc.get('phase', '')):<9}"
                f"{str(doc.get('ep_size', '')):>3} {_doc_status(doc):<24}"
                f"{str(metrics.get('headline_tokens_per_rank', '')):>5}"
                f"{_fnum(metrics.get('roundtrip_us_p99'), '.1f'):>19}  "
                f"{correctness.get('passed')}"
            )
    if failed:
        out.append("\nFailed EP attempts:")
        for doc in failed:
            failure = doc.get("failure") or {}
            out.append(
                f"  {doc.get('backend', '?')}/{doc.get('phase', '?')} "
                f"case={doc.get('case_id') or 'manual'} attempt={doc.get('attempt_id', '1')} "
                f"mode={failure.get('failure_mode', 'unknown')} rc={failure.get('return_code', '?')}"
            )
    return "\n".join(out)


def render_markdown(docs: list[dict]) -> str:
    complete = _completed(docs)
    failed = _failed(docs)
    out = ["## CollectiveX EP results"]
    if complete:
        out += [
            "",
            "| backend | phase | mode | dtype | resource | ep | routing | publication | T* | roundtrip p50 us | roundtrip p99 us | correct |",
            "|---|---|---|---|---|--:|---|---|--:|--:|--:|:--:|",
        ]
        for doc in complete:
            metrics = doc.get("metrics") or {}
            correctness = doc.get("correctness") or {}
            shape = doc.get("shape") or {}
            out.append(
                f"| `{doc.get('backend', '')}` | {doc.get('phase', '')} | {doc.get('mode', '')} | "
                f"{shape.get('dispatch_dtype', '-')} | {doc.get('resource_mode', '')} | "
                f"{doc.get('ep_size', '')} | {shape.get('routing', '-')} | {_doc_status(doc)} | "
                f"{metrics.get('headline_tokens_per_rank', '-')} | "
                f"{_fnum(metrics.get('roundtrip_us_p50'), '.1f')} | "
                f"{_fnum(metrics.get('roundtrip_us_p99'), '.1f')} | "
                f"{'yes' if correctness.get('passed') else 'no'} |"
            )
        for doc in complete:
            out += _sweep_table(doc)
    if failed:
        out += [
            "\n### Failed attempts\n",
            "| backend | phase | case | attempt | failure | rc |",
            "|---|---|---|--:|---|--:|",
        ]
        for doc in failed:
            failure = doc.get("failure") or {}
            out.append(
                f"| `{doc.get('backend', '')}` | {doc.get('phase', '')} | "
                f"`{doc.get('case_id') or 'manual'}` | {doc.get('attempt_id', '1')} | "
                f"{failure.get('failure_mode', 'unknown')} | {failure.get('return_code', '-')} |"
            )
    if not docs:
        out.append("\n> No EP result files found.")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="CollectiveX EP result summary")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runner")
    parser.add_argument("--ts")
    parser.add_argument("--markdown", action="store_true",
                        help="emit reporting-only GitHub summary markdown")
    args = parser.parse_args()

    docs = load_results(args.results_dir, args.runner, args.ts)
    if args.markdown:
        print(render_markdown(docs))
        return 0

    print(render_plain(docs))
    valid = sum(_execution_valid(doc) for doc in docs)
    if valid == 0:
        print("ERROR: no complete, valid EP result was produced.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
