#!/usr/bin/env python3
"""Render a small native-v1 shard summary and gate on a successful case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Emitted case-attempt documents this summary reads, discriminated by record_type.
# This is a best-effort renderer over whatever raw attempts a shard produced; it
# validates nothing.
CASE_RECORD_TYPE = "case-attempt"


def load_results(directory: str, runner: str | None, timestamp: str | None) -> list[dict]:
    documents: list[dict] = []
    for path in sorted(Path(directory).glob("*.json")):
        if runner and not path.name.startswith(f"{runner}_"):
            continue
        if timestamp and timestamp not in path.name:
            continue
        try:
            with path.open() as handle:
                document = json.load(handle)
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and document.get("record_type") == CASE_RECORD_TYPE:
            documents.append(document)
    return documents


def _identity(document: dict) -> tuple[str, str, str, str, int]:
    case = document["case"]
    routing = case["shape"]["routing"]
    sku = document["identity"]["case_factors"]["sku"]
    return (
        sku, case["suite"], routing, case["phase"],
        case.get("ep_size", case.get("ep", 0)),
    )


def _headline(document: dict) -> tuple[int | str, float | str, float | str]:
    rows = document["measurement"]["rows"]
    row = next((item for item in rows if item["tokens_per_rank"] == 64), rows[len(rows) // 2])
    latency = row["components"]["roundtrip"]["percentiles_us"]
    return row["tokens_per_rank"], latency["p50"], latency["p99"]


def render(documents: list[dict], markdown: bool) -> str:
    documents = sorted(documents, key=_identity)
    if markdown:
        lines = [
            "## CollectiveX EP results", "",
            "| ver | sku | backend | suite | phase | routing | ep | outcome | T* | p50 us | p99 us |",
            "|--:|---|---|---|---|---|--:|---|--:|--:|--:|",
        ]
        for document in documents:
            sku, suite, routing, phase, ep = _identity(document)
            backend = document["case"]["backend"]
            token, p50, p99 = _headline(document)
            lines.append(
                f"| {document['version']} | {sku} | `{backend}` | {suite} | {phase} | "
                f"{routing} | {ep} | "
                f"{document['outcome']['status']} | {token} | {p50} | {p99} |"
            )
        if not documents:
            lines.append("\n> No valid native outcome documents found.")
        return "\n".join(lines)
    lines = ["CollectiveX EP results", "======================"]
    for document in documents:
        sku, suite, routing, phase, ep = _identity(document)
        backend = document["case"]["backend"]
        token, _, p99 = _headline(document)
        lines.append(
            f"  v{document['version']} {sku:<10} {backend:<16} {suite:<13} {phase:<7} "
            f"{routing} ep{ep} "
            f"{document['outcome']['status']} T={token} roundtrip_p99_us={p99}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CollectiveX native v1 outcomes")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runner")
    parser.add_argument("--ts")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    documents = load_results(args.results_dir, args.runner, args.ts)
    print(render(documents, args.markdown))
    if args.markdown:
        return 0
    return 0 if any(
        document["outcome"]["status"] == "success"
        for document in documents
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
