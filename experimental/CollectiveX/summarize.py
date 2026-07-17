#!/usr/bin/env python3
"""Render a small shard summary; benchmark gating remains in the harness."""
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
    factors = document["identity"]["case_factors"]
    case = factors["case"]
    return (
        factors["sku"], case["suite"], case["routing"], case["phase"], case["ep"],
    )


def _headline(document: dict) -> tuple[int | str, float | str, float | str]:
    rows = document["measurement"]["rows"]
    row = next((item for item in rows if item["tokens_per_rank"] == 64), rows[len(rows) // 2])
    latency = row["components"]["roundtrip"]["percentiles_us"]
    return row["tokens_per_rank"], latency["p50"], latency["p99"]


def render(documents: list[dict]) -> str:
    documents = sorted(documents, key=_identity)
    invalid = [d for d in documents if d["outcome"]["status"] != "success"]
    lines = ["## CollectiveX EP results", ""]
    if invalid:
        # The leg is already red (ep_harness.run_sweep returns nonzero on a non-success
        # outcome); call the count out loudly so it is not lost in the per-row table.
        lines.append(
            f"> **{len(invalid)} of {len(documents)} outcome(s) INVALID** — "
            "the leg fails; see the outcome column below."
        )
        lines.append("")
    lines += [
        "| ver | sku | backend | suite | phase | routing | ep | outcome | T* | p50 us | p99 us |",
        "|--:|---|---|---|---|---|--:|---|--:|--:|--:|",
    ]
    for document in documents:
        sku, suite, routing, phase, ep = _identity(document)
        backend = document["identity"]["case_factors"]["case"]["backend"]
        token, p50, p99 = _headline(document)
        lines.append(
            f"| {document['version']} | {sku} | `{backend}` | {suite} | {phase} | "
            f"{routing} | {ep} | "
            f"{document['outcome']['status']} | {token} | {p50} | {p99} |"
        )
    if not documents:
        lines.append("\n> No valid native outcome documents found.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CollectiveX native v1 outcomes")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runner")
    parser.add_argument("--ts")
    args = parser.parse_args()
    documents = load_results(args.results_dir, args.runner, args.ts)
    print(render(documents))
    # Pure renderer — never gates CI. The per-case leg gate lives in ep_harness.run_sweep: a
    # non-success outcome returns nonzero and fails the shard (see collx_run_shard). A dead
    # "exit 1 when no success doc" gate here lost its only caller in 41caeaa0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
