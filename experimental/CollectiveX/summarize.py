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


def _identity(document: dict) -> tuple[str, str, str, str, str, str, int, str]:
    factors = document["identity"]["case_factors"]
    case = factors["case"]
    # backend and precision are part of the sort key so a cell's per-backend and
    # per-precision (bf16/fp8) attempts sort adjacently instead of interleaving.
    # mode is here because without it a cell's normal and low-latency rows are
    # indistinguishable: same sku/backend/phase/ep/precision, wildly different
    # latencies, no column telling them apart. Decode carries both on most SKUs.
    return (
        factors["sku"], case["backend"], case["suite"], case["routing"],
        case["mode"], case["phase"], case["ep"], case["precision"],
    )


def _headline(document: dict) -> tuple:
    """Headline row, with the skew bracket beside it.

    `p50`/`p99` are the cross-rank MAX: a layer is not finished until its slowest rank is, so
    MAX is the completion cost. But entry stagger is charged to it, and how much is a property
    of the BACKEND, not just the fleet — on identical h200 low-latency cells the per-iteration
    spread is 9.2us for deepep-v2 and uccl-ep (which share a kernel family) against 2.0us for
    nccl-ep, flat across the whole ladder. So MAX alone taxes some backends more than others,
    and two cells whose MAX gap is smaller than that spread are not separated by the data.
    `min50` is the same iterations reduced with MIN — the skew-excluded floor — and `skew` is
    the per-iteration MAX-MIN. Read the pair as a bracket, not the two ends as rival metrics.
    """
    rows = document["measurement"]["rows"]
    row = next((item for item in rows if item["tokens_per_rank"] == 64), rows[len(rows) // 2])
    latency = row["components"]["roundtrip"]["percentiles_us"]

    def percentile(block: str, name: str) -> float | str:
        # Absent on rows measured before the skew diagnostics were emitted.
        component = (row.get(block) or {}).get(name) or row.get(block) or {}
        return (component.get("percentiles_us") or {}).get("p50", "-")

    return (
        row["tokens_per_rank"], latency["p50"], latency["p99"],
        percentile("cross_rank_min_us", "roundtrip"),
        percentile("cross_rank_spread_us", ""),
    )


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
        "| ver | sku | backend | mode | precision | suite | phase | routing | ep | outcome "
        "| T* | p50 us | p99 us | min50 us | skew us |",
        "|--:|---|---|---|---|---|---|---|--:|---|--:|--:|--:|--:|--:|",
    ]
    for document in documents:
        sku, backend, suite, routing, mode, phase, ep, precision = _identity(document)
        token, p50, p99, min50, skew = _headline(document)
        lines.append(
            f"| {document['version']} | {sku} | `{backend}` | {mode} | {precision} | {suite} | "
            f"{phase} | {routing} | {ep} | "
            f"{document['outcome']['status']} | {token} | {p50} | {p99} | {min50} | {skew} |"
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
