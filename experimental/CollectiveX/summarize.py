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
        if timestamp and timestamp not in path.name:
            continue
        try:
            with path.open() as handle:
                document = json.load(handle)
        except (OSError, ValueError):
            continue
        if not (isinstance(document, dict) and document.get("record_type") == CASE_RECORD_TYPE):
            continue
        # Filter on the SKU the row declares, not on the filename. Results are named
        # `<case_id>_<ts>-cNNN.json` and case_id joins its factors with "-", so the old
        # `startswith(f"{runner}_")` test could never match: the summary would have
        # rendered an empty table rather than failing.
        if runner and document.get("identity", {}).get("case_factors", {}).get("sku") != runner:
            continue
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


def _topology(document: dict) -> str:
    """Scale-up shape, because the same `ep` label is not the same hardware.

    GB200/GB300 run 4 GPUs per node in a 72-GPU MNNVL domain, so their EP8 spans two trays while
    every other SKU's EP8 is one node over NVLink or XGMI. Printing it stops the false comparison.
    """
    topology = document.get("topology") or {}
    per_node = topology.get("gpus_per_node")
    domain = topology.get("scale_up_domain")
    nodes = topology.get("nodes")
    if per_node is None or domain is None:
        return "-"
    return f"{nodes}x{per_node}/d{domain}"


def _wire_basis(document: dict) -> str:
    """Which copy basis this backend's kernels actually move.

    Low-latency deepep-v2/uccl-ep/nccl-ep receive one copy per (token, expert); MoRI's IntraNodeLL
    deduplicates by destination rank -- ~1.5x different combine traffic at EP8, so not equal work.
    """
    rows = document["measurement"]["rows"]
    copies = (rows[0] if rows else {}).get("logical_copies") or {}
    return {"per-assignment": "assign", "rank-deduplicated": "dedup"}.get(copies.get("wire"), "-")


def _headline(document: dict) -> tuple:
    """Headline row, with the skew bracket beside it.

    `p50`/`p99` are the chained pair period where the row carries one — what a decode loop pays
    per MoE layer, cross-rank median over back-to-back pairs. Otherwise `roundtrip`, an
    idle-pipeline latency reduced by cross-rank MAX. Different quantities, so the last tuple
    element reports which the row carries and `render` footnotes the table accordingly.

    That MAX charges entry stagger to the operation, by an amount that is a property of the
    backend — on identical h200 low-latency cells the per-iteration spread is 9.2us for
    deepep-v2/uccl-ep against 2.0us for nccl-ep. `min50` (MIN over the same iterations) and `skew`
    (per-iteration MAX-MIN) bracket it.
    """
    rows = document["measurement"]["rows"]
    if not rows:
        # Degrade rather than crash: one malformed shard must not lose the whole table. No row
        # means nothing to attribute, so it votes on neither footnote.
        return ("-", "-", "-", "-", "-", None)
    row = next((item for item in rows if item["tokens_per_rank"] == 64), rows[len(rows) // 2])
    period = (row["components"].get("pair_period") or {}).get("percentiles_us")
    latency = period or row["components"]["roundtrip"]["percentiles_us"]

    def percentile(block: str, name: str) -> float | str:
        # Absent on rows measured before the skew diagnostics were emitted.
        component = (row.get(block) or {}).get(name) or row.get(block) or {}
        return (component.get("percentiles_us") or {}).get("p50", "-")

    return (
        row["tokens_per_rank"], latency["p50"], latency["p99"],
        percentile("cross_rank_min_us", "roundtrip"),
        percentile("cross_rank_spread_us", ""),
        period is not None,
    )


def _kv_cell(rows: list[dict], kind: str, op: str, batch: str = "min"):
    """The largest-ISL row of a (kind, op) family — the bandwidth-bound
    point — at its smallest or largest measured batch. Paged cells read the
    largest measured block size (the production one when several ran)."""
    matching = [r for r in rows if r.get("kind") == kind and r.get("op") == op]
    if kind == "paged" and matching:
        block = max(r["page_tokens"] for r in matching)
        matching = [r for r in matching if r["page_tokens"] == block]
    if not matching:
        return "-", "-"
    isl = max(r["isl"] for r in matching)
    pick = min if batch == "min" else max
    row = pick((r for r in matching if r["isl"] == isl),
               key=lambda r: r.get("batch", 1))
    return row["gbps_p50"], row["latency_ms"]["p50"]


def render_kv(documents: list[dict]) -> list[str]:
    """kv-transfer table: paged bandwidth at the bandwidth-bound ISL plus the
    contiguous baseline, and the paged latency. Verify failures flip the
    outcome column (and the leg already failed in CI)."""
    lines = ["", "## CollectiveX KV-transfer results", "",
             "| ver | sku | backend | fabric | workload | precision | outcome | op "
             "| paged GB/s b1 | paged GB/s bmax "
             "| contig GB/s | paged ms b1 |",
             "|--:|---|---|---|---|---|---|---|--:|--:|--:|--:|"]
    for document in documents:
        factors = document["identity"]["case_factors"]
        case = factors["case"]
        rows = document["measurement"]["rows"]
        # Cells read the pull lane when measured, else the push lane (a
        # backend may serve one direction only, e.g. mooncake on Pollara
        # where upstream ionic RDMA READ is broken); the op column names
        # which lane the row's numbers come from.
        op = next((candidate for candidate in ("pull", "push")
                   if _kv_cell(rows, "paged", candidate)[0] != "-"), "pull")
        paged_gbps, paged_ms = _kv_cell(rows, "paged", op)
        paged_bmax, _ = _kv_cell(rows, "paged", op, batch="max")
        bulk_gbps, _ = _kv_cell(rows, "bulk", op)
        lines.append(
            f"| {document['version']} | {factors['sku']} | `{case['backend']}` | "
            f"{case['mode']} | {case['workload']} | {case['precision']} | "
            f"{document['outcome']['status']} | {op} | {paged_gbps} | {paged_bmax} | "
            f"{bulk_gbps} | {paged_ms} |"
        )
    lines.append("")
    lines.append("> Paged rows move requests' KV as vLLM's packed block-major descriptor "
                 "lists (one contiguous descriptor per physical block per cache group) "
                 "over randomized block tables; b1/bmax = requests posted per burst "
                 "(GB/s is burst-aggregate); contig is the single-descriptor contiguous "
                 "baseline (host-observed goodput, not proven wire utilization); op "
                 "names the measured direction. GB/s at the largest ISL "
                 "(bandwidth-bound).")
    return lines


def render(documents: list[dict]) -> str:
    kv_documents = sorted(
        (d for d in documents
         if d["identity"]["case_factors"]["case"].get("suite") == "kv-transfer"),
        key=_identity)
    documents = sorted(
        (d for d in documents
         if d["identity"]["case_factors"]["case"].get("suite") != "kv-transfer"),
        key=_identity)
    invalid = [d for d in documents + kv_documents if d["outcome"]["status"] != "success"]
    lines = ["## CollectiveX EP results", ""]
    if invalid:
        # The leg is already red (the benchmark entrypoints return nonzero on a
        # non-success outcome); call the count out loudly so it is not lost in the tables.
        lines.append(
            f"> **{len(invalid)} of {len(documents) + len(kv_documents)} outcome(s) INVALID** — "
            "the leg fails; see the outcome column below."
        )
        lines.append("")
    lines += [
        "| ver | sku | backend | mode | precision | suite | phase | routing | ep | topo "
        "| wire | outcome | T* | p50* us | p99* us | min50 us | skew us |",
        "|--:|---|---|---|---|---|---|---|--:|---|---|---|--:|--:|--:|--:|--:|",
    ]
    chained = []
    for document in documents:
        sku, backend, suite, routing, mode, phase, ep, precision = _identity(document)
        token, p50, p99, min50, skew, row_chained = _headline(document)
        if row_chained is not None:
            chained.append(row_chained)
        topo, wire = _topology(document), _wire_basis(document)
        lines.append(
            f"| {document['version']} | {sku} | `{backend}` | {mode} | {precision} | {suite} | "
            f"{phase} | {routing} | {ep} | {topo} | {wire} | "
            f"{document['outcome']['status']} | {token} | {p50} | {p99} | {min50} | {skew} |"
        )
    if not documents and not kv_documents:
        lines.append("\n> No valid native outcome documents found.")
    if kv_documents:
        lines += render_kv(kv_documents)
    # The starred columns can hold two different quantities, so the table always says which — and
    # says so loudly when it holds both, since a mixed column silently compares a steady-state
    # period against an idle-pipeline latency.
    if chained:
        carrying = sum(1 for flag in chained if flag)
        fallbacks = chained.count(False)
        period_note = ("`*` chained pair period (back-to-back pairs, cross-rank median) — "
                       "what a decode loop pays per layer")
        if fallbacks and carrying:
            lines.append(
                f"\n> {period_note}; **{fallbacks} of {len(chained)} row(s) predate it** and "
                "fall back to the drained `roundtrip` (cross-rank MAX). The two are different "
                "quantities — do not rank across them."
            )
        elif fallbacks:
            lines.append(
                "\n> `*` drained `roundtrip` (cross-rank MAX): no row here carries a chained "
                "pair period."
            )
        else:
            lines.append(f"\n> {period_note}.")
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
