#!/usr/bin/env python3
"""Per-component EP bandwidth, computed purely as a consumer of case-attempt artifacts.

Two views over the bytes + latency every record already emits — no harness or GPU change,
and this gates nothing (see docs/methodology.md):

  A. effective bandwidth -- per-GPU GB/s at each ladder point: bytes / component latency.
  B. alpha/beta fit -- OLS of p50 latency vs bytes, latency ~= alpha + bytes/beta, which
     separates the bandwidth term from the per-call floor that dominates small T.

Bytes are the WIRE basis (`wire_byte_provenance`): one copy per unique (token, dest-rank)
pair for rank-deduplicating layouts, one copy per (token, expert) for the low-latency
kernels that do not deduplicate — whichever the row's `logical_copies.wire` declares. That
makes GB/s comparable across backends and modes. Two adjustments still apply when reading
against a link's peak: these bytes include the copies a rank routes to itself, which never
cross the interconnect, so fabric traffic is (1 - routing.locality.local_rank_fraction) x
these figures; and beta is a MARGINAL rate, so it legitimately sits above every measured
point. Pre-wire artifacts fall back to the rank-deduplicated `byte_provenance`, which for
per-assignment LL rows understates the wire rate (a lower bound, ~34% low on nccl-ep LL
EP8 at T=128), never overstates it.

A line through a ladder is a MODEL and a positive slope alone is not evidence, so every Fit
carries its own quality and beta is withheld unless it clears both gates below. Deliberately
NOT added: a per-SKU peak table to check beta against; that hardware knowledge belongs in
platform_config.json, ages badly, and would mislead in its own right.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import load_results  # noqa: E402  (sibling consumer, stdlib-only)

# stage moves nothing (bytes always 0); isolated_sum is a derived percentile sum, not a real
# chained rate. Only these three carry a measured latency + bytes.
COMPONENTS = ("dispatch", "combine", "roundtrip")

# Below this the line is not describing the ladder: two rungs of noise once yielded 1018 GB/s
# per GPU on a B200 -- above the link's per-direction peak -- at R2 = 0.29, printed
# indistinguishably from an R2 = 0.99 fit. A judgement call; good fits observed at 0.99+.
FIT_MIN_R2 = 0.9
# R2 alone is NOT enough -- a latency-bound dispatch ladder came back at 3763 GB/s with
# R2 = 0.92, because alpha explained nearly everything and the slope was not identifiable.
# So also require the transfer term to be a real share of the top rung's latency; below this
# the honest reading is "not bandwidth-limited here", not "very fast".
FIT_MIN_BANDWIDTH_SHARE = 0.25
# If the smallest rung is not within this fraction of the largest, alpha (the intercept at
# zero bytes) is an extrapolation, not a measured floor -- prefill ladders start at T=1024.
ALPHA_NEAR_ZERO_FRACTION = 0.1
MIN_FIT_POINTS = 3


class Fit(NamedTuple):
    """A fitted ladder. `r2`, `points` and `bandwidth_share` are part of the result, not
    diagnostics: a caller that ignores them can print an impossible bandwidth."""
    alpha_us: float
    beta_gbps: float
    r2: float
    points: int
    alpha_extrapolated: bool
    excluded_rows: int
    bandwidth_share: float  # transfer term / largest measured latency, at the top rung

    @property
    def beta_is_reliable(self) -> bool:
        return self.r2 >= FIT_MIN_R2 and self.bandwidth_share >= FIT_MIN_BANDWIDTH_SHARE


def _ep(document: dict) -> int:
    return int(document["identity"]["case_factors"]["case"]["ep"])


def _algbw_per_gpu(total_logical_bytes: int, latency_us: float, ep: int) -> float | None:
    """Per-GPU effective GB/s, or None when the latency cannot yield a rate. Bytes are the
    AGGREGATE world payload (routed_copies, routing.py), hence the divide by EP size."""
    if latency_us <= 0:
        return None
    return total_logical_bytes / (latency_us * 1e-6) / 1e9 / ep


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """(intercept, slope, r2) of a 2-parameter least-squares fit (stdlib only)."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
             / sum((x - mean_x) ** 2 for x in xs))
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    # ss_tot == 0: no variance to explain, so the fit cannot be credited with anything.
    return intercept, slope, (1.0 - ss_res / ss_tot) if ss_tot else 0.0


def fit_alpha_beta(document: dict, component: str, pct: str = "p50") -> Fit | None:
    """Fit one component's ladder, or None when no fit is defensible at all.

    Regress against the emitted bytes, never T (routed_copies is sub-linear in T at high
    fan-out); p50 by default, since p99 carries tail noise. Rungs that failed the correctness
    gate are excluded — a corrupt payload is not a measurement of how fast correct data moves.
    """
    usable, excluded = [], 0
    for row in document["measurement"]["rows"]:
        percentiles = row["components"][component]["percentiles_us"]
        if not percentiles:
            continue
        if not row.get("correctness", {}).get("passed", True):
            excluded += 1
            continue
        usable.append((_wire_bytes(row, component), percentiles[pct]))
    if len(usable) < MIN_FIT_POINTS or len({x for x, _ in usable}) < 2:
        return None  # too few points, or zero-variance x (e.g. a degenerate ladder)
    xs, ys = [x for x, _ in usable], [y for _, y in usable]
    alpha_us, slope, r2 = _fit_line(xs, ys)
    if slope <= 0:
        return None  # anti-correlated / noise-dominated: no meaningful bandwidth term
    return Fit(  # slope is us per aggregate byte; 1e-3/slope = aggregate GB/s; /ep = per-GPU
        alpha_us=alpha_us,
        beta_gbps=1e-3 / slope / _ep(document),
        r2=r2,
        points=len(usable),
        alpha_extrapolated=min(xs) > ALPHA_NEAR_ZERO_FRACTION * max(xs),
        excluded_rows=excluded,
        bandwidth_share=(slope * max(xs) / max(ys)) if max(ys) > 0 else 0.0,
    )


def _wire_bytes(row: dict, component: str) -> int:
    """Bytes the kernels actually move for this component.

    Prefers `wire_byte_provenance` (per-assignment for token-expert LL receives); artifacts
    written before that field carry only the rank-deduplicated `byte_provenance`, which for
    those LL rows is a LOWER BOUND on wire traffic (~topk/fanout below it), so a bandwidth
    derived from the fallback understates the wire rate rather than overstating it.
    """
    provenance = row.get("wire_byte_provenance") or row["byte_provenance"]
    return provenance[component]["total_logical_bytes"]


def _format_fit(component: str, fit: Fit) -> str:
    """One component's fit, withholding what the data does not support."""
    if not fit.beta_is_reliable:
        why = "poor fit" if fit.r2 < FIT_MIN_R2 else "not bandwidth-bound"
        return (f"{component}: beta=unreliable [{why}] "
                f"(R2={fit.r2:.2f}, transfer share={fit.bandwidth_share:.2f}, n={fit.points})")
    alpha = f"alpha={fit.alpha_us:6.1f}us" + ("*" if fit.alpha_extrapolated else " ")
    return (f"{component}: beta={fit.beta_gbps:6.1f} GB/s {alpha} "
            f"(R2={fit.r2:.3f}, n={fit.points})")


def _cell(row: dict, component: str, ep: int) -> str:
    percentiles = row["components"][component]["percentiles_us"]
    if not percentiles:
        return f"{component}=n/a"
    nbytes = _wire_bytes(row, component)
    p50 = _algbw_per_gpu(nbytes, percentiles["p50"], ep)
    p99 = _algbw_per_gpu(nbytes, percentiles["p99"], ep)
    return f"{component}=n/a" if p50 is None or p99 is None \
        else f"{component}={p50:6.1f}/{p99:<6.1f}"


def _sort_key(document: dict):
    case = document["identity"]["case_factors"]["case"]
    # mode sorts before phase: a cell's normal and low-latency rows are otherwise
    # indistinguishable here, and they are different kernel families.
    return (document["identity"]["case_factors"]["sku"], case["backend"],
            case["mode"], case["precision"], case["phase"], case["ep"],
            document.get("generated_at", ""))


def _provenance(document: dict) -> str:
    """Enough identity to tell two attempts of one case apart — CI result directories
    routinely hold both, and they can disagree."""
    run_id = document["identity"].get("allocation_factors", {}).get("run_id")
    attempt = document["identity"].get("attempt_ordinal")
    parts = ((document.get("generated_at") or "")[:19],
             f"run {run_id}" if run_id else "", f"attempt {attempt}" if attempt else "")
    return " · ".join(p for p in parts if p)


def render(documents: list[dict]) -> str:
    lines = [
        "## CollectiveX EP bandwidth (per-GPU, wire-basis payload)",
        "",
        "GB/s at p50/p99 *latency* -- the p99-latency figure is worst-case bandwidth, not a "
        f"'p99 bandwidth'. `fit`: latency ~= alpha + bytes/beta over the ladder; beta is "
        f"withheld below R2 {FIT_MIN_R2} or when the ladder never became bandwidth-bound, `*` "
        "marks an extrapolated alpha, and rungs failing the correctness gate are excluded.",
        "",
    ]
    for document in sorted(documents, key=_sort_key):
        case = document["identity"]["case_factors"]["case"]
        ep = _ep(document)
        lines.append(
            f"### {document['identity']['case_factors']['sku']} `{case['backend']}` "
            f"{case['mode']} {case['precision']} {case['phase']} ep{ep} — "
            f"{document['outcome']['status']}"
        )
        if provenance := _provenance(document):
            lines.append(f"  ({provenance})")
        for row in document["measurement"]["rows"]:
            cross_node = (row["routing"].get("locality") or {}).get("cross_node_fraction")
            suffix = f"  xnode={cross_node * 100:4.0f}%" if cross_node is not None else ""
            if not row.get("correctness", {}).get("passed", True):
                suffix += "  [correctness FAILED]"
            cells = "  ".join(_cell(row, component, ep) for component in COMPONENTS)
            lines.append(f"  T={row['tokens_per_rank']:<5} {cells}{suffix}")
        fits = [(c, f) for c in COMPONENTS if (f := fit_alpha_beta(document, c))]
        dropped = max((f.excluded_rows for _, f in fits), default=0)
        note = f"  (excluded {dropped} gate-failed rung(s))" if dropped else ""
        lines.append("  fit " + ("   ".join(_format_fit(c, f) for c, f in fits)
                                 if fits else "insufficient points") + note)
        lines.append("")
    if not documents:
        lines.append("> No case-attempt documents found.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CollectiveX per-component EP bandwidth (pure consumer; gates nothing)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runner")
    parser.add_argument("--ts")
    parser.add_argument("--all", action="store_true",
                        help="include non-success outcomes (skipped by default)")
    args = parser.parse_args()
    documents = load_results(args.results_dir, args.runner, args.ts)
    if not args.all:
        documents = [d for d in documents if d["outcome"]["status"] == "success"]
    print(render(documents))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
