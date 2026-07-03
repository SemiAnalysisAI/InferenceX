#!/usr/bin/env python3
"""CollectiveX — result aggregator (the end-of-sweep collector).

The sweep workflow (collectivex-sweep.yml) fans out one matrix CELL per SHARD
(platform × backend × mode × resource), each cell sweeping its cases in a single
allocation and emitting a handful of per-case result JSONs. Instead of leaving
thousands of individual files scattered across the repo, this aggregator COLLECTS
every shard's results into ONE compact line-delimited file:

    results/aggregate/collectivex_ep.ndjson     # one result doc per line

That aggregate is a transient input to bundle validation and the future local
publisher; the per-case JSONs stay inside the run as transient shard intermediates. Within a shard, a
config that was re-run keeps only its NEWEST usable doc (newest generated_at with
publication_status/status in official|comparable-experimental|valid), with
genuinely-failed configs preserved when they have no usable counterpart. The hygiene
rule is folded into the merge so the aggregate is already canonical.

  python3 aggregate_results.py --in-dir <shards_root> --out results/aggregate/collectivex_ep.ndjson

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os

USABLE = {"official", "comparable-experimental", "valid"}


def _first(*values):
    """Return the first available value while preserving false/zero identity fields."""
    return next((value for value in values if value is not None), None)


def _failed_key(d: dict) -> str:
    """Scheduled identity for legacy failed records that predate top-level ``case_id``."""
    failure = d.get("failure") if isinstance(d.get("failure"), dict) else {}
    raw_case = failure.get("case") if isinstance(failure.get("case"), dict) else {}
    case = dict(raw_case)
    shape = d.get("shape") if isinstance(d.get("shape"), dict) else {}
    quant = shape.get("quant") if isinstance(shape.get("quant"), dict) else {}
    eplb_doc = d.get("eplb")
    eplb = eplb_doc.get("enabled") if isinstance(eplb_doc, dict) else eplb_doc
    workload = d.get("workload_name")
    if workload is None:
        workload_doc = d.get("workload")
        workload = (workload_doc.get("workload_id") if isinstance(workload_doc, dict)
                    else workload_doc)
    routing_doc = d.get("routing_profile")
    routing = routing_doc.get("routing") if isinstance(routing_doc, dict) else routing_doc

    # Current failed records already carry these fields in failure.case. Top-level aliases keep
    # older records distinct whenever that scheduled identity was available there instead.
    fallbacks = {
        "suite": d.get("suite"),
        "workload": workload,
        "backend": d.get("backend"),
        "phase": d.get("phase"),
        "ep": d.get("ep_size"),
        "mode": d.get("mode"),
        "dispatch_dtype": _first(shape.get("dispatch_dtype"), d.get("dispatch_dtype")),
        "contract": d.get("measurement_contract"),
        "routing": _first(shape.get("routing"), routing),
        "eplb": eplb,
        "combine_quant_mode": _first(quant.get("combine_quant_mode"),
                                      d.get("combine_quant_mode")),
        "resource_mode": d.get("resource_mode"),
        "tokens_ladder": _first(
            (d.get("reproduction") or {}).get("tokens_ladder")
            if isinstance(d.get("reproduction"), dict) else None,
            d.get("tokens_ladder"),
        ),
    }
    for field, value in fallbacks.items():
        case[field] = _first(case.get(field), value)

    identity = {
        "family": d.get("family"),
        "runner": d.get("runner"),
        "topology_class": d.get("topology_class"),
        "case": case,
    }
    return "failed:" + json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _key(d: dict) -> str:
    """Config identity used to keep newest-per-config."""
    if d.get("case_id"):
        return "case:" + str(d["case_id"])
    if d.get("comparison_key"):
        return str(d["comparison_key"])
    keys = [g.get("comparison_key") for g in d.get("groups", []) if g.get("comparison_key")]
    if keys:
        return "|".join(sorted(str(k) for k in keys))
    if d.get("record_type") == "failed-case":
        return _failed_key(d)
    return "|".join(str(d.get(k, "")) for k in ("family", "runner", "backend", "phase",
                                                "measurement_contract"))


def _usable(d: dict) -> bool:
    return (d.get("publication_status") or d.get("status")) in USABLE


def _document(value, source: str) -> dict:
    if not isinstance(value, dict):
        raise SystemExit(f"aggregate: {source} is not a JSON object")
    return value


def _iter_docs(in_dir: str):
    """Yield (source, doc) for every result doc under in_dir — both per-file *.json and
    line-delimited *.ndjson (so aggregates can be re-merged idempotently)."""
    for root, _dirs, files in os.walk(in_dir):
        for f in files:
            if f.startswith("env_") or f == "analysis.json":
                continue
            p = os.path.join(root, f)
            if f.endswith(".ndjson"):
                with open(p) as fh:
                    for line_number, line in enumerate(fh, 1):
                        line = line.strip()
                        if line:
                            try:
                                value = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise SystemExit(
                                    f"aggregate: malformed NDJSON at {p}:{line_number}: {exc}"
                                ) from exc
                            yield p, _document(value, f"{p}:{line_number}")
            elif f.endswith(".json"):
                try:
                    with open(p) as fh:
                        value = json.load(fh)
                except (OSError, json.JSONDecodeError) as exc:
                    raise SystemExit(f"aggregate: malformed JSON at {p}: {exc}") from exc
                yield p, _document(value, p)


def aggregate(in_dir: str) -> list:
    """Collect every result doc into one newest terminal outcome per config."""
    groups: dict = {}
    for _src, d in _iter_docs(in_dir):
        groups.setdefault(_key(d), []).append(d)
    out = []
    for _k, docs in groups.items():
        usable = sorted([d for d in docs if _usable(d)],
                        key=lambda d: d.get("generated_at", ""), reverse=True)
        if usable:
            out.append(usable[0])
        else:
            # a config that ONLY ever failed: keep its newest record (preserve failed cases)
            out.append(sorted(docs, key=lambda d: d.get("generated_at", ""), reverse=True)[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX result aggregator")
    ap.add_argument("--in-dir", default="results", help="root to walk for shard result files")
    ap.add_argument("--out", default="results/aggregate/collectivex_ep.ndjson")
    a = ap.parse_args()

    docs = aggregate(a.in_dir)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for d in docs:
            fh.write(json.dumps(d, separators=(",", ":")) + "\n")
    skus = sorted({str(d.get("runner", "?")).split("_")[0].split("-")[0] for d in docs})
    backs = sorted({str(d.get("backend") or d.get("op") or "?") for d in docs})
    print(f"aggregate: {len(docs)} docs -> {a.out}  (SKUs={skus} backends={backs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
