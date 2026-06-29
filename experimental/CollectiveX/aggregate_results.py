#!/usr/bin/env python3
"""CollectiveX — result aggregator (the end-of-sweep collector).

The sweep workflow (collectivex-sweep.yml) fans out one matrix CELL per SHARD
(platform × backend × mode × resource), each cell sweeping its cases in a single
allocation and emitting a handful of per-case result JSONs. Instead of leaving
thousands of individual files scattered across the repo, this aggregator COLLECTS
every shard's results into ONE compact line-delimited file:

    results/aggregate/collectivex_ep.ndjson     # one result doc per line

That single artifact is the deliverable the plotter + the app read; the per-case
JSONs stay inside the run as transient shard intermediates. Within a shard, a
config that was re-run keeps only its NEWEST usable doc (newest generated_at with
publication_status/status in official|comparable-experimental|valid), with
genuinely-failed configs preserved when they have no usable counterpart — the same
hygiene prune_results.py applies, folded into the merge so the aggregate is already
canonical.

  python3 aggregate_results.py --in-dir <shards_root> --out results/aggregate/collectivex_ep.ndjson
  python3 aggregate_results.py --in-dir results --explode results   # ndjson -> per-doc (for the plotter)

Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os

USABLE = {"official", "comparable-experimental", "valid"}


def _key(d: dict) -> str:
    """Config identity used to keep newest-per-config (mirrors prune_results._doc_key)."""
    if d.get("comparison_key"):
        return str(d["comparison_key"])
    keys = [g.get("comparison_key") for g in d.get("groups", []) if g.get("comparison_key")]
    if keys:
        return "|".join(sorted(str(k) for k in keys))
    return "|".join(str(d.get(k, "")) for k in ("family", "runner", "backend", "phase",
                                                "measurement_contract"))


def _usable(d: dict) -> bool:
    return (d.get("publication_status") or d.get("status")) in USABLE


def _iter_docs(in_dir: str):
    """Yield (source, doc) for every result doc under in_dir — both per-file *.json and
    line-delimited *.ndjson (so aggregates can be re-merged idempotently)."""
    for root, _dirs, files in os.walk(in_dir):
        for f in files:
            if f.startswith("env_") or f == "analysis.json":
                continue
            p = os.path.join(root, f)
            if f.endswith(".ndjson"):
                for line in open(p):
                    line = line.strip()
                    if line:
                        try:
                            yield p, json.loads(line)
                        except Exception:
                            pass
            elif f.endswith(".json"):
                try:
                    yield p, json.load(open(p))
                except Exception:
                    pass


def aggregate(in_dir: str, keep_per_key: int = 3) -> list:
    """Collect every result doc, keep newest KEEP_PER_KEY usable per config (+ orphan failures)."""
    groups: dict = {}
    for _src, d in _iter_docs(in_dir):
        groups.setdefault(_key(d), []).append(d)
    out = []
    for _k, docs in groups.items():
        usable = sorted([d for d in docs if _usable(d)],
                        key=lambda d: d.get("generated_at", ""), reverse=True)
        if usable:
            out.extend(usable[:keep_per_key])
        else:
            # a config that ONLY ever failed: keep its newest record (preserve failed cases)
            out.append(sorted(docs, key=lambda d: d.get("generated_at", ""), reverse=True)[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX result aggregator")
    ap.add_argument("--in-dir", default="results", help="root to walk for shard result files")
    ap.add_argument("--out", default="results/aggregate/collectivex_ep.ndjson")
    ap.add_argument("--keep-per-key", type=int, default=3)
    ap.add_argument("--explode", metavar="DIR",
                    help="instead of merging, write each ndjson doc in --in-dir back to a per-doc "
                         "JSON under DIR (so the existing plotter glob can read an aggregate)")
    a = ap.parse_args()

    if a.explode:
        os.makedirs(a.explode, exist_ok=True)
        n = 0
        for _src, d in _iter_docs(a.in_dir):
            name = (d.get("artifact_name") or
                    f"{d.get('runner','x')}_{d.get('backend',d.get('op','x'))}_"
                    f"{d.get('phase','na')}_{d.get('generated_at','')}".replace(":", "-"))
            with open(os.path.join(a.explode, f"{name}.json"), "w") as fh:
                json.dump(d, fh)
            n += 1
        print(f"explode: wrote {n} per-doc JSON to {a.explode}")
        return 0

    docs = aggregate(a.in_dir, a.keep_per_key)
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
