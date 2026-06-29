#!/usr/bin/env python3
"""CollectiveX — prune results/ to the fresh canonical set.

The results/ dir accumulates every GHA download across sessions (885+ files): many are SUPERSEDED
debug re-runs of the same config, stale runs from older code, or failed-case stubs that now have a
valid newer counterpart. This prunes to the FRESH canonical set:

  * group every result by its comparison_key (the config identity the plot/aggregator uses);
  * within a group, keep the newest KEEP_PER_KEY runs whose publication_status/status is usable
    (official | comparable-experimental | valid) — newest by generated_at;
  * move everything else (older-than-KEEP valids, and failed/invalid runs that have >=1 usable run in
    their group) to results/.superseded/ (NOT hard-deleted — recoverable; already out of the plot glob).

Keeping KEEP_PER_KEY>1 preserves the repeat-run aggregation (median + error bands across runs, a
P0 deliverable) while removing the long tail of stale debug duplicates. A failed-case with NO usable
counterpart is KEPT (the "preserve genuinely-failed cases" deliverable). env_*.json + analysis.json
are kept. Stdlib only.

  python3 prune_results.py --results-dir results            # prune (move to .superseded)
  python3 prune_results.py --results-dir results --dry-run   # just report
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

KEEP_PER_KEY = 3                       # newest usable runs to keep per config (repeat-run aggregation)
USABLE = {"official", "comparable-experimental", "valid"}


def _doc_key(d: dict) -> str:
    """Config identity: top-level comparison_key (EP), else family+runner+a stable signature."""
    if d.get("comparison_key"):
        return str(d["comparison_key"])
    # collective families (kv-cache/copy-engine/nccl/rl-mesh/allreduce-fw): derive from group keys.
    keys = [g.get("comparison_key") for g in d.get("groups", []) if g.get("comparison_key")]
    if keys:
        return "|".join(sorted(str(k) for k in keys))
    return "|".join(str(d.get(k, "")) for k in ("family", "runner", "backend", "phase", "measurement_contract"))


def _usable(d: dict) -> bool:
    ps = d.get("publication_status") or d.get("status")
    return ps in USABLE


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune CollectiveX results/ to the fresh canonical set")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--keep-per-key", type=int, default=KEEP_PER_KEY)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rd = a.results_dir
    sup = os.path.join(rd, ".superseded")
    files = [f for f in os.listdir(rd) if f.endswith(".json")
             and not f.startswith("env_") and f != "analysis.json"]
    docs = []  # (fname, key, generated_at, usable, is_failed)
    for f in files:
        try:
            d = json.load(open(os.path.join(rd, f)))
        except Exception:
            continue
        docs.append((f, _doc_key(d), d.get("generated_at") or d.get("generated_at", ""),
                     _usable(d), f.startswith("failed_") or d.get("record_type") == "failed-case"))

    # group by key
    groups: dict = {}
    for rec in docs:
        groups.setdefault(rec[1], []).append(rec)

    move = []
    for key, recs in groups.items():
        usable = sorted([r for r in recs if r[3]], key=lambda r: r[2], reverse=True)
        keep = set(r[0] for r in usable[:a.keep_per_key])
        for r in recs:
            f, _, _, is_usable, is_failed = r
            if f in keep:
                continue
            # keep a failed/unusable run ONLY if its group has NO usable run at all
            if (is_failed or not is_usable) and not usable:
                continue
            move.append(f)

    print(f"prune: {len(files)} result files, {len(groups)} configs, keep<= {a.keep_per_key}/config -> "
          f"move {len(move)} superseded/stale to {sup}")
    if a.dry_run:
        for f in sorted(move)[:20]:
            print("  would move:", f)
        return 0
    os.makedirs(sup, exist_ok=True)
    for f in move:
        try:
            shutil.move(os.path.join(rd, f), os.path.join(sup, f))
        except Exception as e:
            print(f"  WARN move {f}: {e!r}")
    print(f"pruned -> {len([x for x in os.listdir(rd) if x.endswith('.json')])} json kept in {rd}, "
          f"{len(os.listdir(sup))} in .superseded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
