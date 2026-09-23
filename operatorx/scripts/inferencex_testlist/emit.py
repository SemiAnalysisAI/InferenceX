"""Aggregate per-row op enumerations into deduped testlist JSON files."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

from .enumerate import OpTriple


# Which top-level op_type lands in which testlist file.
_OP_TO_TESTLIST = {
    "gemm": "gemm",
    "attention_mha": "attention",
    "attention_mla": "attention",
    "topk_routing": "moe",
    "moe_forward": "moe",
    "allreduce": "collectives",
    "allgather": "collectives",
    "reduce_scatter": "collectives",
    "alltoall": "collectives",
    "dispatch": "collectives",
    "combine": "collectives",
    "kvcache_swap": "memory",
}


def _dedupe_key(op_type: str, args: dict[str, Any]) -> tuple:
    """Treat entries with the same op_type + args as the same testlist entry."""
    return (op_type, tuple(sorted(args.items())))


def write_testlists(
    ops_with_names: list[OpTriple], out_dir: str, merge_with_existing: bool = True
) -> dict[str, int]:
    """Write deduped testlists. Returns counts per file.

    If `merge_with_existing` is True and the testlist files already exist,
    union the new entries into them so we don't drop hand-curated shapes.
    """
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[tuple]] = defaultdict(set)

    # Preload existing entries so we union rather than overwrite.
    if merge_with_existing:
        for fname in set(_OP_TO_TESTLIST.values()):
            path = os.path.join(out_dir, f"{fname}.json")
            if os.path.isfile(path):
                try:
                    existing = json.load(open(path))
                except Exception:
                    existing = []
                for entry in existing:
                    if "type" not in entry or "args" not in entry:
                        continue
                    key = _dedupe_key(entry["type"], entry["args"])
                    if key in seen[fname]:
                        continue
                    seen[fname].add(key)
                    by_file[fname].append(entry)

    for op_type, args, name in ops_with_names:
        target = _OP_TO_TESTLIST.get(op_type)
        if target is None:
            continue
        key = _dedupe_key(op_type, args)
        if key in seen[target]:
            continue
        seen[target].add(key)
        entry: dict[str, Any] = {"type": op_type, "args": args}
        if name:
            entry["name"] = name
        by_file[target].append(entry)

    os.makedirs(out_dir, exist_ok=True)
    counts: dict[str, int] = {}
    for fname, entries in by_file.items():
        path = os.path.join(out_dir, f"{fname}.json")
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)
            f.write("\n")
        counts[fname] = len(entries)
    return counts
