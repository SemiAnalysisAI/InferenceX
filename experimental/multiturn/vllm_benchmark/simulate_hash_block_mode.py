#!/usr/bin/env python3
"""Dry-run the trace_replay_tester --hash-block-mode code path, pure-logic.

Matches the schema of neon_trace_simulation.json:
    {
      "trace": "<path>",
      "total_requests": N,
      "summary": {...},
      "requests": [
        {"req": i, "action": ..., "in_tokens": N, "hash_id_count": N,
         "prev_hash_id_count": N, "kept": N, "removed": N, "new": N,
         "kept_tokens": N, "full_reset": bool},
        ...
      ]
    }

The "action" classification reproduces the legacy build_messages logic so this
file can be compared directly to neon_trace_simulation.json. In addition it
records hash-block-mode-specific fields so you can see what the hash-block
code path WOULD produce without having to run the real benchmark:
  - shared_prefix_with_any_prior
  - hash_block_hit_tokens
  - hash_block_miss_tokens

Usage:
    # Single trace (same as neon_trace_simulation.json):
    python3 simulate_hash_block_mode.py \\
        --trace kv-cache-tester/traces_neon/trace_0001.json \\
        --output neon_trace_simulation_hash_block.json

    # Aggregate across a whole directory:
    python3 simulate_hash_block_mode.py \\
        --trace-dir kv-cache-tester/traces_neon \\
        --output neon_trace_simulation_hash_block_agg.json
"""

import argparse
import json
from pathlib import Path


def classify_action(curr_seq: tuple, prefix_len: int, is_first: bool):
    """Classify what hash-block mode's build_messages branch does for this request.

    Hash-block mode has exactly TWO code paths (see trace_replay_tester.py:1306-1323):
      1. FIRST            — first request in the session (no prior)
      2. HASH_BLOCK_FILL  — deterministic content from this request's hash_ids

    The sub-labels below refine HASH_BLOCK_FILL by how much prefix overlaps
    with any prior request (what the prefix cache will actually hit):

      - HIT          : full hash_id prefix match with some prior request
      - PARTIAL_HIT  : non-zero but not full prefix match
      - MISS         : zero prefix overlap with any prior request
      - NO_HASH_IDS  : request has an empty hash_ids list (fallback text path)
    """
    if is_first:
        return "FIRST"
    if not curr_seq:
        return "NO_HASH_IDS"
    if prefix_len == 0:
        return "MISS"
    if prefix_len == len(curr_seq):
        return "HIT"
    return "PARTIAL_HIT"


def longest_shared_prefix(curr_seq, prior_seqs) -> int:
    best = 0
    for p in prior_seqs:
        lim = min(len(curr_seq), len(p))
        k = 0
        while k < lim and curr_seq[k] == p[k]:
            k += 1
        if k > best:
            best = k
            if best == len(curr_seq):
                break
    return best


def simulate_trace(trace_path: Path) -> dict:
    with open(trace_path) as f:
        raw = json.load(f)

    block_size = raw.get("block_size", 64)
    trace_id = raw.get("id", trace_path.stem)

    prior_seqs: list[tuple[int, ...]] = []
    counts = {"FIRST": 0, "HIT": 0, "PARTIAL_HIT": 0, "MISS": 0, "NO_HASH_IDS": 0}
    set_seen: set = set()
    set_hit = set_total = 0
    per_request = []
    total_in = total_hb_hit = 0
    prev_hash_id_count = 0

    for i, req in enumerate(raw.get("requests", [])):
        if req.get("type") == "subagent":
            continue

        curr_seq = tuple(req.get("hash_ids") or [])
        in_tokens = req.get("in") or req.get("input_tokens") or 0
        total_in += in_tokens

        # Longest hash_id prefix shared with ANY prior request → what vLLM's
        # prefix cache will actually hit under hash-block mode.
        prefix_len = longest_shared_prefix(curr_seq, prior_seqs)
        hb_hit_tokens = min(prefix_len * block_size, in_tokens)
        hb_miss_tokens = max(0, in_tokens - hb_hit_tokens)
        total_hb_hit += hb_hit_tokens

        action = classify_action(curr_seq, prefix_len, is_first=(len(prior_seqs) == 0))
        counts[action] = counts.get(action, 0) + 1

        # Infinite-independent-cache upper bound
        set_hit += sum(1 for x in curr_seq if x in set_seen)
        set_total += len(curr_seq)

        per_request.append({
            "req": i,
            "action": action,
            "in_tokens": in_tokens,
            "hash_id_count": len(curr_seq),
            "prev_hash_id_count": prev_hash_id_count,
            "shared_prefix_with_any_prior": prefix_len,
            "hash_block_hit_tokens": hb_hit_tokens,
            "hash_block_miss_tokens": hb_miss_tokens,
        })

        set_seen.update(curr_seq)
        prior_seqs.append(curr_seq)
        prev_hash_id_count = len(curr_seq)

    hb_rate = total_hb_hit / total_in if total_in else 0.0
    set_rate = set_hit / set_total if set_total else 0.0

    return {
        "trace": str(trace_path),
        "mode": "hash-block",
        "total_requests": len(per_request),
        "summary": {
            "first": counts["FIRST"],
            "hit": counts["HIT"],
            "partial_hit": counts["PARTIAL_HIT"],
            "miss": counts["MISS"],
            "no_hash_ids": counts["NO_HASH_IDS"],
            "hash_block_hit_tokens": total_hb_hit,
            "hash_block_input_tokens": total_in,
            "hash_block_cache_hit_rate": round(hb_rate, 4),
            "set_cache_hit_blocks": set_hit,
            "set_cache_total_blocks": set_total,
            "set_cache_hit_rate": round(set_rate, 4),
        },
        "requests": per_request,
    }


def simulate_dir(trace_dir: Path) -> dict:
    trace_files = sorted(trace_dir.glob("trace_*.json"))
    if not trace_files:
        raise SystemExit(f"No trace_*.json in {trace_dir}")

    print(f"Dry-running hash-block mode on {len(trace_files)} traces...")
    per_trace = []
    totals = {
        "first": 0, "hit": 0, "partial_hit": 0, "miss": 0, "no_hash_ids": 0,
        "hash_block_hit_tokens": 0, "hash_block_input_tokens": 0,
        "set_cache_hit_blocks": 0, "set_cache_total_blocks": 0,
        "total_requests": 0,
    }
    for tf in trace_files:
        sim = simulate_trace(tf)
        s = sim["summary"]
        totals["total_requests"] += sim["total_requests"]
        for k in ("first", "hit", "partial_hit", "miss", "no_hash_ids",
                  "hash_block_hit_tokens", "hash_block_input_tokens",
                  "set_cache_hit_blocks", "set_cache_total_blocks"):
            totals[k] += s[k]
        per_trace.append({
            "trace": tf.name,
            "total_requests": sim["total_requests"],
            "summary": s,
        })

    hb_rate = (totals["hash_block_hit_tokens"] / totals["hash_block_input_tokens"]
               if totals["hash_block_input_tokens"] else 0.0)
    set_rate = (totals["set_cache_hit_blocks"] / totals["set_cache_total_blocks"]
                if totals["set_cache_total_blocks"] else 0.0)

    return {
        "trace_dir": str(trace_dir),
        "mode": "hash-block",
        "num_traces": len(trace_files),
        "aggregate": {
            **totals,
            "hash_block_cache_hit_rate": round(hb_rate, 4),
            "set_cache_hit_rate": round(set_rate, 4),
        },
        "per_trace": per_trace,
    }


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="Single trace JSON — same schema as neon_trace_simulation.json")
    src.add_argument("--trace-dir", help="Directory of trace_*.json — aggregate across all")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if args.trace:
        data = simulate_trace(Path(args.trace))
    else:
        data = simulate_dir(Path(args.trace_dir))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)

    s = data.get("summary") or data["aggregate"]
    n_req = data.get("total_requests") or data["aggregate"]["total_requests"]
    print(f"\nSummary ({n_req:,} requests, hash-block mode):")
    print(f"  FIRST:        {s['first']}")
    print(f"  HIT:          {s['hit']}            (full prefix match)")
    print(f"  PARTIAL_HIT:  {s['partial_hit']}    (some prefix overlap)")
    print(f"  MISS:         {s['miss']}           (no prefix overlap w/ any prior)")
    print(f"  NO_HASH_IDS:  {s['no_hash_ids']}    (fallback text path)")
    print(f"  hash-block prefix cache hit rate: {s.get('hash_block_cache_hit_rate', 0):.1%}")
    print(f"  infinite-set cache hit rate:      {s.get('set_cache_hit_rate', 0):.1%}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
