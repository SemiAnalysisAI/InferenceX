#!/usr/bin/env python3
"""
Sample conversations from WildChat dataset based on criteria.

Usage:
    python sample_wildchat.py --num-convs 50 --turns 5 --output sample.json
    python sample_wildchat.py --num-convs 100 --min-turns 3 --max-turns 10 --output sample.json
    python sample_wildchat.py --num-convs 50 --turns 5 --min-tokens 100 --max-tokens 1000 --output sample.json

    # Sample with a target turn distribution (JSON: {turn_count: proportion})
    python sample_wildchat.py --num-convs 5000 --distribution '{"1":0.61,"2":0.19,"3":0.09,"4":0.05,"5":0.024,"6-8":0.024,"9+":0.012}' --output sample.json
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime

from datasets import load_dataset

NUM_PROC = os.cpu_count()


def parse_distribution(dist_str: str) -> dict[int, float]:
    """Parse a distribution spec like {"1":0.61,"2":0.19,"3":0.09,"6-8":0.024,"9+":0.012}.

    Supports:
      - Single turn: "3" -> {3: weight}
      - Range: "6-8" -> {6: weight/3, 7: weight/3, 8: weight/3}
      - Open-ended: "9+" -> {9: weight} (will sample from 9 and above)

    Returns: {turn_count: proportion} with proportions normalized to sum to 1.
    """
    raw = json.loads(dist_str)
    expanded = {}
    for key, weight in raw.items():
        key = key.strip()
        if key.endswith("+"):
            # Open-ended range like "9+"
            turn = int(key[:-1])
            expanded[f"{turn}+"] = weight
        elif "-" in key:
            # Range like "6-8"
            lo, hi = key.split("-")
            lo, hi = int(lo), int(hi)
            per_turn = weight / (hi - lo + 1)
            for t in range(lo, hi + 1):
                expanded[t] = expanded.get(t, 0) + per_turn
        else:
            t = int(key)
            expanded[t] = expanded.get(t, 0) + weight

    # Normalize
    total = sum(expanded.values())
    if abs(total - 1.0) > 0.05:
        print(f"Warning: distribution weights sum to {total:.3f}, expected ~1.0. Normalizing.")
    return {k: v / total for k, v in expanded.items()}


def sample_by_distribution(
    ds, distribution: dict, num_convs: int, seed: int,
    min_tokens: int | None = None, max_tokens: int | None = None,
):
    """Sample from dataset matching a target turn distribution.

    Returns a list of indices into the dataset.
    """
    random.seed(seed)

    # Group dataset indices by turn count
    turn_col = ds["turn"]
    token_col = ds["user_token_count"]
    by_turn = {}  # turn_count -> list of indices
    for i, (tc, tok) in enumerate(zip(turn_col, token_col)):
        if min_tokens is not None and tok < min_tokens:
            continue
        if max_tokens is not None and tok > max_tokens:
            continue
        by_turn.setdefault(tc, []).append(i)

    # Compute target counts per turn bucket
    all_indices = []
    for key, proportion in distribution.items():
        target = round(num_convs * proportion)
        if target == 0:
            continue

        if isinstance(key, str) and key.endswith("+"):
            # Open-ended: sample from turn >= N
            min_turn = int(key[:-1])
            pool = []
            for tc, indices in by_turn.items():
                if tc >= min_turn:
                    pool.extend(indices)
        else:
            turn = int(key)
            pool = by_turn.get(turn, [])

        if len(pool) == 0:
            print(f"  Warning: No conversations with turn={key}, skipping {target} samples")
            continue

        if len(pool) < target:
            print(f"  Warning: Only {len(pool)} conversations with turn={key}, requested {target}")
            target = len(pool)

        sampled = random.sample(pool, target)
        all_indices.extend(sampled)
        print(f"  turn={key}: sampled {len(sampled)} (target {round(num_convs * proportion)})")

    random.shuffle(all_indices)
    return all_indices


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description="Sample conversations from WildChat")
    parser.add_argument(
        "--num-convs",
        type=int,
        default=None,
        help="Number of conversations to sample (default: all matching)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Exact number of turns (mutually exclusive with min/max-turns)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=None,
        help="Minimum number of turns",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Minimum user token count",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum user token count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default=None,
        help='Target turn distribution as JSON, e.g. \'{"1":0.61,"2":0.19,"3":0.09,"4":0.05,"5":0.024,"6-8":0.024,"9+":0.012}\'. '
             "Mutually exclusive with --turns/--min-turns/--max-turns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show matching count and token distribution, don't write output",
    )
    args = parser.parse_args()

    # Validate args
    if args.turns is not None and (args.min_turns is not None or args.max_turns is not None):
        parser.error("--turns is mutually exclusive with --min-turns/--max-turns")
    if args.distribution is not None and any(x is not None for x in [args.turns, args.min_turns, args.max_turns]):
        parser.error("--distribution is mutually exclusive with --turns/--min-turns/--max-turns")

    print("Loading dataset...")
    ds = load_dataset("inferencemax/WildChat-4.8M-4o-tokcount", split="train")
    print(f"Loaded {len(ds):,} conversations")

    # Show turn count distribution
    dist = Counter(ds["turn"])
    print(f"\nTurn count distribution (top 10):")
    for count, freq in sorted(dist.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count} turns: {freq:,} conversations")

    # Distribution-based sampling
    if args.distribution is not None:
        if args.num_convs is None:
            parser.error("--num-convs is required when using --distribution")
        distribution = parse_distribution(args.distribution)
        print(f"\nTarget distribution:")
        for key, pct in sorted(distribution.items(), key=lambda x: str(x[0])):
            print(f"  turn={key}: {pct:.1%} ({round(args.num_convs * pct)} convs)")

        if args.dry_run:
            import numpy as np
            # Show availability per bucket
            turn_col = ds["turn"]
            token_col = ds["user_token_count"]
            by_turn = {}
            for tc, tok in zip(turn_col, token_col):
                if args.min_tokens is not None and tok < args.min_tokens:
                    continue
                if args.max_tokens is not None and tok > args.max_tokens:
                    continue
                by_turn.setdefault(tc, []).append(tok)

            print(f"\n--- Dry Run Summary ---")
            print(f"Requested total: {args.num_convs:,}")
            for key, pct in sorted(distribution.items(), key=lambda x: str(x[0])):
                target = round(args.num_convs * pct)
                if isinstance(key, str) and key.endswith("+"):
                    min_turn = int(key[:-1])
                    available = sum(len(v) for tc, v in by_turn.items() if tc >= min_turn)
                else:
                    available = len(by_turn.get(int(key), []))
                status = "OK" if available >= target else "SHORT"
                print(f"  turn={key}: need {target}, available {available:,} [{status}]")
            return

        print(f"\nSampling by distribution...")
        indices = sample_by_distribution(
            ds, distribution, args.num_convs, args.seed,
            args.min_tokens, args.max_tokens,
        )
        sampled = ds.select(indices)
        print(f"Total sampled: {len(sampled):,}")

        # Show actual distribution
        actual_dist = Counter(sampled["turn"])
        print(f"\nActual turn distribution:")
        for t in sorted(actual_dist.keys()):
            print(f"  {t} turns: {actual_dist[t]:,} ({100*actual_dist[t]/len(sampled):.1f}%)")

        # Write output
        output = []
        for item in sampled:
            output.append({
                "conversation_hash": item["conversation_hash"],
                "model": item["model"],
                "user_token_count": item["user_token_count"],
                "turn_count": item["turn"],
                "conversation": item["conversation"],
            })

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

        print(f"Saved {len(output)} conversations to {args.output}")
        return

    # Build filter criteria
    turns = args.turns
    min_turns = args.min_turns
    max_turns = args.max_turns
    min_tokens = args.min_tokens
    max_tokens = args.max_tokens

    def matches_criteria(example):
        turn_count = example["turn"]

        if turns is not None and turn_count != turns:
            return False
        if min_turns is not None and turn_count < min_turns:
            return False
        if max_turns is not None and turn_count > max_turns:
            return False
        if min_tokens is not None and example["user_token_count"] < min_tokens:
            return False
        if max_tokens is not None and example["user_token_count"] > max_tokens:
            return False
        return True

    print(f"Filtering...")
    filtered = ds.filter(matches_criteria, num_proc=NUM_PROC, desc="Filtering")
    print(f"After filtering: {len(filtered):,} conversations")

    if args.dry_run:
        import numpy as np
        tokens = np.array(filtered["user_token_count"])
        turn_counts = np.array(filtered["turn"])
        print(f"\n--- Dry Run Summary ---")
        print(f"Matching conversations: {len(filtered):,}")
        if args.num_convs is not None:
            print(f"Requested sample size:  {args.num_convs:,}")
            print(f"Can fulfill:            {'YES' if len(filtered) >= args.num_convs else 'NO'}")
        print(f"\nToken count distribution (user_token_count):")
        print(f"  min:    {tokens.min():,.0f}")
        print(f"  25th:   {np.percentile(tokens, 25):,.0f}")
        print(f"  median: {np.median(tokens):,.0f}")
        print(f"  75th:   {np.percentile(tokens, 75):,.0f}")
        print(f"  max:    {tokens.max():,.0f}")
        print(f"  mean:   {tokens.mean():,.0f}")
        print(f"  std:    {tokens.std():,.0f}")
        print(f"\nTurn count distribution:")
        turn_dist = Counter(turn_counts)
        for t in sorted(turn_dist.keys()):
            print(f"  {t} turns: {turn_dist[t]:,}")
        return

    if args.num_convs is None:
        sampled = filtered
    else:
        if len(filtered) < args.num_convs:
            print(f"Warning: Only {len(filtered)} conversations match criteria, requested {args.num_convs}")
            sample_size = len(filtered)
        else:
            sample_size = args.num_convs

        random.seed(args.seed)
        indices = random.sample(range(len(filtered)), sample_size)
        sampled = filtered.select(indices)

    # Convert to list of dicts for JSON output
    output = []
    for item in sampled:
        output.append({
            "conversation_hash": item["conversation_hash"],
            "model": item["model"],
            "user_token_count": item["user_token_count"],
            "turn_count": item["turn"],
            "conversation": item["conversation"],
        })

    # Write JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

    print(f"Saved {len(output)} conversations to {args.output}")


if __name__ == "__main__":
    main()
