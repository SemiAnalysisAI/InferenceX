#!/usr/bin/env python3
"""Generate a synthetic multi-turn benchmark dataset in mooncake_trace JSONL format.

Distributions are driven by a YAML config file (see configs/qwen_trace_profile.yaml
for the default). Output is compatible with AIPerf's mooncake_trace loader — flat
JSONL with session_id + input_length + output_length + delay.

Usage:
    python generate_synthetic_dataset.py --config configs/qwen_trace_profile.yaml
    python generate_synthetic_dataset.py --config configs/qwen_trace_profile.yaml \
        --num-conversations 10000 --seed 123 --output my_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# YAML parser (minimal, stdlib-only)
# ---------------------------------------------------------------------------

def _parse_yaml(path: str) -> dict:
    """Minimal YAML parser sufficient for our config format.

    Supports:
      - Top-level and nested mappings (indent-based)
      - Inline lists of dicts: [{min: 1, max: 50, weight: 5}, ...]
      - Inline dicts: {mean: 362, stddev: 322}
      - Scalar values (int, float, string)
      - Comments (#)
      - Quoted string keys ("0", "default", "2-4")
    """
    with open(path) as f:
        lines = f.readlines()

    def _parse_inline_value(val: str) -> Any:
        """Parse a scalar or inline collection."""
        val = val.strip()
        if not val:
            return None

        # Inline list: [...]
        if val.startswith("["):
            return _parse_inline_list(val)
        # Inline dict: {...}
        if val.startswith("{"):
            return _parse_inline_dict(val)
        # Quoted string
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            return val[1:-1]
        # Number
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            pass
        # Boolean / null
        if val.lower() in ("true", "yes"):
            return True
        if val.lower() in ("false", "no"):
            return False
        if val.lower() in ("null", "~"):
            return None
        return val

    def _parse_inline_dict(s: str) -> dict:
        """Parse {key: val, key: val, ...}"""
        s = s.strip()
        assert s.startswith("{") and s.endswith("}"), f"Bad inline dict: {s}"
        inner = s[1:-1].strip()
        if not inner:
            return {}
        result = {}
        for part in _split_top_level(inner):
            k, _, v = part.partition(":")
            key = k.strip().strip("'\"")
            result[key] = _parse_inline_value(v)
        return result

    def _parse_inline_list(s: str) -> list:
        """Parse [{...}, {...}, ...] or [scalar, scalar, ...]"""
        s = s.strip()
        assert s.startswith("[") and s.endswith("]"), f"Bad inline list: {s}"
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = _split_top_level(inner)
        return [_parse_inline_value(p) for p in parts]

    def _split_top_level(s: str) -> list[str]:
        """Split on commas not inside braces/brackets."""
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in s:
            if ch in ("{", "["):
                depth += 1
                current.append(ch)
            elif ch in ("}", "]"):
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def _indent_level(line: str) -> int:
        return len(line) - len(line.lstrip())

    # ---- Main parse loop ---
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    # For list items (- ...) under a key
    list_key_stack: list[tuple[int, str, dict]] = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1

        # Strip comment
        stripped = raw.split("#")[0].rstrip()
        if not stripped or stripped.isspace():
            continue

        indent = _indent_level(stripped)
        content = stripped.strip()

        # List item under a mapping key
        if content.startswith("- "):
            item_val = content[2:].strip()
            parsed = _parse_inline_value(item_val)
            # Find the parent dict to attach this list to
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else root
            # Find which key this list belongs to — last key at lower indent
            if list_key_stack and list_key_stack[-1][0] < indent:
                key = list_key_stack[-1][1]
                target = list_key_stack[-1][2]
                if key not in target or not isinstance(target[key], list):
                    target[key] = []
                target[key].append(parsed)
            continue

        # key: value
        if ":" in content:
            colon_pos = content.index(":")
            key = content[:colon_pos].strip().strip("'\"")
            val_part = content[colon_pos + 1 :].strip()

            # Pop stack to find parent
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else root

            if val_part:
                parent[key] = _parse_inline_value(val_part)
            else:
                # Nested mapping — create a new dict
                parent[key] = {}
                stack.append((indent, parent[key]))
                list_key_stack.append((indent, key, parent))

    return root


# ---------------------------------------------------------------------------
# Distribution sampling
# ---------------------------------------------------------------------------

def _resolve_distribution(dist_config: dict[str, Any], turn: int) -> Any:
    """Look up the distribution spec for a given turn number.

    Priority: exact match > range match > default.
    """
    # Exact match
    if str(turn) in dist_config:
        return dist_config[str(turn)]

    # Range match — find narrowest range containing turn
    best = None
    best_width = float("inf")
    for key, val in dist_config.items():
        if "-" in key and key != "default":
            parts = key.split("-")
            if len(parts) == 2:
                try:
                    lo, hi = int(parts[0]), int(parts[1])
                    if lo <= turn <= hi:
                        width = hi - lo
                        if width < best_width:
                            best = val
                            best_width = width
                except ValueError:
                    continue

    if best is not None:
        return best

    # Default
    if "default" in dist_config:
        return dist_config["default"]

    raise ValueError(f"No distribution found for turn {turn} in config")


def _sample_distribution(rng: random.Random, dist: Any) -> int:
    """Sample a single integer from a distribution spec.

    dist can be:
      - list of {min, max, weight} dicts       (bucketed uniform)
      - dict with {mean, stddev}                (normal, clipped >= 1)
      - dict with {type: lognormal, mu, sigma}  (lognormal, log-space params)
    """
    if isinstance(dist, list):
        # Bucketed: pick bucket by weight, uniform within [min, max)
        weights = [b["weight"] for b in dist]
        bucket = rng.choices(dist, weights=weights, k=1)[0]
        return rng.randint(bucket["min"], bucket["max"] - 1)
    elif isinstance(dist, dict) and dist.get("type") == "lognormal":
        # Lognormal: mu and sigma are log-space parameters.
        # X = exp(N(mu, sigma)), so median = exp(mu), mean = exp(mu + sigma^2/2).
        val = math.exp(rng.gauss(dist["mu"], dist["sigma"]))
        return max(1, round(val))
    elif isinstance(dist, dict) and "mean" in dist:
        # Normal distribution, clipped to >= 1
        val = rng.gauss(dist["mean"], dist["stddev"])
        return max(1, round(val))
    else:
        raise ValueError(f"Unknown distribution format: {dist}")


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def _build_turn_count_cdf(turn_count_config: dict) -> list[tuple[int, float]]:
    """Build a list of (turn_count, cumulative_weight) for weighted sampling."""
    items = sorted((int(k), v) for k, v in turn_count_config.items())
    total = sum(w for _, w in items)
    cdf = []
    cumulative = 0.0
    for tc, w in items:
        cumulative += w / total
        cdf.append((tc, cumulative))
    return cdf


def _sample_turn_count(rng: random.Random, cdf: list[tuple[int, float]]) -> int:
    """Sample a turn count from the CDF."""
    r = rng.random()
    for tc, c in cdf:
        if r <= c:
            return tc
    return cdf[-1][0]


def generate_dataset(config: dict, num_conversations: int, seed: int) -> list[dict]:
    """Generate synthetic multi-turn conversations as mooncake_trace records."""
    rng = random.Random(seed)

    system_prompt_length = config.get("system_prompt_length", 0)
    delay_cfg = config.get("delay", {"min_ms": 1000, "max_ms": 3000})
    isl_config = config["isl"]
    osl_config = config["osl"]
    turn_count_cdf = _build_turn_count_cdf(config["turn_count"])

    records: list[dict] = []

    for conv_idx in range(num_conversations):
        session_id = f"conv_{conv_idx}"
        num_turns = _sample_turn_count(rng, turn_count_cdf)

        for turn in range(num_turns):
            # Sample ISL (new user tokens)
            isl_dist = _resolve_distribution(isl_config, turn)
            input_length = _sample_distribution(rng, isl_dist)

            # Add system prompt to turn 0
            if turn == 0:
                input_length += system_prompt_length

            # Sample OSL
            osl_dist = _resolve_distribution(osl_config, turn)
            output_length = _sample_distribution(rng, osl_dist)

            record: dict[str, Any] = {
                "session_id": session_id,
                "input_length": input_length,
                "output_length": output_length,
            }

            if turn == 0:
                record["timestamp"] = 0
            else:
                delay = rng.randint(delay_cfg["min_ms"], delay_cfg["max_ms"])
                record["delay"] = delay

            records.append(record)

    return records


def print_summary(config: dict, num_conversations: int, seed: int) -> None:
    """Print a summary of the resolved configuration."""
    print("=" * 60)
    print("Synthetic Multi-Turn Dataset Generator")
    print("=" * 60)
    print(f"  Conversations:       {num_conversations}")
    print(f"  Seed:                {seed}")
    print(f"  System prompt:       {config.get('system_prompt_length', 0)} tokens")

    delay = config.get("delay", {})
    print(f"  Delay:               [{delay.get('min_ms', '?')}, {delay.get('max_ms', '?')}] ms")

    print("\n  Turn count distribution:")
    tc = config["turn_count"]
    total_w = sum(tc.values())
    for k in sorted(tc, key=int):
        pct = tc[k] / total_w * 100
        print(f"    {k} turns: {pct:.1f}%")

    # Estimate expected total turns
    expected_turns = sum(int(k) * v / total_w for k, v in tc.items())
    print(f"  Expected turns/conv: {expected_turns:.2f}")
    print(f"  Expected total turns: ~{int(expected_turns * num_conversations)}")

    for label, cfg in [("ISL", config["isl"]), ("OSL", config["osl"])]:
        print(f"\n  {label} distributions:")
        for key in sorted(cfg.keys(), key=lambda x: (x != "default", x)):
            dist = cfg[key]
            if isinstance(dist, list):
                ranges = ", ".join(
                    f"[{b['min']}-{b['max']}]×{b['weight']}" for b in dist
                )
                print(f"    turn {key}: bucketed — {ranges}")
            elif isinstance(dist, dict) and dist.get("type") == "lognormal":
                mu, sigma = dist["mu"], dist["sigma"]
                median = math.exp(mu)
                mean = math.exp(mu + sigma**2 / 2)
                print(f"    turn {key}: lognormal(μ={mu}, σ={sigma}) → median={median:.0f}, mean={mean:.0f}")
            elif isinstance(dist, dict) and "mean" in dist:
                print(f"    turn {key}: normal(μ={dist['mean']}, σ={dist['stddev']})")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-turn benchmark dataset (mooncake_trace format)"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--num-conversations", type=int, default=5000, help="Number of conversations"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", default="synthetic_multiturn.jsonl", help="Output JSONL file"
    )
    args = parser.parse_args()

    config = _parse_yaml(args.config)
    print_summary(config, args.num_conversations, args.seed)

    records = generate_dataset(config, args.num_conversations, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Stats
    num_sessions = len({r["session_id"] for r in records})
    print(f"\nWrote {len(records)} records ({num_sessions} conversations) to {output_path}")


if __name__ == "__main__":
    main()
