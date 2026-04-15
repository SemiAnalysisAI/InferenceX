#!/usr/bin/env python3
"""Generate synthetic AIPerf-style trace sessions for kv-cache-tester-compatible replay."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def lognormal_sigma(p50: float, p95: float) -> float:
    return math.log(p95 / p50) / 1.645


def sample_tokens(rng: random.Random, p50: float, p95: float, min_v: int, max_v: int) -> int:
    sigma = lognormal_sigma(p50, p95)
    mu = math.log(p50)
    sampled = int(round(rng.lognormvariate(mu, sigma)))
    return max(min_v, min(max_v, sampled))


def generate_sessions(count: int, seed: int) -> dict:
    rng = random.Random(seed)
    sessions = []

    # Target coding-workload distributions:
    # ISL p50~8k, p95~32k
    # OSL p50~512, p95~2k
    for _ in range(count):
        num_turns = rng.randint(4, 18)
        turns = []
        for _ in range(num_turns):
            turns.append(
                {
                    "role": "user",
                    "content_token_count": sample_tokens(
                        rng,
                        p50=8000,
                        p95=32000,
                        min_v=512,
                        max_v=65536,
                    ),
                    "target_output_tokens": sample_tokens(
                        rng,
                        p50=512,
                        p95=2000,
                        min_v=64,
                        max_v=4096,
                    ),
                }
            )
        sessions.append({"turns": turns})

    return {"sessions": sessions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AIPerf traces")
    parser.add_argument("--sessions", type=int, default=100, help="Number of sessions")
    parser.add_argument("--seed", type=int, default=993, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("aiperf_synthetic_traces.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = generate_sessions(args.sessions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
