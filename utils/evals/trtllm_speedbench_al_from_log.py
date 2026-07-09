#!/usr/bin/env python3
"""Parse TRT-LLM iteration logs into SpeedBench AL counters."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


GEN_TOKENS_RE = re.compile(r"'num_generation_tokens':\s*(?P<tokens>[0-9]+)")
ITER_LINE_RE = re.compile(r"\biter\s*=\s*[0-9]+,.*\bstates\s*=")
METRICS_GET_RE = re.compile(r'GET\s+/(?:prometheus/)?metrics\b')


@dataclass(frozen=True)
class TrtLogMetrics:
    samples: int
    accepted_tokens: int
    proposed_draft_tokens: int
    verify_steps: int
    generated_tokens: int

    @property
    def acceptance_length(self) -> float:
        return self.generated_tokens / self.verify_steps


def _read_log_suffix(path: Path, start_offset: int) -> list[str]:
    with path.open("rb") as f:
        if start_offset > 0:
            f.seek(start_offset)
        return f.read().decode(errors="ignore").splitlines()


def parse_trtllm_iteration_log(
    path: Path,
    mtp: int,
    start_offset: int = 0,
    stop_at_metrics_get: bool = True,
) -> TrtLogMetrics | None:
    if mtp <= 0:
        raise ValueError("mtp must be positive")
    if not path.is_file():
        return None

    samples = 0
    accepted = 0
    proposed = 0
    verify_steps = 0
    generated = 0
    max_tokens_per_step = mtp + 1

    for line in _read_log_suffix(path, start_offset):
        if samples and stop_at_metrics_get and METRICS_GET_RE.search(line):
            break
        if not ITER_LINE_RE.search(line):
            continue
        match = GEN_TOKENS_RE.search(line)
        if not match:
            continue

        gen_tokens = int(match.group("tokens"))
        if gen_tokens <= 0:
            continue

        # SpeedBench AL is issued at max-concurrency=1 today, where each
        # generation iteration is one verification step. Keep a batched fallback
        # for postmortem logs by assuming no step can emit more than mtp + 1
        # tokens per active request.
        steps = max(1, math.ceil(gen_tokens / max_tokens_per_step))
        samples += 1
        verify_steps += steps
        generated += gen_tokens
        accepted += max(gen_tokens - steps, 0)
        proposed += steps * mtp

    if samples == 0 or verify_steps <= 0:
        return None

    return TrtLogMetrics(
        samples=samples,
        accepted_tokens=accepted,
        proposed_draft_tokens=proposed,
        verify_steps=verify_steps,
        generated_tokens=generated,
    )


def cmd_tsv(args: argparse.Namespace) -> int:
    metrics = parse_trtllm_iteration_log(
        Path(args.log),
        args.num_speculative_tokens,
        start_offset=max(args.start_offset, 0),
        stop_at_metrics_get=not args.no_stop_at_metrics_get,
    )
    if metrics is None:
        return 1

    print(
        f"{metrics.acceptance_length:.4f}\t"
        f"{metrics.accepted_tokens}\t"
        f"{metrics.verify_steps}\t"
        f"{metrics.proposed_draft_tokens}\t"
        f"{metrics.samples}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--num-speculative-tokens", type=int, required=True)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument(
        "--no-stop-at-metrics-get",
        action="store_true",
        help="Do not stop parsing at the next /metrics request after samples appear.",
    )
    parser.set_defaults(func=cmd_tsv)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI should return concise diagnostics
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
