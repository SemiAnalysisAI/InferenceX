"""Validate SPEED-Bench coding results and emit the DSv4.1 Flash golden AL."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

COUNTERS = ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total")


def read_counters(text: str) -> tuple[float, float]:
    values = []
    for name in COUNTERS:
        samples = re.findall(r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?$", text, re.MULTILINE)
        if not samples:
            raise ValueError(f"Missing acceptance counter: {name}")
        numbers = [float(sample) for sample in samples]
        if any(not math.isfinite(n) or n < 0 for n in numbers):
            raise ValueError(f"Invalid acceptance counter: {name}")
        values.append(sum(numbers))
    return values[0], values[1]


def measure_al(before: str, after: str, benchmark: dict, draft_length: int = 5) -> float:
    # SPEED-Bench Qualitative coding has 80 prompts. Partial/errorful runs must
    # never silently become a golden value, even if their metric delta is valid.
    if benchmark.get("completed") != 80 or benchmark.get("failed", 0) != 0:
        raise ValueError("Expected all 80 coding prompts to complete successfully")
    old_acc, old_drafts = read_counters(before)
    new_acc, new_drafts = read_counters(after)
    accepted, drafts = new_acc - old_acc, new_drafts - old_drafts
    if drafts <= 0 or accepted < 0 or accepted > draft_length * drafts:
        raise ValueError("Invalid acceptance deltas or counter reset")
    return round(1 + accepted / drafts, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["on", "off"], required=True)
    parser.add_argument("--thinking-kwargs", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--category", choices=["coding"], required=True)
    parser.add_argument("--output-len", type=int, required=True)
    args = parser.parse_args()
    lines = [
        "# Acceptance Length (AL) measured with SPEED-Bench Qualitative coding.",
        f"# model: {args.model} | image: {args.image} | TP: {args.tp}",
        f"# temperature: 1.0 | output_len: {args.output_len}",
        f"# thinking_on chat_template_kwargs: {args.thinking_kwargs}",
        '# thinking_off chat_template_kwargs: {"thinking":false}',
        "# method: dspark | draft_sample_method: probabilistic | rejection_sample_method: block",
        "# enable_adaptive_verification: false | engram cpu_offload: true",
        "# key = num_speculative_tokens; AL includes the target verification token.",
        "deepseek-v4.1-flash:",
    ]
    for mode in args.modes:
        root = args.results_dir
        al = measure_al((root / f"before_{mode}.prom").read_text(),
                        (root / f"after_{mode}.prom").read_text(),
                        json.loads((root / f"speedbench_{mode}_mtp5.json").read_text()))
        lines += [f"  thinking_{mode}:", f"    5: {al:.2f}"]
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
