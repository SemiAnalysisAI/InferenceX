#!/usr/bin/env python3
"""Validate an aggregate benchmark result JSON before artifact upload.

Checks structural and physical invariants for both fixed-seq and agentic results,
deriving the expected keys from the data rather than assuming a fixed percentile set.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

LATENCY_FAMILIES = ("ttft", "tpot", "itl", "e2el")
INTERACTIVITY_FAMILY = "intvty"
THROUGHPUT_KEYS = ("tput_per_gpu", "output_tput_per_gpu", "input_tput_per_gpu")
AGENTIC_SCENARIO = "agentic-coding"

# Prefix-style percentile keys: p75_tpot, p99.9_intvty.
_PCTL_KEY = re.compile(r"^p(\d+(?:\.\d+)?)_(.+)$")


def is_number(value: Any) -> bool:
    """Real int/float, excluding bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_positive_int(value: Any) -> bool:
    """Integer greater than zero (bools excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_non_negative_int(value: Any) -> bool:
    """Integer at least zero (bools excluded)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def fmt_pctl(rank: float) -> str:
    """Render a percentile as the keys do: 90.0 -> '90', 99.9 -> '99.9'."""
    return str(int(rank)) if rank == int(rank) else str(rank)


def percentiles_present(data: dict[str, Any], family: str) -> dict[float, str]:
    """Map percentile rank -> key for every p<rank>_<family> key in data."""
    found: dict[float, str] = {}
    for key in data:
        match = _PCTL_KEY.match(key)
        if match and match.group(2) == family:
            rank = float(match.group(1))
            if 0 <= rank <= 100:
                found[rank] = key
    return found


def malformed_percentile_keys(data: dict[str, Any], families: Iterable[str]) -> list[str]:
    """Keys shaped like p<rank>_<family> whose rank is outside the 0-100 range."""
    bad: list[str] = []
    family_set = set(families)
    for key in data:
        match = _PCTL_KEY.match(key)
        if match and match.group(2) in family_set:
            rank = float(match.group(1))
            if not 0 <= rank <= 100:
                bad.append(key)
    return sorted(bad)


def check_identity(data: dict[str, Any]) -> list[str]:
    """Identity strings, run dimensions, and topology metadata are well-formed."""
    errors: list[str] = []
    for key in ("hw", "framework", "precision", "model", "infmax_model_prefix"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a non-empty string")
    if not is_positive_int(data.get("conc")):
        errors.append("conc must be a positive integer")
    # Agentic runs are variable-length and carry no isl/osl.
    if data.get("scenario_type") != AGENTIC_SCENARIO:
        for key in ("isl", "osl"):
            if not is_positive_int(data.get(key)):
                errors.append(f"{key} must be a positive integer")
    is_multinode = data.get("is_multinode")
    if not isinstance(is_multinode, bool):
        errors.append("is_multinode must be present as a bool")
        return errors
    if is_multinode:
        for key in ("prefill_tp", "prefill_ep", "prefill_num_workers"):
            if not is_positive_int(data.get(key)):
                errors.append(f"{key} must be a positive integer for multinode topology")
        for key in ("decode_tp", "decode_ep", "decode_num_workers"):
            if not is_non_negative_int(data.get(key)):
                errors.append(f"{key} must be a non-negative integer for multinode topology")
        for key in ("prefill_dp_attention", "decode_dp_attention"):
            if key not in data:
                errors.append(f"{key} is required for multinode topology")
    else:
        for key in ("tp", "ep"):
            if not is_positive_int(data.get(key)):
                errors.append(f"{key} must be a positive integer for single-node topology")
        if "dp_attention" not in data:
            errors.append("dp_attention is required for single-node topology")
    return errors


def numeric_paths(value: Any, path: str = "") -> Iterable[tuple[str, float]]:
    """Yield (path, number) for every numeric leaf in nested JSON-like data."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from numeric_paths(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_paths(child, f"{path}[{index}]")


def check_numeric_finite(data: dict[str, Any]) -> list[str]:
    """No numeric field may be NaN or +/-Infinity."""
    return [
        f"{path} must be finite"
        for path, value in numeric_paths(data)
        if not math.isfinite(value)
    ]


def check_throughput(data: dict[str, Any]) -> list[str]:
    """Per-GPU throughput fields are present, finite, and positive."""
    errors: list[str] = []
    for key in THROUGHPUT_KEYS:
        value = data.get(key)
        if not is_number(value) or not math.isfinite(value) or value <= 0:
            errors.append(f"{key} must be a positive finite number")
    return errors


def check_percentile_families(data: dict[str, Any]) -> list[str]:
    """Families that report percentiles must report the same ranks, and intvty mirrors
    tpot key-for-key. A family a run does not emit at all is not required."""
    errors: list[str] = []
    for key in malformed_percentile_keys(data, (*LATENCY_FAMILIES, INTERACTIVITY_FAMILY)):
        errors.append(f"{key} is a malformed percentile key")

    present = {family: percentiles_present(data, family) for family in LATENCY_FAMILIES}
    with_pctls = {family: ranks for family, ranks in present.items() if ranks}
    if with_pctls:
        union = set().union(*(set(ranks) for ranks in with_pctls.values()))
        for family, ranks in with_pctls.items():
            for missing in sorted(union - set(ranks)):
                errors.append(
                    f"metric '{family}' missing percentile p{fmt_pctl(missing)} "
                    "that other metrics report"
                )

    tpot_ranks = set(present["tpot"])
    intvty_ranks = set(percentiles_present(data, INTERACTIVITY_FAMILY))
    for missing in sorted(tpot_ranks - intvty_ranks):
        errors.append(
            f"p{fmt_pctl(missing)}_tpot present but p{fmt_pctl(missing)}_intvty missing"
        )
    for extra in sorted(intvty_ranks - tpot_ranks):
        errors.append(
            f"p{fmt_pctl(extra)}_intvty present but p{fmt_pctl(extra)}_tpot missing"
        )
    return errors


def _monotonic(data: dict[str, Any], family: str, increasing: bool) -> list[str]:
    """One family's percentile values are non-negative, finite, and monotonic in rank."""
    entries = percentiles_present(data, family)
    errors: list[str] = []
    prev_key = ""
    prev_val: float | None = None
    for rank in sorted(entries):
        key = entries[rank]
        value = data[key]
        if not is_number(value) or not math.isfinite(value):
            errors.append(f"{key} must be a finite percentile value")
            prev_val = None
            continue
        if value < 0:
            errors.append(f"{key} must be non-negative")
        if prev_val is not None:
            if increasing and value < prev_val:
                errors.append(
                    f"{family} percentiles must be non-decreasing: "
                    f"{prev_key}={prev_val} > {key}={value}"
                )
            if not increasing and value > prev_val:
                errors.append(
                    f"{family} percentiles must be non-increasing: "
                    f"{prev_key}={prev_val} < {key}={value}"
                )
        prev_key, prev_val = key, value
    return errors


def check_monotonicity(data: dict[str, Any]) -> list[str]:
    """Latency percentiles are non-decreasing in P. Interactivity is 1000/tpot for
    fixed-seq (non-increasing in P) but a measured percentile of 1/itl for agentic
    (non-decreasing)."""
    errors: list[str] = []
    for family in LATENCY_FAMILIES:
        errors += _monotonic(data, family, increasing=True)
    agentic = data.get("scenario_type") == AGENTIC_SCENARIO
    errors += _monotonic(data, INTERACTIVITY_FAMILY, increasing=agentic)
    return errors


def validate(data: dict[str, Any]) -> list[str]:
    """Return all validation errors for one aggregate result."""
    errors: list[str] = []
    errors += check_identity(data)
    errors += check_numeric_finite(data)
    errors += check_throughput(data)
    errors += check_percentile_families(data)
    errors += check_monotonicity(data)
    return errors


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    """CLI: validate one aggregate result JSON; exit 1 with messages on failure."""
    parser = argparse.ArgumentParser(
        description="Validate an InferenceX aggregate result JSON."
    )
    parser.add_argument("agg_json", type=Path)
    args = parser.parse_args()

    try:
        data = load_json(args.agg_json)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"failed to load JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("agg JSON must be an object", file=sys.stderr)
        return 1

    errors = validate(data)
    if errors:
        print(f"Agg result validation failed for {args.agg_json}:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Agg result validated: {args.agg_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
