"""Fail required-power jobs if returned AgentX aggregates lack valid power."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from infx.results.power import POWER_METRIC_SCHEMA_VERSION


def validate_power(path: Path, *, disagg: bool = False) -> list[str]:
    """Check the producer's final verdict without rewriting audit evidence."""
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [f"failed to read result: {exc}"]
    if not isinstance(result, dict):
        return ["result must be a JSON object"]

    errors = []
    schema = result.get("power_metric_schema_version")
    if type(schema) is not int or schema != POWER_METRIC_SCHEMA_VERSION:
        errors.append(f"power_metric_schema_version must be {POWER_METRIC_SCHEMA_VERSION}")
    valid = result.get("power_valid")
    if type(valid) not in (int, float) or valid != 1:
        errors.append("power_valid must be numeric 1")
    fields = [
        "avg_power_w",
        "avg_total_gpu_power_w",
        "total_gpu_energy_j",
        "joules_per_output_token",
    ]
    if disagg:
        fields.extend(
            (
                "prefill_gpu_energy_j",
                "decode_gpu_energy_j",
                "prefill_joules_per_input_token",
                "decode_joules_per_output_token",
            )
        )
    for field in fields:
        value = result.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            errors.append(f"{field} must be a finite positive number")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument(
        "--disagg", action="store_true", help="Require separate prefill/decode energy"
    )
    args = parser.parse_args()
    failed = False
    for path in args.results:
        for error in validate_power(path, disagg=args.disagg):
            print(f"ERROR: {path}: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
