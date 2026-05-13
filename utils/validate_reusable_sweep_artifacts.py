#!/usr/bin/env python3
"""Validate that reused sweep artifacts match the current merge-run matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


FIXED_SEQ_KEYS = ("1k1k", "8k1k")


def as_bool(value: Any) -> bool:
    """Parse booleans stored as bools or strings."""
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def as_int(value: Any, default: int = 0) -> int:
    """Parse integers from workflow/JSON values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    with open(path) as handle:
        return json.load(handle)


def expected_benchmark_keys(config: dict[str, Any]) -> set[tuple[Any, ...]]:
    """Build expected benchmark identity keys from process_changelog output."""
    expected: set[tuple[Any, ...]] = set()

    for seq_key in FIXED_SEQ_KEYS:
        for entry in config.get("single_node", {}).get(seq_key, []) or []:
            expected.add(
                (
                    "single",
                    entry["runner"],
                    entry["model-prefix"],
                    entry["framework"],
                    entry["precision"],
                    entry.get("spec-decoding", "none"),
                    as_bool(entry.get("disagg", False)),
                    as_int(entry["isl"]),
                    as_int(entry["osl"]),
                    as_int(entry["tp"]),
                    as_int(entry.get("ep", 1)),
                    as_bool(entry.get("dp-attn", False)),
                    as_int(entry["conc"]),
                )
            )

        for entry in config.get("multi_node", {}).get(seq_key, []) or []:
            prefill = entry["prefill"]
            decode = entry["decode"]
            decode_workers = as_int(decode.get("num-worker", 0))
            expected_decode_tp = as_int(decode.get("tp", 0)) if decode_workers > 0 else 0
            expected_decode_ep = as_int(decode.get("ep", 0)) if decode_workers > 0 else 0
            for conc in entry["conc"]:
                expected.add(
                    (
                        "multi",
                        entry["runner"],
                        entry["model-prefix"],
                        entry["framework"],
                        entry["precision"],
                        entry.get("spec-decoding", "none"),
                        as_bool(entry.get("disagg", False)),
                        as_int(entry["isl"]),
                        as_int(entry["osl"]),
                        as_int(prefill.get("tp", 0)),
                        as_int(prefill.get("ep", 1)),
                        as_bool(prefill.get("dp-attn", False)),
                        as_int(prefill.get("num-worker", 0)),
                        expected_decode_tp,
                        expected_decode_ep,
                        as_bool(decode.get("dp-attn", False)),
                        decode_workers,
                        as_int(conc),
                    )
                )

    return expected


def actual_benchmark_keys(artifacts_dir: Path) -> set[tuple[Any, ...]]:
    """Build actual benchmark identity keys from results_bmk/agg_bmk.json."""
    actual: set[tuple[Any, ...]] = set()
    results_dir = artifacts_dir / "results_bmk"
    for path in results_dir.glob("*.json"):
        data = load_json(path)
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("scenario_type") == "agentic-coding":
                continue
            if as_bool(row.get("is_multinode", False)):
                actual.add(
                    (
                        "multi",
                        row.get("hw"),
                        row.get("infmax_model_prefix"),
                        row.get("framework"),
                        row.get("precision"),
                        row.get("spec_decoding", "none"),
                        as_bool(row.get("disagg", False)),
                        as_int(row.get("isl")),
                        as_int(row.get("osl")),
                        as_int(row.get("prefill_tp")),
                        as_int(row.get("prefill_ep", 1)),
                        as_bool(row.get("prefill_dp_attention", False)),
                        as_int(row.get("prefill_num_workers", 0)),
                        as_int(row.get("decode_tp")),
                        as_int(row.get("decode_ep", 1)),
                        as_bool(row.get("decode_dp_attention", False)),
                        as_int(row.get("decode_num_workers", 0)),
                        as_int(row.get("conc")),
                    )
                )
            else:
                actual.add(
                    (
                        "single",
                        row.get("hw"),
                        row.get("infmax_model_prefix"),
                        row.get("framework"),
                        row.get("precision"),
                        row.get("spec_decoding", "none"),
                        as_bool(row.get("disagg", False)),
                        as_int(row.get("isl")),
                        as_int(row.get("osl")),
                        as_int(row.get("tp")),
                        as_int(row.get("ep", 1)),
                        as_bool(row.get("dp_attention", False)),
                        as_int(row.get("conc")),
                    )
                )
    return actual


def expected_eval_jobs(config: dict[str, Any]) -> int:
    """Count expected eval-only matrix jobs."""
    return len(config.get("evals", []) or []) + len(config.get("multinode_evals", []) or [])


def validate_eval_artifacts(artifacts_dir: Path, expected_jobs: int) -> list[str]:
    """Validate eval aggregate/raw artifacts when eval jobs are expected."""
    if expected_jobs == 0:
        return []

    errors: list[str] = []
    eval_agg_files = list((artifacts_dir / "eval_results_all").glob("*.json"))
    if not eval_agg_files:
        errors.append("missing eval_results_all aggregate artifact")
    else:
        row_count = 0
        for path in eval_agg_files:
            data = load_json(path)
            if isinstance(data, list):
                row_count += len(data)
        if row_count == 0:
            errors.append("eval_results_all contains no rows")

    raw_eval_dirs = [
        path
        for path in artifacts_dir.iterdir()
        if path.is_dir() and path.name.startswith("eval_") and path.name != "eval_results_all"
    ]
    if len(raw_eval_dirs) < expected_jobs:
        errors.append(
            f"expected at least {expected_jobs} raw eval artifact dirs, found {len(raw_eval_dirs)}"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    args = parser.parse_args()

    config = load_json(args.config_json)
    if not isinstance(config, dict):
        raise ValueError("config JSON must be an object")
    if not args.artifacts_dir.is_dir():
        raise ValueError(f"artifacts directory does not exist: {args.artifacts_dir}")

    errors: list[str] = []
    expected_bmk = expected_benchmark_keys(config)
    actual_bmk = actual_benchmark_keys(args.artifacts_dir)

    if expected_bmk:
        if not actual_bmk:
            errors.append("missing results_bmk benchmark aggregate artifact")
        missing = expected_bmk - actual_bmk
        if missing:
            errors.append(
                f"reused benchmark artifacts are missing {len(missing)} expected row(s)"
            )
            for key in sorted(missing)[:20]:
                errors.append(f"  missing: {key}")
            if len(missing) > 20:
                errors.append(f"  ... and {len(missing) - 20} more")

    errors.extend(validate_eval_artifacts(args.artifacts_dir, expected_eval_jobs(config)))

    if errors:
        print("Reusable sweep artifact validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "Reusable sweep artifacts validated: "
        f"{len(expected_bmk)} benchmark row(s), {expected_eval_jobs(config)} eval job(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
