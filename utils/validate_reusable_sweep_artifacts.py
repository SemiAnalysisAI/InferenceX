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


def bool_str(value: Any) -> str:
    """Render booleans as GitHub Actions does in filenames."""
    return "true" if as_bool(value) else "false"


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


def expected_eval_artifact_prefixes(config: dict[str, Any]) -> list[str]:
    """Build expected raw eval result artifact prefixes from matrix entries."""
    prefixes: list[str] = []
    for entry in config.get("evals", []) or []:
        exp_name = entry["exp-name"]
        prefixes.append(
            f"eval_{exp_name}_{exp_name}_{entry['precision']}_{entry['framework']}_"
            f"tp{as_int(entry['tp'])}-ep{as_int(entry.get('ep', 1))}-"
            f"dpa{bool_str(entry.get('dp-attn', False))}_"
            f"disagg-{bool_str(entry.get('disagg', False))}_"
            f"spec-{entry.get('spec-decoding', 'none')}_conc{as_int(entry['conc'])}_"
        )

    for entry in config.get("multinode_evals", []) or []:
        exp_name = entry["exp-name"]
        prefill = entry["prefill"]
        decode = entry["decode"]
        conc = "x".join(str(as_int(value)) for value in entry["conc"])
        prefixes.append(
            f"eval_{exp_name}_{exp_name}_{entry['precision']}_{entry['framework']}_"
            f"prefill-tp{as_int(prefill['tp'])}-ep{as_int(prefill.get('ep', 1))}-"
            f"dp{bool_str(prefill.get('dp-attn', False))}-"
            f"nw{as_int(prefill.get('num-worker', 0))}_"
            f"decode-tp{as_int(decode.get('tp', 0))}-ep{as_int(decode.get('ep', 0))}-"
            f"dp{bool_str(decode.get('dp-attn', False))}-"
            f"nw{as_int(decode.get('num-worker', 0))}_"
            f"disagg-{bool_str(entry.get('disagg', False))}_"
            f"spec-{entry.get('spec-decoding', 'none')}_conc{conc}_"
        )
    return prefixes


def validate_eval_artifacts(artifacts_dir: Path, expected_prefixes: list[str]) -> list[str]:
    """Validate eval aggregate/raw artifacts when eval jobs are expected."""
    if not expected_prefixes:
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
        path.name
        for path in artifacts_dir.iterdir()
        if path.is_dir() and path.name.startswith("eval_") and path.name != "eval_results_all"
    ]
    missing = [
        prefix
        for prefix in expected_prefixes
        if not any(name.startswith(prefix) for name in raw_eval_dirs)
    ]
    if missing:
        errors.append(
            f"missing {len(missing)} expected raw eval result artifact dir(s)"
        )
        for prefix in sorted(missing)[:20]:
            errors.append(f"  missing eval artifact prefix: {prefix}")
        if len(missing) > 20:
            errors.append(f"  ... and {len(missing) - 20} more")

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
    expected_eval_prefixes = expected_eval_artifact_prefixes(config)

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

    errors.extend(validate_eval_artifacts(args.artifacts_dir, expected_eval_prefixes))

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
