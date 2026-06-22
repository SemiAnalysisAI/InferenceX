#!/usr/bin/env python3
"""Validate reused sweep artifacts for internal consistency."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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


def json_rows(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield mapping rows from aggregate or point JSON files."""
    for path in paths:
        data = load_json(path)
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict):
                yield path, row


def benchmark_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build a fixed-sequence identity from one result row."""
    if as_bool(row.get("is_multinode", False)):
        return (
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
    return (
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


def actual_benchmark_key_rows(
    artifacts_dir: Path,
) -> list[tuple[Any, ...]]:
    """Build actual fixed-sequence identity rows from results_bmk."""
    paths = (artifacts_dir / "results_bmk").glob("*.json")
    return [
        benchmark_key(row)
        for _, row in json_rows(paths)
        if row.get("scenario_type") != "agentic-coding"
    ]


def actual_benchmark_keys(artifacts_dir: Path) -> set[tuple[Any, ...]]:
    """Build the set of actual fixed-sequence identities."""
    return set(actual_benchmark_key_rows(artifacts_dir))


def agentic_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build an agentic identity from one point result."""
    if as_bool(row.get("is_multinode", False)):
        return (
            "multi",
            row.get("hw"),
            row.get("infmax_model_prefix"),
            row.get("framework"),
            row.get("precision"),
            row.get("spec_decoding", "none"),
            as_bool(row.get("disagg", False)),
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
    return (
        "single",
        row.get("hw"),
        row.get("infmax_model_prefix"),
        row.get("framework"),
        row.get("precision"),
        as_int(row.get("tp")),
        as_int(row.get("ep", 1)),
        as_bool(row.get("dp_attention", False)),
        as_int(row.get("conc")),
        row.get("offloading", "none"),
    )


def agentic_point_files(artifacts_dir: Path) -> list[Path]:
    """Return downloaded bmk_agentic point-result JSON files."""
    paths: list[Path] = []
    for artifact_dir in artifacts_dir.glob("bmk_agentic_*"):
        if artifact_dir.is_dir():
            paths.extend(artifact_dir.rglob("*.json"))
    return sorted(set(paths))


def agentic_keys_from_paths(paths: Iterable[Path]) -> list[tuple[Any, ...]]:
    """Build agentic identity rows from aggregate or point-result paths."""
    return [
        agentic_key(row)
        for _, row in json_rows(paths)
        if row.get("scenario_type") == "agentic-coding"
    ]


def actual_agentic_keys(artifacts_dir: Path) -> set[tuple[Any, ...]]:
    """Build actual agentic identities from aggregate and point results."""
    paths = list((artifacts_dir / "results_bmk").glob("*.json"))
    paths.extend(agentic_point_files(artifacts_dir))
    return set(agentic_keys_from_paths(paths))


def validate_identity_set(
    label: str,
    expected: set[tuple[Any, ...]],
    actual: set[tuple[Any, ...]],
) -> list[str]:
    """Return detailed errors for an exact identity-set comparison."""
    errors: list[str] = []
    missing = expected - actual
    extra = actual - expected
    if missing:
        errors.append(f"{label} artifacts are missing {len(missing)} expected row(s)")
        for key in sorted(missing, key=repr)[:20]:
            errors.append(f"  missing: {key}")
        if len(missing) > 20:
            errors.append(f"  ... and {len(missing) - 20} more")
    if extra:
        errors.append(f"{label} artifacts contain {len(extra)} unexpected row(s)")
        for key in sorted(extra, key=repr)[:20]:
            errors.append(f"  unexpected: {key}")
        if len(extra) > 20:
            errors.append(f"  ... and {len(extra) - 20} more")
    return errors


def duplicate_identity_errors(
    label: str,
    identities: Iterable[tuple[Any, ...]],
) -> list[str]:
    """Reject duplicate rows that set equality would otherwise hide."""
    counts = Counter(identities)
    duplicates = {
        identity: count
        for identity, count in counts.items()
        if count > 1
    }
    if not duplicates:
        return []

    duplicate_rows = sum(count - 1 for count in duplicates.values())
    errors = [
        f"{label} artifacts contain {duplicate_rows} duplicate row(s)"
    ]
    for identity, count in sorted(
        duplicates.items(),
        key=lambda item: repr(item[0]),
    )[:20]:
        errors.append(f"  duplicate x{count}: {identity}")
    if len(duplicates) > 20:
        errors.append(f"  ... and {len(duplicates) - 20} more identities")
    return errors


def validate_fixed_artifacts(
    artifacts_dir: Path,
) -> list[str]:
    """Validate fixed-sequence aggregate rows for duplicate identities."""
    actual_rows = actual_benchmark_key_rows(artifacts_dir)
    return duplicate_identity_errors("fixed-sequence", actual_rows)


def validate_agentic_artifacts(
    artifacts_dir: Path,
) -> list[str]:
    """Validate agentic point, raw, and aggregate artifacts agree."""
    point_rows = agentic_keys_from_paths(agentic_point_files(artifacts_dir))
    errors = duplicate_identity_errors("agentic point", point_rows)

    results_bmk = artifacts_dir / "results_bmk"
    if results_bmk.is_dir():
        aggregate_rows = agentic_keys_from_paths(results_bmk.glob("*.json"))
        errors.extend(
            duplicate_identity_errors("agentic aggregate", aggregate_rows)
        )
        errors.extend(
            validate_identity_set(
                "agentic aggregate",
                set(point_rows),
                set(aggregate_rows),
            )
        )

    point_names = {
        path.relative_to(artifacts_dir).parts[0].removeprefix("bmk_")
        for path in agentic_point_files(artifacts_dir)
    }
    raw_names = {
        path.name
        for path in artifacts_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("agentic_")
        and path.name != "agentic_aggregated"
    }
    if point_names != raw_names:
        missing_raw = point_names - raw_names
        extra_raw = raw_names - point_names
        for name in sorted(missing_raw):
            errors.append(f"missing raw agentic artifact dir: {name}")
        for name in sorted(extra_raw):
            errors.append(f"unexpected raw agentic artifact dir: {name}")

    aggregate_dir = artifacts_dir / "agentic_aggregated"
    summary_path = aggregate_dir / "summary.csv"
    if aggregate_dir.exists():
        if not summary_path.is_file():
            errors.append("missing agentic_aggregated/summary.csv")
        else:
            with open(summary_path, newline="") as handle:
                summary_rows = [
                    str(row.get("exp_name") or "")
                    for row in csv.DictReader(handle)
                    if row.get("exp_name")
                ]
            duplicate_names = [
                name
                for name, count in Counter(summary_rows).items()
                if count > 1
            ]
            for name in sorted(duplicate_names):
                errors.append(
                    f"agentic aggregate has duplicate experiment: {name}"
                )
            summary_names = set(summary_rows)
            if summary_names != raw_names:
                for name in sorted(raw_names - summary_names):
                    errors.append(f"agentic aggregate is missing experiment: {name}")
                for name in sorted(summary_names - raw_names):
                    errors.append(
                        f"agentic aggregate has unexpected experiment: {name}"
                    )

    return errors


def normalized_runner(value: Any) -> str:
    """Normalize runner labels that aggregates may uppercase."""
    return str(value or "").lower()


def eval_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build an eval identity from one aggregate row."""
    if as_bool(row.get("is_multinode", False)):
        return (
            "multi",
            normalized_runner(row.get("hw")),
            row.get("model_prefix", row.get("infmax_model_prefix")),
            row.get("framework"),
            row.get("precision"),
            row.get("spec_decoding", "none"),
            as_int(row.get("isl", 8192), 8192),
            as_int(row.get("osl", 1024), 1024),
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
    return (
        "single",
        normalized_runner(row.get("hw")),
        row.get("model_prefix", row.get("infmax_model_prefix")),
        row.get("framework"),
        row.get("precision"),
        row.get("spec_decoding", "none"),
        as_int(row.get("isl", 8192), 8192),
        as_int(row.get("osl", 1024), 1024),
        as_int(row.get("tp")),
        as_int(row.get("ep", 1)),
        as_bool(row.get("dp_attention", False)),
        as_int(row.get("conc")),
    )


def raw_eval_artifact_dirs(artifacts_dir: Path) -> list[Path]:
    """Return raw eval result artifacts, excluding aggregate and debug artifacts."""
    return sorted(
        path
        for path in artifacts_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("eval_")
        and path.name != "eval_results_all"
        and not path.name.startswith("eval_server_logs_")
        and not path.name.startswith("eval_gpu_metrics_")
    )


def raw_eval_key_rows(
    artifacts_dir: Path,
) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Build logical eval identities from each raw artifact's metadata."""
    rows: list[tuple[Any, ...]] = []
    errors: list[str] = []
    for artifact_dir in raw_eval_artifact_dirs(artifacts_dir):
        meta_path = artifact_dir / "meta_env.json"
        if not meta_path.is_file():
            errors.append(
                f"raw eval artifact {artifact_dir.name!r} is missing meta_env.json"
            )
            continue
        try:
            meta = load_json(meta_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(
                f"raw eval artifact {artifact_dir.name!r} has invalid "
                f"meta_env.json: {exc}"
            )
            continue
        if not isinstance(meta, dict):
            errors.append(
                f"raw eval artifact {artifact_dir.name!r} has non-object "
                "meta_env.json"
            )
            continue
        eval_concs = meta.get("completed_eval_concs")
        if isinstance(meta.get("eval_concs"), list):
            if not isinstance(eval_concs, list):
                errors.append(
                    f"raw eval artifact {artifact_dir.name!r} has invalid "
                    "batched concurrency metadata"
                )
                continue
            rows.extend(
                eval_key({**meta, "conc": eval_conc})
                for eval_conc in eval_concs
            )
        else:
            rows.append(eval_key(meta))
    return rows, errors


def validate_eval_artifacts(
    artifacts_dir: Path,
) -> list[str]:
    """Validate raw and aggregate eval artifacts agree."""
    raw_rows, errors = raw_eval_key_rows(artifacts_dir)
    errors.extend(duplicate_identity_errors("raw eval", raw_rows))

    aggregate_dir = artifacts_dir / "eval_results_all"
    aggregate_files = list(aggregate_dir.glob("*.json"))
    if raw_rows or aggregate_dir.exists():
        if not aggregate_files:
            errors.append("missing eval_results_all aggregate artifact")
        else:
            row_count = 0
            aggregate_rows: list[tuple[Any, ...]] = []
            for path in aggregate_files:
                data = load_json(path)
                if isinstance(data, list):
                    row_count += len(data)
                    aggregate_rows.extend(
                        eval_key(row)
                        for row in data
                        if isinstance(row, dict)
                    )
            if row_count == 0:
                errors.append("eval_results_all contains no rows")
            errors.extend(
                duplicate_identity_errors(
                    "eval aggregate",
                    aggregate_rows,
                )
            )
            errors.extend(
                validate_identity_set(
                    "eval aggregate",
                    set(raw_rows),
                    set(aggregate_rows),
                )
            )

    return errors


def validate_run_stats(artifacts_dir: Path, required: bool) -> list[str]:
    """Require run-stats when fixed-sequence collection should have run."""
    if not required:
        return []
    if list((artifacts_dir / "run-stats").glob("*.json")):
        return []
    return ["missing run-stats artifact for fixed-sequence benchmarks"]


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.artifacts_dir.is_dir():
        raise ValueError(
            f"artifacts directory does not exist: {args.artifacts_dir}"
        )

    fixed_rows = actual_benchmark_key_rows(args.artifacts_dir)
    agentic_rows = agentic_keys_from_paths(
        agentic_point_files(args.artifacts_dir)
    )
    eval_rows, _ = raw_eval_key_rows(args.artifacts_dir)

    errors = validate_fixed_artifacts(args.artifacts_dir)
    errors.extend(validate_agentic_artifacts(args.artifacts_dir))
    errors.extend(validate_eval_artifacts(args.artifacts_dir))
    errors.extend(validate_run_stats(args.artifacts_dir, bool(fixed_rows)))
    if not fixed_rows and not agentic_rows and not eval_rows:
        errors.append("no reusable benchmark, agentic, or eval result rows found")

    if errors:
        print("Reusable sweep artifact validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "Reusable sweep artifacts validated: "
        f"{len(set(fixed_rows))} fixed-sequence row(s), "
        f"{len(set(agentic_rows))} agentic row(s), "
        f"{len(set(eval_rows))} eval row(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
