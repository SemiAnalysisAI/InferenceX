#!/usr/bin/env python3
"""Validate reused sweep artifacts against the exact target matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


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


def json_rows(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield mapping rows from aggregate or point JSON files."""
    for path in paths:
        data = load_json(path)
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict):
                yield path, row


def expected_benchmark_keys(config: dict[str, Any]) -> set[tuple[Any, ...]]:
    """Build expected fixed-sequence identities from process_changelog output."""
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
            decode_tp = as_int(decode.get("tp", 0)) if decode_workers > 0 else 0
            decode_ep = as_int(decode.get("ep", 0)) if decode_workers > 0 else 0
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
                        decode_tp,
                        decode_ep,
                        as_bool(decode.get("dp-attn", False)),
                        decode_workers,
                        as_int(conc),
                    )
                )

    return expected


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


def actual_benchmark_keys(artifacts_dir: Path) -> set[tuple[Any, ...]]:
    """Build actual fixed-sequence identities from results_bmk."""
    paths = (artifacts_dir / "results_bmk").glob("*.json")
    return {
        benchmark_key(row)
        for _, row in json_rows(paths)
        if row.get("scenario_type") != "agentic-coding"
    }


def expected_agentic_keys(config: dict[str, Any]) -> set[tuple[Any, ...]]:
    """Build expected agentic point-result identities."""
    expected: set[tuple[Any, ...]] = set()
    for entry in config.get("single_node", {}).get("agentic", []) or []:
        expected.add(
            (
                "single",
                entry["runner"],
                entry["model-prefix"],
                entry["framework"],
                entry["precision"],
                as_int(entry["tp"]),
                as_int(entry.get("ep", 1)),
                as_bool(entry.get("dp-attn", False)),
                as_int(entry["conc"]),
                entry.get("offloading", "none"),
            )
        )

    for entry in config.get("multi_node", {}).get("agentic", []) or []:
        prefill = entry["prefill"]
        decode = entry["decode"]
        expected.add(
            (
                "multi",
                entry["runner"],
                entry["model-prefix"],
                entry["framework"],
                entry["precision"],
                entry.get("spec-decoding", "none"),
                as_bool(entry.get("disagg", False)),
                as_int(prefill.get("tp", 0)),
                as_int(prefill.get("ep", 1)),
                as_bool(prefill.get("dp-attn", False)),
                as_int(prefill.get("num-worker", 0)),
                as_int(decode.get("tp", 0)),
                as_int(decode.get("ep", 1)),
                as_bool(decode.get("dp-attn", False)),
                as_int(decode.get("num-worker", 0)),
                as_int(entry["conc"]),
            )
        )
    return expected


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


def actual_agentic_keys(artifacts_dir: Path) -> set[tuple[Any, ...]]:
    """Build actual agentic identities from aggregate and point results."""
    paths = list((artifacts_dir / "results_bmk").glob("*.json"))
    paths.extend(agentic_point_files(artifacts_dir))
    return {
        agentic_key(row)
        for _, row in json_rows(paths)
        if row.get("scenario_type") == "agentic-coding"
    }


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


def validate_agentic_artifacts(
    artifacts_dir: Path,
    expected: set[tuple[Any, ...]],
) -> list[str]:
    """Validate exact agentic point, raw, and aggregate artifact coverage."""
    actual = actual_agentic_keys(artifacts_dir)
    errors = validate_identity_set("agentic", expected, actual)

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
        if not expected:
            errors.append("unexpected agentic_aggregated artifact")
        elif not summary_path.is_file():
            errors.append("missing agentic_aggregated/summary.csv")
        else:
            with open(summary_path, newline="") as handle:
                summary_names = {
                    str(row.get("exp_name") or "")
                    for row in csv.DictReader(handle)
                    if row.get("exp_name")
                }
            if summary_names != raw_names:
                for name in sorted(raw_names - summary_names):
                    errors.append(f"agentic aggregate is missing experiment: {name}")
                for name in sorted(summary_names - raw_names):
                    errors.append(
                        f"agentic aggregate has unexpected experiment: {name}"
                    )

    return errors


def expected_eval_jobs(config: dict[str, Any]) -> int:
    """Count expected eval-only matrix jobs."""
    return len(config.get("evals", []) or []) + len(
        config.get("multinode_evals", []) or []
    )


def normalized_runner(value: Any) -> str:
    """Normalize runner labels that aggregates may uppercase."""
    return str(value or "").lower()


def expected_eval_keys(config: dict[str, Any]) -> set[tuple[Any, ...]]:
    """Build expected eval aggregate identities."""
    expected: set[tuple[Any, ...]] = set()
    for entry in config.get("evals", []) or []:
        expected.add(
            (
                "single",
                normalized_runner(entry["runner"]),
                entry["model-prefix"],
                entry["framework"],
                entry["precision"],
                entry.get("spec-decoding", "none"),
                as_int(entry["tp"]),
                as_int(entry.get("ep", 1)),
                as_bool(entry.get("dp-attn", False)),
                as_int(entry["conc"]),
            )
        )

    for entry in config.get("multinode_evals", []) or []:
        prefill = entry["prefill"]
        decode = entry["decode"]
        expected.add(
            (
                "multi",
                normalized_runner(entry["runner"]),
                entry["model-prefix"],
                entry["framework"],
                entry["precision"],
                entry.get("spec-decoding", "none"),
                as_int(prefill.get("tp", 0)),
                as_int(prefill.get("ep", 1)),
                as_bool(prefill.get("dp-attn", False)),
                as_int(prefill.get("num-worker", 0)),
                as_int(decode.get("tp", 0)),
                as_int(decode.get("ep", 1)),
                as_bool(decode.get("dp-attn", False)),
                as_int(decode.get("num-worker", 0)),
                as_int(entry.get("eval-conc", entry["conc"][0])),
            )
        )
    return expected


def eval_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build an eval identity from one aggregate row."""
    if as_bool(row.get("is_multinode", False)):
        return (
            "multi",
            normalized_runner(row.get("hw")),
            row.get("model_prefix"),
            row.get("framework"),
            row.get("precision"),
            row.get("spec_decoding", "none"),
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
        row.get("model_prefix"),
        row.get("framework"),
        row.get("precision"),
        row.get("spec_decoding", "none"),
        as_int(row.get("tp")),
        as_int(row.get("ep", 1)),
        as_bool(row.get("dp_attention", False)),
        as_int(row.get("conc")),
    )


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
            f"{entry['runner']}_"
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
            f"{entry['runner']}_"
        )
    return prefixes


def raw_eval_artifact_names(artifacts_dir: Path) -> set[str]:
    """Return eval result artifacts, excluding aggregate and debug artifacts."""
    return {
        path.name
        for path in artifacts_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("eval_")
        and path.name != "eval_results_all"
        and not path.name.startswith("eval_server_logs_")
        and not path.name.startswith("eval_gpu_metrics_")
    }


def validate_eval_artifacts(
    artifacts_dir: Path,
    expected_prefixes: list[str],
    expected_aggregate_keys: set[tuple[Any, ...]] | None = None,
) -> list[str]:
    """Validate exact eval aggregate and raw artifact coverage."""
    errors: list[str] = []
    expected = set(expected_prefixes)
    actual_names = raw_eval_artifact_names(artifacts_dir)
    matched: set[str] = set()
    unexpected: set[str] = set()

    for name in actual_names:
        matches = {prefix for prefix in expected if name.startswith(prefix)}
        if len(matches) == 1:
            matched.update(matches)
        elif not matches:
            unexpected.add(name)
        else:
            errors.append(
                f"eval artifact {name!r} matches multiple expected identities"
            )

    missing = expected - matched
    if missing:
        errors.append(f"missing {len(missing)} expected raw eval result artifact dir(s)")
        for prefix in sorted(missing)[:20]:
            errors.append(f"  missing eval artifact prefix: {prefix}")
        if len(missing) > 20:
            errors.append(f"  ... and {len(missing) - 20} more")
    if unexpected:
        errors.append(f"found {len(unexpected)} unexpected raw eval artifact dir(s)")
        for name in sorted(unexpected)[:20]:
            errors.append(f"  unexpected eval artifact: {name}")
        if len(unexpected) > 20:
            errors.append(f"  ... and {len(unexpected) - 20} more")

    aggregate_dir = artifacts_dir / "eval_results_all"
    aggregate_files = list(aggregate_dir.glob("*.json"))
    if expected:
        if not aggregate_files:
            errors.append("missing eval_results_all aggregate artifact")
        else:
            row_count = 0
            aggregate_keys: set[tuple[Any, ...]] = set()
            for path in aggregate_files:
                data = load_json(path)
                if isinstance(data, list):
                    row_count += len(data)
                    aggregate_keys.update(
                        eval_key(row)
                        for row in data
                        if isinstance(row, dict)
                    )
            if row_count == 0:
                errors.append("eval_results_all contains no rows")
            if expected_aggregate_keys is not None:
                errors.extend(
                    validate_identity_set(
                        "eval aggregate",
                        expected_aggregate_keys,
                        aggregate_keys,
                    )
                )
    elif aggregate_dir.exists():
        errors.append("unexpected eval_results_all aggregate artifact")

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
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    args = parser.parse_args()

    config = load_json(args.config_json)
    if not isinstance(config, dict):
        raise ValueError("config JSON must be an object")
    if not args.artifacts_dir.is_dir():
        raise ValueError(
            f"artifacts directory does not exist: {args.artifacts_dir}"
        )

    expected_bmk = expected_benchmark_keys(config)
    actual_bmk = actual_benchmark_keys(args.artifacts_dir)
    expected_agentic = expected_agentic_keys(config)
    expected_eval_prefixes = expected_eval_artifact_prefixes(config)
    expected_eval_aggregate = expected_eval_keys(config)

    errors = validate_identity_set(
        "fixed-sequence",
        expected_bmk,
        actual_bmk,
    )
    if expected_bmk and not (args.artifacts_dir / "results_bmk").is_dir():
        errors.insert(0, "missing results_bmk benchmark aggregate artifact")
    errors.extend(
        validate_agentic_artifacts(args.artifacts_dir, expected_agentic)
    )
    errors.extend(
        validate_eval_artifacts(
            args.artifacts_dir,
            expected_eval_prefixes,
            expected_eval_aggregate,
        )
    )
    errors.extend(validate_run_stats(args.artifacts_dir, bool(expected_bmk)))

    if errors:
        print("Reusable sweep artifact validation failed:", file=sys.stderr)
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "Reusable sweep artifacts validated: "
        f"{len(expected_bmk)} fixed-sequence row(s), "
        f"{len(expected_agentic)} agentic row(s), "
        f"{expected_eval_jobs(config)} eval job(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
