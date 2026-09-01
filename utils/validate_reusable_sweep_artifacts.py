#!/usr/bin/env python3
"""Validate reused sweep artifacts for internal consistency."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


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
            as_int(row.get("prefill_pp", 1), 1),
            as_int(row.get("prefill_dcp_size", 1), 1),
            as_int(row.get("prefill_pcp_size", 1), 1),
            as_int(row.get("prefill_ep", 1)),
            as_bool(row.get("prefill_dp_attention", False)),
            as_int(row.get("prefill_num_workers", 0)),
            as_int(row.get("decode_tp")),
            as_int(row.get("decode_pp", 1), 1),
            as_int(row.get("decode_dcp_size", 1), 1),
            as_int(row.get("decode_pcp_size", 1), 1),
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
        as_int(row.get("pp", 1), 1),
        as_int(row.get("dcp_size", 1), 1),
        as_int(row.get("pcp_size", 1), 1),
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


def freeze_identity_value(value: Any) -> Any:
    """Convert nested JSON values into deterministic, hashable identities."""
    if isinstance(value, dict):
        return tuple(
            sorted(
                (key, freeze_identity_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_identity_value(item) for item in value)
    return value


def agentic_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build an agentic identity from one point result."""
    if "kv_offloading" in row:
        kv_offloading = row.get("kv_offloading") or "none"
        offload_key: Any = (
            kv_offloading,
            freeze_identity_value(row.get("kv_offload_backend") or "")
            if kv_offloading != "none"
            else "",
        )
    else:
        offload_key = row.get("offloading", "none")

    if as_bool(row.get("is_multinode", False)):
        key = (
            "multi",
            row.get("hw"),
            row.get("infmax_model_prefix"),
            row.get("framework"),
            row.get("precision"),
            row.get("spec_decoding", "none"),
            as_bool(row.get("disagg", False)),
            as_int(row.get("prefill_tp")),
            as_int(row.get("prefill_pp", 1), 1),
            as_int(row.get("prefill_dcp_size", 1), 1),
            as_int(row.get("prefill_pcp_size", 1), 1),
            as_int(row.get("prefill_ep", 1)),
            as_bool(row.get("prefill_dp_attention", False)),
            as_int(row.get("prefill_num_workers", 0)),
            as_int(row.get("decode_tp")),
            as_int(row.get("decode_pp", 1), 1),
            as_int(row.get("decode_dcp_size", 1), 1),
            as_int(row.get("decode_pcp_size", 1), 1),
            as_int(row.get("decode_ep", 1)),
            as_bool(row.get("decode_dp_attention", False)),
            as_int(row.get("decode_num_workers", 0)),
            as_int(row.get("conc")),
        )
        if "kv_offloading" in row or "offloading" in row:
            return (*key, offload_key)
        return key
    return (
        "single",
        row.get("hw"),
        row.get("infmax_model_prefix"),
        row.get("framework"),
        row.get("precision"),
        as_int(row.get("tp")),
        as_int(row.get("pp", 1), 1),
        as_int(row.get("dcp_size", 1), 1),
        as_int(row.get("pcp_size", 1), 1),
        as_int(row.get("ep", 1)),
        as_bool(row.get("dp_attention", False)),
        as_int(row.get("conc")),
        offload_key,
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
        if path.is_dir() and path.name.startswith("agentic_")
    }
    if point_names != raw_names:
        missing_raw = point_names - raw_names
        extra_raw = raw_names - point_names
        for name in sorted(missing_raw):
            errors.append(f"missing raw agentic artifact dir: {name}")
        for name in sorted(extra_raw):
            errors.append(f"unexpected raw agentic artifact dir: {name}")

    return errors


def normalized_runner(value: Any) -> str:
    """Normalize runner labels that aggregates may uppercase."""
    return str(value or "").lower()


LEGACY_EVAL_SUITE = "<legacy-eval-suite>"
def invalid_eval_suite(row: dict[str, Any]) -> bool:
    """Return whether an explicit eval-suite identity is malformed."""
    suite = row.get("eval_suite")
    return "eval_suite" in row and (
        not isinstance(suite, str) or not suite
    )





def eval_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build an eval identity from one aggregate row."""
    if as_bool(row.get("is_multinode", False)):
        return (
            "multi",
            normalized_runner(row.get("hw")),
            row.get("model_prefix", row.get("infmax_model_prefix")),
            row.get("framework"),
            row.get("precision"),
            row.get("eval_suite", LEGACY_EVAL_SUITE),
            row.get("spec_decoding", "none"),
            as_int(row.get("isl", 8192), 8192),
            as_int(row.get("osl", 1024), 1024),
            as_int(row.get("prefill_tp")),
            as_int(row.get("prefill_pp", 1), 1),
            as_int(row.get("prefill_dcp_size", 1), 1),
            as_int(row.get("prefill_pcp_size", 1), 1),
            as_int(row.get("prefill_ep", 1)),
            as_bool(row.get("prefill_dp_attention", False)),
            as_int(row.get("prefill_num_workers", 0)),
            as_int(row.get("decode_tp")),
            as_int(row.get("decode_pp", 1), 1),
            as_int(row.get("decode_dcp_size", 1), 1),
            as_int(row.get("decode_pcp_size", 1), 1),
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
        row.get("eval_suite", LEGACY_EVAL_SUITE),
        row.get("spec_decoding", "none"),
        as_int(row.get("isl", 8192), 8192),
        as_int(row.get("osl", 1024), 1024),
        as_int(row.get("tp")),
        as_int(row.get("pp", 1), 1),
        as_int(row.get("dcp_size", 1), 1),
        as_int(row.get("pcp_size", 1), 1),
        as_int(row.get("ep", 1)),
        as_bool(row.get("dp_attention", False)),
        as_int(row.get("conc")),
    )


def eval_result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build a task-level eval result identity."""
    return (*eval_key(row), row.get("task"))


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


def _positive_int(value: Any) -> bool:
    """Return whether value is a positive JSON integer (not a boolean)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _raw_meta_contributions(
    artifact_name: str,
    meta: dict[str, Any],
) -> tuple[
    list[tuple[tuple[Any, ...], Optional[int]]],
    bool,
    list[str],
]:
    """Validate raw eval metadata and return its logical contributions."""
    prefix = f"raw eval artifact {artifact_name!r}"
    if "eval_concs" not in meta:
        conc = meta.get("conc")
        if not _positive_int(conc):
            return [], False, [f"{prefix} has invalid legacy concurrency"]
        return [(eval_key(meta), None)], False, []

    expected = meta.get("eval_concs")
    completed = meta.get("completed_eval_concs")
    failed = meta.get("failed_eval_concs", [])
    fields = (
        ("eval_concs", expected),
        ("completed_eval_concs", completed),
        ("failed_eval_concs", failed),
    )
    errors: list[str] = []
    if not all(isinstance(values, list) for _, values in fields):
        return [], True, [f"{prefix} has invalid batched concurrency metadata"]

    for field, values in fields:
        if any(not _positive_int(value) for value in values):
            errors.append(f"{prefix} has invalid {field}")
            continue
        if len(set(values)) != len(values):
            errors.append(f"{prefix} has duplicate {field}")
    if errors:
        return [], True, errors

    expected_set = set(expected)
    completed_set = set(completed)
    failed_set = set(failed)
    if not expected_set:
        errors.append(f"{prefix} has empty eval_concs")
    if not completed_set:
        errors.append(f"{prefix} has no completed eval concurrencies")
    if not completed_set <= expected_set:
        errors.append(f"{prefix} completed unexpected eval concurrencies")
    if not failed_set <= expected_set:
        errors.append(f"{prefix} failed unexpected eval concurrencies")
    if completed_set & failed_set:
        errors.append(f"{prefix} has overlapping completed and failed concurrencies")
    if completed_set | failed_set != expected_set:
        errors.append(f"{prefix} has unaccounted eval concurrencies")
    if failed_set:
        errors.append(f"{prefix} reports failed eval concurrencies")
    if errors:
        return [], True, errors

    return (
        [(eval_key({**meta, "conc": conc}), conc) for conc in completed],
        True,
        [],
    )


def raw_eval_key_rows(
    artifacts_dir: Path,
) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Build and validate logical identities from raw eval artifacts."""
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
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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
        if invalid_eval_suite(meta):
            errors.append(
                f"raw eval artifact {artifact_dir.name!r} has invalid "
                "eval_suite"
            )
            continue

        contributions, batched, meta_errors = _raw_meta_contributions(
            artifact_dir.name,
            meta,
        )
        errors.extend(meta_errors)
        if meta_errors:
            continue

        result_paths = _recognized_eval_result_paths(
            artifact_dir.glob("results*.json")
        )
        if batched:
            expected = set(meta["eval_concs"])
            for path in result_paths:
                conc = _result_concurrency(path.name)
                if conc is None:
                    errors.append(
                        f"raw eval artifact {artifact_dir.name!r} has batched "
                        f"result {path.name!r} without a concurrency suffix"
                    )
                elif conc not in expected:
                    errors.append(
                        f"raw eval artifact {artifact_dir.name!r} has result "
                        f"{path.name!r} for unexpected concurrency {conc}"
                    )

        for _, conc in contributions:
            candidates = [
                path
                for path in result_paths
                if not batched or _result_concurrency(path.name) == conc
            ]
            conc_label = f" for concurrency {conc}" if conc is not None else ""
            if not candidates:
                errors.append(
                    f"raw eval artifact {artifact_dir.name!r} has no "
                    f"recognized eval result{conc_label}"
                )
                continue
            latest = max(candidates, key=_result_order)
            result_error = _raw_result_error(latest)
            if result_error is not None:
                errors.append(
                    f"raw eval artifact {artifact_dir.name!r} latest result "
                    f"{latest.name!r}{conc_label} {result_error}"
                )
                continue
            result_data = load_json(latest)
            result_tasks = result_data["results"]
            contribution_meta = (
                {**meta, "conc": conc} if conc is not None else meta
            )
            rows.extend(
                eval_result_key({**contribution_meta, "task": task})
                for task in result_tasks
            )
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
                try:
                    data = load_json(path)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"eval aggregate {path.name!r} is invalid JSON: {exc}"
                    )
                    continue
                if not isinstance(data, list):
                    errors.append(
                        f"eval aggregate {path.name!r} is not a list"
                    )
                    continue
                row_count += len(data)
                for index, row in enumerate(data):
                    if not isinstance(row, dict):
                        errors.append(
                            f"eval aggregate {path.name!r} row {index} "
                            "is not an object"
                        )
                        continue
                    if invalid_eval_suite(row):
                        errors.append(
                            f"eval aggregate {path.name!r} row {index} "
                            "has invalid eval_suite"
                        )
                        continue
                    aggregate_rows.append(eval_result_key(row))
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


# ── Dedupe reran eval artifacts ───────────────────────────────────────────────
#
# A flaky eval retried several times leaves multiple raw ``eval_*`` dirs and
# multiple ``eval_results_all`` rows for one logical eval identity, which the
# checks above would otherwise reject. ``dedupe_reran_evals`` collapses those to
# the latest result per identity (by timestamp or legacy mtime) so a legitimate
# rerun does not fail validation. It only acts on identities that have a clear
# latest result, ordered by a filename timestamp or legacy mtime. Identities
# with no result file are left in place for validation to reject. Eval-only;
# fixed-sequence and agentic artifacts are untouched.

# lm-eval result files are ``results_<ISO>.json`` (optionally a ``_concN`` /
# staging suffix). Timestamped names and legacy mtimes are both converted to
# epoch nanoseconds so mixed naming schemes have one coherent ordering.
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?")
_EVAL_RESULT_FORMAT = "inferencex-eval-v1"

# Batched result files carry their concurrency as a ``_concN`` suffix (kept in
# sync with ``collect_eval_results.CONC_SUFFIX_RE``).
_CONC_SUFFIX_RE = re.compile(r"_conc(\d+)(?:_\d+)?\.json$")


def _result_concurrency(name: str) -> Optional[int]:
    """Extract a batched eval concurrency from a staged result file name."""
    match = _CONC_SUFFIX_RE.search(name)
    return int(match.group(1)) if match else None


def _result_timestamp(name: str) -> Optional[str]:
    """Extract the sortable lm-eval timestamp from a result file name."""
    match = _TIMESTAMP_RE.search(name)
    return match.group(0) if match else None


def _timestamp_ns(stamp: str) -> int:
    """Convert an lm-eval filename timestamp to UTC epoch nanoseconds."""
    date, clock = stamp.split("T", 1)
    hms, separator, fraction = clock.partition(".")
    parsed = datetime.strptime(
        f"{date}T{hms}",
        "%Y-%m-%dT%H-%M-%S",
    ).replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = parsed - epoch
    fractional_ns = (
        int((fraction + "000000000")[:9]) if separator else 0
    )
    return (
        delta.days * 86_400_000_000_000
        + delta.seconds * 1_000_000_000
        + fractional_ns
    )


def _result_order(path: Path) -> tuple[int, str]:
    """Return one deterministic recency key for timestamped and legacy files."""
    stamp = _result_timestamp(path.name)
    try:
        recency = (
            _timestamp_ns(stamp)
            if stamp is not None
            else path.stat().st_mtime_ns
        )
    except ValueError:
        recency = path.stat().st_mtime_ns
    return recency, path.name


def _recognized_eval_result_paths(paths: Iterable[Path]) -> list[Path]:
    """Return result JSONs carrying a collector-recognized eval marker."""
    recognized: list[Path] = []
    for path in paths:
        try:
            data = load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and (
            "lm_eval_version" in data
            or data.get("result_format") == _EVAL_RESULT_FORMAT
        ):
            recognized.append(path)
    return recognized


def _raw_result_error(path: Path) -> Optional[str]:
    """Return a structural error for a raw result, or None when reusable."""
    try:
        data = load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"is malformed JSON: {exc}"
    if not isinstance(data, dict):
        return "is not an object"
    if "integration_error" in data:
        return "reports an integration error"
    if (
        "lm_eval_version" not in data
        and data.get("result_format") != _EVAL_RESULT_FORMAT
    ):
        return "has no recognized eval result format"

    results = data.get("results")
    if not isinstance(results, dict) or not results:
        return "has empty or malformed results"
    configs = data.get("configs", {})
    if not isinstance(configs, dict):
        return "has malformed configs"

    sample_counts = data.get("n-samples")
    if "n-samples" in data and not isinstance(sample_counts, dict):
        return "has malformed effective sample counts"

    for task, metrics in results.items():
        if not isinstance(task, str) or not task:
            return "has an invalid task name"
        if not isinstance(metrics, dict) or not metrics:
            return f"has empty or malformed results for task {task!r}"
        task_config = configs.get(task, {})
        if not isinstance(task_config, dict):
            return f"has malformed config for task {task!r}"
        metric_list = task_config.get("metric_list", [])
        filter_list = task_config.get("filter_list", [])
        if not isinstance(metric_list, list) or not isinstance(filter_list, list):
            return f"has malformed config for task {task!r}"
        if metric_list:
            first_metric = metric_list[0]
            if (
                not isinstance(first_metric, dict)
                or not isinstance(first_metric.get("metric"), str)
                or not first_metric["metric"]
            ):
                return f"has malformed metric config for task {task!r}"
            base_metric = first_metric["metric"]
        else:
            base_metric = "exact_match"
        if filter_list:
            if any(
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"]
                for item in filter_list
            ):
                return f"has malformed filter config for task {task!r}"
            configured_names = [
                f"{base_metric},{item['name']}"
                for item in filter_list
            ]
            strict_names = [
                name
                for name in configured_names
                if "strict" in name or "resolved" in name
            ]
            fallback_names = [
                name
                for name in configured_names
                if "flex" in name or "extract" in name
            ]
            primary_names = strict_names or fallback_names or configured_names
        else:
            primary_names = ["acc" if "acc" in metrics else base_metric]
        if not primary_names or any(name not in metrics for name in primary_names):
            return f"has no score for task {task!r}"

        for name in primary_names:
            score = metrics[name]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or score < 0
                or score > 1
            ):
                return (
                    f"has invalid score {name!r} for task {task!r}: "
                    f"{score!r}"
                )
        if sample_counts is not None:
            task_counts = sample_counts.get(task)
            if not isinstance(task_counts, dict) or "effective" not in task_counts:
                return f"has malformed effective sample count for task {task!r}"
            effective = task_counts["effective"]
            if (
                isinstance(effective, bool)
                or not isinstance(effective, (int, float))
                or not math.isfinite(effective)
                or effective <= 0
            ):
                return (
                    f"has invalid effective sample count for task {task!r}: "
                    f"{effective!r}"
                )
    return None


def _raw_dir_contributions(
    artifact_dir: Path,
) -> tuple[list[tuple[tuple[Any, ...], Optional[int]]], dict[str, Any], bool]:
    """Return validated (identity, conc) contributions and raw metadata."""
    try:
        meta = load_json(artifact_dir / "meta_env.json")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], {}, False
    if not isinstance(meta, dict):
        return [], {}, False
    if invalid_eval_suite(meta):
        return [], meta, False
    contributions, batched, errors = _raw_meta_contributions(
        artifact_dir.name,
        meta,
    )
    if errors:
        return [], meta, batched
    return contributions, meta, batched


def _source_names_raw_dir(source: Any, artifact_name: str) -> bool:
    """Return whether an aggregate source path names this exact raw directory."""
    return artifact_name in re.split(r"[\\/]+", str(source or ""))


def _eval_winners(artifacts_dir: Path) -> dict[tuple[Any, ...], str]:
    """Pick structurally valid, aggregate-backed latest raw results."""
    aggregate_sources: dict[tuple[Any, ...], list[Any]] = {}
    aggregate_dir = artifacts_dir / "eval_results_all"
    for path in sorted(aggregate_dir.glob("*.json")):
        try:
            data = load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict) and not invalid_eval_suite(row):
                aggregate_sources.setdefault(eval_key(row), []).append(
                    row.get("source")
                )

    best: dict[
        tuple[Any, ...],
        tuple[tuple[int, str], str, Path],
    ] = {}
    for artifact_dir in raw_eval_artifact_dirs(artifacts_dir):
        contributions, _, batched = _raw_dir_contributions(artifact_dir)
        result_paths = _recognized_eval_result_paths(
            artifact_dir.glob("results*.json")
        )
        for key, key_conc in contributions:
            candidates = [
                path
                for path in result_paths
                if not batched or _result_concurrency(path.name) == key_conc
            ]
            if not candidates:
                continue
            latest = max(candidates, key=_result_order)
            candidate = (_result_order(latest), artifact_dir.name, latest)
            current = best.get(key)
            if current is None or candidate[:2] > current[:2]:
                best[key] = candidate

    winners: dict[tuple[Any, ...], str] = {}
    for key, (_, artifact_name, path) in best.items():
        if _raw_result_error(path) is not None:
            continue
        if any(
            _source_names_raw_dir(source, artifact_name)
            for source in aggregate_sources.get(key, [])
        ):
            winners[key] = artifact_name
    return winners


def _dedupe_eval_aggregate(
    artifacts_dir: Path, winners: dict[tuple[Any, ...], str]
) -> list[str]:
    """Keep one aggregate row per winning identity across all aggregate files."""
    eval_dir = artifacts_dir / "eval_results_all"
    if not eval_dir.is_dir():
        return []

    loaded: dict[Path, list[Any]] = {}
    groups: dict[
        tuple[Any, ...],
        list[tuple[Path, int, dict[str, Any]]],
    ] = {}
    for agg_path in sorted(eval_dir.glob("*.json")):
        try:
            data = load_json(agg_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        loaded[agg_path] = data
        for index, row in enumerate(data):
            if isinstance(row, dict) and not invalid_eval_suite(row):
                groups.setdefault(eval_result_key(row), []).append(
                    (agg_path, index, row)
                )

    keep = {
        path: set(range(len(data)))
        for path, data in loaded.items()
    }
    winner_result_names: dict[tuple[Any, ...], str] = {}
    for key, artifact_name in winners.items():
        artifact_dir = artifacts_dir / artifact_name
        contributions, _, batched = _raw_dir_contributions(artifact_dir)
        conc = next(
            (candidate_conc for candidate_key, candidate_conc in contributions
             if candidate_key == key),
            None,
        )
        candidates = [
            path
            for path in _recognized_eval_result_paths(
                artifact_dir.glob("results*.json")
            )
            if not batched or _result_concurrency(path.name) == conc
        ]
        if candidates:
            winner_result_names[key] = max(candidates, key=_result_order).name

    for key, entries in groups.items():
        artifact_key = key[:-1]
        winner = winners.get(artifact_key)
        if winner is None or len(entries) == 1:
            continue
        matching = [
            entry
            for entry in entries
            if _source_names_raw_dir(entry[2].get("source"), winner)
        ]
        winner_result_name = winner_result_names.get(artifact_key)
        exact_matching = [
            entry
            for entry in matching
            if re.split(
                r"[\\/]+",
                str(entry[2].get("source") or ""),
            )[-1] == winner_result_name
        ]
        if not exact_matching:
            continue
        chosen = max(
            exact_matching,
            key=lambda entry: (entry[0].name, entry[1]),
        )
        chosen_location = chosen[0], chosen[1]
        for path, index, _ in entries:
            if (path, index) != chosen_location:
                keep[path].discard(index)

    messages: list[str] = []
    for agg_path, data in loaded.items():
        kept = [
            row
            for index, row in enumerate(data)
            if index in keep[agg_path]
        ]
        if len(kept) == len(data):
            continue
        agg_path.write_text(json.dumps(kept, indent=2))
        messages.append(
            f"{agg_path.name}: kept {len(kept)} of {len(data)} eval row(s)"
        )
    return messages


def _prune_raw_eval_dir(
    artifact_dir: Path, winners: dict[tuple[Any, ...], str]
) -> Optional[str]:
    """Drop a raw dir's identities that a newer dir supersedes."""
    contributions, meta, batched = _raw_dir_contributions(artifact_dir)
    if not contributions:
        return None
    name = artifact_dir.name

    def superseded(key: tuple[Any, ...]) -> bool:
        winner = winners.get(key)
        return winner is not None and winner != name

    if not batched:
        if superseded(contributions[0][0]):
            shutil.rmtree(artifact_dir)
            return f"removed superseded raw eval dir {name!r}"
        return None

    losing = {
        conc for key, conc in contributions if conc is not None and superseded(key)
    }
    if not losing:
        return None
    for path in artifact_dir.glob("results*.json"):
        if _result_concurrency(path.name) in losing:
            path.unlink()
    remaining = [
        conc
        for conc in meta.get("completed_eval_concs", [])
        if conc not in losing
    ]
    if not remaining:
        shutil.rmtree(artifact_dir)
        return f"removed superseded batched raw eval dir {name!r}"
    meta["eval_concs"] = remaining
    meta["completed_eval_concs"] = remaining
    (artifact_dir / "meta_env.json").write_text(json.dumps(meta))
    dropped = ",".join(str(conc) for conc in sorted(losing))
    return (
        f"pruned superseded conc(s) {dropped} from batched raw eval dir {name!r}"
    )


def dedupe_reran_evals(artifacts_dir: Path) -> list[str]:
    """Collapse reran eval duplicates in place; return a change log."""
    winners = _eval_winners(artifacts_dir)
    messages = _dedupe_eval_aggregate(artifacts_dir, winners)
    for artifact_dir in raw_eval_artifact_dirs(artifacts_dir):
        message = _prune_raw_eval_dir(artifact_dir, winners)
        if message:
            messages.append(message)
    return messages


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.artifacts_dir.is_dir():
        raise ValueError(
            f"artifacts directory does not exist: {args.artifacts_dir}"
        )

    # Collapse reran (flaky) eval duplicates to the latest result before
    # validating, so a legitimate rerun does not fail the consistency checks.
    dedupe_messages = dedupe_reran_evals(args.artifacts_dir)
    if dedupe_messages:
        print("Collapsed reran eval duplicates (kept latest result per identity):")
        for message in dedupe_messages:
            print(f"  {message}")

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
