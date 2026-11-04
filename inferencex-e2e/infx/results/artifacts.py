"""Identity and consistency checks for downloaded sweep artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


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
            *(
                ("dwdp", as_int(row["prefill_dwdp_size"], 1))
                if as_int(row.get("prefill_dwdp_size", 1), 1) > 1
                else ()
            ),
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
        return tuple(sorted((key, freeze_identity_value(item)) for key, item in value.items()))
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
            *(
                ("dwdp", as_int(row["prefill_dwdp_size"], 1))
                if as_int(row.get("prefill_dwdp_size", 1), 1) > 1
                else ()
            ),
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
        row.get("spec_decoding", "none"),
        as_bool(row.get("disagg", False)),
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
    """Return downloaded bmk_agentic point-result JSON files, latest run attempt only."""
    paths: set[Path] = set()
    for artifact_dir in artifacts_dir.glob("bmk_agentic_*"):
        if artifact_dir.is_dir():
            paths.update(artifact_dir.rglob("*.json"))
    return sorted(paths - superseded_attempt_files(paths))


def superseded_attempt_files(paths: Iterable[Path]) -> set[Path]:
    """Files whose every row is a point rerun by a later run_attempt elsewhere."""
    attempts: dict[Path, list[tuple[tuple[Any, ...], int]]] = {}
    latest: dict[tuple[Any, ...], int] = {}
    for path in paths:
        try:
            rows = [row for _, row in json_rows([path])]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not rows or not all(isinstance(row.get("run_attempt"), int) for row in rows):
            continue
        attempts[path] = [(agentic_key(row), row["run_attempt"]) for row in rows]
        for key, attempt in attempts[path]:
            latest[key] = max(attempt, latest.get(key, attempt))
    return {
        path
        for path, rows in attempts.items()
        if all(attempt < latest[key] for key, attempt in rows)
    }


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
        errors.extend(f"  missing: {key}" for key in sorted(missing, key=repr)[:20])
        if len(missing) > 20:
            errors.append(f"  ... and {len(missing) - 20} more")
    if extra:
        errors.append(f"{label} artifacts contain {len(extra)} unexpected row(s)")
        errors.extend(f"  unexpected: {key}" for key in sorted(extra, key=repr)[:20])
        if len(extra) > 20:
            errors.append(f"  ... and {len(extra) - 20} more")
    return errors


def duplicate_identity_errors(
    label: str,
    identities: Iterable[tuple[Any, ...]],
) -> list[str]:
    """Return errors naming each ``identities`` entry seen more than once, which set equality would hide."""
    counts = Counter(identities)
    duplicates = {identity: count for identity, count in counts.items() if count > 1}
    if not duplicates:
        return []

    duplicate_rows = sum(count - 1 for count in duplicates.values())
    errors = [f"{label} artifacts contain {duplicate_rows} duplicate row(s)"]
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
    """Return duplicate-identity errors for the fixed-sequence rows in ``artifacts_dir``."""
    actual_rows = actual_benchmark_key_rows(artifacts_dir)
    return duplicate_identity_errors("fixed-sequence", actual_rows)


def validate_agentic_artifacts(
    artifacts_dir: Path,
) -> list[str]:
    """Return errors where the agentic point, raw and aggregate artifacts in ``artifacts_dir`` disagree.

    Raw dirs of attempts replaced by a later rerun still download and are ignored.
    """
    point_files = agentic_point_files(artifacts_dir)
    point_rows = agentic_keys_from_paths(point_files)
    errors = duplicate_identity_errors("agentic point", point_rows)

    results_bmk = artifacts_dir / "results_bmk"
    if results_bmk.is_dir():
        aggregate_rows = agentic_keys_from_paths(results_bmk.glob("*.json"))
        errors.extend(duplicate_identity_errors("agentic aggregate", aggregate_rows))
        errors.extend(
            validate_identity_set(
                "agentic aggregate",
                set(point_rows),
                set(aggregate_rows),
            )
        )

    point_names = {
        path.relative_to(artifacts_dir).parts[0].removeprefix("bmk_") for path in point_files
    }
    superseded_names = {
        path.relative_to(artifacts_dir).parts[0].removeprefix("bmk_")
        for path in artifacts_dir.glob("bmk_agentic_*/**/*.json")
    } - point_names
    raw_names = {
        path.name
        for path in artifacts_dir.iterdir()
        if path.is_dir() and path.name.startswith("agentic_")
    } - superseded_names
    if point_names != raw_names:
        missing_raw = point_names - raw_names
        extra_raw = raw_names - point_names
        for name in sorted(missing_raw):
            errors.append(f"missing raw agentic artifact dir: {name}")
        for name in sorted(extra_raw):
            errors.append(f"unexpected raw agentic artifact dir: {name}")

    return errors


def validate_run_stats(artifacts_dir: Path, required: bool) -> list[str]:
    """Return an error when ``required`` and ``artifacts_dir`` holds no run-stats JSON, else []."""
    if not required:
        return []
    if list((artifacts_dir / "run-stats").glob("*.json")):
        return []
    return ["missing run-stats artifact for fixed-sequence benchmarks"]
