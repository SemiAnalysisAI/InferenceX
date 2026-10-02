"""Validate and deduplicate downloaded eval artifacts."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from infx.results.artifacts import (
    as_bool,
    as_int,
    duplicate_identity_errors,
    load_json,
    validate_identity_set,
)
from infx.results.evals import (
    read_eval_results,
    result_concurrency,
    result_error,
    result_order,
    select_latest_result,
)


def normalized_runner(value: Any) -> str:
    """Normalize runner labels that aggregates may uppercase."""
    return str(value or "").lower()


LEGACY_EVAL_SUITE = "<legacy-eval-suite>"


def invalid_eval_suite(row: dict[str, Any]) -> bool:
    """Return whether an explicit eval-suite identity is malformed."""
    suite = row.get("eval_suite")
    return "eval_suite" in row and (not isinstance(suite, str) or not suite)


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
    list[tuple[tuple[Any, ...], int | None]],
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


@dataclass
class _RawEvalArtifact:
    path: Path
    meta: dict[str, Any]
    contributions: list[tuple[tuple[Any, ...], int | None]]
    batched: bool
    errors: list[str]

    @cached_property
    def results(self) -> dict[Path, dict[str, Any]]:
        return read_eval_results(self.path.glob("results*.json"))

    @classmethod
    def read(cls, path: Path) -> _RawEvalArtifact:
        artifact = cls(path, {}, [], False, [])
        prefix = f"raw eval artifact {path.name!r}"
        meta_path = path / "meta_env.json"
        if not meta_path.is_file():
            artifact.errors.append(f"{prefix} is missing meta_env.json")
            return artifact
        try:
            meta = load_json(meta_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            artifact.errors.append(f"{prefix} has invalid meta_env.json: {exc}")
            return artifact
        if not isinstance(meta, dict):
            artifact.errors.append(f"{prefix} has non-object meta_env.json")
            return artifact
        artifact.meta = meta
        if invalid_eval_suite(meta):
            artifact.errors.append(f"{prefix} has invalid eval_suite")
            return artifact
        artifact.contributions, artifact.batched, artifact.errors = _raw_meta_contributions(
            path.name, meta
        )
        return artifact


def raw_eval_key_rows(
    artifacts_dir: Path,
) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Build and validate logical identities from raw eval artifacts."""
    rows: list[tuple[Any, ...]] = []
    errors: list[str] = []
    for artifact_dir in raw_eval_artifact_dirs(artifacts_dir):
        artifact = _RawEvalArtifact.read(artifact_dir)
        errors.extend(artifact.errors)
        if artifact.errors:
            continue
        meta = artifact.meta
        if artifact.batched:
            expected = set(meta["eval_concs"])
            for path in artifact.results:
                conc = result_concurrency(path.name)
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

        for _, conc in artifact.contributions:
            latest = select_latest_result(artifact.results, concurrency=conc)
            conc_label = f" for concurrency {conc}" if conc is not None else ""
            if latest is None:
                errors.append(
                    f"raw eval artifact {artifact_dir.name!r} has no "
                    f"recognized eval result{conc_label}"
                )
                continue
            error = result_error(artifact.results[latest])
            if error is not None:
                errors.append(
                    f"raw eval artifact {artifact_dir.name!r} latest result "
                    f"{latest.name!r}{conc_label} {error}"
                )
                continue
            result_tasks = artifact.results[latest]["results"]
            contribution_meta = {**meta, "conc": conc} if conc is not None else meta
            rows.extend(
                eval_result_key({**contribution_meta, "task": task}) for task in result_tasks
            )
    return rows, errors


def validate_eval_artifacts(
    artifacts_dir: Path,
) -> list[str]:
    """Validate raw and aggregate eval artifacts agree."""
    return inspect_eval_artifacts(artifacts_dir)[1]


def inspect_eval_artifacts(
    artifacts_dir: Path,
) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Return raw result identities and all raw/aggregate consistency errors."""
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
                    errors.append(f"eval aggregate {path.name!r} is invalid JSON: {exc}")
                    continue
                if not isinstance(data, list):
                    errors.append(f"eval aggregate {path.name!r} is not a list")
                    continue
                row_count += len(data)
                for index, row in enumerate(data):
                    if not isinstance(row, dict):
                        errors.append(f"eval aggregate {path.name!r} row {index} is not an object")
                        continue
                    if invalid_eval_suite(row):
                        errors.append(
                            f"eval aggregate {path.name!r} row {index} has invalid eval_suite"
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

    return raw_rows, errors


def _source_names_raw_dir(source: Any, artifact_name: str) -> bool:
    """Return whether an aggregate source path names this exact raw directory."""
    return artifact_name in re.split(r"[\\/]+", str(source or ""))


def _dedupe_eval_aggregate(
    loaded: dict[Path, list[Any]], winners: dict[tuple[Any, ...], Path]
) -> list[str]:
    """Keep one aggregate row per winning identity across all aggregate files."""
    groups: dict[
        tuple[Any, ...],
        list[tuple[Path, int, dict[str, Any]]],
    ] = {}
    for agg_path, data in loaded.items():
        for index, row in enumerate(data):
            if isinstance(row, dict) and not invalid_eval_suite(row):
                groups.setdefault(eval_result_key(row), []).append((agg_path, index, row))

    keep = {path: set(range(len(data))) for path, data in loaded.items()}
    for key, entries in groups.items():
        artifact_key = key[:-1]
        winner = winners.get(artifact_key)
        if winner is None or len(entries) == 1:
            continue
        matching = [
            entry
            for entry in entries
            if _source_names_raw_dir(entry[2].get("source"), winner.parent.name)
        ]
        exact_matching = [
            entry
            for entry in matching
            if re.split(
                r"[\\/]+",
                str(entry[2].get("source") or ""),
            )[-1]
            == winner.name
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
        kept = [row for index, row in enumerate(data) if index in keep[agg_path]]
        if len(kept) == len(data):
            continue
        agg_path.write_text(json.dumps(kept, indent=2))
        messages.append(f"{agg_path.name}: kept {len(kept)} of {len(data)} eval row(s)")
    return messages


def _prune_raw_eval_dir(
    artifact: _RawEvalArtifact, winners: dict[tuple[Any, ...], Path]
) -> str | None:
    """Drop a raw dir's identities that a newer dir supersedes."""
    if not artifact.contributions:
        return None
    artifact_dir = artifact.path
    meta = artifact.meta
    name = artifact_dir.name

    def superseded(key: tuple[Any, ...]) -> bool:
        winner = winners.get(key)
        return winner is not None and winner.parent.name != name

    if not artifact.batched:
        if superseded(artifact.contributions[0][0]):
            shutil.rmtree(artifact_dir)
            return f"removed superseded raw eval dir {name!r}"
        return None

    losing = {conc for key, conc in artifact.contributions if conc is not None and superseded(key)}
    if not losing:
        return None
    for path in artifact_dir.glob("results*.json"):
        if result_concurrency(path.name) in losing:
            path.unlink()
    remaining = [conc for conc in meta.get("completed_eval_concs", []) if conc not in losing]
    if not remaining:
        shutil.rmtree(artifact_dir)
        return f"removed superseded batched raw eval dir {name!r}"
    meta["eval_concs"] = remaining
    meta["completed_eval_concs"] = remaining
    (artifact_dir / "meta_env.json").write_text(json.dumps(meta))
    dropped = ",".join(str(conc) for conc in sorted(losing))
    return f"pruned superseded conc(s) {dropped} from batched raw eval dir {name!r}"


def dedupe_reran_evals(artifacts_dir: Path) -> list[str]:
    """Collapse reran eval duplicates in place; return a change log."""
    aggregate_sources: dict[tuple[Any, ...], list[Any]] = {}
    aggregates = {}
    for path in sorted((artifacts_dir / "eval_results_all").glob("*.json")):
        try:
            data = load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        aggregates[path] = data
        for row in data:
            if isinstance(row, dict) and not invalid_eval_suite(row):
                aggregate_sources.setdefault(eval_key(row), []).append(row.get("source"))

    best: dict[
        tuple[Any, ...],
        tuple[tuple[int, str], str, Path, dict[str, Any]],
    ] = {}
    artifacts = []
    for path in raw_eval_artifact_dirs(artifacts_dir):
        artifact = _RawEvalArtifact.read(path)
        artifacts.append(artifact)
        results = artifact.results
        for key, key_conc in artifact.contributions:
            latest = select_latest_result(results, concurrency=key_conc)
            if latest is None:
                continue
            candidate = (result_order(latest), artifact.path.name, latest, results[latest])
            current = best.get(key)
            if current is None or candidate[:2] > current[:2]:
                best[key] = candidate

    winners: dict[tuple[Any, ...], Path] = {}
    for key, (_, artifact_name, path, data) in best.items():
        if result_error(data) is not None:
            continue
        if any(
            _source_names_raw_dir(source, artifact_name)
            for source in aggregate_sources.get(key, [])
        ):
            winners[key] = path

    messages = _dedupe_eval_aggregate(aggregates, winners)
    for artifact in artifacts:
        message = _prune_raw_eval_dir(artifact, winners)
        if message:
            messages.append(message)
    return messages
