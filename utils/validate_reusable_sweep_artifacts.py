#!/usr/bin/env python3
"""Validate reused sweep artifacts for internal consistency."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
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


# ── Dedupe reran eval artifacts ───────────────────────────────────────────────
#
# A flaky eval retried several times leaves multiple raw ``eval_*`` dirs and
# multiple ``eval_results_all`` rows for one logical eval identity, which the
# checks above would otherwise reject. ``dedupe_reran_evals`` collapses those to
# the latest result per identity (by lm-eval result timestamp) so a legitimate
# rerun does not fail validation. It only acts on identities that have a clear
# latest result; genuinely ambiguous duplicates (no result timestamp to order
# them by) are left in place for validation to reject. Eval-only; fixed-sequence
# and agentic artifacts are untouched.

# lm-eval result files are ``results_<ISO>.json`` (optionally a ``_concN`` /
# staging suffix). The timestamp uses dashes throughout, so it is fixed-width
# and lexicographically sortable.
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?")

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


def _raw_dir_contributions(
    artifact_dir: Path,
) -> tuple[list[tuple[tuple[Any, ...], Optional[int]]], dict[str, Any], bool]:
    """Return (identity, conc) pairs a raw dir contributes, plus its meta.

    Mirrors ``raw_eval_key_rows``: a batched artifact contributes one identity
    per ``completed_eval_concs`` entry; a legacy artifact contributes one from
    its meta. ``conc`` is the batched concurrency (``None`` for legacy).
    """
    meta = load_json(artifact_dir / "meta_env.json")
    if not isinstance(meta, dict):
        return [], {}, False
    batched = isinstance(meta.get("eval_concs"), list)
    if batched:
        concs = meta.get("completed_eval_concs")
        if not isinstance(concs, list):
            return [], meta, True
        return (
            [(eval_key({**meta, "conc": conc}), as_int(conc)) for conc in concs],
            meta,
            True,
        )
    return [(eval_key(meta), None)], meta, False


def _eval_winners(artifacts_dir: Path) -> dict[tuple[Any, ...], str]:
    """Pick the raw dir holding the latest result for each eval identity.

    Ranks only by the lm-eval result timestamp, so an identity appears here
    only when at least one of its raw dirs carries a real result. Identities
    with no timestamped result get no winner and are left untouched.
    """
    best: dict[tuple[Any, ...], tuple[str, str]] = {}
    for artifact_dir in raw_eval_artifact_dirs(artifacts_dir):
        contributions, _, batched = _raw_dir_contributions(artifact_dir)
        if not contributions:
            continue
        for path in artifact_dir.glob("results*.json"):
            stamp = _result_timestamp(path.name)
            if stamp is None:
                continue
            conc = _result_concurrency(path.name)
            for key, key_conc in contributions:
                # A batched result file is tagged with its conc; only let it
                # compete for the matching identity.
                if batched and conc is not None and key_conc != conc:
                    continue
                candidate = (stamp, artifact_dir.name)
                if best.get(key) is None or candidate > best[key]:
                    best[key] = candidate
    return {key: name for key, (_, name) in best.items()}


def _dedupe_eval_aggregate(
    artifacts_dir: Path, winners: dict[tuple[Any, ...], str]
) -> list[str]:
    """Keep one aggregate row per winning identity (its winner dir's row)."""
    eval_dir = artifacts_dir / "eval_results_all"
    if not eval_dir.is_dir():
        return []
    messages: list[str] = []
    for agg_path in sorted(eval_dir.glob("*.json")):
        data = load_json(agg_path)
        if not isinstance(data, list):
            continue
        groups: dict[tuple[Any, ...], list[int]] = {}
        keep: set[int] = set()
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                keep.add(idx)
                continue
            groups.setdefault(eval_key(row), []).append(idx)
        for key, indices in groups.items():
            winner = winners.get(key)
            # Only collapse identities with a clear latest result; leave
            # ambiguous duplicates for validation to reject.
            if winner is None or len(indices) == 1:
                keep.update(indices)
                continue
            keep.add(
                next(
                    (
                        idx
                        for idx in indices
                        if winner in str(data[idx].get("source") or "")
                    ),
                    max(
                        indices,
                        key=lambda idx: _result_timestamp(
                            str(data[idx].get("source") or "")
                        )
                        or "",
                    ),
                )
            )
        if len(keep) != len(data):
            kept = [row for idx, row in enumerate(data) if idx in keep]
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
        if as_int(conc) not in losing
    ]
    if not remaining:
        shutil.rmtree(artifact_dir)
        return f"removed superseded batched raw eval dir {name!r}"
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
