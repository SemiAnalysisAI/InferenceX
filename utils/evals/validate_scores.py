#!/usr/bin/env python3
"""Validate eval scores against minimum thresholds.

Reads lm-eval results JSON files and checks that scored metrics meet the
required minimum.  Thresholds are configured per-task, with optional per-model
overrides, in a JSON config file (default: utils/evals/thresholds.json):

    {
      "default": { "gsm8k": 0.85, "gpqa_diamond_cot_n_shot": 0.30 },
      "models": {
        "dsv4": { "gsm8k": 0.90 },
        "glm5": { "gsm8k": 0.92 }
      }
    }

The model is identified by its `infmax_model_prefix` (e.g. "dsv4", "glm5"),
read from meta_env.json in the current directory -- written alongside the
results*.json files by the eval harness.  For each task the threshold is
resolved most-specific-first:

    models[<prefix>][<task>]  ->  default[<task>]  ->  --min-score

Models without an entry under "models" (or runs where the prefix can't be
determined) fall back to the global default, then to --min-score.

A legacy flat config ({"gsm8k": 0.85, ...}) is still accepted and treated as
the global default with no per-model overrides.

Usage:
    python3 utils/evals/validate_scores.py
    python3 utils/evals/validate_scores.py --expected-concs "1 2 4 8"
    python3 utils/evals/validate_scores.py --thresholds my_thresholds.json
    python3 utils/evals/validate_scores.py --model-prefix dsv4
    python3 utils/evals/validate_scores.py --min-score 0.90  # flat fallback
"""
import argparse
from collections import Counter
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

CONC_SUFFIX_RE = re.compile(r"_conc(\d+)(?:_\d+)?\.json$")


def _validate_threshold_map(values: object, location: str) -> dict[str, float]:
    """Validate and normalize one task-to-threshold mapping."""
    if not isinstance(values, dict):
        raise ValueError(f"{location} must be a JSON object")

    normalized = {}
    for task, value in values.items():
        if not isinstance(task, str) or not task:
            raise ValueError(f"{location} contains an invalid task name")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                f"{location}.{task} must be a finite number between 0 and 1"
            )
        normalized[task] = float(value)
    return normalized


def load_config(path: str) -> dict:
    """Load thresholds config, normalized to {"default": {...}, "models": {...}}.

    Accepts both the per-model format ({"default": {...}, "models": {...}}) and
    the legacy flat format ({task: min_score}), which is treated as the global
    default with no per-model overrides.
    """
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("thresholds config must be a JSON object")
    if "default" not in cfg and "models" not in cfg:
        # Legacy flat format: the whole object is the per-task default.
        return {
            "default": _validate_threshold_map(cfg, "default"),
            "models": {},
        }

    unknown_keys = sorted(set(cfg) - {"default", "models"})
    if unknown_keys:
        raise ValueError(
            "thresholds config contains unsupported keys: "
            + ", ".join(unknown_keys)
        )

    models = cfg.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models must be a JSON object")
    normalized_models = {}
    for prefix, thresholds in models.items():
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("models contains an invalid model prefix")
        normalized_models[prefix] = _validate_threshold_map(
            thresholds,
            f"models.{prefix}",
        )

    return {
        "default": _validate_threshold_map(cfg.get("default", {}), "default"),
        "models": normalized_models,
    }


def detect_model_prefix(meta_env_path: str, override: str | None) -> str | None:
    """Resolve the model prefix: explicit override > meta_env.json > env var."""
    if override:
        return override
    try:
        with open(meta_env_path) as f:
            prefix = json.load(f).get("infmax_model_prefix")
        if isinstance(prefix, str) and prefix and prefix != "unknown":
            return prefix
    except (json.JSONDecodeError, OSError, AttributeError):
        pass
    env_prefix = os.environ.get("MODEL_PREFIX")
    if env_prefix and env_prefix != "unknown":
        return env_prefix
    return None


def resolve_threshold(config: dict, prefix: str | None, task: str, fallback: float):
    """Return (min_score, source) for a task, most-specific-first."""
    models = config.get("models", {})
    if prefix and task in models.get(prefix, {}):
        return models[prefix][task], f"models.{prefix}"
    default = config.get("default", {})
    if task in default:
        return default[task], "default"
    return fallback, "min-score"


def parse_expected_concs(raw_value: str | None) -> list[int] | None:
    """Parse a workflow-provided, space-separated concurrency list."""
    if raw_value is None:
        return None
    if not raw_value.strip():
        raise ValueError("expected concurrency list is empty")

    values = raw_value.split()
    if not all(re.fullmatch(r"[1-9][0-9]*", value) for value in values):
        raise ValueError(
            "expected concurrencies must be positive integers separated by spaces"
        )

    concs = [int(value) for value in values]
    if len(set(concs)) != len(concs):
        raise ValueError("expected concurrency list contains duplicates")
    return concs


def _is_valid_conc_list(
    values: object,
    *,
    allow_empty: bool = True,
) -> bool:
    """Return whether a value is a list of unique positive integer concurrencies."""
    return (
        isinstance(values, list)
        and (allow_empty or bool(values))
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in values
        )
        and len(set(values)) == len(values)
    )


def validate_batch_manifest(
    meta_env_path: str,
    result_files: list[str],
    expected_concs: list[int] | None = None,
) -> list[str]:
    """Validate that eval artifacts cover every workflow-requested concurrency."""
    errors = []
    if expected_concs is not None and not _is_valid_conc_list(
        expected_concs,
        allow_empty=False,
    ):
        return ["workflow supplied an invalid expected concurrency list"]
    if not result_files:
        errors.append("eval produced no result files")

    try:
        with open(meta_env_path) as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError("metadata root must be a JSON object")
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        if expected_concs is not None or any(
            CONC_SUFFIX_RE.search(Path(result_file).name)
            for result_file in result_files
        ):
            errors.append(
                f"eval metadata {meta_env_path} is unavailable or invalid: {exc}"
            )
        return errors

    eval_exit_code = meta.get("eval_exit_code")
    if expected_concs is not None and (
        not isinstance(eval_exit_code, int)
        or isinstance(eval_exit_code, bool)
        or eval_exit_code < 0
    ):
        errors.append(
            "eval metadata must contain a non-negative integer eval_exit_code"
        )
    elif "eval_exit_code" in meta and (
        not isinstance(eval_exit_code, int)
        or isinstance(eval_exit_code, bool)
        or eval_exit_code < 0
    ):
        errors.append("eval metadata contains an invalid eval_exit_code")
    elif isinstance(eval_exit_code, int) and eval_exit_code != 0:
        errors.append(f"eval command failed with exit code {eval_exit_code}")

    metadata_expected = meta.get("eval_concs")
    metadata_is_batched = "eval_concs" in meta
    if expected_concs is None and not metadata_is_batched:
        if any(
            CONC_SUFFIX_RE.search(Path(result_file).name)
            for result_file in result_files
        ):
            errors.append(
                "concurrency-suffixed eval results exist but batched metadata is missing"
            )
        return errors

    if expected_concs is not None and len(expected_concs) == 1 and not metadata_is_batched:
        metadata_conc = meta.get("conc")
        if (
            not isinstance(metadata_conc, int)
            or isinstance(metadata_conc, bool)
            or metadata_conc != expected_concs[0]
        ):
            errors.append(
                "eval metadata concurrency "
                f"{metadata_conc!r} does not match workflow request "
                f"{expected_concs[0]}"
            )
        if len(result_files) != 1:
            errors.append(
                "non-batched eval must produce exactly one result file; "
                f"found {len(result_files)}"
            )
        suffixed_results = [
            result_file
            for result_file in result_files
            if CONC_SUFFIX_RE.search(Path(result_file).name)
        ]
        if suffixed_results:
            errors.append(
                "non-batched eval produced concurrency-suffixed result files"
            )
        return errors

    if not metadata_is_batched:
        errors.append(
            "workflow requested multiple eval concurrencies but batched metadata is missing"
        )
        expected_set = set(expected_concs or [])
    else:
        completed = meta.get("completed_eval_concs")
        failed = meta.get("failed_eval_concs")
        if not (
            _is_valid_conc_list(metadata_expected, allow_empty=False)
            and _is_valid_conc_list(completed)
            and _is_valid_conc_list(failed)
        ):
            errors.append(
                "batched eval metadata must contain unique, positive-integer "
                "concurrency lists"
            )
            expected_set = set(expected_concs or [])
        else:
            metadata_expected_set = set(metadata_expected)
            expected_set = set(expected_concs or metadata_expected)
            completed_set = set(completed)
            failed_set = set(failed)
            overlap = sorted(completed_set & failed_set)
            if overlap:
                errors.append(
                    "batched eval metadata marks concurrency as both completed "
                    "and failed: "
                    + ", ".join(str(value) for value in overlap)
                )

            if (
                expected_concs is not None
                and metadata_expected_set != expected_set
            ):
                missing = sorted(expected_set - metadata_expected_set)
                unexpected = sorted(metadata_expected_set - expected_set)
                if missing:
                    errors.append(
                        "batched eval metadata is missing workflow concurrency: "
                        + ", ".join(str(value) for value in missing)
                    )
                if unexpected:
                    errors.append(
                        "batched eval metadata has unexpected concurrency: "
                        + ", ".join(str(value) for value in unexpected)
                    )
            if failed_set:
                errors.append(
                    "batched eval failed for concurrency: "
                    + ", ".join(str(value) for value in sorted(failed_set))
                )
            if completed_set != expected_set:
                missing = sorted(expected_set - completed_set)
                unexpected = sorted(completed_set - expected_set)
                if missing:
                    errors.append(
                        "batched eval is missing completed concurrency: "
                        + ", ".join(str(value) for value in missing)
                    )
                if unexpected:
                    errors.append(
                        "batched eval completed unexpected concurrency: "
                        + ", ".join(str(value) for value in unexpected)
                    )

    actual_conc_counts = Counter()
    for result_file in result_files:
        match = CONC_SUFFIX_RE.search(Path(result_file).name)
        if match is None:
            errors.append(
                f"batched eval result lacks a concurrency suffix: {result_file}"
            )
            continue
        actual_conc_counts[int(match.group(1))] += 1

    duplicate_results = sorted(
        conc for conc, count in actual_conc_counts.items() if count > 1
    )
    if duplicate_results:
        errors.append(
            "batched eval has duplicate result files for concurrency: "
            + ", ".join(str(value) for value in duplicate_results)
        )

    actual_concs = set(actual_conc_counts)
    missing_results = sorted(expected_set - actual_concs)
    unexpected_results = sorted(actual_concs - expected_set)
    if missing_results:
        errors.append(
            "batched eval is missing result files for concurrency: "
            + ", ".join(str(value) for value in missing_results)
        )
    if unexpected_results:
        errors.append(
            "batched eval has unexpected result files for concurrency: "
            + ", ".join(str(value) for value in unexpected_results)
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate eval scores")
    parser.add_argument(
        "--min-score", type=float, default=0.85,
        help="Fallback minimum score when no threshold config matches (default: 0.85)",
    )
    parser.add_argument(
        "--thresholds", default=None,
        help="Path to thresholds JSON config (default: utils/evals/thresholds.json)",
    )
    parser.add_argument(
        "--meta-env", default="meta_env.json",
        help="Path to meta_env.json used to detect the model prefix (default: meta_env.json)",
    )
    parser.add_argument(
        "--model-prefix", default=None,
        help="Override the detected model prefix (default: read from meta_env.json / $MODEL_PREFIX)",
    )
    parser.add_argument(
        "--metric-prefix", default="exact_match,",
        help="Only check metrics whose name starts with this prefix (default: 'exact_match,')",
    )
    parser.add_argument(
        "--results-glob", default="results*.json",
        help="Glob pattern for result files (default: 'results*.json')",
    )
    parser.add_argument(
        "--expected-concs",
        default=None,
        help=(
            "Space-separated concurrencies requested by the workflow. When set, "
            "metadata and result coverage must match exactly."
        ),
    )
    args = parser.parse_args()

    try:
        expected_concs = parse_expected_concs(args.expected_concs)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not math.isfinite(args.min_score) or not 0 <= args.min_score <= 1:
        print(
            "FAIL: --min-score must be a finite number between 0 and 1",
            file=sys.stderr,
        )
        return 1
    if not args.metric_prefix:
        print("FAIL: --metric-prefix must not be empty", file=sys.stderr)
        return 1

    # Load thresholds config
    thresholds_path = args.thresholds or str(Path(__file__).parent / "thresholds.json")
    try:
        config = load_config(thresholds_path)
        print(f"Loaded thresholds from {thresholds_path}")
    except (json.JSONDecodeError, OSError, ValueError) as e:
        print(
            f"FAIL: could not load thresholds from {thresholds_path}: {e}",
            file=sys.stderr,
        )
        return 1

    # Identify the model so per-model thresholds can apply
    prefix = detect_model_prefix(args.meta_env, args.model_prefix)
    if prefix and prefix in config.get("models", {}):
        print(f"Model prefix: {prefix} (per-model thresholds apply)")
    elif prefix:
        print(f"Model prefix: {prefix} (no per-model override; using global default)")
    else:
        print("Model prefix: <unknown> (using global default thresholds)")

    failed = False
    checked = 0
    result_files = sorted(glob.glob(args.results_glob))
    expected_metric_set: set[tuple[str, str]] | None = None
    expected_metric_source: str | None = None

    manifest_errors = validate_batch_manifest(
        args.meta_env,
        result_files,
        expected_concs=expected_concs,
    )
    for error in manifest_errors:
        print(f"FAIL: {error}", file=sys.stderr)
        failed = True
    if not manifest_errors:
        if expected_concs is not None:
            print(
                "PASS: eval produced every requested concurrency: "
                + ", ".join(str(value) for value in expected_concs)
            )
        else:
            try:
                with open(args.meta_env) as f:
                    if "eval_concs" in json.load(f):
                        print("PASS: batched eval produced every requested concurrency")
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    "WARN: could not inspect eval metadata for batched concurrency "
                    f"status: {exc}",
                    file=sys.stderr,
                )

    for f in result_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"FAIL: could not read eval result {f}: {exc}", file=sys.stderr)
            failed = True
            continue

        file_checked = 0
        file_metric_set: set[tuple[str, str]] = set()
        results = data.get("results", {}) if isinstance(data, dict) else {}
        if not isinstance(results, dict):
            results = {}
        for task, metrics in results.items():
            if not isinstance(metrics, dict):
                print(
                    f"FAIL: {Path(f).name}: {task} result is not a JSON object",
                    file=sys.stderr,
                )
                failed = True
                continue
            min_score, source = resolve_threshold(config, prefix, task, args.min_score)
            task_has_metric = False
            task_checked = 0
            for name, val in metrics.items():
                if not name.startswith(args.metric_prefix) or "stderr" in name:
                    continue
                task_has_metric = True
                file_metric_set.add((task, name))
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    print(
                        f"FAIL: {Path(f).name}: {task} {name} has non-numeric "
                        f"value {val!r}",
                        file=sys.stderr,
                    )
                    failed = True
                    continue
                checked += 1
                file_checked += 1
                task_checked += 1
                if not math.isfinite(val):
                    print(
                        f"FAIL: {Path(f).name}: {task} {name} is not finite",
                        file=sys.stderr,
                    )
                    failed = True
                elif not 0 <= val <= 1:
                    print(
                        f"FAIL: {Path(f).name}: {task} {name} = {val:.4f} "
                        "is outside [0, 1]",
                        file=sys.stderr,
                    )
                    failed = True
                elif val < min_score:
                    print(
                        f"FAIL: {Path(f).name}: {task} {name} = {val:.4f} "
                        f"(< {min_score} from {source})",
                        file=sys.stderr,
                    )
                    failed = True
                else:
                    print(
                        f"PASS: {Path(f).name}: {task} {name} = {val:.4f} "
                        f"(>= {min_score} from {source})"
                    )
            if not task_has_metric:
                print(
                    f"FAIL: {Path(f).name}: {task} has no metric matching "
                    f"prefix {args.metric_prefix!r}",
                    file=sys.stderr,
                )
                failed = True
            elif task_checked == 0:
                failed = True
        if file_checked == 0:
            print(
                f"FAIL: {Path(f).name} has no numeric metrics matching "
                f"prefix {args.metric_prefix!r}",
                file=sys.stderr,
            )
            failed = True
        elif expected_metric_set is None:
            expected_metric_set = file_metric_set
            expected_metric_source = Path(f).name
        elif file_metric_set != expected_metric_set:
            missing_metrics = sorted(expected_metric_set - file_metric_set)
            unexpected_metrics = sorted(file_metric_set - expected_metric_set)
            if missing_metrics:
                print(
                    f"FAIL: {Path(f).name} is missing metrics present in "
                    f"{expected_metric_source}: "
                    + ", ".join(f"{task}/{metric}" for task, metric in missing_metrics),
                    file=sys.stderr,
                )
            if unexpected_metrics:
                print(
                    f"FAIL: {Path(f).name} has unexpected metrics compared with "
                    f"{expected_metric_source}: "
                    + ", ".join(
                        f"{task}/{metric}" for task, metric in unexpected_metrics
                    ),
                    file=sys.stderr,
                )
            failed = True

    if checked == 0:
        print(
            "FAIL: no metrics matched prefix '{}'".format(args.metric_prefix),
            file=sys.stderr,
        )

    return 1 if (failed or checked == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
