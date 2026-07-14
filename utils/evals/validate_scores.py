#!/usr/bin/env python3
"""Validate eval scores against per-task and per-model thresholds."""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

CONC_SUFFIX_RE = re.compile(r"_conc(\d+)(?:_\d+)?\.json$")


def load_config(path: str) -> dict:
    """Load YAML or JSON thresholds, including legacy flat configs."""
    with open(path) as f:
        text = f.read()
    try:
        import yaml
    except ModuleNotFoundError:
        # Runner hosts may lack PyYAML.
        try:
            cfg = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                f"PyYAML is not installed and {path} is not JSON; "
                "install it with 'pip install pyyaml'"
            ) from None
    else:
        try:
            cfg = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError("thresholds config must be a mapping")
    if "default" not in cfg and "models" not in cfg:
        # Legacy flat format: the whole object is the per-task default.
        return {"default": cfg, "models": {}}
    return {"default": cfg.get("default", {}), "models": cfg.get("models", {})}


def detect_model_prefix(meta_env_path: str, override: str | None) -> str | None:
    """Resolve the model prefix: explicit override > meta_env.json > env var."""
    if override:
        return override
    try:
        with open(meta_env_path) as f:
            prefix = json.load(f).get("infmax_model_prefix")
        if prefix and prefix != "unknown":
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


def validate_batch_manifest(
    meta_env_path: str,
    result_files: list[str],
    expected_concs: list[int] | None = None,
) -> list[str]:
    """Validate that a batched eval produced every requested concurrency."""
    try:
        with open(meta_env_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        if expected_concs is not None or any(
            CONC_SUFFIX_RE.search(Path(result_file).name)
            for result_file in result_files
        ):
            return [
                "batched eval result files exist but "
                f"{meta_env_path} is unavailable or invalid: {exc}"
            ]
        return []

    if expected_concs is not None and "eval_concs" not in meta:
        if len(expected_concs) > 1:
            return ["workflow requested multiple concurrencies but batched eval metadata is missing"]
        errors = []
        if meta.get("conc") != expected_concs[0]:
            errors.append("eval metadata concurrency does not match workflow request")
        if len(result_files) != 1:
            errors.append("eval must produce exactly one result file")
        return errors
    if "eval_concs" not in meta:
        return []

    metadata_expected = meta.get("eval_concs")
    completed = meta.get("completed_eval_concs")
    failed = meta.get("failed_eval_concs")
    if not all(
        isinstance(values, list)
        for values in (metadata_expected, completed, failed)
    ):
        return ["batched eval metadata must contain list-valued concurrency fields"]
    if not all(
        isinstance(value, int) and value > 0
        for values in (metadata_expected, completed, failed)
        for value in values
    ):
        return ["batched eval metadata contains an invalid concurrency"]

    errors = []
    metadata_expected_set = set(metadata_expected)
    expected_set = set(expected_concs or metadata_expected)
    completed_set = set(completed)
    failed_set = set(failed)
    if len(metadata_expected_set) != len(metadata_expected):
        errors.append("batched eval metadata contains duplicate expected concurrencies")
    if len(completed_set) != len(completed):
        errors.append("batched eval metadata contains duplicate completed concurrencies")
    if expected_concs is not None and metadata_expected_set != expected_set:
        errors.append("batched eval metadata does not match workflow concurrencies")
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

    actual_concs = set()
    for result_file in result_files:
        match = CONC_SUFFIX_RE.search(Path(result_file).name)
        if match is None:
            errors.append(
                f"batched eval result lacks a concurrency suffix: {result_file}"
            )
            continue
        actual_concs.add(int(match.group(1)))

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
    # Keep merged CI logs ordered.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Validate eval scores")
    parser.add_argument(
        "--min-score", type=float, default=0.85,
        help="Fallback minimum score when no threshold config matches (default: 0.85)",
    )
    parser.add_argument(
        "--thresholds", default=None,
        help="Path to thresholds config, YAML or JSON (default: utils/evals/thresholds.yaml)",
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
        help="Space-separated concurrencies requested by the workflow",
    )
    args = parser.parse_args()

    expected_concs = None
    if args.expected_concs is not None:
        try:
            expected_concs = [int(value) for value in args.expected_concs.split()]
        except ValueError:
            expected_concs = []
        if (
            not expected_concs
            or any(value <= 0 for value in expected_concs)
            or len(set(expected_concs)) != len(expected_concs)
        ):
            print("FAIL: expected concurrencies must be unique positive integers", file=sys.stderr)
            return 1

    # Load thresholds config
    config = {"default": {}, "models": {}}
    thresholds_path = args.thresholds
    if thresholds_path is None:
        default_path = Path(__file__).parent / "thresholds.yaml"
        if default_path.exists():
            thresholds_path = str(default_path)
    if thresholds_path:
        try:
            config = load_config(thresholds_path)
            print(f"Loaded thresholds from {thresholds_path}")
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"WARN: could not load thresholds from {thresholds_path}: {e}", file=sys.stderr)

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

    manifest_errors = validate_batch_manifest(
        args.meta_env,
        result_files,
        expected_concs,
    )
    for error in manifest_errors:
        print(f"FAIL: {error}", file=sys.stderr)
        failed = True
    if not manifest_errors:
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
        match = CONC_SUFFIX_RE.search(Path(f).name)
        conc_label = f"[conc={match.group(1)}] " if match else ""
        with open(f) as fh:
            data = json.load(fh)
        for task, metrics in data.get("results", {}).items():
            min_score, source = resolve_threshold(config, prefix, task, args.min_score)
            for name, val in metrics.items():
                if not name.startswith(args.metric_prefix) or "stderr" in name:
                    continue
                if not isinstance(val, (int, float)):
                    continue
                checked += 1
                if val < min_score:
                    print(
                        f"FAIL: {conc_label}{task} {name} = {val:.4f} (< {min_score} from {source})",
                        file=sys.stderr,
                    )
                    failed = True
                else:
                    print(
                        f"PASS: {conc_label}{task} {name} = {val:.4f} (>= {min_score} from {source})"
                    )

    if checked == 0:
        print("WARN: no metrics matched prefix '{}'".format(args.metric_prefix), file=sys.stderr)

    return 1 if (failed or checked == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
