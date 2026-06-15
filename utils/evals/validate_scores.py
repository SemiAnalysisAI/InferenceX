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
    python3 utils/evals/validate_scores.py --thresholds my_thresholds.json
    python3 utils/evals/validate_scores.py --model-prefix dsv4
    python3 utils/evals/validate_scores.py --min-score 0.90  # flat fallback
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path


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
    args = parser.parse_args()

    # Load thresholds config
    config = {"default": {}, "models": {}}
    thresholds_path = args.thresholds
    if thresholds_path is None:
        default_path = Path(__file__).parent / "thresholds.json"
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

    for f in sorted(glob.glob(args.results_glob)):
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
                        f"FAIL: {task} {name} = {val:.4f} (< {min_score} from {source})",
                        file=sys.stderr,
                    )
                    failed = True
                else:
                    print(f"PASS: {task} {name} = {val:.4f} (>= {min_score} from {source})")

    if checked == 0:
        print("WARN: no metrics matched prefix '{}'".format(args.metric_prefix), file=sys.stderr)

    return 1 if (failed or checked == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
