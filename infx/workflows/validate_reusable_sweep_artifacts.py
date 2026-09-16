"""Validate reused sweep artifacts for internal consistency."""

import argparse
import sys
from pathlib import Path

from infx.results.artifacts import (
    actual_agentic_keys as actual_agentic_keys,
    actual_benchmark_key_rows as actual_benchmark_key_rows,
    actual_benchmark_keys as actual_benchmark_keys,
    agentic_key as agentic_key,
    agentic_keys_from_paths as agentic_keys_from_paths,
    agentic_point_files as agentic_point_files,
    as_bool as as_bool,
    as_int as as_int,
    benchmark_key as benchmark_key,
    duplicate_identity_errors as duplicate_identity_errors,
    freeze_identity_value as freeze_identity_value,
    json_rows as json_rows,
    load_json as load_json,
    validate_agentic_artifacts as validate_agentic_artifacts,
    validate_fixed_artifacts as validate_fixed_artifacts,
    validate_identity_set as validate_identity_set,
    validate_run_stats as validate_run_stats,
)
from infx.results.eval_artifacts import (
    dedupe_reran_evals as dedupe_reran_evals,
    eval_key as eval_key,
    eval_result_key as eval_result_key,
    inspect_eval_artifacts,
    invalid_eval_suite as invalid_eval_suite,
    normalized_runner as normalized_runner,
    raw_eval_artifact_dirs as raw_eval_artifact_dirs,
    raw_eval_key_rows as raw_eval_key_rows,
    validate_eval_artifacts as validate_eval_artifacts,
)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.artifacts_dir.is_dir():
        raise ValueError(f"artifacts directory does not exist: {args.artifacts_dir}")

    # Collapse reran (flaky) eval duplicates to the latest result before
    # validating, so a legitimate rerun does not fail the consistency checks.
    dedupe_messages = dedupe_reran_evals(args.artifacts_dir)
    if dedupe_messages:
        print("Collapsed reran eval duplicates (kept latest result per identity):")
        for message in dedupe_messages:
            print(f"  {message}")

    fixed_rows = actual_benchmark_key_rows(args.artifacts_dir)
    agentic_rows = agentic_keys_from_paths(agentic_point_files(args.artifacts_dir))
    eval_rows, eval_errors = inspect_eval_artifacts(args.artifacts_dir)

    errors = validate_fixed_artifacts(args.artifacts_dir)
    errors.extend(validate_agentic_artifacts(args.artifacts_dir))
    errors.extend(eval_errors)
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
