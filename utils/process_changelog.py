import argparse
import json
import re
import subprocess
from collections import defaultdict

import yaml
from constants import GENERATE_SWEEPS_PY_SCRIPT, MASTER_CONFIGS
from matrix_logic.generate_sweep_configs import seq_len_to_str
from matrix_logic.validation import (
    ChangelogEntry,
    ChangelogMatrixEntry,
    load_config_files,
)

SCENARIO_TYPES = ("fixed-seq-len", "agentic-coding")


def get_added_lines(base_ref: str, head_ref: str, filepath: str) -> str:
    result = subprocess.run(
        ["git", "diff", base_ref, head_ref, "--", filepath],
        capture_output=True,
        text=True,
    )

    added_lines = []
    for line in result.stdout.split("\n"):
        if line.startswith("-") and not line.startswith("---"):
            deleted_content = line[1:]
            # Allow whitespace-only or empty line deletions
            if deleted_content.strip():
                # Don't allow deletions in the changelog
                # By convention, it should act as a running log of performance changes,
                # so we only want to see additions
                raise ValueError(
                    f"Deletions are not allowed in {filepath}. "
                    f"Only additions to the changelog are permitted. "
                    f"Found deleted line: {deleted_content}"
                )
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

    return "\n".join(added_lines)


def trim_conc(entries: list[dict]) -> list[dict]:
    """Trim each parallelism config's concurrency sweep to its lowest point.

    Non-full-sweep PRs only need a single concurrency point per parallelism
    config to validate a change runs end-to-end, so the shared cluster stays
    clear. Push-to-main and ``full-sweep-enabled`` PRs skip this reduction.

    The retained value is the minimum configured concurrency — independent of
    the source ordering of ``conc-list`` / ``conc-start``.

    Input comes from ``json.loads(subprocess.stdout)`` so ``conc`` is always
    ``int`` (single-node) or ``list`` (multi-node); other single-node fields
    are hashable scalars.

    - Single-node entries: group by every other field and keep only the entry
      with the lowest ``conc`` per group.
    - Multi-node entries: trim the ``conc`` list in place to ``[min(conc)]``.
    """
    groups: dict[tuple, list[int]] = {}
    out: list[dict] = []

    for entry in entries:
        if entry.get("prefill") is not None:
            conc = entry.get("conc")
            if isinstance(conc, list) and len(conc) > 1:
                entry = {**entry, "conc": [min(conc)]}
            out.append(entry)
            continue

        key = tuple(sorted((k, v) for k, v in entry.items() if k != "conc"))
        groups.setdefault(key, []).append(len(out))
        out.append(entry)

    drop: set[int] = set()
    for idxs in groups.values():
        if len(idxs) > 1:
            keep = min(idxs, key=lambda i: out[i]["conc"])
            drop.update(i for i in idxs if i != keep)
    return [e for i, e in enumerate(out) if i not in drop]


def get_config_keys_from_master(
    config_keys: list[str], master_config: dict
) -> list[str]:
    resolved_keys = {}
    for key in config_keys:
        if "*" in key:
            pattern = re.compile(re.escape(key).replace(r"\*", ".*"))
            matched_keys = [k for k in master_config if pattern.fullmatch(k)]
            if not matched_keys:
                raise ValueError(
                    f"No config keys matched the wildcard pattern '{key}' in master configs."
                )
            for matched_key in matched_keys:
                resolved_keys.setdefault(matched_key, None)
        elif key not in master_config:
            raise ValueError(f"Config key '{key}' not found in master configs.")
        else:
            resolved_keys.setdefault(key, None)
    return list(resolved_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", type=str, required=True)
    parser.add_argument("--head-ref", type=str, required=True)
    parser.add_argument("--changelog-file", type=str, required=True)
    parser.add_argument("--trim-conc", action="store_true")
    parser.add_argument(
        "--all-evals",
        action="store_true",
        help="Expand every changelog entry's eval selection without changing throughput.",
    )
    parser.add_argument(
        "--evals-only",
        action="store_true",
        help="Suppress throughput for every changelog entry without expanding eval selection.",
    )
    args = parser.parse_args()

    added_yaml = get_added_lines(args.base_ref, args.head_ref, args.changelog_file)

    if not added_yaml.strip():
        raise ValueError("No additions found in the changelog file.")

    changelog_data = yaml.safe_load(added_yaml)

    if not changelog_data:
        raise ValueError("No valid YAML entries found in the changelog additions.")

    final_results = {
        "single_node": defaultdict(list),
        "multi_node": defaultdict(list),
        "evals": [],
        "multinode_evals": [],
        "changelog_metadata": {
            "base_ref": args.base_ref,
            "head_ref": args.head_ref,
            "entries": changelog_data,
        },
    }

    all_benchmark_results = []
    all_eval_results = []
    # Track benchmark coverage per scenario so overlapping changelog entries
    # with disjoint scenario filters do not suppress each other.
    benchmark_scenarios_seen = defaultdict(set)
    eval_configs_seen = set()

    master_config = load_config_files(MASTER_CONFIGS)
    resolved_entries = []
    for entry_data in changelog_data:
        entry = ChangelogEntry.model_validate(entry_data)
        all_configs = get_config_keys_from_master(
            entry.config_keys, master_config
        )
        resolved_entries.append((entry, all_configs))

    # Process all-evals entries first so their broader eval matrix wins when
    # the same config appears in multiple changelog entries.
    resolved_entries.sort(key=lambda item: not item[0].all_evals)

    for entry, all_configs in resolved_entries:
        entry_scenarios = tuple(entry.scenario_type or SCENARIO_TYPES)
        expand_all_evals = args.all_evals or entry.all_evals
        suppress_throughput = (
            args.evals_only
            or entry.evals_only
            or entry.all_evals
        )

        if not suppress_throughput:
            # Generate benchmark entries (no evals)
            benchmark_groups = defaultdict(list)
            for config in all_configs:
                unseen_scenarios = tuple(
                    scenario for scenario in SCENARIO_TYPES
                    if (
                        scenario in entry_scenarios
                        and scenario not in benchmark_scenarios_seen[config]
                    )
                )
                if unseen_scenarios:
                    benchmark_scenarios_seen[config].update(unseen_scenarios)
                    benchmark_groups[unseen_scenarios].append(config)

            for scenarios, benchmark_configs in benchmark_groups.items():
                base_cmd = [
                    "python3",
                    GENERATE_SWEEPS_PY_SCRIPT,
                    "test-config",
                    "--config-keys",
                    *benchmark_configs,
                    "--config-files",
                    *MASTER_CONFIGS,
                    "--no-evals",
                ]
                if scenarios != SCENARIO_TYPES:
                    base_cmd.extend(["--scenario-type", *scenarios])
                try:
                    result = subprocess.run(
                        base_cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(e.stderr)
                    raise
                all_benchmark_results.extend(json.loads(result.stdout))

        # Evals only apply to fixed-sequence scenarios. Do not mark a config as
        # seen when an agentic-only entry generates no eval matrix.
        if "fixed-seq-len" not in entry_scenarios:
            continue

        eval_configs = [c for c in all_configs if c not in eval_configs_seen]
        if eval_configs:
            eval_configs_seen.update(eval_configs)
            eval_flags = ["--evals-only"]
            if expand_all_evals:
                eval_flags.append("--all-evals")
            base_cmd = [
                "python3",
                GENERATE_SWEEPS_PY_SCRIPT,
                "test-config",
                "--config-keys",
                *eval_configs,
                "--config-files",
                *MASTER_CONFIGS,
                *eval_flags,
                "--scenario-type",
                "fixed-seq-len",
            ]
            try:
                eval_result = subprocess.run(
                    base_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(e.stderr)
                raise
            all_eval_results.extend(json.loads(eval_result.stdout))

    if args.trim_conc:
        all_benchmark_results = trim_conc(all_benchmark_results)

    for result in all_benchmark_results:
        if result.get("scenario-type") == "agentic-coding":
            if result.get("prefill") is not None:
                final_results["multi_node"]["agentic"].append(result)
            else:
                final_results["single_node"]["agentic"].append(result)
        elif "prefill" in result and result["prefill"] is not None:
            seq_len_str = seq_len_to_str(result["isl"], result["osl"])
            final_results["multi_node"][seq_len_str].append(result)
        else:
            seq_len_str = seq_len_to_str(result["isl"], result["osl"])
            final_results["single_node"][seq_len_str].append(result)

    final_results["evals"] = [e for e in all_eval_results if e.get("prefill") is None]
    final_results["multinode_evals"] = [e for e in all_eval_results if e.get("prefill") is not None]

    # Validate final results structure
    validated = ChangelogMatrixEntry.model_validate(final_results)
    print(validated.model_dump_json(by_alias=True, exclude_none=True))


if __name__ == "__main__":
    main()
