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
    """Trim each parallelism config's concurrency sweep to its highest point.

    Non-full-sweep PRs only need a single concurrency point per parallelism
    config to validate a change runs end-to-end, so the shared cluster stays
    clear. Push-to-main and ``full-sweep-enabled`` PRs skip this reduction.

    The retained value is the maximum configured concurrency — independent of
    the source ordering of ``conc-list`` / ``conc-start``.

    Input comes from ``json.loads(subprocess.stdout)`` so ``conc`` is always
    ``int`` (single-node) or ``list`` (multi-node); other single-node fields
    are hashable scalars.

    - Single-node entries: group by every other field and keep only the entry
      with the highest ``conc`` per group.
    - Multi-node entries: trim the ``conc`` list in place to ``[max(conc)]``.
    """
    groups: dict[tuple, list[int]] = {}
    out: list[dict] = []

    for entry in entries:
        if entry.get("prefill") is not None:
            conc = entry.get("conc")
            if isinstance(conc, list) and len(conc) > 1:
                entry = {**entry, "conc": [max(conc)]}
            out.append(entry)
            continue

        key = tuple(sorted((k, v) for k, v in entry.items() if k != "conc"))
        groups.setdefault(key, []).append(len(out))
        out.append(entry)

    drop: set[int] = set()
    for idxs in groups.values():
        if len(idxs) > 1:
            keep = max(idxs, key=lambda i: out[i]["conc"])
            drop.update(i for i in idxs if i != keep)
    return [e for i, e in enumerate(out) if i not in drop]


def get_config_keys_from_master(
    config_keys: list[str], master_config: dict
) -> list[str]:
    resolved_keys = set()
    for key in config_keys:
        if "*" in key:
            pattern = re.compile(re.escape(key).replace(r"\*", ".*"))
            matched_keys = [k for k in master_config if pattern.fullmatch(k)]
            if not matched_keys:
                raise ValueError(
                    f"No config keys matched the wildcard pattern '{key}' in master configs."
                )
            resolved_keys.update(matched_keys)
        elif key not in master_config:
            raise ValueError(f"Config key '{key}' not found in master configs.")
        else:
            resolved_keys.add(key)
    return list(resolved_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", type=str, required=True)
    parser.add_argument("--head-ref", type=str, required=True)
    parser.add_argument("--changelog-file", type=str, required=True)
    parser.add_argument("--trim-conc", action="store_true")
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
    # Deduplicate repeated configs separately for benchmarks and evals.
    # An evals-only entry should not prevent a later regular entry from
    # generating benchmarks for the same config, and vice versa.
    benchmark_configs_seen = set()
    eval_configs_seen = set()

    for entry_data in changelog_data:
        entry = ChangelogEntry.model_validate(entry_data)
        all_configs = get_config_keys_from_master(
            entry.config_keys, load_config_files(MASTER_CONFIGS)
        )

        if not entry.evals_only:
            # Generate benchmark entries (no evals)
            benchmark_configs = [c for c in all_configs if c not in benchmark_configs_seen]
            if benchmark_configs:
                benchmark_configs_seen.update(benchmark_configs)
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

        # Generate eval entries separately
        eval_configs = [c for c in all_configs if c not in eval_configs_seen]
        if eval_configs:
            eval_configs_seen.update(eval_configs)
            base_cmd = [
                "python3",
                GENERATE_SWEEPS_PY_SCRIPT,
                "test-config",
                "--config-keys",
                *eval_configs,
                "--config-files",
                *MASTER_CONFIGS,
                "--evals-only",
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
        seq_len_str = seq_len_to_str(result["isl"], result["osl"])
        if "prefill" in result and result["prefill"] is not None:
            final_results["multi_node"][seq_len_str].append(result)
        else:
            final_results["single_node"][seq_len_str].append(result)

    final_results["evals"] = [e for e in all_eval_results if e.get("prefill") is None]
    final_results["multinode_evals"] = [e for e in all_eval_results if e.get("prefill") is not None]

    # Validate final results structure
    validated = ChangelogMatrixEntry.model_validate(final_results)
    print(validated.model_dump_json(by_alias=True))


if __name__ == "__main__":
    main()
