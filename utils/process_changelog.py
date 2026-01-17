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
            # Don't allow deletions in the changelog
            # By convention, it should act as a running log of performance changes,
            # so we only want to see additions
            raise ValueError(
                f"Deletions are not allowed in {filepath}. "
                f"Only additions to the changelog are permitted. "
                f"Found deleted line: {line[1:]}"
            )
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

    return "\n".join(added_lines)


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
        "changelog_metadata": {
            "base_ref": args.base_ref,
            "head_ref": args.head_ref,
            "entries": changelog_data,
        },
    }

    all_results = []
    # Deduplicate repeated configs, if for some reason a config key appears multiple times
    # in one commit, we don't want to run that config two times (there will just be twice as many
    # data points for that config, which is not useful)
    all_configs_to_run = set()

    for entry_data in changelog_data:
        entry = ChangelogEntry.model_validate(entry_data)
        configs_to_run = get_config_keys_from_master(
            entry.config_keys, load_config_files(MASTER_CONFIGS)
        )

        # Skip configs already processed
        configs_to_run = [c for c in configs_to_run if c not in all_configs_to_run]
        if not configs_to_run:
            continue
        all_configs_to_run.update(configs_to_run)

        try:
            result = subprocess.run(
                [
                    "python3",
                    GENERATE_SWEEPS_PY_SCRIPT,
                    "test-config",
                    "--config-keys",
                    *configs_to_run,
                    "--config-files",
                    *MASTER_CONFIGS,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            raise

        all_results.extend(json.loads(result.stdout))

    for result in all_results:
        seq_len_str = seq_len_to_str(result["isl"], result["osl"])
        if "prefill" in result and result["prefill"] is not None:
            final_results["multi_node"][seq_len_str].append(result)
        else:
            final_results["single_node"][seq_len_str].append(result)

    # Validate final results structure
    validated = ChangelogMatrixEntry.model_validate(final_results)
    print(validated.model_dump_json(by_alias=True))


if __name__ == "__main__":
    main()
