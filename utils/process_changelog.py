import yaml
import json
import re
import argparse
import subprocess

from pydantic import BaseModel, ConfigDict, Field
from pprint import pprint

from matrix_logic.validation import load_config_files

MASTER_CONFIGS = [".github/configs/amd-master.yaml",
                  ".github/configs/nvidia-master.yaml"]
RUNNER_CONFIG = ".github/configs/runners.yaml"


class ChangelogEntry(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    config_keys: list[str] = Field(alias='config-keys', min_length=1)
    description: str


def get_added_lines(base_ref, head_ref, filepath):
    result = subprocess.run(
        ["git", "diff", base_ref, head_ref, "--", filepath],
        capture_output=True,
        text=True
    )

    added_lines = []
    for line in result.stdout.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            # Don't allow deletions in the changelog
            # By convention, it should act as a running log of performance changes,
            # so we only want to see additions
            raise ValueError(
                f"Deletions are not allowed in {filepath}. "
                f"Only additions to the changelog are permitted. "
                f"Found deleted line: {line[1:]}"
            )
        elif line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])

    return '\n'.join(added_lines)


def get_config_keys_from_master(config_keys: list[str], master_config: dict) -> list[str]:
    resolved_keys = set()
    for key in config_keys:
        if "*" in key:
            pattern = re.compile(re.escape(key).replace(r"\*", ".*"))
            matched_keys = [k for k in master_config if pattern.fullmatch(k)]
            if not matched_keys:
                raise ValueError(
                    f"No config keys matched the wildcard pattern '{key}' in master configs.")
            resolved_keys.update(matched_keys)
        elif key not in master_config:
            raise ValueError(f"Config key '{key}' not found in master configs.")
        else:
            resolved_keys.add(key)
    pprint(list(resolved_keys))
    return list(resolved_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-ref', type=str, required=True)
    parser.add_argument('--head-ref', type=str, required=True)
    parser.add_argument('--changelog-file', type=str, required=True)
    args = parser.parse_args()

    added_yaml = get_added_lines(
        args.base_ref, args.head_ref, args.changelog_file)

    if not added_yaml.strip():
        print("No new changelog entries found")
        return

    changelog_data = yaml.safe_load(added_yaml)

    if not changelog_data:
        print("No new changelog entries found")
        return

    all_results = []
    for entry_data in changelog_data:
        entry = ChangelogEntry.model_validate(entry_data)
        configs_to_run = get_config_keys_from_master(
            entry.config_keys, load_config_files(MASTER_CONFIGS))

        try:
            result = subprocess.run([
                "python3", "utils/matrix_logic/generate_sweep_configs.py", "test-config",
                "--config-keys", *configs_to_run,
                "--config-files", *MASTER_CONFIGS,
                "--runner-config", RUNNER_CONFIG
            ],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(e.stderr)

        all_results.extend(json.loads(result.stdout))

    print(json.dumps(all_results))


if __name__ == "__main__":
    main()
