import yaml
import json
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

    config_keys: list[str] = Field(alias='config-keys')
    description: str


def get_added_lines(base_ref, head_ref, filepath):
    result = subprocess.run(
        ["git", "diff", base_ref, head_ref, "--", filepath],
        capture_output=True,
        text=True
    )

    added_lines = []
    for line in result.stdout.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])

    return '\n'.join(added_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-ref', type=str, required=True)
    parser.add_argument('--head-ref', type=str, required=True)
    parser.add_argument('--changelog-file', type=str, required=True)
    args = parser.parse_args()

    master_config_data = load_config_files(MASTER_CONFIGS)

    added_yaml = get_added_lines(
        args.base_ref, args.head_ref, args.changelog_file)

    if not added_yaml.strip():
        print("No new changelog entries found")
        return

    changelog_data = yaml.safe_load(added_yaml)
    pprint(changelog_data)

    if not changelog_data:
        print("No new changelog entries found")
        return

    all_results = []
    for entry_data in changelog_data:
        entry = ChangelogEntry.model_validate(entry_data)

        try:
            result = subprocess.run([
                "python3", "utils/matrix_logic/generate_sweep_configs.py", "test-config",
                "--config-keys", *entry.config_keys,
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

        # for config_key in entry.config_keys:
        #     if config_key not in master_config_data:
        #         raise ValueError(
        #             f"Config key '{config_key}' does not exist in master config files."
        #         )

        # # print(f"Config keys: {entry.config_keys}")
        # # print(f"Seq lens: {entry.seq_lens}")
        # # print(f"Description: {entry.description}")


if __name__ == "__main__":
    main()
