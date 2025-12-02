import yaml
import json
import argparse

from pydantic import BaseModel, ConfigDict, Field

from matrix_logic.validation import load_config_files, load_runner_file

MASTER_CONFIGS = [".github/configs/amd-master.yaml",
                  ".github/configs/nvidia-master.yaml"]


class ChangelogEntry(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    
    config_keys: list[str] = Field(alias='config-keys')
    description: str
    seq_lens: list[str] = Field(alias='seq-lens')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--changelog-file',
        type=str,
        required=True,
        help='Path to the changelog YAML file'
    )

    args = parser.parse_args()

    master_config_data = load_config_files(MASTER_CONFIGS)

    with open(args.changelog_file, 'r') as f:
        changelog_data = yaml.safe_load(f)

    for entry_data in changelog_data:
        entry = ChangelogEntry.model_validate(entry_data)
        
        # Make sure the specfied config keys actually exist in the master config files
        for config_key in entry.config_keys:
            if config_key not in master_config_data.keys():
                raise ValueError(
                    f"Config key '{config_key}' does not exist in master config files."
                )


if __name__ == "__main__":
    main()