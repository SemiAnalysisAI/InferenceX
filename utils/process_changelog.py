import yaml
import json
import argparse

from pydantic import BaseModel, ConfigDict, Field
from matrix_logic.validation import load_config_files, load_runner_file

MASTER_CONFIGS = [".github/configs/amd-master.yaml",
                  ".github/configs/nvidia-master.yaml"]


class ChangelogEntry(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    
    config_key: str = Field(alias='config-key')
    description: str


class Config(BaseModel):
    changelog: list[ChangelogEntry]


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

    changelog_path = args.changelog_file
    with open(changelog_path, 'r') as f:
        changelog_data = yaml.safe_load(f)
        
    changelog_data = Config.model_validate(changelog_data)
    
    # Validate all config-keys actually exist in the master config files
    for entry in changelog_data.changelog:
        if entry.config_key not in master_config_data.keys():
            raise ValueError(
                f"Changelog entry with config-key '{entry.config_key}' does not exist in master config files."
            )
         


if __name__ == "__main__":
    main()
