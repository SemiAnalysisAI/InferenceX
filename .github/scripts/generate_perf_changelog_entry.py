#!/usr/bin/env python3
"""Generate perf-changelog.yaml entries from PR config diffs.

When a PR modifies .github/configs/*-master.yaml files, this script analyzes
the diff to determine which config keys were affected and what changed, then
appends appropriate entries to perf-changelog.yaml.

Usage:
    python generate_perf_changelog_entry.py \
        --base-sha <merge-base-sha> \
        --head-sha <head-sha> \
        --pr-number <pr-number> \
        --pr-title <pr-title> \
        --repo <owner/repo>
"""

import argparse
import re
import subprocess
import sys

import yaml


MASTER_CONFIGS = [
    ".github/configs/nvidia-master.yaml",
    ".github/configs/amd-master.yaml",
]

CHANGELOG_FILE = "perf-changelog.yaml"


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def get_changed_config_files(base_sha: str, head_sha: str) -> list[str]:
    """Return list of master config files that were modified in the PR."""
    diff_output = run_git(["diff", "--name-only", base_sha, head_sha])
    changed_files = diff_output.strip().split("\n")
    return [f for f in changed_files if f in MASTER_CONFIGS]


def get_config_at_ref(filepath: str, ref: str) -> dict:
    """Load a YAML config file at a specific git ref."""
    try:
        content = run_git(["show", f"{ref}:{filepath}"])
        return yaml.safe_load(content) or {}
    except subprocess.CalledProcessError:
        return {}


def get_diff_for_file(base_sha: str, head_sha: str, filepath: str) -> str:
    """Get the unified diff for a specific file."""
    return run_git(["diff", base_sha, head_sha, "--", filepath])


def find_changed_config_keys(
    base_config: dict, head_config: dict
) -> dict[str, str]:
    """Compare two config dicts and return changed keys with change type.

    Returns a dict of {config_key: change_type} where change_type is
    'added', 'modified', or 'removed'.
    """
    changes = {}
    all_keys = set(base_config.keys()) | set(head_config.keys())

    for key in all_keys:
        if key not in base_config:
            changes[key] = "added"
        elif key not in head_config:
            changes[key] = "removed"
        elif base_config[key] != head_config[key]:
            changes[key] = "modified"

    return changes


def describe_config_changes(
    key: str,
    change_type: str,
    base_config: dict,
    head_config: dict,
) -> list[str]:
    """Generate human-readable descriptions of what changed for a config key."""
    descriptions = []

    if change_type == "added":
        cfg = head_config[key]
        framework = cfg.get("framework", "unknown")
        image = cfg.get("image", "unknown")
        descriptions.append(f"Add new {key} configuration ({framework})")
        descriptions.append(f"Image: {image}")
        if cfg.get("multinode"):
            descriptions.append("Multi-node configuration with disaggregation")
        if cfg.get("spec-decoding"):
            descriptions.append(
                f"Speculative decoding: {cfg.get('spec-decoding')}"
            )
        return descriptions

    if change_type == "removed":
        descriptions.append(f"Remove {key} configuration")
        return descriptions

    # change_type == "modified"
    base_cfg = base_config[key]
    head_cfg = head_config[key]

    # Check image change
    base_image = base_cfg.get("image", "")
    head_image = head_cfg.get("image", "")
    if base_image != head_image:
        # Extract version info if possible
        descriptions.append(f"Update image from {base_image} to {head_image}")

    # Check runner change
    if base_cfg.get("runner") != head_cfg.get("runner"):
        descriptions.append(
            f"Change runner from {base_cfg.get('runner')} "
            f"to {head_cfg.get('runner')}"
        )

    # Check multinode/disagg changes
    if base_cfg.get("multinode") != head_cfg.get("multinode"):
        if head_cfg.get("multinode"):
            descriptions.append("Enable multi-node support")
        else:
            descriptions.append("Disable multi-node support")

    if base_cfg.get("disagg") != head_cfg.get("disagg"):
        if head_cfg.get("disagg"):
            descriptions.append("Enable disaggregated prefill/decode")
        else:
            descriptions.append("Disable disaggregated prefill/decode")

    # Check search space changes
    base_seq = base_cfg.get("seq-len-configs", [])
    head_seq = head_cfg.get("seq-len-configs", [])
    if base_seq != head_seq:
        descriptions.append("Update search space and concurrency configurations")

    # If no specific changes detected, add generic description
    if not descriptions:
        descriptions.append(f"Update {key} configuration parameters")

    return descriptions


def changelog_already_has_pr(pr_number: int) -> bool:
    """Check if perf-changelog.yaml already has an entry for this PR."""
    try:
        with open(CHANGELOG_FILE) as f:
            content = f.read()
        return f"/pull/{pr_number}" in content
    except FileNotFoundError:
        return False


def generate_entry(
    config_keys: list[str],
    descriptions: list[str],
    pr_number: int,
    repo: str,
) -> dict:
    """Generate a single perf-changelog entry."""
    return {
        "config-keys": config_keys,
        "description": descriptions,
        "pr-link": f"https://github.com/{repo}/pull/{pr_number}",
    }


def append_entry_to_changelog(entry: dict) -> None:
    """Append a new entry to perf-changelog.yaml."""
    # Read existing content
    try:
        with open(CHANGELOG_FILE) as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""

    # Format the new entry as YAML manually for consistent formatting
    lines = ["\n- config-keys:"]
    for key in entry["config-keys"]:
        lines.append(f"    - {key}")
    lines.append("  description:")
    for desc in entry["description"]:
        lines.append(f'    - "{desc}"')
    lines.append(f"  pr-link: {entry['pr-link']}")
    lines.append("")

    new_entry_text = "\n".join(lines)

    with open(CHANGELOG_FILE, "w") as f:
        f.write(existing.rstrip("\n") + "\n" + new_entry_text)


def main():
    parser = argparse.ArgumentParser(
        description="Generate perf-changelog.yaml entries from PR config diffs"
    )
    parser.add_argument(
        "--base-sha",
        required=True,
        help="Base SHA (merge base) to diff against",
    )
    parser.add_argument(
        "--head-sha",
        required=True,
        help="Head SHA (merged commit) to diff against",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        required=True,
        help="PR number for the pr-link field",
    )
    parser.add_argument(
        "--pr-title",
        default="",
        help="PR title (used as fallback description)",
    )
    parser.add_argument(
        "--repo",
        default="InferenceMAX/InferenceMAX",
        help="Repository in owner/repo format",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the entry without writing to file",
    )
    args = parser.parse_args()

    # Check if changelog already has this PR
    if changelog_already_has_pr(args.pr_number):
        print(
            f"perf-changelog.yaml already contains an entry for PR #{args.pr_number}. "
            "Skipping auto-generation.",
            file=sys.stderr,
        )
        return

    # Find which master config files changed
    changed_configs = get_changed_config_files(args.base_sha, args.head_sha)
    if not changed_configs:
        print("No master config files changed. Nothing to do.", file=sys.stderr)
        return

    # Analyze changes per config file
    all_descriptions: dict[str, list[str]] = {}

    for config_file in changed_configs:
        base_config = get_config_at_ref(config_file, args.base_sha)
        head_config = get_config_at_ref(config_file, args.head_sha)

        changes = find_changed_config_keys(base_config, head_config)

        for key, change_type in changes.items():
            descs = describe_config_changes(
                key, change_type, base_config, head_config
            )
            all_descriptions[key] = descs

    if not all_descriptions:
        print("No config key changes detected. Nothing to do.", file=sys.stderr)
        return

    # Group config keys by similar descriptions to create compact entries
    desc_to_keys: dict[tuple, list[str]] = {}
    for key, descs in all_descriptions.items():
        desc_tuple = tuple(descs)
        desc_to_keys.setdefault(desc_tuple, []).append(key)

    entries = []
    for descs, keys in desc_to_keys.items():
        keys.sort()
        entry = generate_entry(
            config_keys=keys,
            descriptions=list(descs),
            pr_number=args.pr_number,
            repo=args.repo,
        )
        entries.append(entry)

    for entry in entries:
        if args.dry_run:
            print("--- Generated entry ---")
            print(yaml.dump([entry], default_flow_style=False))
        else:
            append_entry_to_changelog(entry)
            print(
                f"Appended entry for config keys: {entry['config-keys']}",
                file=sys.stderr,
            )

    if not args.dry_run:
        print(
            f"Successfully added {len(entries)} entries to {CHANGELOG_FILE}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
