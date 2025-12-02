#!/usr/bin/env python3
import yaml
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_username():
    try:
        username = subprocess.check_output(
            ["git", "config", "github.user"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
        if username:
            return f"@{username}"
    except subprocess.CalledProcessError:
        pass

    try:
        return subprocess.check_output(
            ["git", "config", "user.name"],
            text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main():
    changelog_path = Path("perf-changelog.yaml")

    if not changelog_path.exists():
        return

    with open(changelog_path) as f:
        changelog = yaml.safe_load(f)

    if not changelog or 'changelog' not in changelog:
        return

    modified = False
    today = datetime.now().strftime('%Y-%m-%d %H:%M')
    author = get_git_username()

    for entry in changelog['changelog']:
        if not entry.get('date'):
            entry['date'] = today
            modified = True

        if not entry.get('author'):
            entry['author'] = author
            modified = True

    if modified:
        with open(changelog_path, 'w') as f:
            yaml.dump(changelog, f, sort_keys=False, default_flow_style=False)


if __name__ == "__main__":
    main()
