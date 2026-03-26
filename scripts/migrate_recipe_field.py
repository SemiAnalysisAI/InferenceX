#!/usr/bin/env python3
"""
Migrate nvidia-master.yaml: extract CONFIG_FILE from additional-settings into a dedicated recipe field.

For each prefill/decode worker config that has a CONFIG_FILE= entry in additional-settings:
  1. Extract the path (stripping the 'recipes/' prefix since it's now implicit)
  2. Add it as a 'recipe' field on the worker config
  3. Remove the CONFIG_FILE= entry from additional-settings
  4. Remove the additional-settings key entirely if it becomes empty
  5. Remove any comment lines referencing the srt-slurm GitHub URL above the CONFIG_FILE entry

Usage:
    python scripts/migrate_recipe_field.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path


def migrate(content: str) -> str:
    """Transform YAML content: move CONFIG_FILE from additional-settings to recipe field."""
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect 'additional-settings:' lines
        match = re.match(r'^(\s+)additional-settings:\s*$', line)
        if not match:
            result.append(line)
            i += 1
            continue

        indent = match.group(1)
        item_indent = indent + '- '
        comment_indent = indent + '# '

        # Collect subsequent lines that are items or comments belonging to this block
        settings_lines = []
        config_file_value = None
        comment_lines_before_config = []
        j = i + 1

        while j < len(lines):
            l = lines[j]
            # Comment line at the right indent level (part of this block)
            if re.match(rf'^{re.escape(indent)}#', l) or re.match(rf'^{re.escape(indent)}\s+#', l):
                comment_lines_before_config.append((j, l))
                j += 1
                continue
            # List item
            item_match = re.match(rf'^{re.escape(indent)}- "(.*)"$', l) or re.match(rf"^{re.escape(indent)}- '(.*)'$", l)
            if item_match:
                value = item_match.group(1)
                if value.startswith('CONFIG_FILE='):
                    config_file_value = value.split('=', 1)[1]
                    # Mark comment lines before this CONFIG_FILE for removal
                    # (they're srt-slurm GitHub URL references)
                    srt_comments = [
                        (idx, cl) for idx, cl in comment_lines_before_config
                        if 'srt-slurm' in cl or 'CONFIG_FILE' in cl
                    ]
                    settings_lines.append((j, l, 'config_file', srt_comments))
                else:
                    settings_lines.append((j, l, 'keep', []))
                comment_lines_before_config = []
                j += 1
                continue
            # Anything else means end of this additional-settings block
            break

        if config_file_value is None:
            # No CONFIG_FILE found, keep everything as-is
            result.append(line)
            i += 1
            continue

        # Strip 'recipes/' prefix from the path (it's now implicit)
        recipe_path = config_file_value
        if recipe_path.startswith('recipes/'):
            recipe_path = recipe_path[len('recipes/'):]

        # Emit the recipe field at the same indent as additional-settings
        result.append(f'{indent}recipe: {recipe_path}')

        # Check if there are remaining (non-CONFIG_FILE) settings
        remaining = [(idx, l, kind, comments) for idx, l, kind, comments in settings_lines if kind == 'keep']

        if remaining:
            # Keep additional-settings with only the non-CONFIG_FILE items
            result.append(line)  # additional-settings:
            for idx, l, kind, comments in remaining:
                result.append(l)
        # else: drop additional-settings entirely (it's now empty)

        # Also skip any srt-slurm comment lines that were above CONFIG_FILE
        srt_comment_indices = set()
        for idx, l, kind, comments in settings_lines:
            if kind == 'config_file':
                for cidx, cl in comments:
                    srt_comment_indices.add(cidx)

        # Skip the lines we've already processed
        i = j
        continue

    return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser(description='Migrate CONFIG_FILE to recipe field')
    parser.add_argument('--dry-run', action='store_true', help='Print output without writing')
    parser.add_argument('--file', default='.github/configs/nvidia-master.yaml',
                        help='Path to master config file')
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: {filepath} not found", file=sys.stderr)
        sys.exit(1)

    content = filepath.read_text()
    migrated = migrate(content)

    if args.dry_run:
        print(migrated)
    else:
        filepath.write_text(migrated)
        print(f"Migrated {filepath}")


if __name__ == '__main__':
    main()
