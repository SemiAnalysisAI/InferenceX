"""Decide whether a PR description's author checklist is complete.

The checklist is the task list under the ``## Checklist`` heading of the PR template.
Every item must be checked except conditional ones ("If ...", "Before merging ..."),
which may stay unchecked until they apply.
"""

from __future__ import annotations

import re
import sys

CHECKLIST_HEADING = re.compile(r"^#{1,6}\s+Checklist\s*$")
ANY_HEADING = re.compile(r"^#{1,6}\s")
SECTION_END = re.compile(r"^\s*</details>")
TASK_ITEM = re.compile(r"^- \[([ xX])\]\s+(.*)$")
CONDITIONAL = re.compile(r"^(If|Before merging)\b")


def checklist_complete(body: str) -> bool:
    """Return True when the Checklist section exists and every unconditional item is checked."""
    items: list[tuple[bool, str]] = []
    in_section = False
    for line in body.splitlines():
        if CHECKLIST_HEADING.match(line):
            in_section = True
            continue
        if in_section and (ANY_HEADING.match(line) or SECTION_END.match(line)):
            break
        if in_section and (match := TASK_ITEM.match(line)):
            items.append((match.group(1) != " ", match.group(2).lstrip("*").strip()))
    return bool(items) and all(checked or CONDITIONAL.match(text) for checked, text in items)


def main() -> int:
    """Read a PR description from stdin and print true or false."""
    print("true" if checklist_complete(sys.stdin.read()) else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
