"""Preserve recipe formatting while injecting environment into schema-2 roles."""

import re


def rewrite_role_environments(content: str, values: tuple[tuple[str, str], ...]) -> tuple[str, int]:
    """Set worker variables without touching frontend, service or client env."""
    # Restrict schema-2 injection to worker roles, excluding frontend,
    # benchmark and service env blocks. Keep comments and YAML anchors.
    roles = re.search(r"(?ms)^roles:\n.*?(?=^\S|\Z)", content)
    if roles is None:
        return content, 0
    count = 0

    def inject_role(match: re.Match[str]) -> str:
        nonlocal count
        block = match.group(0)
        env = re.search(r"(?m)^    env:([^\n]*)$", block)
        variables = "".join(f'\n      {key}: "{value}"' for key, value in values)
        if env:
            suffix = env.group(1).strip()
            if suffix.startswith("*"):
                replacement = "    env:\n      <<: " + suffix + variables
            elif not suffix or suffix.startswith(("&", "#")):
                replacement = env.group(0) + variables
            else:
                raise ValueError("synthetic acceptance requires a block-style role env")
            block = block[:env.start()] + replacement + block[env.end():]
        else:
            header_end = block.index("\n")
            block = block[:header_end] + "\n    env:" + variables + block[header_end:]
        count += 1
        return block

    rewritten_roles = re.sub(
        r"(?ms)^  (?:agg|prefill|decode):\n.*?(?=^  \S|\Z)",
        inject_role,
        roles.group(0),
    )
    rewritten = content[:roles.start()] + rewritten_roles + content[roles.end():]
    return rewritten, count
