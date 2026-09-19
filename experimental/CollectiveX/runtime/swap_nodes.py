"""Preserve exclusions for existing Slurm nodes, ignoring retired node names."""

import subprocess
import sys


def existing_exclusions(expression: str) -> str:
    requested = subprocess.check_output(
        ["scontrol", "show", "hostnames", expression], text=True
    ).splitlines()
    current = set(
        subprocess.check_output(
            ["sinfo", "-N", "-h", "-o", "%N"], text=True
        ).splitlines()
    )
    if not current:
        raise ValueError("Slurm returned no nodes; refusing to discard exclusions")
    return ",".join(node for node in requested if node in current)


if __name__ == "__main__":
    print(existing_exclusions(sys.argv[1]))
