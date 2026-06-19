#!/usr/bin/env python3

from importlib.util import find_spec
from pathlib import Path


def main() -> None:
    """Keep idle DP-attention ranks available for scheduler collectives."""
    package_spec = find_spec("sglang")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("Unable to locate the installed SGLang package")

    package_path = Path(next(iter(package_spec.submodule_search_locations)))
    scheduler_path = package_path / "srt/managers/scheduler.py"
    source = scheduler_path.read_text()
    old = "        self.invariant_checker._check_tree_cache()\n"
    new = (
        "        if not self.server_args.enable_dp_attention:\n"
        "            self.invariant_checker._check_tree_cache()\n"
    )

    if old in source:
        scheduler_path.write_text(source.replace(old, new, 1))
    elif new not in source:
        raise RuntimeError(f"Unexpected SGLang scheduler source: {scheduler_path}")


if __name__ == "__main__":
    main()
