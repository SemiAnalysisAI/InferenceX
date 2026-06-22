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
    pool_old = "        if not self.enable_hisparse:\n"
    pool_new = (
        "        if (\n"
        "            not self.enable_hisparse\n"
        "            and not self.server_args.enable_dp_attention\n"
        "        ):\n"
    )
    tree_old = "        self.invariant_checker._check_tree_cache()\n"
    tree_new = (
        "        if not self.server_args.enable_dp_attention:\n"
        "            self.invariant_checker._check_tree_cache()\n"
    )

    if pool_old in source:
        source = source.replace(pool_old, pool_new, 1)
    elif pool_new not in source:
        raise RuntimeError(f"Unexpected SGLang scheduler source: {scheduler_path}")

    if tree_old in source:
        source = source.replace(tree_old, tree_new, 1)
    elif tree_new not in source:
        raise RuntimeError(f"Unexpected SGLang scheduler source: {scheduler_path}")

    scheduler_path.write_text(source)


if __name__ == "__main__":
    main()
