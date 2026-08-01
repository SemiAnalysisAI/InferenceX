"""Reject incomplete hybrid Mooncake saves before writing partial keys."""

from __future__ import annotations

import importlib.util
from pathlib import Path

OLD_SAVE_PROLOGUE = """        try:
            if token_len == 0:
                return

            if self._should_skip_request(req_id):
"""

NEW_SAVE_PROLOGUE = """        try:
            if token_len == 0:
                return

            if len(block_ids_per_group) != len(self.token_databases):
                # A full external hit can reach this older Kimi image without
                # block tables for every hybrid group. Never write only the
                # groups that happen to be present: that would publish a
                # corrupt partial prefix. The prefill-side Mooncake copy
                # remains valid and this decode-side extension can be safely
                # recomputed on its next turn.
                logger.warning(
                    "Skipping incomplete hybrid Mooncake save for request %s: "
                    "metadata has %d groups, worker expects %d",
                    req_id,
                    len(block_ids_per_group),
                    len(self.token_databases),
                )
                return

            if self._should_skip_request(req_id):
"""


def patch_worker(package_dir: Path) -> None:
    """Add the hybrid group-count guard to the Mooncake sending worker."""
    target = (
        package_dir / "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"
    )
    source = target.read_text()
    if "Skipping incomplete hybrid Mooncake save" in source:
        print(f"[kimi-k3-mooncake-save-groups] Already patched {target}")
        return
    if OLD_SAVE_PROLOGUE not in source:
        raise RuntimeError(f"Unexpected vLLM source layout in {target}")
    source = source.replace(OLD_SAVE_PROLOGUE, NEW_SAVE_PROLOGUE, 1)
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"[kimi-k3-mooncake-save-groups] Patched {target}")


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed vllm package")
    patch_worker(Path(next(iter(spec.submodule_search_locations))))


if __name__ == "__main__":
    main()
