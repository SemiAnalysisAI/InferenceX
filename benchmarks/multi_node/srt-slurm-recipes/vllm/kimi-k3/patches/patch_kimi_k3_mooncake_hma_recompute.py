"""Backport conservative KV-load recovery for Kimi-K3's hybrid cache."""

from __future__ import annotations

import importlib.util
from pathlib import Path


OLD_SINGLE_GROUP_RECOVERY = """            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )
"""

NEW_HYBRID_RECOVERY = """            req_block_id_groups = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            if len(req_block_id_groups) != 1:
                # Mooncake reports load failures as physical block IDs without
                # identifying the hybrid KV group. Attention and Mamba groups
                # can use different logical block sizes, so there is no safe
                # common token boundary at which to truncate them. Conservatively
                # recompute the complete prefix for every matching hybrid request.
                # This is a rare eviction-race path and preserves correctness.
                req_hybrid_block_ids = {
                    block_id
                    for group_block_ids in req_block_id_groups
                    for block_id in group_block_ids
                }
                if req_hybrid_block_ids.isdisjoint(invalid_block_ids):
                    continue

                request.num_computed_tokens = 0
                total_affected_tokens += max(req_num_computed_tokens, 0)
                if evict_blocks:
                    blocks_to_evict.update(req_hybrid_block_ids)
                affected_req_ids.add(req_id)
                continue

            (req_block_ids,) = req_block_id_groups
"""


def patch_scheduler(package_dir: Path) -> None:
    """Teach the scheduler to recover rather than crash on hybrid load loss."""
    target = package_dir / "v1/core/sched/scheduler.py"
    source = target.read_text()
    if "req_hybrid_block_ids = {" in source:
        print(f"[kimi-k3-mooncake-hma] Already patched {target}")
        return
    if OLD_SINGLE_GROUP_RECOVERY not in source:
        raise RuntimeError(f"Unexpected vLLM source layout in {target}")
    source = source.replace(
        OLD_SINGLE_GROUP_RECOVERY,
        NEW_HYBRID_RECOVERY,
        1,
    )
    compile(source, str(target), "exec")
    target.write_text(source)
    print(f"[kimi-k3-mooncake-hma] Patched {target}")


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the installed vllm package")
    patch_scheduler(Path(next(iter(spec.submodule_search_locations))))


if __name__ == "__main__":
    main()
