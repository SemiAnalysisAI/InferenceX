#!/usr/bin/env python3
"""Use each vLLM KV allocation's own logical size during CPU offload setup."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


OLD_BLOCK = """        logical_storage_bytes = self.kv_cache_config.kv_cache_tensors[0].size

        # The DMA backend copies whole blocks as base + block_id * stride(0),
"""
NEW_BLOCK = """        logical_storage_bytes_by_layer = {
            layer: cache_tensor.size
            for cache_tensor in self.kv_cache_config.kv_cache_tensors
            for layer in cache_tensor.layers
        }

        # The DMA backend copies whole blocks as base + block_id * stride(0),
"""
OLD_STORAGE = """            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
"""
NEW_STORAGE = """            logical_storage_bytes = logical_storage_bytes_by_layer[name]
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
"""


def installed_worker_path() -> Path:
    """Return the SimpleCPUOffload worker module from the installed vLLM."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm package is not installed")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    return package_root / "v1/simple_kv_offload/worker.py"


def patch_worker(worker_path: Path) -> bool:
    """Patch per-allocation sizing and return whether the source changed."""
    source = worker_path.read_text()
    if NEW_BLOCK in source and NEW_STORAGE in source:
        return False
    if NEW_BLOCK in source or NEW_STORAGE in source:
        raise RuntimeError(f"partially patched vLLM worker at {worker_path}")
    if source.count(OLD_BLOCK) != 1 or source.count(OLD_STORAGE) != 1:
        raise RuntimeError(
            f"unsupported vLLM SimpleCPUOffload worker at {worker_path}"
        )

    patched = source.replace(OLD_BLOCK, NEW_BLOCK).replace(OLD_STORAGE, NEW_STORAGE)
    worker_path.write_text(patched)
    return True


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"Usage: {argv[0]} [WORKER_PATH]", file=sys.stderr)
        return 2

    try:
        worker_path = (
            Path(argv[1]).resolve() if len(argv) == 2 else installed_worker_path()
        )
        changed = patch_worker(worker_path)
    except (OSError, RuntimeError) as error:
        print(f"ERROR: failed to patch vLLM CPU offload: {error}", file=sys.stderr)
        return 1

    state = "Patched" if changed else "Already patched"
    print(f"{state} vLLM SimpleCPUOffload per-allocation sizing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
