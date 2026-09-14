#!/usr/bin/env python3
"""Split heterogeneous vLLM KV backing storage into valid CPU offload regions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OLD_SETUP = """        logical_storage_bytes = self.kv_cache_config.kv_cache_tensors[0].size

        # The DMA backend copies whole blocks as base + block_id * stride(0),
"""
NEW_SETUP = """        logical_storage_bytes = self.kv_cache_config.kv_cache_tensors[0].size
        split_storage_by_layer = any(
            logical_storage_bytes
            % (num_blocks * cache_tensor.block_stride)
            != 0
            for cache_tensor in self.kv_cache_config.kv_cache_tensors
        )

        # The DMA backend copies whole blocks as base + block_id * stride(0),
"""
OLD_LOOP = """        unique_gpu_caches: dict[str, torch.Tensor] = {}
        seen: set[tuple[torch.device, int]] = set()
        for name, tensor in kv_caches.items():
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            if key in seen:
                continue
            seen.add(key)

            physical_per_block, remainder = divmod(tensor.shape[0], num_blocks)
            assert remainder == 0, (
                f"KV cache {name!r} has {tensor.shape[0]} physical blocks, which "
                f"is not divisible by {num_blocks} scheduler blocks"
            )
            block_bytes = tensor.stride(0) * tensor.element_size() * physical_per_block
            raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(storage)
            assert raw.numel() >= logical_storage_bytes, (
                f"KV cache {name!r} storage has {raw.numel()} bytes, smaller "
                f"than the configured {logical_storage_bytes}-byte allocation"
            )
            regions = raw[:logical_storage_bytes].view(-1, num_blocks, block_bytes)
            for idx, region in enumerate(regions):
                key_name = name if len(regions) == 1 else f"{name}.{idx}"
                unique_gpu_caches[key_name] = region
"""
NEW_LOOP = """        unique_gpu_caches: dict[str, torch.Tensor] = {}
        seen: set[tuple[torch.device, int, int, int]] = set()
        for name, tensor in kv_caches.items():
            physical_per_block, remainder = divmod(tensor.shape[0], num_blocks)
            assert remainder == 0, (
                f"KV cache {name!r} has {tensor.shape[0]} physical blocks, which "
                f"is not divisible by {num_blocks} scheduler blocks"
            )
            block_bytes = tensor.stride(0) * tensor.element_size() * physical_per_block
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(storage)

            if split_storage_by_layer:
                region_offset = tensor.storage_offset() * tensor.element_size()
                region_bytes = num_blocks * block_bytes
            else:
                region_offset = 0
                region_bytes = logical_storage_bytes

            key = (tensor.device, storage.data_ptr(), region_offset, region_bytes)
            if key in seen:
                continue
            seen.add(key)

            region_end = region_offset + region_bytes
            assert raw.numel() >= region_end, (
                f"KV cache {name!r} storage has {raw.numel()} bytes, smaller "
                f"than the required {region_end}-byte region"
            )
            regions = raw[region_offset:region_end].view(
                -1, num_blocks, block_bytes
            )
            for idx, region in enumerate(regions):
                key_name = name if len(regions) == 1 else f"{name}.{idx}"
                unique_gpu_caches[key_name] = region
"""


def installed_worker_path() -> Path:
    """Return the SimpleCPUOffload worker module from the installed vLLM."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm package is not installed")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    return package_root / "v1/simple_kv_offload/worker.py"


def patch_worker(worker_path: Path) -> bool:
    """Patch heterogeneous layer-region sizing and return whether source changed."""
    source = worker_path.read_text()
    if NEW_SETUP in source and NEW_LOOP in source:
        return False
    if NEW_SETUP in source or NEW_LOOP in source:
        raise RuntimeError(f"partially patched vLLM worker at {worker_path}")
    if source.count(OLD_SETUP) != 1 or source.count(OLD_LOOP) != 1:
        raise RuntimeError(
            f"unsupported vLLM SimpleCPUOffload worker at {worker_path}"
        )

    patched = source.replace(OLD_SETUP, NEW_SETUP).replace(OLD_LOOP, NEW_LOOP)
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
    print(f"{state} vLLM SimpleCPUOffload heterogeneous layer regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
