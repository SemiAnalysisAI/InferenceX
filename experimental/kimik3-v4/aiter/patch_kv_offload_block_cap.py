#!/usr/bin/env python3
"""Cap the simple-KV-offload CPU block count so pipeline stages agree on it.

`SimpleCPUOffloadScheduler._derive_cpu_config` sizes the CPU pool from the *global*
GPU KV config (`num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes`) and then
hands out block ids up to that count. Each `SimpleCPUOffloadWorker`, however, sizes
its own pinned buffers from **its own pipeline stage's** tensors
(`cpu_capacity_bytes // sum(per_tensor_bytes_per_block)`).

Under pipeline parallelism the stages can hold different numbers of unique KV
tensors, so they arrive at different block counts. Kimi-K3 TP8xPP2 on MI355X:
PP0 sees 3 unique KV tensors and allocates 47095 blocks, PP1 sees 5 and allocates
only 28257 — while the scheduler issues ids against 47095. Once the working set
crosses 28257 (conc>=16 with dspark), PP1's `copy_blocks` writes past the end of
its pinned buffer and every PP1 worker takes a SIGSEGV inside
`hipMemcpyBatchAsync`. Nothing bounds-checks the id, and raising
TOTAL_CPU_DRAM_GB cannot help: both counts scale together, so PP1 stays at ~60%
of the scheduler's count.

Fix: clamp the scheduler to the smallest per-stage count. Set
`KV_OFFLOAD_MAX_CPU_BLOCKS` to the lowest "allocating N CPU blocks" value logged
by `worker.py` across stages (grep `SimpleCPUOffloadWorker:` in the rank logs).
Workers keep their own sizing; over-provisioned stages simply leave rows unused.

The clamp is applied by re-deriving the config with a reduced capacity rather than
rewriting `num_blocks`, so `kv_cache_tensors` sizes stay consistent with it.

Same import discipline as patch_gemm_n6288_chunk: no torch/vllm at module scope.
"""
from __future__ import annotations

import os
import sys


def install() -> None:
    if os.environ.get("_KV_OFFLOAD_BLOCK_CAP_INSTALLING") == "1":
        return
    cap = int(os.environ.get("KV_OFFLOAD_MAX_CPU_BLOCKS", "0") or 0)
    if cap <= 0:
        return
    os.environ["_KV_OFFLOAD_BLOCK_CAP_INSTALLING"] = "1"
    try:
        from vllm.v1.simple_kv_offload import manager as mgr

        cls = mgr.SimpleCPUOffloadScheduler
        if getattr(cls._derive_cpu_config, "_kimik3_block_cap", False):
            print("[patch_kv_offload_block_cap] already installed", file=sys.stderr, flush=True)
            return

        # cls._derive_cpu_config is already the bare function for a @staticmethod;
        # __func__ only exists on the descriptor in cls.__dict__.
        _raw = cls.__dict__["_derive_cpu_config"]
        _orig = _raw.__func__ if isinstance(_raw, staticmethod) else _raw

        def _derive_cpu_config(gpu_config, cpu_capacity_bytes):
            cfg = _orig(gpu_config, cpu_capacity_bytes)
            if cfg.num_blocks <= cap:
                return cfg
            is_packed = any(t.block_stride for t in gpu_config.kv_cache_tensors)
            gpu_total_bytes = (
                gpu_config.kv_cache_tensors[0].size
                if is_packed
                else sum(t.size for t in gpu_config.kv_cache_tensors)
            )
            capped_bytes = cap * gpu_total_bytes // gpu_config.num_blocks
            capped = _orig(gpu_config, capped_bytes)
            print(
                "[patch_kv_offload_block_cap] capped CPU pool %d -> %d blocks "
                "(cap=%d, capacity %.2f -> %.2f GB)"
                % (
                    cfg.num_blocks,
                    capped.num_blocks,
                    cap,
                    cpu_capacity_bytes / (1024**3),
                    capped_bytes / (1024**3),
                ),
                file=sys.stderr,
                flush=True,
            )
            return capped

        _derive_cpu_config._kimik3_block_cap = True  # type: ignore[attr-defined]
        cls._derive_cpu_config = staticmethod(_derive_cpu_config)
        print(
            "[patch_kv_offload_block_cap] installed "
            f"SimpleCPUOffloadScheduler._derive_cpu_config (cap={cap} blocks)",
            file=sys.stderr,
            flush=True,
        )
    finally:
        os.environ.pop("_KV_OFFLOAD_BLOCK_CAP_INSTALLING", None)


def main() -> int:
    try:
        install()
    except Exception as exc:
        print(f"[patch_kv_offload_block_cap] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
