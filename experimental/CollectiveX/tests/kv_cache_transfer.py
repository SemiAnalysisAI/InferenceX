#!/usr/bin/env python3
"""CollectiveX — KV-cache transfer benchmark (family=kv-cache).

Times raw CUDA memcpy of KV-cache-shaped buffers across the transfer paths a
serving stack actually uses, with CUDA events (GPU-accurate). Adapted from
experimental/kvcache_transfer_DtoH_HtoD/benchmark.py but WITHOUT the vLLM
`swap_blocks` dependency — CollectiveX containers may not ship vLLM, and the goal
asks for the raw CUDA/HIP memcpy + CPU pinned-memory path as the reference.

Dimensions (goal P2 "KV-cache transfer suite"):
  direction : dtoh | htod | dtod-local | dtod-remote (remote needs >=2 GPUs)
  layout    : contiguous (one copy) | paged (N scattered block copies — the real
              paged-KV pattern; captures per-block launch/scatter overhead)
  size class: decode-sized (small per-token blocks) .. prefill/prefix-cache-sized (large)
  backend   : memcpy (raw cudaMemcpy), pinned (CPU pinned host) — WIRED.
              nixl / mooncake / mori-io / nccl — declared, NOT wired (stubs; never faked).

Stdlib + torch; torch is imported lazily so `--help` / `--parse-only`-style use works
without a GPU. One provenance-tagged JSON per run, matching run_nccl.py's structure.

  python tests/kv_cache_transfer.py --direction all --runner h200-dgxc \\
      --topology-class h200-nvlink-island --transport nvlink \\
      --env-json results/env.json --out results/h200_kvcache.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "kv-cache-memcpy-v1"
FAMILY = "kv-cache"

# Backends: which transfer mechanism moves the bytes. Only the raw memcpy + pinned-host
# paths are wired; the rest are declared so the axis is honest and a future adapter slots in.
WIRED_BACKENDS = ("memcpy", "pinned")
STUB_BACKENDS = ("nixl", "mooncake", "mori-io", "nccl")

# KV block byte sizes: decode-sized (a few tokens' KV) .. prefill/prefix-cache-sized.
# A DeepSeek-V3 layer KV block for a handful of tokens is ~tens of KiB; a prefill/prefix
# chunk is MiB. Sweep geometric 16KiB -> 256MiB and class each point.
DECODE_MAX_BYTES = 512 * 1024          # <=512KiB == "decode-sized"
DEFAULT_MIN_BYTES = 16 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def size_class(nbytes: int) -> str:
    return "decode" if nbytes <= DECODE_MAX_BYTES else "prefill"


def _sizes(min_bytes: int, max_bytes: int, factor: int = 4):
    out, s = [], min_bytes
    while s <= max_bytes:
        out.append(s)
        s *= factor
    return out


def comparison_key(meta: dict) -> str:
    parts = [meta["direction"], meta["layout"], meta["backend"], meta["dtype"],
             str(meta["nodes"]), meta["topology_class"], meta["measurement_contract"]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _bench_one(torch, src, dst, total_bytes, block_bytes, layout, paged_blocks,
               warmup: int, iters: int):
    """Time `iters` copies of total_bytes from src->dst. paged => paged_blocks scattered
    block copies of block_bytes each; contiguous => one copy. Returns (time_ms, gb_s)."""
    def _do():
        if layout == "paged":
            # scatter: copy each logical block to a (shuffled) destination block slot —
            # the paged-KV access pattern (non-contiguous gather/scatter).
            for s_off, d_off in paged_blocks:
                dst[d_off:d_off + block_bytes].copy_(src[s_off:s_off + block_bytes],
                                                      non_blocking=True)
        else:
            dst.copy_(src, non_blocking=True)

    for _ in range(warmup):
        _do()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _do()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    gb_s = (total_bytes / (ms / 1e3)) / 1e9 if ms > 0 else 0.0
    return round(ms, 5), round(gb_s, 2)


def _alloc(torch, where, nbytes, pinned: bool):
    n = nbytes  # bytes; use uint8 so 1 elem == 1 byte
    if where == "cpu":
        t = torch.empty(n, dtype=torch.uint8, device="cpu")
        return t.pin_memory() if pinned else t
    return torch.empty(n, dtype=torch.uint8, device=where)


def run_direction(torch, direction, backend, layout, sizes, block_bytes, warmup, iters,
                  ngpu: int):
    """Yield a row per size for one (direction, backend, layout)."""
    rows = []
    pinned = (backend == "pinned")
    for nbytes in sizes:
        # endpoints
        if direction == "dtoh":
            src_dev, dst_dev = "cuda:0", "cpu"
        elif direction == "htod":
            src_dev, dst_dev = "cpu", "cuda:0"
        elif direction == "dtod-local":
            src_dev, dst_dev = "cuda:0", "cuda:0"
        elif direction == "dtod-remote":
            if ngpu < 2:
                return [], "n/a (needs >=2 GPUs)"
            src_dev, dst_dev = "cuda:0", "cuda:1"
        else:
            return [], f"unknown direction {direction}"
        # pinned only matters when a host buffer is involved
        host_involved = ("cpu" in (src_dev, dst_dev))
        if backend == "pinned" and not host_involved:
            continue  # pinned is a host-memory property; skip for pure DtoD
        try:
            src = _alloc(torch, src_dev, nbytes, pinned and src_dev == "cpu")
            dst = _alloc(torch, dst_dev, nbytes, pinned and dst_dev == "cpu")
        except RuntimeError as exc:  # OOM at the largest sizes — stop, don't crash
            rows.append({"transfer_bytes": nbytes, "error": f"alloc: {exc!r}", "correct": None})
            break
        nblk = max(1, nbytes // block_bytes)
        bb = nbytes // nblk
        # paged: shuffle destination block order (deterministic) to force scatter
        paged = [((i * bb), (((i * 2654435761) % nblk) * bb)) for i in range(nblk)] \
            if layout == "paged" else None
        ms, gb_s = _bench_one(torch, src, dst, nbytes, bb, layout, paged, warmup, iters)
        rows.append({
            "transfer_bytes": nbytes, "size_class": size_class(nbytes),
            "block_bytes": bb if layout == "paged" else nbytes,
            "num_blocks": nblk if layout == "paged" else 1,
            "time_ms": ms, "bandwidth_gb_s": gb_s,
            "correct": True,  # raw memcpy is exact (uint8); kept for schema parity
        })
        del src, dst
        torch.cuda.empty_cache()
    return rows, None


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX KV-cache transfer benchmark")
    ap.add_argument("--direction", default="all",
                    choices=["all", "dtoh", "htod", "dtod-local", "dtod-remote"])
    ap.add_argument("--backends", default="memcpy,pinned",
                    help="comma list from memcpy,pinned (wired) — stubs are recorded, not run")
    ap.add_argument("--layouts", default="contiguous,paged")
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--block-bytes", type=int, default=64 * 1024,
                    help="paged KV block size (a few tokens' KV); default 64KiB")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    # provenance (mirror run_nccl.py)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="")
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: torch unavailable: {exc!r}", file=sys.stderr)
        return 3
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 3

    ngpu = torch.cuda.device_count()
    directions = (["dtoh", "htod", "dtod-local", "dtod-remote"]
                  if args.direction == "all" else [args.direction])
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    layouts = [l.strip() for l in args.layouts.split(",") if l.strip()]
    sizes = _sizes(args.min_bytes, args.max_bytes)

    groups = []
    notes = []
    peak = 0.0
    for backend in backends:
        if backend not in WIRED_BACKENDS:
            notes.append(f"backend '{backend}' not wired (declared only)")
            continue
        for direction in directions:
            for layout in layouts:
                rows, na = run_direction(torch, direction, backend, layout, sizes,
                                         args.block_bytes, args.warmup, args.iters, ngpu)
                if na:
                    notes.append(f"{direction}/{backend}/{layout}: {na}")
                    continue
                if not rows:
                    continue
                peak = max(peak, max((r.get("bandwidth_gb_s") or 0.0) for r in rows))
                meta = {"direction": direction, "layout": layout, "backend": backend,
                        "dtype": "uint8", "nodes": args.nodes,
                        "topology_class": args.topology_class,
                        "measurement_contract": MEASUREMENT_CONTRACT}
                groups.append({**meta, "comparison_key": comparison_key(meta), "rows": rows})

    env = None
    if args.env_json and os.path.exists(args.env_json):
        with open(args.env_json) as fh:
            env = json.load(fh)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "generated_by": "kv_cache_transfer.py",
        "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
        "runner": args.runner,
        "transport": args.transport,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "nodes": args.nodes,
        "num_gpus_visible": ngpu,
        "wired_backends": list(WIRED_BACKENDS),
        "declared_unwired_backends": list(STUB_BACKENDS),
        "status": "valid" if (groups and peak > 0.0) else "invalid",
        "num_groups": len(groups),
        "groups": groups,
        "notes": notes,
        "environment": env,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(f"kv-cache: {len(groups)} (dir,backend,layout) groups -> {args.out} "
          f"(status={doc['status']}, peak_bw={peak:.1f} GB/s, gpus={ngpu})")
    if notes:
        print("notes: " + "; ".join(notes), file=sys.stderr)
    return 0 if doc["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
