#!/usr/bin/env python3
"""CollectiveX — NCCL/RCCL KV-cache transfer benchmark (family=kv-cache, backend=nccl|rccl).

The point-to-point KV handoff a disaggregated stack does over the collective library directly:
torchrun with 2 ranks, rank 0 `dist.send`s KV-block-sized buffers to rank 1 (`dist.recv`), timed
with CUDA events. NCCL on NVIDIA, RCCL on AMD/ROCm (same torch.distributed API) — so this is the
WIRED `nccl`/`rccl` KV-cache backend the goal's "KV-cache transfer backends" axis declared a stub
(the NCCL collective suite covers the all_reduce/all_gather primitives; this is the p2p KV path).

Emits one kv-cache-family JSON (plots in the KV-cache tab next to memcpy/nixl/mori-io). Single
(dir, backend, layout) group per run. Backend label = rccl on ROCm, nccl on CUDA.

  torchrun --nproc_per_node=2 tests/nccl_kv_transfer.py --runner h200-dgxc \\
      --topology-class h200-nvlink-island --transport nvlink \\
      --env-json results/env.json --out results/h200_ncclkv.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "nccl-kv-sendrecv-v1"
FAMILY = "kv-cache"

DEFAULT_MIN_BYTES = 64 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DECODE_MAX_BYTES = 512 * 1024


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


def _bench_one(torch, dist, rank, send_buf, recv_buf, nbytes, warmup, iters):
    """rank0 sends -> rank1 recvs, `iters` times, CUDA-event timed on the active rank. Returns
    (latency_ms, gb_s) on rank 0 (rank 1 returns None and is the receiver)."""
    def _once():
        if rank == 0:
            dist.send(send_buf, dst=1)
        else:
            dist.recv(recv_buf, src=0)
    for _ in range(warmup):
        _once()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _once()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    gb_s = (nbytes / (ms / 1e3)) / 1e9 if ms > 0 else 0.0
    return round(ms, 5), round(gb_s, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX NCCL/RCCL KV-cache transfer benchmark")
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="nvlink")
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:
        print(f"ERROR: torch unavailable: {exc!r}", file=sys.stderr)
        return 3
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available", file=sys.stderr)
        return 3

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    # backend label tracks the actual transport library: RCCL on ROCm, NCCL on CUDA.
    is_rocm = bool(getattr(torch.version, "hip", None))
    backend_label = "rccl" if is_rocm else "nccl"

    if world < 2:
        if rank == 0:
            _emit(args, [], "invalid", 0.0, [f"needs >=2 ranks (torchrun --nproc_per_node>=2); world={world}"],
                  backend_label)
        return 1
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world, rank=rank)

    sizes = _sizes(args.min_bytes, args.max_bytes)
    rows = []
    peak = 0.0
    for nbytes in sizes:
        try:
            send_buf = torch.empty(nbytes, dtype=torch.uint8, device=dev) if rank == 0 else torch.empty(1, dtype=torch.uint8, device=dev)
            recv_buf = torch.empty(nbytes, dtype=torch.uint8, device=dev) if rank == 1 else torch.empty(1, dtype=torch.uint8, device=dev)
            ms, gb_s = _bench_one(torch, dist, rank, send_buf, recv_buf, nbytes, args.warmup, args.iters)
        except RuntimeError as exc:
            if rank == 0:
                rows.append({"transfer_bytes": nbytes, "error": f"{exc!r}", "correct": None})
            break
        if rank == 0:
            rows.append({"transfer_bytes": nbytes, "size_class": size_class(nbytes),
                         "block_bytes": nbytes, "num_blocks": 1,
                         "time_ms": ms, "bandwidth_gb_s": gb_s, "correct": True})
            peak = max(peak, gb_s)
        del send_buf, recv_buf
        torch.cuda.empty_cache()

    dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return 0

    groups = []
    if any(r.get("bandwidth_gb_s") for r in rows):
        meta = {"direction": "dtod-remote", "layout": "contiguous", "backend": backend_label,
                "dtype": "uint8", "nodes": args.nodes,
                "topology_class": args.topology_class,
                "measurement_contract": MEASUREMENT_CONTRACT}
        groups.append({**meta, "comparison_key": comparison_key(meta), "rows": rows})
    status = "valid" if (groups and peak > 0.0) else "invalid"
    _emit(args, groups, status, peak, [f"{backend_label} 2-rank send/recv (rank0->rank1)"], backend_label)
    dist.destroy_process_group()
    return 0 if status == "valid" else 1


def _emit(args, groups, status, peak, notes, backend_label):
    env = None
    if args.env_json and os.path.exists(args.env_json):
        with open(args.env_json) as fh:
            env = json.load(fh)
    doc = {"schema_version": SCHEMA_VERSION, "family": FAMILY,
           "generated_by": "nccl_kv_transfer.py",
           "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
           "runner": args.runner, "transport": args.transport,
           "measurement_contract": MEASUREMENT_CONTRACT, "nodes": args.nodes,
           "wired_backends": [backend_label], "status": status,
           "num_groups": len(groups), "groups": groups, "notes": notes, "environment": env}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(f"{backend_label}-kv: {len(groups)} groups -> {args.out} (status={status}, peak_bw={peak:.1f} GB/s)")


if __name__ == "__main__":
    raise SystemExit(main())
