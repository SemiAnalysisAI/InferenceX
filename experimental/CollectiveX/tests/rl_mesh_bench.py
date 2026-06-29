#!/usr/bin/env python3
"""CollectiveX — RL mesh-to-mesh transfer benchmark (family=rl-mesh).

In RL post-training the TRAINER mesh (updated weights) must hand parameters to the
GENERATOR/rollout mesh, and rollouts flow back — an NCCL M2N / "NCCL Xfer" pattern between
two DISJOINT device meshes. This benchmark splits the world into a trainer half and a
generator half and times weight-sized tensor transfer between them, both directions, under
two redistribution patterns:

  paired       : trainer rank i  -> generator rank i        (1:1 send/recv, matched ranks)
  redistribute : every trainer rank -> every generator rank (disjoint all-to-all reshard,
                 the realistic case when trainer-TP != generator-TP)

Run under torchrun (multi-process); world is split in half (needs >=2 ranks, even count).
CUDA-event timed; one provenance-tagged JSON like run_nccl.py. Stdlib + torch (torch only
needed at runtime; --help works without it).

  torchrun --nproc_per_node=8 tests/rl_mesh_bench.py --runner h200-dgxc \\
      --topology-class h200-nvlink-island --transport nvlink \\
      --env-json results/env.json --out results/h200_rl_mesh.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "rl-mesh-xfer-v1"
FAMILY = "rl-mesh"

# Weight-shard byte sizes a trainer->generator handoff moves: a single large tensor (a fused
# QKV / MLP weight) up to a whole layer's params. Sweep 1 MiB .. 1 GiB.
DEFAULT_MIN_BYTES = 1 << 20
DEFAULT_MAX_BYTES = 1 << 30


def _sizes(lo, hi, factor=4):
    out, s = [], lo
    while s <= hi:
        out.append(s)
        s *= factor
    return out


def comparison_key(meta: dict) -> str:
    parts = [meta["direction"], meta["pattern"], str(meta["world_size"]),
             meta["topology_class"], meta["measurement_contract"]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _bench(fn, torch, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX RL mesh-to-mesh transfer benchmark")
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="nvlink")
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: torch unavailable: {exc!r}", file=sys.stderr)
        return 3

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 2 or world % 2 != 0:
        if rank == 0:
            print(f"ERROR: rl-mesh needs an even world_size >= 2 (got {world})", file=sys.stderr)
        return 5
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    half = world // 2
    is_trainer = rank < half
    # peer for the paired (1:1) pattern: trainer i <-> generator (i+half)
    paired_peer = (rank + half) if is_trainer else (rank - half)
    sizes = _sizes(args.min_bytes, args.max_bytes)
    groups = []
    peak = 0.0

    def _buf(nbytes):
        return torch.empty(nbytes, dtype=torch.uint8, device=dev)

    # PAIRED 1:1 send/recv, timed on the trainer side per direction.
    for direction in ("trainer_to_generator", "generator_to_trainer"):
        rows = []
        sender_is_trainer = (direction == "trainer_to_generator")
        i_send = (is_trainer == sender_is_trainer)  # this rank sends in this direction
        for nbytes in sizes:
            buf = _buf(nbytes)

            def step():
                if i_send:
                    dist.send(buf, dst=paired_peer)
                else:
                    dist.recv(buf, src=paired_peer)
            try:
                ms = _bench(step, torch, args.warmup, args.iters)
            except RuntimeError as exc:
                rows.append({"transfer_bytes": nbytes, "error": repr(exc), "correct": None})
                break
            gb_s = (nbytes / (ms / 1e3)) / 1e9 if ms > 0 else 0.0
            # reduce timing across ranks (max = slowest pair) for a stable number
            t = torch.tensor([ms], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            ms_max = float(t.item())
            gb_s = (nbytes / (ms_max / 1e3)) / 1e9 if ms_max > 0 else 0.0
            peak = max(peak, gb_s)
            rows.append({"transfer_bytes": nbytes, "time_ms": round(ms_max, 5),
                         "bandwidth_gb_s": round(gb_s, 2), "correct": True})
        meta = {"direction": direction, "pattern": "paired", "world_size": world,
                "trainer_ranks": half, "generator_ranks": world - half,
                "topology_class": args.topology_class, "measurement_contract": MEASUREMENT_CONTRACT}
        groups.append({**meta, "comparison_key": comparison_key(meta), "rows": rows})

    # REDISTRIBUTE: disjoint all-to-all (trainer half scatters to all generator ranks). Each
    # sender sends nbytes/half to each receiver in the other mesh; timed via batched isend/irecv.
    for direction in ("trainer_to_generator", "generator_to_trainer"):
        rows = []
        senders = range(0, half) if direction == "trainer_to_generator" else range(half, world)
        receivers = range(half, world) if direction == "trainer_to_generator" else range(0, half)
        am_sender = rank in senders
        am_receiver = rank in receivers
        for nbytes in sizes:
            chunk = max(1, nbytes // half)
            sbuf = _buf(chunk)

            def step():
                reqs = []
                if am_sender:
                    for dst in receivers:
                        reqs.append(dist.isend(sbuf, dst=dst))
                if am_receiver:
                    for src in senders:
                        rbuf = _buf(chunk)
                        reqs.append(dist.irecv(rbuf, src=src))
                for r in reqs:
                    r.wait()
            try:
                ms = _bench(step, torch, args.warmup, args.iters)
            except RuntimeError as exc:
                rows.append({"transfer_bytes": nbytes, "error": repr(exc), "correct": None})
                break
            t = torch.tensor([ms], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            ms_max = float(t.item())
            # effective payload moved per receiver = nbytes (half chunks of nbytes/half)
            gb_s = (nbytes / (ms_max / 1e3)) / 1e9 if ms_max > 0 else 0.0
            peak = max(peak, gb_s)
            rows.append({"transfer_bytes": nbytes, "time_ms": round(ms_max, 5),
                         "bandwidth_gb_s": round(gb_s, 2), "correct": True})
        meta = {"direction": direction, "pattern": "redistribute", "world_size": world,
                "trainer_ranks": half, "generator_ranks": world - half,
                "topology_class": args.topology_class, "measurement_contract": MEASUREMENT_CONTRACT}
        groups.append({**meta, "comparison_key": comparison_key(meta), "rows": rows})

    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return 0

    env = None
    if args.env_json and os.path.exists(args.env_json):
        with open(args.env_json) as fh:
            env = json.load(fh)
    doc = {
        "schema_version": SCHEMA_VERSION, "family": FAMILY,
        "generated_by": "rl_mesh_bench.py",
        "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
        "runner": args.runner, "transport": args.transport,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "world_size": world, "trainer_ranks": half, "generator_ranks": world - half,
        "status": "valid" if (groups and peak > 0.0) else "invalid",
        "peak_bandwidth_gb_s": round(peak, 2),
        "num_groups": len(groups), "groups": groups, "environment": env,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(f"rl-mesh: {len(groups)} (direction,pattern) groups -> {args.out} "
          f"(status={doc['status']}, peak_bw={peak:.1f} GB/s, world={world} trainer={half})")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if doc["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
