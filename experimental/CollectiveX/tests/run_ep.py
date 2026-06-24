#!/usr/bin/env python3
"""CollectiveX — EP dispatch/combine benchmark entrypoint (run under torchrun).

Picks a backend adapter (DeepEP or MoRI), runs the source-tokens-per-rank sweep
via ep_harness, and writes one provenance-tagged JSON doc. Dispatch and combine
are timed SEPARATELY (see ep_harness); only T varies along the resulting line.

  torchrun --nproc_per_node=8 tests/run_ep.py --backend mori \\
      --phase decode --runner mi355x-amds --topology-class mi355x-xgmi \\
      --transport xgmi --env-json results/env.json --out results/mi355x_mori_decode.json

  torchrun --nproc_per_node=8 tests/run_ep.py --backend deepep \\
      --phase prefill --runner b200-dgxc --topology-class b200-nvlink-island \\
      --transport nvlink --env-json results/env.json --out results/b200_deepep_prefill.json
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the sibling tests/ modules importable when run as `tests/run_ep.py` under
# torchrun (it executes the file as __main__, not as a package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ep_harness  # noqa: E402  (stdlib-only; safe before torch)


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX EP dispatch/combine sweep")
    ap.add_argument("--backend", required=True, choices=["deepep", "mori"])
    ep_harness.add_common_args(ap)
    args = ap.parse_args()

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: torch unavailable: {exc!r}", file=sys.stderr)
        return 3

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    # MoRI inits its shmem on a process group it registers as "default" and wants
    # the gloo+nccl combo with an explicit device_id (per its reference test);
    # DeepEP uses a plain nccl group.
    if not dist.is_initialized():
        if args.backend == "mori":
            dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank,
                                    world_size=world_size, device_id=device)
        else:
            dist.init_process_group("nccl")

    if args.backend == "mori":
        from ep_mori import MoRIBackend as Backend
    else:
        from ep_deepep import DeepEPBackend as Backend

    backend = Backend(args, rank, world_size, local_rank, device)
    if rank == 0:
        print(f"[run_ep] backend={args.backend} phase={args.phase} world={world_size} "
              f"ep_size={world_size // max(1, args.num_ep_groups)} hidden={args.hidden} "
              f"topk={args.topk} experts={args.experts} dtype={args.dispatch_dtype}")

    rc = ep_harness.run_sweep(args, backend, torch, dist, device, rank, world_size)
    # finalize() handles backend-specific teardown: DeepEP returns rc cleanly;
    # MoRI hard-exits past its post-shmem_finalize teardown assertion.
    return backend.finalize(rc)


if __name__ == "__main__":
    raise SystemExit(main())
