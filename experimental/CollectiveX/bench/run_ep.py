#!/usr/bin/env python3
"""CollectiveX v1 EP benchmark entrypoint for torchrun or rank environments."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

# Make the sibling bench/ modules importable when run as `bench/run_ep.py` under
# torchrun (it executes the file as __main__, not as a package).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import ep_harness  # noqa: E402  (stdlib-only; safe before torch)


def _loaded_collective_version() -> str | None:
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            paths = {
                os.path.realpath(line.rstrip().split()[-1])
                for line in handle
                if any(name in line for name in ("libnccl.so", "librccl.so"))
                and os.path.isfile(line.rstrip().split()[-1])
            }
        if len(paths) != 1:
            return None
        version = ctypes.c_int()
        library = ctypes.CDLL(paths.pop())
        if library.ncclGetVersion(ctypes.byref(version)) != 0:
            return None
        return ep_harness.format_collective_version(version.value)
    except (AttributeError, OSError):
        return None


def _runtime_info(torch, *, vendor: str) -> dict:
    """Return the runtime versions needed to compare and debug results."""
    runtime_kind = "cuda" if vendor == "nvidia" else "hip"
    collective_kind = "nccl" if vendor == "nvidia" else "rccl"
    return {
        "accelerator_runtime": getattr(torch.version, runtime_kind),
        "collective_library": {"kind": collective_kind, "version": _loaded_collective_version()},
        "framework": str(torch.__version__),
        "vendor": vendor,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX EP dispatch/combine sweep")
    ap.add_argument("--backend", required=True, choices=["deepep-v2", "mori"])
    ep_harness.add_common_args(ap)
    args = ap.parse_args()

    if not ep_harness.is_case_id(args.case_id):
        print(f"ERROR: invalid native case ID {args.case_id!r}", file=sys.stderr)
        return 2
    # Seed and timing arrive baked into the case argv from the single
    # configs/sweep.json source; there is no separate canonical constant to
    # cross-check against.

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

    vendor = "amd" if torch.version.hip else "nvidia"
    device_name = torch.cuda.get_device_name(device)
    args.runtime_device_product = device_name
    args.image = os.environ.get("COLLECTIVEX_IMAGE", "")
    _run = {
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA")
        or os.environ.get("GITHUB_SHA"),
    }
    args.git_run = _run if any(_run.values()) else None

    # Import the backend class only after torch initializes. The selected mode is an
    # explicit case dimension; adapters do not infer it from the token ladder.
    if args.backend == "mori":
        from ep_mori import MoRIBackend as Backend
    else:
        from ep_deepep_v2 import DeepEPV2Backend as Backend

    # MoRI registers the default GPU process group with its SHMEM runtime. Keep that
    # group device-only so scale-out does not also depend on a host Gloo fabric.
    if not dist.is_initialized():
        if args.backend == "mori":
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                device_id=device,
            )
        else:
            # PR #605 reuses PyTorch's NCCL communicator through ``_comm_ptr``. Supplying
            # device_id eagerly forms it before ElasticBuffer construction.
            dist.init_process_group("nccl", device_id=device)

    args.runtime = _runtime_info(torch, vendor=vendor)

    # Construct + run inside a try so a backend exception (esp. a new adapter on GPU) prints its
    # FULL traceback to STDOUT — torchrun captures per-rank stdout but only summarizes stderr, so an
    # uncaught exception is otherwise invisible in CI. Print on every rank (prefixed) then re-raise.
    try:
        backend = Backend(args, rank, world_size, local_rank, device)
        if rank == 0:
            print(
                f"[run_ep] backend={args.backend} phase={args.phase} mode={args.mode} "
                f"world={world_size} ep_size={world_size} hidden={args.hidden} "
                f"topk={args.topk} experts={args.experts} precision={args.precision} "
                f"routing={args.routing} seed={args.seed}"
            )
        rc = ep_harness.run_sweep(args, backend, torch, dist, device, rank, world_size)
    except Exception:
        import traceback

        print(
            f"[run_ep][rank{rank}] backend={args.backend} FAILED:\n"
            + traceback.format_exc(),
            flush=True,
        )
        raise
    # finalize() handles backend-specific teardown: DeepEP returns rc cleanly;
    # MoRI hard-exits past its post-shmem_finalize teardown assertion.
    return backend.finalize(rc)


if __name__ == "__main__":
    raise SystemExit(main())
