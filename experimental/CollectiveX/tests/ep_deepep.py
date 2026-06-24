#!/usr/bin/env python3
"""CollectiveX EP backend adapter — DeepEP (NVIDIA), normal mode.

The harness owns the deterministic shared routing trace, the comm-only timing, and
the doc; this file owns only DeepEP's API calls and its correctness reference.
`make_problem` materializes the harness-provided rank slice (no RNG here), so every
SKU runs the identical routed workload.

Correctness (per DeepEP's intranode test): a pure dispatch->combine round trip with no
expert compute reconstructs x only after dividing by the number of ranks each token was
sent to, so the harness expects combined ≈ x * is_token_in_rank.sum(dim=1).
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist

try:
    from deep_ep import Buffer  # type: ignore
    import deep_ep  # for version/provenance
except Exception as exc:  # pragma: no cover - needs the built DeepEP
    print("ERROR: deep_ep import failed — DeepEP must be present/built at job setup. "
          f"{exc!r}", file=sys.stderr)
    raise


def _deepep_version() -> str:
    try:
        import importlib.metadata as _md
        return _md.version("deep_ep")
    except Exception:
        return getattr(deep_ep, "__version__", "unknown")


class DeepEPBackend:
    name = "deepep"
    combine_needs_redispatch = False  # DeepEP combine reuses the handle (its own bench does too)

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = args.mode
        self.group = dist.group.WORLD
        if args.mode == "ll":
            raise NotImplementedError("DeepEP low-latency (LL) path is wired in Phase 3; use --mode normal")
        if args.dispatch_dtype == "fp8":
            if rank == 0:
                print("WARN: deepep fp8 dispatch is wired in Phase 3; using bf16 (provenance reflects bf16).",
                      file=sys.stderr)
            args.dispatch_dtype = "bf16"
        # Intranode normal mode: NVLink buffer only. ONE buffer size for ALL points
        # (review: a phase-dependent 2/4 GiB made the shared T=128 point differ between
        # the decode and prefill sweeps). 4 GiB holds T up to 4096 (validated).
        num_nvl_bytes = int(os.environ.get("CX_DEEPEP_NVL_BYTES", str(4 * 1024 * 1024 * 1024)))
        self.buffer = Buffer(self.group, num_nvl_bytes, 0)
        try:
            Buffer.set_num_sms(args.num_sms)
        except Exception as exc:  # pragma: no cover - version dependent
            if rank == 0:
                print(f"WARN: could not set num_sms={args.num_sms}: {exc!r}", file=sys.stderr)
        ver = _deepep_version()
        dev_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.backend_provenance = {
            "deepep_version": ver,
            "deepep_commit": os.environ.get("DEEPEP_COMMIT") or f"pkg-{ver}",
            "num_sms": args.num_sms, "device_sms": dev_sms,
            "resource_mode": "fixed-num-sms", "num_nvl_bytes": num_nvl_bytes,
        }

    def buffer_cap(self, args):
        return None  # NVLink buffer is large; no hard per-T ceiling like MoRI's heap

    def make_problem(self, T, idx, weights, x):
        # idx[T,topk] int64, weights[T,topk] f32, x[T,hidden] bf16 — the shared trace slice.
        return types.SimpleNamespace(T=T, x=x, topk_idx=idx.to(torch.int64),
                                     topk_weights=weights.to(torch.float32))

    def dispatch(self, p):
        (num_tokens_per_rank, _, num_tokens_per_expert,
         is_token_in_rank, _) = self.buffer.get_dispatch_layout(p.topk_idx, self.args.experts)
        recv_x, _recv_idx, recv_topk_weights, _, handle, _ = self.buffer.dispatch(
            p.x, topk_idx=p.topk_idx, topk_weights=p.topk_weights,
            num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert)
        return types.SimpleNamespace(
            recv_x=recv_x, recv_topk_weights=recv_topk_weights, handle=handle,
            is_token_in_rank=is_token_in_rank)

    def stage(self, p, h):
        # comm-only contract: "expert outputs" already exist as recv_x; nothing to stage.
        return None

    def combine(self, p, h):
        combined_x, _, _ = self.buffer.combine(h.recv_x, h.handle, topk_weights=h.recv_topk_weights)
        return combined_x

    def expected(self, p, h):
        ranks_per_token = h.is_token_in_rank.sum(dim=1, keepdim=True).clamp(min=1).float()
        return p.x.float() * ranks_per_token, p.T

    def recv_tokens(self, h):
        return int(h.recv_x.shape[0])

    def finalize(self, rc):
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
