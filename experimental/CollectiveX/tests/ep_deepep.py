#!/usr/bin/env python3
"""CollectiveX EP backend adapter — DeepEP (NVIDIA), normal mode.

Ports the validated dispatch/combine sequence from the old run_deepep.py into the
ep_harness Backend protocol. The harness owns the token sweep + separated timing;
this file owns only DeepEP's API calls and its correctness reference.

  !!! DeepEP's Python API is VERSION-SENSITIVE (V2 moved NVSHMEM->NCCL and unified
  the APIs). The dispatch/combine block follows the documented normal-mode
  intranode API; validate against the deep_ep commit actually built at job time
  (rebuild-deepep) and recorded in provenance.

Correctness (per DeepEP's tests/legacy/test_intranode.py): a pure dispatch->combine
round trip with no expert compute reconstructs x only after dividing by the number
of ranks each token was sent to, i.e. combined_x / is_token_in_rank.sum(dim=1).
So the harness expects combined ≈ x * ranks_per_token.
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist

try:
    from deep_ep import Buffer  # type: ignore
except Exception as exc:  # pragma: no cover - needs the built DeepEP
    print("ERROR: deep_ep import failed — DeepEP must be built at job setup "
          f"(rebuild-deepep). {exc!r}", file=sys.stderr)
    raise


class DeepEPBackend:
    name = "deepep"
    mode = "normal"
    measurement_contract = "deepep-normal-v1"
    combine_needs_redispatch = False  # DeepEP combine reuses the handle (its own bench does too)

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.group = dist.group.WORLD
        # Intranode normal mode: NVLink buffer only (no RDMA for single node). Size
        # to hold the largest sweep point's routed traffic. Prefill's large-T points
        # (up to 4096 tok/rank) need a bigger buffer than decode — validated on
        # B200 (EP8) and GB200 (EP4) at 4 GiB through T=4096; decode is fine at 2 GiB.
        # Override with CX_DEEPEP_NVL_BYTES.
        _default_nvl = (4 if args.phase == "prefill" else 2) * 1024 * 1024 * 1024
        num_nvl_bytes = int(os.environ.get("CX_DEEPEP_NVL_BYTES", str(_default_nvl)))
        self.buffer = Buffer(self.group, num_nvl_bytes, 0)
        try:
            Buffer.set_num_sms(args.num_comm_sms)
        except Exception as exc:  # pragma: no cover - version dependent
            if rank == 0:
                print(f"WARN: could not set num_sms={args.num_comm_sms}: {exc!r}", file=sys.stderr)
        self.backend_provenance = {
            "deepep_commit": os.environ.get("DEEPEP_COMMIT", "unknown"),
            "num_nvl_bytes": num_nvl_bytes,
            "num_comm_sms": args.num_comm_sms,
        }
        if args.dispatch_dtype == "fp8" and rank == 0:
            print("WARN: deepep fp8 dispatch payload not wired for the exact-reconstruction "
                  "gate yet; using bf16. (provenance reflects bf16.)", file=sys.stderr)
            args.dispatch_dtype = "bf16"

    def buffer_cap(self, args):
        return None  # NVLink buffer is large; no hard per-T ceiling like MoRI's heap

    def make_problem(self, T):
        a = self.args
        H, topk, E = a.hidden, a.topk, a.experts
        x = torch.randn((T, H), dtype=torch.bfloat16, device=self.device)
        if a.routing == "zipf":
            probs = (1.0 / torch.arange(1, E + 1, device=self.device).float())
            topk_idx = torch.multinomial(probs.expand(T, E), topk, replacement=False).to(torch.int64)
        else:  # balanced / uniform: topk distinct experts drawn uniformly per token
            topk_idx = torch.stack([
                torch.randperm(E, device=self.device)[:topk] for _ in range(T)
            ]).to(torch.int64)
        topk_weights = torch.softmax(
            torch.randn((T, topk), device=self.device, dtype=torch.float32), dim=-1)
        return types.SimpleNamespace(T=T, x=x, topk_idx=topk_idx, topk_weights=topk_weights)

    def dispatch(self, p):
        # ===================== DeepEP normal-mode dispatch =====================
        (num_tokens_per_rank, _, num_tokens_per_expert,
         is_token_in_rank, _) = self.buffer.get_dispatch_layout(p.topk_idx, self.args.experts)
        recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = self.buffer.dispatch(
            p.x, topk_idx=p.topk_idx, topk_weights=p.topk_weights,
            num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert)
        # =======================================================================
        return types.SimpleNamespace(
            recv_x=recv_x, recv_topk_weights=recv_topk_weights, handle=handle,
            is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert)

    def stage(self, p, h):
        # DeepEP combine consumes recv_x directly (no separate registered buffer to
        # stage into) — the "expert outputs" are recv_x itself for a pure round trip.
        return None

    def combine(self, p, h):
        combined_x, _, _ = self.buffer.combine(h.recv_x, h.handle, topk_weights=h.recv_topk_weights)
        return combined_x

    def expected(self, p, h):
        # combined ≈ x * (#ranks each token was dispatched to)
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
