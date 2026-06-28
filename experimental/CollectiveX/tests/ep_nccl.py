"""CollectiveX — NCCL all-to-all expert-parallel backend (cross-node EP, goal 182).

The canonical "token-shuffle" EP built on torch.distributed's NCCL ``all_to_all_single``: dispatch
routes each token-copy to the rank that owns its expert via an uneven all-to-all; combine reverses it
and weighted-sums the top-k copies back into each origin token. With no expert compute the round trip
reconstructs ``x * sum(topk_weights)`` per token.

Why this exists alongside DeepEP/UCCL/MoRI: those use custom one-sided RDMA (DeepEP/NVSHMEM, UCCL's own
ibv verbs, MoRI ionic_rdma). Cross-node, UCCL's ``ibv_reg_mr`` failed with EINVAL -> heap corruption ->
SIGSEGV (run 28326528672) because the cluster's IB HCAs / container lack the GPUDirect-RDMA peer-memory
that custom verbs registration needs. NCCL's collective transport, by contrast, negotiates IB and
*gracefully host-stages* when GPUDirect RDMA is unavailable — so an EP built purely on NCCL collectives
runs cross-node on the same fabric. It is also the reference baseline the fused EP kernels improve upon,
so a same-shape NCCL number is a meaningful comparison point, not just a fallback.

Scope: BF16, normal mode, layout-and-dispatch-v1 (the timed window includes the layout/argsort + both
all-to-alls). RCCL exposes the identical API, so this backend also covers AMD (rccl) cross-node EP.
"""
import os
import types

import torch
import torch.distributed as dist


class NCCLBackend:
    name = "nccl-ep"
    combine_needs_redispatch = False   # dispatch saves the permutation + splits; combine reuses them
    wants_warm_burst = False
    # Pure-collective token shuffle: bf16 only (no fp8 dispatch path), normal mode, single contract.
    SUPPORTED_PRECISIONS = {"bf16"}
    SUPPORTED_MODES = {"normal"}
    SUPPORTED_CONTRACTS = {"layout-and-dispatch-v1"}

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.experts = args.experts
        assert args.experts % world_size == 0, \
            f"NCCL EP needs experts({args.experts}) divisible by world_size({world_size})"
        self.experts_per_rank = args.experts // world_size
        assert args.dispatch_dtype in self.SUPPORTED_PRECISIONS and args.mode in self.SUPPORTED_MODES, \
            f"NCCL EP supports precisions={sorted(self.SUPPORTED_PRECISIONS)} modes={sorted(self.SUPPORTED_MODES)} only"
        self.tolerance = 5e-2   # bf16 round-trip
        try:
            _nccl = ".".join(str(v) for v in torch.cuda.nccl.version())
        except Exception:
            _nccl = "unknown"
        self.backend_provenance = {
            "backend": "nccl-all2all",
            "nccl_version": _nccl,
            "transport": "nccl-all_to_all_single",
            "resource_mode": args.resource_mode,
            "num_sms": None,
            "device_sms": torch.cuda.get_device_properties(device).multi_processor_count,
            "tuned_source": "nccl-collective",
        }

    def buffer_cap(self, args):
        return None   # no fixed pre-allocated buffer; all-to-all sizes itself per step

    def make_problem(self, T, idx, weights, x):
        # idx[T,topk] int64, weights[T,topk] f32, x[T,hidden] bf16 — the shared routing-trace slice.
        return types.SimpleNamespace(T=T, x=x, topk_idx=idx.to(torch.int64),
                                     topk_weights=weights.to(torch.float32), layout=None)

    def dispatch(self, p):
        ws = self.world_size
        x = p.x                                   # [T, H] bf16
        idx = p.topk_idx                          # [T, topk]
        T, H = int(x.shape[0]), int(x.shape[1])
        topk = int(idx.shape[1])
        dev = x.device
        # Flatten the T*topk token-copies; each goes to the rank owning its expert.
        flat_expert = idx.reshape(-1)                                       # [T*topk]
        flat_dest = (flat_expert // self.experts_per_rank).to(torch.int64)  # dest rank per copy
        flat_token = torch.arange(T, device=dev, dtype=torch.int64).repeat_interleave(topk)
        # Group copies by destination rank (stable -> deterministic, invertible permutation).
        order = torch.argsort(flat_dest, stable=True)
        send_counts = torch.bincount(flat_dest, minlength=ws)               # [ws]
        send_x = x.index_select(0, flat_token.index_select(0, order)).contiguous()  # [T*topk, H], send order
        # Exchange per-rank counts so every rank can size its receive buffer.
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        sc = send_counts.tolist()
        rc = recv_counts.tolist()
        total_recv = int(sum(rc))
        recv_x = torch.empty((total_recv, H), dtype=x.dtype, device=dev)
        # The dispatch all-to-all (uneven splits). NCCL routes internode over IB (host-staged if no
        # GPUDirect RDMA) — this is the line that runs cross-node where UCCL's ibv_reg_mr fails.
        dist.all_to_all_single(recv_x, send_x, rc, sc)
        return types.SimpleNamespace(recv_x=recv_x, combine_input=None, order=order,
                                     flat_token=flat_token, flat_w=p.topk_weights.reshape(-1),
                                     send_counts=sc, recv_counts=rc, T=T, H=H, total_recv=total_recv)

    def stage(self, p, h):
        # No expert compute: the expert "output" is the received tokens as-is (the round-trip identity).
        h.combine_input = h.recv_x
        return None

    def combine(self, p, h):
        # Reverse all-to-all: ship expert outputs back to their origin ranks (swap the split lists).
        send_back = torch.empty((int(h.order.shape[0]), h.H), dtype=h.combine_input.dtype,
                                device=h.combine_input.device)
        dist.all_to_all_single(send_back, h.combine_input.contiguous(), h.send_counts, h.recv_counts)
        # send_back is in send (sorted) order; invert the argsort to token-copy order.
        copies = torch.empty_like(send_back)
        copies[h.order] = send_back
        # Weighted reduce of each token's top-k copies into [T, H] (accumulate in fp32 for stability).
        out = torch.zeros((h.T, h.H), dtype=torch.float32, device=send_back.device)
        out.index_add_(0, h.flat_token, copies.float() * h.flat_w.unsqueeze(1))
        return out.to(p.x.dtype)

    def recv_tokens(self, h):
        return int(h.total_recv)

    def expected(self, p, h):
        # Round trip with identity expert: out[t] = sum_k w[t,k] * x[t] = x[t] * sum_k w[t,k].
        wsum = p.topk_weights.sum(dim=1, keepdim=True).float()
        return p.x.float() * wsum, p.T

    def finalize(self, rc):
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
