"""CollectiveX NCCL all-to-all expert-parallel reference backend.

The canonical "token-shuffle" EP built on torch.distributed's NCCL ``all_to_all_single``. Like the
DeepEP-family APIs, dispatch sends one hidden-state copy to each distinct destination rank, even when
multiple selected experts live on that rank. Combine reverses the shuffle and sums those rank copies.

Why this exists alongside DeepEP/UCCL/MoRI: it is the portable collective reference baseline for the
same rank-deduplicated payload and routing metadata. It keeps the library comparison anchored to the
platform collective stack without claiming the custom fused kernels use the same transport algorithm.

Scope: BF16, normal mode, layout-and-dispatch-v1. The timed dispatch includes layout, count exchange,
payload, rank-masked expert indices, gate weights, and source-token metadata; combine returns only
the activation payload. RCCL exposes the same API. The v1 AMD matrix uses this backend at EP8 and EP16.
"""

import os
import re
import types

import torch
import torch.distributed as dist
import contracts
import ep_precision


def _runtime_collective(args, torch_module) -> tuple[str, str]:
    expected = "rccl" if torch_module.version.hip else "nccl"
    fingerprint = getattr(args, "runtime_fingerprint", None)
    collective = fingerprint.get("collective_library") if isinstance(fingerprint, dict) else None
    if (
        not isinstance(collective, dict)
        or collective.get("kind") != expected
        or not isinstance(collective.get("version"), str)
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", collective["version"])
    ):
        raise RuntimeError("loaded collective runtime identity is unavailable")
    return expected, collective["version"]


class NCCLBackend:
    name = "nccl-ep"
    stage_device_work = False
    combine_needs_redispatch = False  # dispatch saves the permutation + splits
    combine_weight_semantics = "unweighted-rank-sum"

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.experts = args.experts
        self.mode = getattr(args, "mode", "normal")
        if self.mode != "normal":
            raise ep_precision.PrecisionError("NCCL/RCCL EP supports normal mode only")
        self.precision_profile_id, self.communication_precision = (
            ep_precision.resolve_precision(
                args,
                backend=self.name,
                mode=self.mode,
                supported_profiles={"d-bf16.c-bf16"},
            )
        )
        if args.experts % world_size:
            raise ValueError(f"experts({args.experts}) must divide world_size({world_size})")
        self.experts_per_rank = args.experts // world_size
        self.tolerance = 5e-2  # bf16 round-trip
        _library, _version = _runtime_collective(args, torch)
        if args.scale_out_transport:
            hcas = os.environ.get("NCCL_IB_HCA", "")
            if os.environ.get("NCCL_NET") != "IB":
                raise RuntimeError("scale-out collective network mode is not IB")
            if not re.fullmatch(
                r"=[A-Za-z][A-Za-z0-9_.-]{0,31}(?::[1-9][0-9]*)?"
                r"(?:,[A-Za-z][A-Za-z0-9_.-]{0,31}(?::[1-9][0-9]*)?)*",
                hcas,
            ):
                raise RuntimeError("scale-out collective HCA allowlist is invalid")
        self.kernel_generation = contracts.collective_kernel_generation(_library)
        self.backend_provenance = {
            "backend": f"{_library}-all2all",
            "backend_lineage": _library,
            "collective_library": _library,
            "nccl_version": _version,
            "transport": f"{_library}-all_to_all_single",
            "resource_mode": "fixed-profile",
            "num_sms": None,
            "device_sms": torch.cuda.get_device_properties(device).multi_processor_count,
            "tuned_source": "nccl-collective",
            "reference_semantics": "rank-deduplicated-payload-plus-routing-metadata-v2",
            "routing_metadata": "expert-index-gate-weight-source-token",
            "dispatch_dtype": "bf16",
            "combine_dtype": "bf16",
        }

    def buffer_cap(self, args):
        return None  # no fixed pre-allocated buffer; all-to-all sizes itself per step

    def make_problem(self, T, idx, weights, x):
        encoding = ep_precision.encode_dispatch(
            torch, x, self.communication_precision
        )
        # idx[T,topk] int64, weights[T,topk] f32, x[T,hidden] bf16 — the shared routing-trace slice.
        return types.SimpleNamespace(
            T=T,
            x=x,
            oracle_x=encoding.semantic,
            dispatch_precision_evidence=encoding.evidence,
            topk_idx=idx.to(torch.int64),
            topk_weights=weights.to(torch.float32),
            layout=None,
        )

    def dispatch(self, p):
        ws = self.world_size
        x = p.x  # [T, H] bf16
        idx = p.topk_idx  # [T, topk]
        T, H = int(x.shape[0]), int(x.shape[1])
        dev = x.device
        # DeepEP dispatches one token per destination rank, not one copy per expert. Build the same
        # rank-deduplicated routing map so NCCL traffic and combine semantics are comparable.
        destinations = (idx // self.experts_per_rank).clamp_(0, ws - 1)
        present = torch.zeros((T, ws), dtype=torch.bool, device=dev)
        present.scatter_(1, destinations, True)
        flat_token, flat_dest = present.nonzero(as_tuple=True)
        # Group rank copies by destination (stable -> deterministic, invertible permutation).
        order = torch.argsort(flat_dest, stable=True)
        ordered_token = flat_token.index_select(0, order)
        ordered_dest = flat_dest.index_select(0, order)
        send_counts = torch.bincount(flat_dest, minlength=ws)  # [ws]
        send_x = x.index_select(0, ordered_token).contiguous()
        send_topk_idx = idx.index_select(0, ordered_token).contiguous()
        expert_start = ordered_dest.unsqueeze(1) * self.experts_per_rank
        local_mask = ((send_topk_idx >= expert_start)
                      & (send_topk_idx < expert_start + self.experts_per_rank))
        send_topk_idx = torch.where(
            local_mask, send_topk_idx - expert_start, torch.full_like(send_topk_idx, -1)
        )
        send_topk_weights = p.topk_weights.index_select(0, ordered_token).contiguous()
        send_topk_weights.masked_fill_(~local_mask, 0)
        send_src_metadata = (ordered_token.to(torch.int64) | (self.rank << 32)).contiguous()
        # Exchange per-rank counts so every rank can size its receive buffer.
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        sc = send_counts.tolist()
        rc = recv_counts.tolist()
        total_recv = int(sum(rc))
        recv_x = torch.empty((total_recv, H), dtype=x.dtype, device=dev)
        recv_topk_idx = torch.empty((total_recv, int(idx.shape[1])), dtype=idx.dtype, device=dev)
        recv_topk_weights = torch.empty((total_recv, int(idx.shape[1])),
                                        dtype=p.topk_weights.dtype, device=dev)
        recv_src_metadata = torch.empty((total_recv,), dtype=torch.int64, device=dev)
        # Dispatch the uneven per-rank splits over the configured collective transport.
        dist.all_to_all_single(recv_x, send_x, rc, sc)
        dist.all_to_all_single(recv_topk_idx, send_topk_idx, rc, sc)
        dist.all_to_all_single(recv_topk_weights, send_topk_weights, rc, sc)
        dist.all_to_all_single(recv_src_metadata, send_src_metadata, rc, sc)
        return types.SimpleNamespace(
            recv_x=recv_x, combine_input=None, order=order, flat_token=flat_token,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights, recv_src_rank=recv_src_metadata >> 32,
            recv_src_token=recv_src_metadata & ((1 << 32) - 1), send_counts=sc, recv_counts=rc,
            T=T, H=H, topk=int(idx.shape[1]), total_recv=total_recv)

    def stage(self, p, h):
        # No expert compute: the expert "output" is the received tokens as-is (the round-trip identity).
        h.combine_input = h.recv_x
        return None

    def combine(self, p, h):
        # Reverse all-to-all: ship expert outputs back to their origin ranks (swap the split lists).
        send_back = torch.empty((int(h.order.shape[0]), h.H), dtype=h.combine_input.dtype,
                                device=h.combine_input.device)
        dist.all_to_all_single(send_back, h.combine_input.contiguous(),
                               h.send_counts, h.recv_counts)
        # send_back is in send (sorted) order; invert the argsort to token-copy order.
        copies = torch.empty_like(send_back)
        copies[h.order] = send_back
        # Sum one copy per destination rank under this reference's explicit unweighted contract.
        out = torch.zeros((h.T, h.H), dtype=torch.float32, device=send_back.device)
        out.index_add_(0, h.flat_token, copies.float())
        return out.to(p.x.dtype)

    def inspect_dispatch(self, p, h):
        valid = h.recv_topk_idx >= 0
        expert_ids = torch.where(
            valid,
            h.recv_topk_idx + self.rank * self.experts_per_rank,
            h.recv_topk_idx,
        )
        return types.SimpleNamespace(
            payload=h.recv_x,
            expert_ids=expert_ids,
            weights=h.recv_topk_weights.masked_fill(~valid, 0),
            local_expert_counts=torch.bincount(
                h.recv_topk_idx[valid], minlength=self.experts_per_rank
            ),
            ordering_contract="source-rank-major-stable-v1",
        )

    def combine_transformed(self, p, h, transformed):
        h.combine_input = transformed.to(h.recv_x.dtype)
        return self.combine(p, h)

    def recv_tokens(self, h):
        return int(h.total_recv)

    def oracle_dispatch_payload(self, payload):
        return payload

    def precision_evidence(self, problem, view=None):
        return ep_precision.precision_evidence(
            torch,
            profile_id=self.precision_profile_id,
            profile=self.communication_precision,
            problem=problem,
            view=view,
        )

    def finalize(self, rc):
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
