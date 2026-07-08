#!/usr/bin/env python3
"""Shared DeepEP-API dispatch/combine surface for the DeepEP and UCCL adapters.

UCCL's ``uccl_deepep.Buffer`` is a drop-in clone of DeepEP's ``deep_ep.Buffer`` low-latency and
normal API, so both adapters run byte-identical mode handling, dispatch/combine, and expert-packed
inspection. That shared operation lives here; each concrete backend keeps only what is genuinely
vendor-specific: its native buffer import, ``create_buffer`` provenance, and process teardown.

This base is deliberately free of any ``deep_ep``/``uccl`` import. The UCCL benchmark image installs
uccl WITHOUT deep_ep, so importing ``ep_uccl`` must never transitively require ``deep_ep`` — an
inherited method resolves module globals from where it is *defined*, so keeping this file
vendor-agnostic is what makes the shared base safe for both images.

Communication is fixed BF16: dispatch and combine move BF16 activations, so the native
``use_fp8``/``use_logfmt`` controls are always driven off and the received buffer is the
semantic payload directly.
"""
from __future__ import annotations

import types

import torch
import torch.distributed as dist
from ep_backend import EPBackend


class DeepEPFamilyBackend(EPBackend):
    # Abstract intermediate: never instantiated or registered (capability.BACKENDS is an
    # explicit dict, and nothing enumerates EPBackend subclasses). The non-empty name only
    # satisfies EPBackend.__init_subclass__; concrete subclasses override it. create_buffer
    # stays abstract here, so this class cannot itself be constructed.
    name = "deepep-family"
    _vendor = "DeepEP"
    SUPPORTED_MODES = ("normal", "low-latency")
    stage_device_work = False
    combine_needs_redispatch = False
    # DeepEP reduces activations and top-k weights independently. The activation
    # tensor must therefore carry the complete local weighted expert sum.
    combine_weight_semantics = "unweighted-rank-sum"
    oracle_layout = "token-rank"
    payload_unit = "token-rank"

    def __init__(self, args, rank, world_size, local_rank, device):
        # Base validates mode against SUPPORTED_MODES (normal / low-latency).
        super().__init__(args, rank, world_size, local_rank, device)
        self.group = dist.group.WORLD
        # Low-latency flips the contract flags and fixes the per-rank cap so
        # buffer_cap can report it to make_inputs before create_buffer runs.
        if self.mode == "low-latency":
            if args.phase != "decode":
                raise ValueError(
                    f"{self._vendor} low-latency mode only supports the decode ladder"
                )
            if args.experts % world_size:
                raise ValueError(
                    f"{self._vendor} low-latency experts must divide the EP group"
                )
            self.combine_needs_redispatch = True
            self.combine_weight_semantics = "gate-weighted-sum"
            self.oracle_layout = "expert-packed"
            self.payload_unit = "token-expert"
            self.max_tokens_per_rank = 128

    def buffer_cap(self, args):
        return self.max_tokens_per_rank if self.mode == "low-latency" else None

    def dispatch(self, p):
        if self.mode == "low-latency":
            recv_x, recv_counts, handle, _, _ = self.buffer.low_latency_dispatch(
                p.x,
                p.topk_idx,
                self.max_tokens_per_rank,
                self.args.experts,
                use_fp8=False,  # BF16 communication path.
                async_finish=False,
                return_recv_hook=False,
            )
            return types.SimpleNamespace(
                recv_x=recv_x,
                recv_counts=recv_counts,
                handle=handle,
            )
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = self.buffer.get_dispatch_layout(p.topk_idx, self.args.experts)
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, _ = self.buffer.dispatch(
            p.dispatch_x,
            topk_idx=p.topk_idx,
            topk_weights=p.topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            async_finish=False,
        )
        return types.SimpleNamespace(
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            recv_counts=recv_counts,
            handle=handle,
        )

    def stage(self, p, h):
        # BF16: the received buffer is already the semantic payload to combine.
        h.combine_input = h.recv_x

    def combine(self, p, h):
        if self.mode == "low-latency":
            combined_x, _, _ = self.buffer.low_latency_combine(
                h.combine_input,
                p.topk_idx,
                p.topk_weights,
                h.handle,
                use_logfmt=False,  # BF16 communication path.
                async_finish=False,
                return_recv_hook=False,
            )
            return combined_x
        combined_x, _, _ = self.buffer.combine(
            h.combine_input, h.handle, async_finish=False
        )
        return combined_x

    def inspect_dispatch(self, p, h):
        valid = h.recv_topk_idx >= 0
        expert_ids = torch.where(
            valid,
            h.recv_topk_idx + self.rank * (self.args.experts // self.world_size),
            h.recv_topk_idx,
        )
        return types.SimpleNamespace(
            payload=h.recv_x,
            expert_ids=expert_ids,
            weights=h.recv_topk_weights.masked_fill(~valid, 0),
            local_expert_counts=torch.tensor(h.recv_counts, device=self.device, dtype=torch.int64),
            ordering_contract="source-rank-major-stable-v1",
        )

    def inspect_expert_dispatch(self, p, h):
        if self.mode != "low-latency":
            raise RuntimeError("expert-packed inspection requires low-latency mode")
        p.recv_counts = tuple(int(value) for value in h.recv_counts.tolist())
        return types.SimpleNamespace(
            payload=h.recv_x,
            local_expert_counts=h.recv_counts,
            source_info=h.handle[0],
            layout_range=h.handle[1],
        )

    def combine_transformed(self, p, h, transformed):
        if self.mode == "low-latency":
            packed = torch.zeros(
                h.recv_x.shape,
                dtype=torch.bfloat16,
                device=h.recv_x.device,
            )
            packed[h.oracle_local_expert_slots, h.oracle_packed_positions] = transformed.to(
                packed.dtype
            )
            combined, _, _ = self.buffer.low_latency_combine(
                packed,
                p.topk_idx,
                p.topk_weights,
                h.handle,
                use_logfmt=False,  # BF16 communication path.
                async_finish=False,
                return_recv_hook=False,
            )
            return combined
        combined, _, _ = self.buffer.combine(
            transformed.to(h.recv_x.dtype), h.handle, async_finish=False
        )
        return combined

    def recv_tokens(self, h):
        if self.mode == "low-latency":
            return int(h.recv_counts.to(torch.int64).sum().item())
        return int(h.recv_x.shape[0])
