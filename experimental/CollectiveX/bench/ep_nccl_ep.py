#!/usr/bin/env python3
"""Vendor-neutral NCCL/RCCL all-to-all EP backend (the collective-library baseline).

Unlike DeepEP/MoRI (fused dispatch/combine kernels), this is the reference
expert-parallel path built directly on the framework collective: a routing-aware
**variable-split** all-to-all. It runs on any RCCL (AMD) or NCCL (NVIDIA) build in
the image — no source pin, no device buffer, no build step.

It is emphatically NOT a plain equal-split ``all_to_all_single``: EP dispatch is a
data-dependent, uneven scatter. This adapter implements the token-rank contract
the oracle expects (``oracle_layout="token-rank"``):

  * Dispatch sends each token ONCE to each *distinct* rank that owns one of its
    top-k experts, carrying that token's top-k row masked to the experts local to
    the destination (global expert id, else -1; weight, else 0). Per-destination
    counts are data-dependent, so the payload/metadata move via
    ``all_to_all_single`` with per-rank input/output split sizes (all-to-all-v).
  * Combine is the reverse variable all-to-all followed by an unweighted rank-sum
    (``index_add_``) back into each source token — matching
    ``combine_weight_semantics="unweighted-rank-sum"``, since the staged payload
    already carries each rank's weighted expert-partial-sum.

Split sizes are inherently host-visible for the collective API, so dispatch reads
the exchanged receive counts to host — the honest cost of the unfused path.
"""
from __future__ import annotations

import types

import torch
import torch.distributed as dist

from ep_backend import EPBackend


class NcclEPBackend(EPBackend):
    name = "nccl-ep"
    SUPPORTED_MODES = ("normal",)
    stage_device_work = False
    combine_needs_redispatch = False
    combine_weight_semantics = "unweighted-rank-sum"
    oracle_layout = "token-rank"
    payload_unit = "token-rank"

    def __init__(self, args, rank, world_size, local_rank, device):
        # normal-mode only (base SUPPORTED_MODES enforces it); vendor-neutral.
        super().__init__(args, rank, world_size, local_rank, device)
        self.group = dist.group.WORLD

    def create_buffer(self, spec):
        # No preallocated communicator buffer: the collective all-to-all sizes each
        # transfer from the routing trace. Record the sizing the driver expects.
        self.max_tokens = spec.max_tokens_per_rank
        self.experts_per_rank = self.args.experts // self.world_size
        self.hidden = self.args.hidden

    # ---- dispatch: token-rank variable-split all-to-all ------------------------------

    def _plan(self, idx):
        """Build the (token, dest-rank) send plan for this rank's routing rows.

        Returns send-order tensors grouped by destination rank plus the split
        sizes both directions. ``send_token`` maps each sent pair back to its
        local source token for the combine rank-sum.
        """
        T = idx.shape[0]
        epr = self.experts_per_rank
        dest = torch.div(idx, epr, rounding_mode="floor")  # [T, K] dest rank per slot
        onehot = torch.zeros(
            (T, self.world_size), dtype=torch.bool, device=idx.device
        )
        rows = torch.arange(T, device=idx.device).unsqueeze(1).expand_as(dest)
        onehot[rows, dest] = True  # token t -> rank r if any of t's experts live on r
        pair_t, pair_r = onehot.nonzero(as_tuple=True)  # [P], [P]
        order = torch.argsort(pair_r, stable=True)  # contiguous per-destination groups
        send_token = pair_t[order].contiguous()
        send_rank = pair_r[order]
        send_counts = torch.bincount(
            send_rank, minlength=self.world_size
        ).to(torch.int64)
        return send_token, send_rank, send_counts

    def dispatch(self, p):
        idx = p.topk_idx.to(torch.int64)
        weights = p.topk_weights.to(torch.float32)
        K = idx.shape[1]
        epr, dev = self.experts_per_rank, self.device
        send_token, send_rank, send_counts = self._plan(idx)
        P = int(send_token.shape[0])

        dest_of_token = torch.div(idx, epr, rounding_mode="floor")  # [T,K]
        keep = dest_of_token[send_token] == send_rank.unsqueeze(1)  # [P,K] slot on dest
        send_eid = torch.where(keep, idx[send_token], idx.new_full((1,), -1))
        send_w = torch.where(keep, weights[send_token], weights.new_zeros((1,)))
        send_payload = p.dispatch_x.index_select(0, send_token).contiguous()

        # Exchange per-rank counts, then the payload + routing metadata with splits.
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        in_splits = send_counts.tolist()
        out_splits = recv_counts.tolist()
        total = int(sum(out_splits))

        recv_payload = torch.empty((total, self.hidden), dtype=p.dispatch_x.dtype, device=dev)
        recv_eid = torch.empty((total, K), dtype=torch.int64, device=dev)
        recv_w = torch.empty((total, K), dtype=torch.float32, device=dev)
        dist.all_to_all_single(recv_payload, send_payload, out_splits, in_splits, group=self.group)
        dist.all_to_all_single(recv_eid, send_eid.contiguous(), out_splits, in_splits, group=self.group)
        dist.all_to_all_single(recv_w, send_w.contiguous(), out_splits, in_splits, group=self.group)

        valid = recv_eid >= 0
        local_index = (recv_eid[valid] - self.rank * epr).to(torch.int64)
        local_expert_counts = torch.bincount(local_index, minlength=epr)

        return types.SimpleNamespace(
            recv_payload=recv_payload,
            recv_eid=recv_eid,
            recv_w=recv_w,
            local_expert_counts=local_expert_counts,
            send_token=send_token,
            in_splits=in_splits,
            out_splits=out_splits,
            P=P,
            T=int(idx.shape[0]),
            total=total,
            combine_input=None,
        )

    def stage(self, p, h):
        # BF16 path: the received payload is already the semantic combine input.
        h.combine_input = h.recv_payload

    def _reverse_combine(self, h, payload):
        """Reverse variable all-to-all + unweighted rank-sum into source tokens."""
        back = torch.empty((h.P, self.hidden), dtype=payload.dtype, device=self.device)
        # Swap the split directions: receive-order groups go back to send order.
        dist.all_to_all_single(
            back, payload.contiguous(), h.in_splits, h.out_splits, group=self.group
        )
        out = torch.zeros((h.T, self.hidden), dtype=torch.float32, device=self.device)
        if h.P:
            out.index_add_(0, h.send_token, back.to(torch.float32))
        return out.to(payload.dtype)

    def combine(self, p, h):
        return self._reverse_combine(h, h.combine_input)

    def recv_tokens(self, h):
        return h.total

    def inspect_dispatch(self, p, h):
        return types.SimpleNamespace(
            payload=h.recv_payload,
            expert_ids=h.recv_eid,
            weights=h.recv_w,
            local_expert_counts=h.local_expert_counts,
            ordering_contract="nccl-ep-token-rank-v1",
        )

    def combine_transformed(self, p, h, transformed):
        return self._reverse_combine(h, transformed)

    def finalize(self, rc):
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            return 1
        return rc
