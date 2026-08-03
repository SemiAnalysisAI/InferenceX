#!/usr/bin/env python3
"""FlashInfer one-sided NVLink EP: the transport a GB200/GB300 deployment actually runs.

The GB SKUs are an NVLink-only domain, and vLLM selects a different all-to-all there than it
does on an RDMA fabric: `--all2all-backend deepep_v2` for RDMA, `flashinfer_nvlink_one_sided`
for NVLink. Benchmarking the GB racks with DeepEP V2 and NCCL EP therefore measured transports
no deployment would choose on that hardware. This adapter closes that gap.

ONE-SIDED, not two-sided. The sibling `flashinfer_nvlink_two_sided` backend is deliberately not
wired: it produces gibberish output on GB200/arm64 (vllm#39722). One-sided means the initiator
writes straight into the target's workspace and then sets a flag — on MNNVL, where peer memory
is directly addressable across the 72-GPU domain, that degenerates to ordinary stores over the
fabric with no rendezvous, which is exactly why it wins on this topology.

Upstream surface (flashinfer.comm.trtllm_moe_alltoall):

    MoeAlltoAll(mapping, max_num_tokens, top_k, num_experts,
                workspace_size_per_rank=..., mnnvl_config=...)
    .dispatch(token_selected_experts, input_payloads, runtime_max_tokens_per_rank)
        -> list of [ep_size, runtime_max_tokens_per_rank, *] workspace-backed receives
    .combine(payload, runtime_max_tokens_per_rank, output=...)
        -> [local_num_tokens, elements_per_token]

Three properties of that API shape this adapter:

  * Strict phase pairing. `dispatch` asserts "called twice without combine", `combine` asserts
    the phase is "dispatched", and the internal state resets after each combine. So a timed
    combine needs a fresh dispatch and a timed dispatch needs a draining combine — the same
    contract MoRI and the DeepEP V2 low-latency path already declare.
  * A PADDED receive. Dispatch returns `[ep_size, runtime_max_tokens_per_rank, hidden]`, not a
    compact buffer, so every oracle view must read valid slots only. Padding is untouched
    workspace memory; reading it would feed garbage to the correctness gate.
  * `runtime_max_tokens_per_rank <= max_num_tokens`, so the workspace is sized once from the
    ladder maximum and each call passes the current rung.

`normal` mode only. FlashInfer exposes one one-sided A2A kernel family, not a separate
decode-optimized one, so there is no honest `low-latency` cell to add: an `ll_backends` row
here would re-measure the same kernel under a mode that promises a different one. The decode
phase is still covered — `normal` runs the full decode and prefill ladders.
"""
from __future__ import annotations

import re
import types

import torch

from ep_backend import EPBackend

# Stamped by the kernel into the expert-id payload of every receive slot it did not fill.
# -1 is safe: real ids are [0, num_experts).
_INVALID_EXPERT = -1

# 0.6.16 ("Port the TensorRT-LLM one-sided A2A optimizations") rewrote the combine
# accumulator from the payload dtype to FP32 with a single narrowing store. Earlier wheels
# round at every level of the top-k reduction tree, which is a few BF16 ulps the plain
# FP32 expectation does not carry, so the oracle needs to know which kernel it is facing.
_COMBINE_FP32_SINCE = (0, 6, 16)


class FlashInferEPBackend(EPBackend):
    name = "flashinfer-ep"
    maturity = "production"  # vLLM --all2all-backend flashinfer_nvlink_one_sided
    # One kernel family; see the module docstring for why there is no low-latency mode.
    SUPPORTED_MODES = ("normal",)
    # BF16 first. The combine side accepts fp8_e4m3fn/uint8 output dtypes and a
    # use_low_precision accumulate, but dispatch FP8 needs the scale payload plumbed as a
    # second input_payload and validated against the oracle's cast round-trip; not this pass.
    SUPPORTED_PRECISIONS = ("bf16",)
    kernel_generation = "flashinfer-mnnvl-one-sided"
    # stage() now copies the received payload into the workspace combine region.
    stage_device_work = True
    # The kernel scatters expert outputs back to the supplying rank; it does not multiply by
    # the routing weights (those ride along as a caller payload, and vLLM applies them in the
    # MoE layer, not in the A2A). Verified against the oracle during bring-up.
    combine_weight_semantics = "unweighted-rank-sum"
    # Set per wheel in create_buffer; see _COMBINE_FP32_SINCE.
    combine_reduction = "topk-slot-tree"
    # Forced by the phase asserts described in the module docstring.
    combine_needs_redispatch = True
    dispatch_needs_combine_cleanup = True
    combine_input_attr = "combine_input"

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
        self._a2a = None
        self._max_tokens = None
        self.experts_per_rank = args.experts // world_size

    # ---- setup -------------------------------------------------------------------------------

    def _topk_idx_dtype(self):
        """int32 — what the kernel reads the routing plane as.

        Declaring it here means `make_problem` casts once, untimed, instead of `dispatch`
        casting on every call inside the measured window. `topk_weights` is already FP32
        from `make_problem`, so both payloads reach the kernel with no conversion.
        """
        return torch.int32

    def buffer_cap(self, args):
        # The workspace is sized from the ladder maximum rather than a fixed slot budget, so
        # there is no cap to clamp the ladder against.
        return None

    def create_buffer(self, spec):
        """Build the one MoeAlltoAll for this group, sized to the ladder maximum.

        The communicator handed to MnnvlConfig must span exactly the EP group: the kernel
        asserts `workspace.size(0) == moe_ep_size`, so a wider group silently mis-sizes it.
        """
        import flashinfer
        from flashinfer.comm import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig
        from flashinfer.comm.trtllm_moe_alltoall import (
            MoeAlltoAll,
            moe_a2a_get_workspace_size_per_rank,
        )

        self._max_tokens = spec.max_tokens_per_rank
        hidden = self.args.hidden
        top_k = self.args.topk
        # Dispatch carries the activation plus the routing metadata the kernel needs per token:
        # int32 expert ids and fp32 gate weights, top_k of each. Combine carries BF16 hidden.
        dispatch_bytes = hidden * 2 + top_k * 4 + top_k * 4
        combine_bytes = hidden * 2
        workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.world_size,
            max_num_tokens=self._max_tokens,
            total_dispatch_payload_size_per_token=dispatch_bytes,
            combine_payload_size_per_token=combine_bytes,
        )
        mapping = Mapping(
            self.world_size,
            self.rank,
            self.args.gpus_per_node,
            tp_size=self.world_size,
            moe_ep_size=self.world_size,
        )
        self._a2a = MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=self._max_tokens,
            top_k=top_k,
            num_experts=self.args.experts,
            workspace_size_per_rank=workspace_size,
            mnnvl_config=MnnvlConfig(comm_backend=_communicator(_ep_group())),
        )
        wheel = tuple(int(n) for n in re.findall(r"\d+", flashinfer.__version__)[:3])
        if wheel >= _COMBINE_FP32_SINCE:
            self.combine_reduction = "domain-fp32"
        # Every rank must finish mapping its workspace before any peer writes into it;
        # vLLM barriers here for the same reason. Scoped to the EP group, not the world.
        torch.distributed.barrier(group=_ep_group())

    # ---- transport contract ------------------------------------------------------------------

    def dispatch(self, p):
        """Dispatch the activation together with the routing metadata the oracle needs back.

        Three payloads, not one. The correctness oracle reads a per-received-row view carrying
        `expert_ids` and `weights` alongside the payload, and the only way to know a received
        row's routing is for the sender to ship it: the kernel moves opaque bytes. This is the
        intended shape — the upstream workspace accounting budgets exactly `top_k * 4` for
        int32 ids plus `top_k * 4` for fp32 weights on top of the hidden bytes.

        `invalid_token_expert_id` + `expert_id_payload_index` are how a receiver tells which
        rows are real: the receive planes are `[ep_size, max_tokens, *]` and a rank only gets
        the tokens that selected one of its experts, so the kernel stamps the sentinel into the
        expert-id payload of every slot it did not fill.
        """
        recv_x, recv_idx, recv_w = self._a2a.dispatch(
            p.topk_idx,
            [p.dispatch_x, p.topk_idx, p.topk_weights],
            p.T,
            invalid_token_expert_id=_INVALID_EXPERT,
            expert_id_payload_index=1,
        )
        return types.SimpleNamespace(
            recv_x=recv_x, recv_idx=recv_idx, recv_w=recv_w,
            tokens=p.T, topk=p.topk_idx.shape[1], combine_input=None,
        )

    def _combine_buffer(self, h):
        """The workspace-resident combine payload region for this rung."""
        return self._a2a.get_combine_payload_tensor_in_workspace(
            h.tokens, h.recv_x.shape[-1], h.recv_x.dtype
        )

    def stage(self, p, h):
        """Materialise the combine payload in the workspace region the API designates.

        A production integration has the expert GEMM write its output straight into this
        region and submits it with `payload_in_workspace=True`, so combine performs no
        staging copy. Copying here rather than handing `combine` a caller-owned tensor keeps
        that copy out of the combine measurement, where production does not pay it; it is
        still executed and reported, as `stage`.
        """
        buffer = self._combine_buffer(h)
        buffer.copy_(h.recv_x)
        h.combine_input = buffer

    def combine(self, p, h):
        out = self._a2a.combine(h.combine_input, h.tokens, payload_in_workspace=True)
        h.out = out
        return out

    def recv_tokens(self, h):
        # Rows the kernel actually filled, across every source plane.
        return int(self._valid_rows(h).sum().item())

    # ---- correctness-oracle views ------------------------------------------------------------

    def _valid_rows(self, h):
        """Boolean mask over the flattened [ep_size * max_tokens] receive slots."""
        idx = h.recv_idx.reshape(-1, h.topk)
        return (idx != _INVALID_EXPERT).any(dim=1)

    def inspect_dispatch(self, p, h):
        """Compact per-received-row view for the correctness oracle.

        The receive is `[ep_size, max_tokens, *]` with a rank's tokens only in the slots the
        kernel filled, so flatten and keep rows whose expert-id payload is not the sentinel.

        Ids come back exactly as sent, i.e. GLOBAL and covering the token's whole top-k —
        including experts owned by OTHER ranks. The oracle builds its expectation as "global id
        where `id // experts_per_rank == rank`, else -1", so the non-local entries have to be
        masked out here too, with their weights zeroed, or every row would disagree.
        """
        keep = self._valid_rows(h)
        hidden = h.recv_x.shape[-1]
        payload = h.recv_x.reshape(-1, hidden)[keep]
        ids = h.recv_idx.reshape(-1, h.topk).to(torch.int64)[keep]
        weights = h.recv_w.reshape(-1, h.topk).to(torch.float32)[keep]
        local = (ids >= 0) & ((ids // self.experts_per_rank) == self.rank)
        expert_ids = torch.where(local, ids, torch.full_like(ids, -1))
        return types.SimpleNamespace(
            payload=payload,
            expert_ids=expert_ids,
            weights=weights.masked_fill(~local, 0.0),
            # Per-local-expert arrival count; the oracle compares it against its own bincount.
            local_expert_counts=torch.bincount(
                (ids[local] - self.rank * self.experts_per_rank),
                minlength=self.experts_per_rank,
            ),
        )

    def combine_transformed(self, p, h, transformed):
        """Combine an oracle-transformed payload through the same kernel as the timed path.

        Scattered into the workspace combine region (see stage) and submitted with
        `payload_in_workspace=True`, so the kernel reads the slots from the region the API
        designates for them rather than from the dispatch-receive buffer.
        """
        buffer = self._combine_buffer(h)
        flat = buffer.view(-1, buffer.shape[-1])
        keep = self._valid_rows(h)
        flat[keep] = transformed.to(buffer.dtype)
        flat[~keep] = 0            # slots the kernel never filled contribute nothing
        return self._a2a.combine(buffer, h.tokens, payload_in_workspace=True)


def _ep_group():
    """The process group spanning the EP world. CollectiveX runs one group per case and the
    default group IS the EP group, so this is the default — named for the kernel's assert
    (`workspace.size(0) == moe_ep_size`), which a wider group would silently violate."""
    import torch.distributed as dist

    return dist.group.WORLD

def _communicator(group):
    """Bridge MnnvlConfig to the process group the harness already established.

    FlashInfer needs a communicator spanning exactly the EP group to exchange MNNVL fabric
    handles; the harness has one in torch.distributed, so wrap that rather than standing up a
    second. The contract is `flashinfer.comm.mnnvl.CommBackend` and it is not optional in any
    part: an earlier version of this adapter implemented only rank/size/allgather/Split and
    the first dispatch died with `CUDA error: unspecified launch failure` (sticky 719) on
    gb200. `barrier` is the reason — handle exchange has to complete on every rank before any
    rank's kernel touches peer memory, and without it the writes land on memory the peer has
    not mapped yet. Built as a subclass of the upstream ABC so a future interface change is an
    import-time error here rather than another asynchronous fault on the cluster.
    """
    import torch.distributed as dist
    from flashinfer.comm.mnnvl import CommBackend

    class _TorchDistCommunicator(CommBackend):
        def __init__(self, process_group):
            self._group = process_group

        def Get_rank(self) -> int:
            return self._group.rank()

        def Get_size(self) -> int:
            return self._group.size()

        def allgather(self, data):
            gathered = [None] * self.Get_size()
            dist.all_gather_object(gathered, data, group=self._group)
            return gathered

        def bcast(self, data, root):
            # broadcast_object_list mutates the list in place.
            payload = [data]
            dist.broadcast_object_list(payload, src=root, group=self._group)
            return payload[0]

        def barrier(self) -> None:
            dist.barrier(group=self._group)

        def Split(self, color, key):
            # The harness's group already IS the EP group, so a split returns the same view.
            return self

    return _TorchDistCommunicator(group)
