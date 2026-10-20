#!/usr/bin/env python3
"""FlashInfer one-sided NVLink EP: the transport a GB200/GB300 deployment actually runs.

The GB SKUs are an NVLink-only domain, and vLLM selects a different all-to-all there than it
does on an RDMA fabric: `--all2all-backend deepep_v2` for RDMA, `flashinfer_nvlink_one_sided`
for NVLink.

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
    .combine(payload, runtime_max_tokens_per_rank, payload_in_workspace=False[, output=...])
        -> [local_num_tokens, elements_per_token]

The contract is vLLM's integration (`FlashInferNVLinkOneSidedManager.combine_into` and the
`flashinfer_nvlink_one_sided` prepare/finalize). Its finalize hands combine the experts' output, a
tensor the caller owns, and never sets `payload_in_workspace`, so combine first stages that payload
into the workspace itself: that copy is part of what vLLM's combine costs, and the timed combine
pays it. Having the expert GEMM write straight into the workspace region instead is TensorRT-LLM's
path (and SGLang's, with its `flashinfer_cutlass` MoE runner only). Wheels from 0.6.16 combine into
a caller-supplied `output=`; before that vLLM copies the returned tensor into its own, and so does
`_combine_into`.

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

import inspect
import os
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


def _wheel_has_fp32_combine(version: str) -> bool:
    """Does this wheel accumulate combine in FP32, per `_COMBINE_FP32_SINCE`?

    Needs real version ordering, not a digit scrape: reading `0.6.16rc1` as 0.6.16 models FP32
    against a per-level-rounding kernel, which can exceed COMBINE_REL_TOL and red a correct run.
    The opposite error costs a few ulps, so anything unparseable answers False.
    """
    from packaging.version import InvalidVersion, Version

    try:
        return Version(version) >= Version(".".join(str(n) for n in _COMBINE_FP32_SINCE))
    except InvalidVersion:
        return False


def _accepts_keyword(fn, name) -> bool:
    """vLLM's `supports_kw(fn, name, allow_var_kwargs=False)`: whether `name` is an explicit
    parameter of `fn`. vLLM picks its combine call by this probe, not by version (`output=` and
    `use_low_precision` both arrived in 0.6.16), so the adapter makes the same choice the same way."""
    try:
        parameter = inspect.signature(fn).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)


def _version_slug(version: str) -> str:
    """The wheel version as a `kernel_generation` segment: a local tag such as `+cu130` keeps its
    text but not its `+`."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", version).strip("-") or "unknown"


# FP8 block size, matching the DeepSeek-V3 recipe every other FP8 backend here uses.
_FP8_BLOCK = 128


def _blockwise_cast_to_fp8(x):
    """Per-128-channel e4m3 quantize: (values [m, n], FP32 scales [m, n//128]).

    The layout vLLM dispatches on this transport: `per_token_group_quant_fp8(x, 128)`, row-major
    FP32 scales, sent as the payload after the values. Local rather than shared on purpose:
    deepep-v2 and uccl-ep must match their own libraries' in-kernel bits, while nothing here
    quantises in-kernel, so this copy only has to round-trip self-consistently. Three contracts,
    not three copies of one.
    """
    m, n = x.shape
    blocks = x.view(m, -1, _FP8_BLOCK)
    amax = blocks.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    values = (blocks * (448.0 / amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n)
    return values, (amax / 448.0).view(m, -1)


def _blockwise_cast_back(values, scales):
    """Inverse of _blockwise_cast_to_fp8, to BF16."""
    m, n = values.shape
    blocks = values.view(m, -1, _FP8_BLOCK).float()
    return (blocks * scales.view(m, -1, 1)).view(m, n).to(torch.bfloat16)


class FlashInferEPBackend(EPBackend):
    name = "flashinfer-ep"
    maturity = "production"  # vLLM --all2all-backend flashinfer_nvlink_one_sided
    # One kernel family and so the base's normal-only SUPPORTED_MODES; see the module docstring
    # for why there is no low-latency mode.
    # FP8 is dispatch-side only: scales ride as a fourth payload and combine stays BF16, so none
    # of the 0.6.16+ combine-quant API is needed. vLLM accepts DeepSeek blockwise FP8 (e4m3, FP32
    # 1x128 scales) on this transport from v0.29.0, so it is an engine path like BF16.
    SUPPORTED_PRECISIONS = ("bf16", "fp8")
    CUDA_GRAPH_MODES = ("normal",)
    # create_buffer appends the installed wheel's version: the kernels differ between wheels (0.6.16
    # brought the FP32 combine and programmatic dependent launch), so an image bump must not pool
    # with the rows before it. "staged": combine stages a caller-owned payload, as in vLLM.
    kernel_generation = "flashinfer-mnnvl-one-sided-staged"
    # combine_weight_semantics stays the base's unweighted-rank-sum: the kernel scatters expert
    # outputs back to the supplying rank without multiplying by the routing weights (those ride
    # along as a caller payload, and vLLM applies them in the MoE layer, not in the A2A).
    # Set per wheel in create_buffer; see _COMBINE_FP32_SINCE.
    combine_reduction = "topk-slot-tree"
    # Forced by the phase asserts described in the module docstring.
    requires_fresh_pair = True

    @property
    def cuda_graph_supported(self) -> bool:
        # Decode only: replay cuts gb200 EP8 T=1 pair period 151 -> 43us but leaves prefill flat.
        return super().cuda_graph_supported and getattr(self.args, "phase", None) == "decode"

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
        self._fp8 = self.precision == "fp8"
        if self._fp8:
            # The codec, bytes and block size of deepep-v2/uccl-ep, so the same label.
            self._enable_fp8("fp8-e4m3fn", _blockwise_cast_to_fp8, _blockwise_cast_back)
        # BF16 stage() is a pointer hand-off, so only the FP8 dequant is device work to hoist.
        self.stage_device_work = self._fp8
        self._a2a = None
        self._max_tokens = None
        self._workspace_sizing = {}
        # Which of vLLM's two combine calls this wheel takes; resolved in create_buffer.
        self._combine_takes_output = False
        self.experts_per_rank = args.experts // world_size

    # ---- setup -------------------------------------------------------------------------------

    def _topk_idx_dtype(self):
        """int32 — what the kernel reads the routing plane as.

        Declaring it here means `make_problem` casts once, untimed, instead of `dispatch`
        casting on every call inside the measured window. `topk_weights` is already FP32
        from `make_problem`, so both payloads reach the kernel with no conversion.
        """
        return torch.int32

    def make_problem(self, T, idx, weights, x):
        """Adds the MoE layer's output, which vLLM's finalize owns and combines into.

        Allocated here, once per rung and untimed, so no timed window or graph capture allocates
        it, and every combine of the rung writes the same tensor.
        """
        problem = super().make_problem(T, idx, weights, x)
        problem.combine_output = torch.empty((T, self.args.hidden), dtype=torch.bfloat16, device=self.device)
        return problem

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
        dispatch_bytes = (
            hidden * self.dispatch_value_bytes
            + self.dispatch_scale_bytes_per_copy
            + top_k * 4 + top_k * 4
        )
        combine_bytes = hidden * 2
        workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.world_size,
            max_num_tokens=self._max_tokens,
            total_dispatch_payload_size_per_token=dispatch_bytes,
            combine_payload_size_per_token=combine_bytes,
        )
        self._workspace_sizing = {
            "capacity_tokens_per_rank": self._max_tokens,
            "dispatch_bytes_per_token": dispatch_bytes,
            "combine_bytes_per_token": combine_bytes,
            "workspace_bytes_per_rank": workspace_size,
        }
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
        self.library_version = flashinfer.__version__
        self.kernel_generation = f"{type(self).kernel_generation}-{_version_slug(self.library_version)}"
        self._combine_takes_output = _accepts_keyword(self._a2a.combine, "output")
        if _wheel_has_fp32_combine(flashinfer.__version__):
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
        # Quantise here, not in make_problem: production runs one fused bf16->fp8 kernel per
        # forward pass immediately before this collective. The scales then ride as their own
        # payload, which shifts the expert ids to index 2 -- four payloads, and the kernel's
        # kMaxPayloads is exactly 4, matching vLLM's own [values, scales, ids, weights] order.
        if self._fp8:
            values, scales = self._quant(p.dispatch_x)
            payloads = [values, scales, p.topk_idx, p.topk_weights]
            expert_id_index = 2
        else:
            payloads = [p.dispatch_x, p.topk_idx, p.topk_weights]
            expert_id_index = 1
        received = self._a2a.dispatch(
            p.topk_idx,
            payloads,
            p.T,
            invalid_token_expert_id=_INVALID_EXPERT,
            expert_id_payload_index=expert_id_index,
        )
        # One received tensor per payload. Under FP8 `recv_x` stays the VALUES plane and the
        # scales ride beside it, so every shape-sensitive consumer keeps working on a tensor.
        if self._fp8:
            recv_x, recv_scales, recv_idx, recv_w = received
        else:
            (recv_x, recv_idx, recv_w), recv_scales = received, None
        return types.SimpleNamespace(
            recv_x=recv_x, recv_scales=recv_scales, recv_idx=recv_idx, recv_w=recv_w,
            tokens=p.T, topk=p.topk_idx.shape[1], combine_input=None,
        )

    def _filled_slot_index(self, p, h):
        """Row indices of the receive slots dispatch actually filled, resolved once per rung.

        Indexing with `_valid_rows`' boolean mask needs the match count on the host, which would
        put a device read inside the timed stage. Routing is fixed per ladder point, so resolve to
        an integer index on first use (always the untimed `warm()`) and cache it on the problem.
        """
        index = getattr(p, "flashinfer_filled_slots", None)
        if index is None:
            index = self._valid_rows(h).nonzero(as_tuple=True)[0]
            p.flashinfer_filled_slots = index
        return index

    def stage(self, p, h):
        """Hand combine the payload vLLM's finalize would: a caller-owned BF16 `[ep, T, hidden]`.

        In vLLM that is the experts' output, written beside the receive, never into the workspace
        combine region, so combine stages it there itself (`payload_in_workspace=False`). With no
        expert the received rows are that output, and under BF16 the receive already is a
        contiguous BF16 `[ep, T, hidden]` tensor disjoint from the combine region, so the hand-off
        is a pointer and the staging copy is timed inside combine.

        Under FP8 the received values are 1-byte, so the dequant into a fresh BF16 tensor stands
        in for the expert's BF16 output: device work, hoisted out of the roundtrip like every
        FP8 stage. Only the filled slots are converted, the ones the kernel's staging copy reads
        (`moeA2APrepareCombineKernel` returns past `recv_counters[source]`); the rest are never read.
        """
        if not self._fp8:
            h.combine_input = h.recv_x
            return
        filled = self._filled_slot_index(p, h)
        hidden = h.recv_x.shape[-1]
        payload = torch.empty(h.recv_x.shape, dtype=torch.bfloat16, device=h.recv_x.device)
        payload.view(-1, hidden)[filled] = _blockwise_cast_back(
            h.recv_x.view(-1, hidden)[filled], h.recv_scales.view(-1, h.recv_scales.shape[-1])[filled]
        )
        h.combine_input = payload

    def combine(self, p, h):
        return self._combine_into(h.combine_input, h.tokens, p.combine_output)

    def _combine_into(self, payload, tokens, output):
        """vLLM's `combine_into`: a caller-owned payload, which combine stages into the workspace,
        combined into the caller's `output`. A wheel without `output=` (before 0.6.16) returns a
        fresh tensor that vLLM copies into its own, so that copy is timed here too.
        `use_low_precision` stays at FlashInfer's default, False, as vLLM's
        VLLM_FLASHINFER_MOE_A2A_LOW_PRECISION_COMBINE leaves it."""
        if self._combine_takes_output:
            self._a2a.combine(payload, tokens, payload_in_workspace=False, output=output)
        else:
            output.copy_(self._a2a.combine(payload, tokens, payload_in_workspace=False))
        return output

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
        if self._fp8:
            # Dequantised with the same pair semantic_payload used, so the oracle compares like
            # for like.
            payload = _blockwise_cast_back(
                payload, h.recv_scales.reshape(-1, h.recv_scales.shape[-1])[keep]
            )
        return self._global_id_view(
            payload,
            h.recv_idx.reshape(-1, h.topk).to(torch.int64)[keep],
            h.recv_w.reshape(-1, h.topk).to(torch.float32)[keep],
            self.experts_per_rank,
        )

    def combine_transformed(self, p, h, transformed):
        """Combine an oracle-transformed payload through the timed path's call (`_combine_into`).

        Scattered into a fresh caller-owned BF16 `[ep, T, hidden]` payload, as the experts'
        output is, so combine's own staging copy moves it into the workspace.
        """
        payload = torch.zeros(h.recv_x.shape, dtype=torch.bfloat16, device=h.recv_x.device)
        payload.view(-1, payload.shape[-1])[self._valid_rows(h)] = transformed.to(torch.bfloat16)
        return self._combine_into(payload, h.tokens, p.combine_output)

    def implementation_details(self):
        """What this run handed the library (`EPBackend.implementation_details`)."""
        from flashinfer.comm import trtllm_moe_alltoall

        return {
            "flashinfer_version": self.library_version,
            # vLLM's contract: combine stages a caller-owned payload; the dispatch receive is
            # consumed in place, as a workspace view.
            "combine_payload": "caller-owned",
            "payload_in_workspace": False,
            "zero_copy": False,
            "combine_output": "output-kw" if self._combine_takes_output else "copy",
            "low_precision_combine": False,
            "dispatch_scale_format": "fp32-per-128-block-row-major" if self._fp8 else None,
            # Left at the library's device default by vLLM and here alike; wheels before 0.6.16
            # have no programmatic dependent launch at all.
            "pdl_available": _accepts_keyword(trtllm_moe_alltoall.moe_a2a_combine, "enable_pdl"),
            # One CTA per token: the grid follows T and there is no SM budget, and NVLink loads
            # and stores need no queue pairs.
            "num_sms": None,
            "num_qps": None,
            "launch_env": {name: value for name, value in sorted(os.environ.items())
                           if name.startswith("TLLM_MOE_A2A_")},
            **self._workspace_sizing,
        }


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
    part: implementing only rank/size/allgather/Split killed the first dispatch with `CUDA
    error: unspecified launch failure` (sticky 719) on gb200. Every method is required because
    `CommBackend` declares them abstract, not because the fabric path calls them: `MnnvlMemory`
    exchanges handles with `allgather` alone. What the ordering actually needs is the explicit
    `torch.distributed.barrier` in create_buffer.
    Built as a subclass of the upstream class, which in the image's 0.6.8.post1 is an ABC, so an
    interface change is an import-time error here rather than another asynchronous fault on the
    cluster. Later wheels (0.6.18.post1, 0.7.0.post1) make `CommBackend` a `typing.Protocol`
    (flashinfer/comm/abstractions.py), under which a missing method raises nothing at import:
    recheck this class against the protocol when the pin moves.
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
