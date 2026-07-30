#!/usr/bin/env python3
"""NCCL EP adapter: NVIDIA's native MoE dispatch/combine on the NCCL Device API.

NCCL EP (github.com/NVIDIA/nccl/contrib/nccl_ep, arXiv 2603.13606) is a ground-up MoE
communication library built on NCCL's Device API — LSA (NVLink load/store) intra-node and
GIN (GPU-Initiated Networking) inter-node — with two algorithms selected per case:
  normal      -> HIGH_THROUGHPUT (HT), the Hybrid-EP-derived prefill/train path. FLAT recv
                 layout ``[N, hidden]`` (one row per received token, no expert structure) with
                 an unweighted rank-sum combine — identical semantics to deepep-v2 normal.
  low-latency -> LOW_LATENCY (LL), the DeepEP-derived decode path. EXPERT_MAJOR recv layout
                 ``[num_local_experts, max_dispatch*num_ranks, hidden]`` with a source-side
                 weighted-kernel-sum combine — identical semantics to deepep-v2 low-latency.
Because both map exactly onto the two combine contracts CollectiveX already models, the
scale_up_domain two-level combine oracle in ep_harness applies unchanged.

BF16 only. NCCL EP's FP8 machinery exists but RELEASE.md lists it unsupported/untested this
release, so this adapter does not override the FP8 encode hooks (SUPPORTED_PRECISIONS=("bf16",)).

Communicator bootstrap: NCCL EP forms its OWN NCCL communicator (separate from PyTorch's
process group) via ``Communicator.init(nranks, rank, unique_id)``. Upstream broadcasts the
unique id with MPI; CollectiveX has no MPI, so rank 0 generates the id and we broadcast its
bytes over the already-initialized torch process group (see ``_bootstrap_comm``).

Python bindings are ``nccl4py`` (``import nccl.core`` + ``nccl.ep``); the API surface used
here is verified against upstream ``bindings/nccl4py/nccl/ep`` and driven exactly as
``contrib/nccl_ep/ep_test.py`` drives it.
"""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.distributed as dist

from ep_backend import EPBackend

try:
    import nccl.core as nccl_core
    import nccl.ep as nccl_ep
    from nccl.ep import (
        Algorithm,
        CombineConfig,
        CombineInputs,
        CombineOutputs,
        DispatchConfig,
        DispatchInputs,
        DispatchOutputs,
        GroupConfig,
        HandleConfig,
        Layout,
        LayoutInfo,
        Tensor,
    )
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: NCCL EP import failed: {exc!r}", file=sys.stderr)
    raise


# ncclUniqueId is a fixed 128-byte blob; we still broadcast the length first so a future size
# change can't silently truncate the id on the non-root ranks.
_UNIQUE_ID_MAX_BYTES = 256


class NCCLEPBackend(EPBackend):
    name = "nccl-ep"
    # One library, two algorithms selected by args.mode. kernel_generation and the combine
    # semantics are switched to their LL values in __init__ (mirrors ep_deepep_v2).
    #   normal      -> HT / FLAT layout / unweighted-rank-sum combine.
    #   low-latency -> LL / EXPERT_MAJOR layout / source-side weighted-kernel-sum combine.
    kernel_generation = "nccl-ep-ht"
    SUPPORTED_MODES = ("normal", "low-latency")
    SUPPORTED_PRECISIONS = ("bf16",)
    stage_device_work = False
    combine_input_attr = "combine_input_t"  # this adapter's combine reads combine_input_t
    combine_needs_redispatch = False
    dispatch_needs_combine_cleanup = False
    combine_weight_semantics = "unweighted-rank-sum"

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
        # NCCL EP group creation requires the NCCL Device API (LSA symmetric memory), which NCCL
        # only advertises (comm.device_api_support) when cuMem allocation is enabled — otherwise
        # ncclEpCreateGroup fails its deviceApiSupport gate with ncclInvalidUsage. The launcher
        # exports this process-wide for every backend; setdefault before the EP comm is formed
        # also covers manual/torchrun invocations. Verified on h100 EP8: absent -> group.create
        # fails (error 5); present -> device_api_support=True and HT+LL groups create.
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
        self.group = dist.group.WORLD
        self.experts_per_rank = args.experts // world_size
        self.num_local_experts = self.experts_per_rank
        self._internode = world_size > int(args.scale_up_domain)
        self._ll = self.mode == "low-latency"
        if self._ll:
            # LL decode kernels apply the top-k gate at the source (weighted), not an
            # unweighted rank sum — the benchmark stages the UNWEIGHTED per-expert transform
            # and the kernel multiplies by the gate. Same contract as deepep-v2 low-latency.
            self.kernel_generation = "nccl-ep-ll"
            self.combine_weight_semantics = "weighted-kernel-sum"
        # NCCL EP's handle is explicitly reusable across dispatch/combine cycles (ep_test.py
        # cached mode redispatches and recombines on one handle), so — unlike DeepEP's legacy
        # low-latency Buffer — no timed component needs a fresh dispatch or a draining combine;
        # both modes keep combine_needs_redispatch / dispatch_needs_combine_cleanup False. If
        # on-metal bring-up ever shows LL result-tensor aliasing, flip both to True for LL.
        self._algorithm = Algorithm.LOW_LATENCY if self._ll else Algorithm.HIGH_THROUGHPUT
        self._layout = Layout.EXPERT_MAJOR if self._ll else Layout.FLAT
        # send_only=0 on every dispatch/combine (no staged execution). Handle.complete() is
        # still required after each call — the HT combine only stages the local send and
        # finishes the cross-rank gather in complete() (see _finish, verified on h100 EP8).
        # FWD pass carries top-k weights on dispatch (HT) and forbids them on the HT combine
        # input (the combine is a plain rank sum).
        self._dispatch_cfg = DispatchConfig(send_only=0, round_scales=0)
        self._combine_cfg = CombineConfig(send_only=0)
        self._comm = None
        self._ep_group = None
        # Exactly ONE handle per group, rebound per problem shape — see _ensure_handle.
        self._handle = None
        self._bound = None

    def buffer_cap(self, args):
        if self._ll:
            # LL pre-allocates the fixed [num_local_experts, cap*num_ranks, hidden] receive
            # buffer, so cap is a hard per-rank dispatch-slot bound (same 256 as ep_deepep_v2 /
            # ep_uccl low-latency; the harness clamps the decode ladder and reports drops).
            return 256
        return None

    # ---- helpers -----------------------------------------------------------------------------

    @staticmethod
    def _t(x):
        """Wrap a torch tensor as an ``nccl.ep.Tensor`` (torch passthrough; the library reads
        the device pointer/shape/dtype internally and anchors the torch buffer's lifetime)."""
        return Tensor(x)

    def _stream(self):
        """Raw handle of torch's current CUDA stream — NCCL EP runs on the same stream torch
        times, so its work is captured by the harness's CUDA-event timing."""
        return torch.cuda.current_stream().cuda_stream

    def _finish(self, h, stream):
        """Complete a dispatch/combine on the handle. Upstream ep_test.py calls this after
        every dispatch and combine unconditionally: with send_only=0 the LL path self-completes
        (combine runs SEND|RECV in one call), but the HT path only stages the local send in the
        combine call and finishes the cross-rank gather in complete() — without it HT combine
        returns just the source rank's own local-expert contribution (verified on h100 EP8)."""
        h.handle.complete(stream=stream)

    def _bootstrap_comm(self):
        """Form NCCL EP's own communicator, broadcasting the unique id over the torch PG.

        Rank 0 generates the id; we ship its raw bytes through the existing torch process
        group (no MPI in CollectiveX) and every rank rebuilds it, then all ranks collectively
        ``Communicator.init``. This communicator is independent of PyTorch's — the torch PG is
        used only for this broadcast and for the harness's timing all-reduces/barriers.
        """
        # Internode EP16 rides GIN over the RDMA fabric. NCCL EP's own ncclEpCreateGroup builds
        # the internode DevComm with ginForceEnable + a RAIL connection type, so we do NOT force
        # NCCL_GIN_TYPE here — forcing GDAKI(3) can prevent NCCL from selecting a fabric-supported
        # GIN transport and trips an illegal-memory-access in the GIN dispatch kernel (on-metal
        # checkpoint; still under cross-node bring-up). Set NCCL_GIN_TYPE in the launcher env if a
        # specific transport must be pinned.
        length = torch.zeros(1, dtype=torch.int64, device=self.device)
        payload = torch.zeros(_UNIQUE_ID_MAX_BYTES, dtype=torch.uint8, device=self.device)
        if self.rank == 0:
            uid_bytes = nccl_core.get_unique_id().as_bytes
            assert 0 < len(uid_bytes) <= _UNIQUE_ID_MAX_BYTES, (
                f"unexpected ncclUniqueId size {len(uid_bytes)}"
            )
            length[0] = len(uid_bytes)
            payload[: len(uid_bytes)] = torch.frombuffer(
                bytearray(uid_bytes), dtype=torch.uint8
            ).to(self.device)
        dist.broadcast(length, src=0)
        dist.broadcast(payload, src=0)
        n = int(length.item())
        uid = nccl_core.UniqueId.from_bytes(bytes(payload[:n].cpu().numpy().tobytes()))
        self._comm = nccl_core.Communicator.init(
            nranks=self.world_size, rank=self.rank, unique_id=uid
        )

    # ---- buffer construction -----------------------------------------------------------------

    def create_buffer(self, spec):
        """Bootstrap the communicator, create the EP group sized from the ladder maximum, and
        allocate the persistent receive/combine buffers reused across every ladder shape."""
        self.max_dispatch = spec.max_tokens_per_rank
        hidden = self.args.hidden
        self._bootstrap_comm()
        # max_recv_tokens_per_rank: HT requires >0 and >= max_dispatch; LL auto-derives when 0.
        # world*max_dispatch is the recv-slot budget (every peer sends all its tokens here).
        max_recv = self.max_dispatch * self.world_size
        config = GroupConfig(
            algorithm=self._algorithm,
            num_experts=self.args.experts,
            max_dispatch_tokens_per_rank=self.max_dispatch,
            max_recv_tokens_per_rank=max_recv,
            max_token_bytes=hidden * 2,  # bfloat16 payload
        )
        self._ep_group = nccl_ep.Group.create(self._comm, config)

        dev = self.device
        if self._ll:
            # EXPERT_MAJOR recv: [num_local_experts, max_dispatch*num_ranks, hidden].
            slots = self.max_dispatch * self.world_size
            self._recv_x = torch.empty(
                (self.num_local_experts, slots, hidden), dtype=torch.bfloat16, device=dev
            )
            # Per-local-expert received-token counts, written by NCCL EP during dispatch.
            self._recv_count = torch.empty(
                (self.num_local_experts,), dtype=torch.int32, device=dev
            )
            # Zeroed scratch the combine oracle scatters the transformed rows into.
            self._combine_scratch = torch.empty_like(self._recv_x)
            self._recv_x_t = self._t(self._recv_x)
            self._recv_count_t = self._t(self._recv_count)
        else:
            # FLAT recv sized to the group's recv-slot budget max_recv_tokens_per_rank =
            # world_size * max_dispatch — the max unique tokens this rank can receive (every peer
            # sends up to max_dispatch and FLAT delivers one row per received token). HT combine
            # copies this whole buffer into the group's expert_input_token IPC staging, which the
            # group sized to exactly max_recv; oversizing it (e.g. ep_test's
            # num_local_experts*max_dispatch, which only equals max_recv there because
            # num_local_experts==n_ranks) overflows that buffer with cudaErrorInvalidValue.
            rows = self.max_dispatch * self.world_size
            self._recv_rows = rows
            self._recv_x = torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev)
            self._recv_w = torch.empty((rows, self.args.topk), dtype=torch.float32, device=dev)
            self._recv_idx = torch.empty((rows, self.args.topk), dtype=torch.int64, device=dev)
            self._recv_x_t = self._t(self._recv_x)
            self._recv_w_t = self._t(self._recv_w)
            self._recv_idx_t = self._t(self._recv_idx)
            # HT FLAT dispatch writes per-local-expert received counts (unpadded int32) here via
            # the metadata path. Passing it in the dispatch LayoutInfo is REQUIRED for the
            # internode GIN dispatch kernel — a null expert_counters is tolerated intranode (EP8)
            # but faults the cross-node kernel (illegal memory access). ep_bench passes this too.
            self._ht_disp_counts = torch.empty(
                (self.num_local_experts,), dtype=torch.int32, device=dev
            )
            self._ht_disp_counts_t = self._t(self._ht_disp_counts)

    def _ensure_handle(self, p):
        """Bind the group's single handle to p's routing, creating it on first use.

        ONE handle per group, rebound per shape — never one handle per shape. `buffer_idx`, the
        LL double-buffer parity selector, is per-HANDLE state, but the buffers it selects are
        offsets into the per-GROUP rdma_buffer: two handles built from the same group config
        resolve to the SAME parity-0/parity-1 count+flag slots and advance their parity
        independently, so one handle's "next buffer, safe to clean" is the other's "current
        buffer, in flight". Interleaving handles therefore corrupts the signalling even when it
        does not hang outright, which makes every latency drawn from such a run suspect. Filed
        upstream as NVIDIA/nccl#2303; reproduced on a stock wheel by ladder [1] (one handle,
        clean) vs ladder [1, 2] (two handles, 64 dispatch + 6 combine receive timeouts -> 719).

        `ncclEpInitHandle` takes no token count and `ncclEpUpdateHandle` is documented as a
        "per-step collective: prepare the handle for the given top-k routing decisions", so
        rebinding IS the intended lifecycle. This also matches the other three backends, which
        each allocate one object sized to the ladder maximum and vary the token count per call.

        Both create_handle and update are collective, and HT additionally performs the metadata
        exchange that fixes the received-token count, so they must run in the same order on
        every rank. They only ever run on a shape CHANGE, and every timed component is preceded
        by an untimed warm() on its own problem — so the collective always lands in warm (or in
        the oracle passes), never inside a timed window. Re-entering with the already-bound
        problem returns immediately without a collective or a sync.
        """
        cached = getattr(p, "_nccl", None)
        if cached is not None:
            if self._bound is not cached:
                self._rebind(cached)
            return cached
        stream = self._stream()
        topk_idx_t = self._t(p.topk_idx)
        h = types.SimpleNamespace(
            in_tokens_t=self._t(p.dispatch_x),
            topk_idx_t=topk_idx_t,
        )
        if not self._ll:
            h.in_weights_t = self._t(p.topk_weights)
        # combined output is restored to original token order: [num_tokens, hidden].
        h.out = torch.empty((p.T, self.args.hidden), dtype=torch.bfloat16, device=self.device)
        h.out_t = self._t(h.out)
        # HT: request the received-token counters at handle creation (populated by the HT
        # metadata exchange). recv_total_counter is the authoritative FLAT row count.
        ht_layout_info = None
        if not self._ll:
            h.recv_total = torch.zeros(1, dtype=torch.int32, device=self.device)
            h.recv_experts = torch.zeros(
                self.num_local_experts, dtype=torch.int32, device=self.device
            )
            ht_layout_info = LayoutInfo(
                expert_counters=self._t(h.recv_experts),
                recv_total_counter=self._t(h.recv_total),
            )
        # LL takes layout_info only on dispatch (the API forbids it on create/update); HT needs
        # it here so this problem's counters receive its own metadata-exchange results.
        h.layout_info = ht_layout_info
        if self._handle is None:
            self._handle = self._ep_group.create_handle(
                self._layout,
                topk_idx_t,
                layout_info=ht_layout_info,
                config=HandleConfig(),
                stream=stream,
            )
            h.handle = self._handle
            torch.cuda.synchronize()
            if not self._ll:
                h.count = int(h.recv_total.item())
            self._bound = h
        else:
            h.handle = self._handle
            self._rebind(h)
        p._nccl = h
        return h

    def _rebind(self, h):
        """Point the single handle at h's routing (collective; untimed callers only).

        Rebinding does not reallocate: `Handle.update` only swaps in the new top-k indices.
        HT re-reads its received-token count because the metadata exchange recomputes it for
        this routing; the value is deterministic per problem, so a later rebind to the same
        problem reproduces it.
        """
        self._handle.update(
            h.topk_idx_t,
            layout_info=None if self._ll else h.layout_info,
            stream=self._stream(),
        )
        torch.cuda.synchronize()
        if not self._ll:
            h.count = int(h.recv_total.item())
        self._bound = h

    # ---- transport contract ------------------------------------------------------------------

    def dispatch(self, p):
        h = self._ensure_handle(p)
        stream = self._stream()
        if self._ll:
            # LL EXPERT_MAJOR: tokens in, 3D per-expert padded tokens out, per-expert recv
            # counts written into expert_counters. No weights on the dispatch (the gate is
            # applied by the combine kernel at the source).
            h.handle.dispatch(
                DispatchInputs(tokens=h.in_tokens_t),
                DispatchOutputs(tokens=self._recv_x_t),
                layout_info=LayoutInfo(expert_counters=self._recv_count_t),
                config=self._dispatch_cfg,
                stream=stream,
            )
            h.recv_x = self._recv_x
            h.recv_count = self._recv_count
        else:
            # HT FLAT: tokens + top-k weights in (FWD requires weights); received tokens,
            # received top-k weights and GLOBAL top-k expert ids out.
            h.handle.dispatch(
                DispatchInputs(tokens=h.in_tokens_t, topk_weights=h.in_weights_t),
                DispatchOutputs(
                    tokens=self._recv_x_t,
                    topk_weights=self._recv_w_t,
                    topk_idx=self._recv_idx_t,
                ),
                layout_info=LayoutInfo(expert_counters=self._ht_disp_counts_t),
                config=self._dispatch_cfg,
                stream=stream,
            )
            h.recv_x = self._recv_x
            h.recv_w = self._recv_w
            h.recv_idx = self._recv_idx
        self._finish(h, stream)
        return h

    def stage(self, p, h):
        # BF16 combine input is the received buffer itself; no device work (value correctness
        # is exercised only through the oracle's combine_transformed path).
        h.combine_input_t = self._recv_x_t

    def combine(self, p, h):
        stream = self._stream()
        if self._ll:
            # Weighted LL combine: the kernel multiplies each expert contribution by the
            # source token's gate (CombineOutputs.topk_weights) before the FP32 accumulation.
            h.handle.combine(
                CombineInputs(tokens=h.combine_input_t),
                CombineOutputs(tokens=h.out_t, topk_weights=self._t(p.topk_weights)),
                config=self._combine_cfg,
                stream=stream,
            )
        else:
            # Unweighted HT combine (FWD forbids input weights): sums the per-token expert
            # aggregates back to each token's home rank, restored to original order.
            h.handle.combine(
                CombineInputs(tokens=h.combine_input_t),
                CombineOutputs(tokens=h.out_t),
                config=self._combine_cfg,
                stream=stream,
            )
        self._finish(h, stream)
        return h.out

    def recv_tokens(self, h):
        if self._ll:
            return int(h.recv_count.sum().item())
        return int(h.count)

    # ---- correctness-oracle views ------------------------------------------------------------

    def _ll_inspect_dispatch(self, p, h):
        """Flat per-slot view over the EXPERT_MAJOR padded receive (mirror of
        ep_deepep_v2._ll_inspect_dispatch): each local expert's valid tokens are packed at the
        front [0:recv_count[e]] of its slot dimension. Flatten to the oracle's compact
        (expert, slot) row-major contract and keep the coordinates for the combine scatter."""
        recv_bf16 = h.recv_x  # [E, S, hidden] BF16
        num_slots = recv_bf16.shape[1]
        counts = h.recv_count.to(torch.int64)  # [E]
        slot_valid = (
            torch.arange(num_slots, device=recv_bf16.device).unsqueeze(0) < counts.unsqueeze(1)
        )
        slot_expert, slot_j = slot_valid.nonzero(as_tuple=True)
        h.slot_expert = slot_expert
        h.slot_j = slot_j
        local_lo = self.rank * self.num_local_experts
        return types.SimpleNamespace(
            payload=recv_bf16[slot_expert, slot_j],
            expert_ids=local_lo + slot_expert.to(torch.int64),
            local_expert_counts=counts,
        )

    def inspect_dispatch(self, p, h):
        if self._ll:
            return self._ll_inspect_dispatch(p, h)
        # HT FLAT normal recv: front-packed to recv_total_counter, one row per received token.
        # recv_idx holds this rank's LOCAL expert indices [0, experts_per_rank) front-packed per
        # row (valid entries first, non-local padded to -1) with recv_w aligned to them — NOT the
        # global top-k. (Verified on h100 EP8: rank-1 token recv_idx=[2,17,-1..] for global experts
        # 34,49; rank 0 looks global only because its local range starts at 0.) So rebase the valid
        # locals to the GLOBAL ids the oracle compares by + rank*experts_per_rank, exactly as
        # ep_uccl/ep_deepep_v2 normal do. The oracle sorts each row over the top-k axis and sums
        # the per-expert transforms, so token order is free and no per-(token,expert) expansion is
        # needed.
        count = int(h.count)
        local_idx = h.recv_idx[:count].to(torch.int64)  # [count, topk] LOCAL ids, -1 non-local
        valid = local_idx >= 0
        expert_ids = torch.where(
            valid, local_idx + self.rank * self.experts_per_rank, local_idx
        )
        weights = h.recv_w[:count].to(torch.float32).masked_fill(~valid, 0)
        return types.SimpleNamespace(
            payload=h.recv_x[:count],  # [count, hidden] BF16
            expert_ids=expert_ids,
            weights=weights,
            local_expert_counts=torch.bincount(
                local_idx[valid], minlength=self.experts_per_rank
            ),
        )

    def _ll_combine_transformed(self, p, h, transformed):
        """Scatter the oracle-transformed rows back into a zeroed EXPERT_MAJOR combine buffer
        at the exact (expert, slot) coordinates inspect read them from, then run the weighted
        LL combine (mirror of ep_deepep_v2._ll_combine_transformed). `transformed` is
        [N, hidden] in that slot order; the kernel applies p.topk_weights, so the staged
        transform is unweighted."""
        combine_buf = self._combine_scratch
        combine_buf.zero_()
        combine_buf[h.slot_expert, h.slot_j] = transformed.to(combine_buf.dtype)
        stream = self._stream()
        h.handle.combine(
            CombineInputs(tokens=self._t(combine_buf)),
            CombineOutputs(tokens=h.out_t, topk_weights=self._t(p.topk_weights)),
            config=self._combine_cfg,
            stream=stream,
        )
        self._finish(h, stream)
        return h.out[: p.T]

    def combine_transformed(self, p, h, transformed):
        if self._ll:
            return self._ll_combine_transformed(p, h, transformed)
        # `transformed` is the oracle's per-received-token combine input [count, hidden]
        # (already summed over the top-k axis, gate folded in). Write it in place over the
        # dispatch-output buffer (recv_x) — the same registered buffer the real expert MLP
        # overwrites — so the combine's cross-rank gather reads it from the slots its routing
        # map expects. Zero the padding tail first; combine sums across the token's unique
        # destination ranks back to each token's home rank.
        self._recv_x.zero_()
        self._recv_x[: transformed.shape[0]].copy_(transformed.to(self._recv_x.dtype))
        stream = self._stream()
        h.handle.combine(
            CombineInputs(tokens=self._recv_x_t),
            CombineOutputs(tokens=h.out_t),
            config=self._combine_cfg,
            stream=stream,
        )
        self._finish(h, stream)
        return h.out

    def finalize(self, rc):
        """Clean teardown: NCCL EP's Device-API objects tear down without MoRI's post-
        shmem_finalize assertion, so we destroy the handles/group/comm and the torch PG in
        order rather than hard-exiting."""
        try:
            dist.barrier()
            self._destroy_handles()
            if self._ep_group is not None:
                self._ep_group.destroy()
            if self._comm is not None:
                self._comm.destroy()
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            return 1
        return rc

    def _destroy_handles(self):
        # One handle for the whole group, so teardown is a single explicit destroy rather than
        # waiting on per-problem GC. The problem namespaces still hold a reference to it for
        # their dispatch/combine calls; dropping _bound first keeps a late rebind from touching
        # a destroyed handle. The group/comm destroy below reclaims the device buffers.
        self._bound = None
        if self._handle is not None:
            self._handle.destroy()
            self._handle = None
