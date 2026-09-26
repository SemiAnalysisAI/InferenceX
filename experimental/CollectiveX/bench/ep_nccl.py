#!/usr/bin/env python3
"""NCCL EP adapter: NVIDIA's native MoE dispatch/combine on the NCCL Device API.

NCCL EP (github.com/NVIDIA/nccl-extensions, arXiv 2603.13606) is a ground-up MoE
communication library built on NCCL's Device API — LSA (NVLink load/store) intra-node and
GIN (GPU-Initiated Networking) inter-node — with two algorithms selected per case:
  normal      -> HIGH_THROUGHPUT (HT), the Hybrid-EP-derived prefill/train path. FLAT recv
                 layout ``[N, hidden]`` (one row per received token, no expert structure) with
                 an unweighted rank-sum combine — identical semantics to deepep-v2 normal.
  low-latency -> LOW_LATENCY (LL), the rank-major decode path. This uses the pre-reduced
                 contract used by inference frameworks, followed by an unweighted rank-sum combine.
Both modes use the existing CollectiveX combine oracle.

BF16 only for now: v0.2 grows a real quantization surface (DispatchQuantizationRecipe.FWD /
DS_FP8E3M4 for FP8 dispatch, experimental CombineQuantizationRecipe.NVFP4), but wiring it into
the fp8_consume model is its own bring-up, so this adapter still does not override the FP8
encode hooks (SUPPORTED_PRECISIONS=("bf16",)).

Communicator bootstrap: NCCL EP forms its OWN NCCL communicator (separate from PyTorch's
process group) via ``Communicator.init(nranks, rank, unique_id)``. Upstream broadcasts the
unique id with MPI; CollectiveX has no MPI, so rank 0 generates the id and we broadcast its
bytes over the already-initialized torch process group (see ``_bootstrap_comm``).

Python bindings are split across two wheels since v0.2: ``nccl-extensions`` owns ``nccl.ep``
(the libnccl_ep.so JIT runtime + Cython bindings) and ``nccl4py`` provides ``nccl.core``
(Communicator/UniqueId). The API surface used here is verified against the published
nccl-extensions wheel and driven exactly as upstream's ep_test.py drives it — every class and
signature this adapter touches is unchanged from v0.1.
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
        ZeroCopyMode,
    )
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: NCCL EP import failed: {exc!r}", file=sys.stderr)
    raise


# ncclUniqueId is a fixed 128-byte blob; we still broadcast the length first so a future size
# change can't silently truncate the id on the non-root ranks.
_UNIQUE_ID_MAX_BYTES = 256

# Low-latency receive sizing, deliberately two numbers, mirroring ep_deepep_v2: _LL_BUFFER_CAP
# sizes the pre-allocated receive (and so the transport footprint), _LL_LADDER_CAP bounds which
# token counts are measured. Separating them lets the ladder be clamped around a kernel defect
# without moving the footprint and silently re-basing the rungs that remain.
#
# The ladder is RESTORED to the full buffer under nccl-ep v0.2. The v0.1 port of DeepEP's
# low-latency combine lacked the PR #642 fence — reduction warps read shared memory, then
# mbarrier_arrive(emptyBarriers[stageIdx]) with no fence between, letting the producer's next
# TMA load overwrite a stage mid-read (observed on gb300 EP8 BF16 at T=256: bimodal 1-in-5
# discrete corruption, max rel err 0.4704 vs healthy 0.0039). The v0.2 wheel ships the fence:
# fence_view_async_shared() before the elect_one_sync mbarrier_arrive in the shipped headers'
# ll_ep.cuh combine recv pipeline — exactly the documented restore condition for this clamp.
# The T=256 rung is back on the ladder; the correctness oracle re-verifies it on every run.
_LL_BUFFER_CAP = 256
_LL_LADDER_CAP = _LL_BUFFER_CAP


class NCCLEPBackend(EPBackend):
    name = "nccl-ep"
    maturity = "candidate"  # NVIDIA's library, but no engine exposes an NCCL-EP selector
    # One library, two algorithms selected by args.mode. kernel_generation and the combine
    # semantics are switched to their LL values in __init__ (mirrors ep_deepep_v2).
    #   normal      -> HT / FLAT layout / unweighted-rank-sum combine.
    #   low-latency -> LL / RANK_MAJOR layout / inference-framework contract.
    # "-routed" marks the generation whose timed HT dispatch charges the per-step
    # ncclEpUpdateHandle (see dispatch()); earlier "nccl-ep-ht" rows excluded it and the
    # docs record that those cannot be separated by any other field — this suffix is the
    # per-row discriminator that change lacked. "v02" marks the nccl-extensions v0.2 mover
    # (new kernels: LL combine fence, B200 EP16 fix, HT gains) so pre-upgrade rows never
    # pool with post-upgrade rows.
    # "-static" marks the combine input bound to the full static receive plane (see
    # `_bind_ht_recv_count`); "-zc" rows before it sliced that input to the received count.
    kernel_generation = "nccl-ep-v02-ht-routed-zc-static"
    SUPPORTED_MODES = ("normal", "low-latency")
    SUPPORTED_PRECISIONS = ("bf16",)
    # LL replays; HT stays eager. Graphed HT's oracle failures were the oracle's own unfenced
    # write into the zero-copy window (fixed in `combine_transformed`); with that fixed, graphed
    # HT is correct but 1.03-1.11x eager's pair period on h100/h200 EP16 (runs 36231927003..
    # 36231931954 vs 36176100175), so eager is HT's best configuration.
    CUDA_GRAPH_MODES = ("low-latency", "normal") if os.environ.get("COLLX_DIAG_HT_GRAPH", "0") == "1" else ("low-latency",)
    stage_device_work = False
    requires_fresh_pair = False
    receive_layout = "token-rank"
    combine_weight_semantics = "unweighted-rank-sum"
    zero_copy = True
    _ll_expert_major = False

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
        # LL layout: rank-major (TensorRT-LLM's NCCL EP contract, the default) or expert-major
        # (DeepEP LL's contract, as vLLM/SGLang decode consume it). Both are native LL layouts.
        layout = os.environ.get("COLLX_NCCL_LL_LAYOUT", "rank-major")
        if layout not in ("rank-major", "expert-major"):
            raise ValueError(f"COLLX_NCCL_LL_LAYOUT must be rank-major or expert-major, got {layout!r}")
        self._ll_expert_major = self._ll and layout == "expert-major"
        # LL rank-major follows the inference-framework contract. Direct windows are scale-up only.
        self.zero_copy = (not self._ll or not self._internode) and not self._ll_expert_major
        if self._ll_expert_major:
            # Weighted source-side combine over a per-expert padded receive: deepep-v2 LL's
            # contract, so this is the like-for-like row against the DeepEP-API backends.
            # "-em" separates this from the pre-#3370 "nccl-ep-v02-ll" rows, whose timed windows
            # also carried a handle.complete() per op; v0.2 needs complete() only after send_only.
            self.kernel_generation = "nccl-ep-v02-ll-em"
            self.receive_layout = "token-expert"
            self.combine_weight_semantics = "weighted-kernel-sum"
        elif self._ll:
            self.kernel_generation = (
                "nccl-ep-v02-ll-rm-zc" if self.zero_copy else "nccl-ep-v02-ll-rm"
            )
            self.receive_layout = "token-rank"
            self.combine_weight_semantics = "unweighted-rank-sum"
            # NCCL LL rank-major suppresses duplicate ranks in top-k order, accumulates the
            # returned BF16 rank rows in FP32, and narrows only at the final output.
            self.combine_reduction = "rank-fp32"
        # NCCL EP's handle is explicitly reusable across dispatch/combine cycles (ep_test.py
        # cached mode redispatches and recombines on one handle), so — unlike DeepEP's legacy
        # low-latency Buffer — no timed component needs a fresh dispatch or a draining combine;
        # both modes keep requires_fresh_pair False.
        self._algorithm = Algorithm.LOW_LATENCY if self._ll else Algorithm.HIGH_THROUGHPUT
        if self._ll:
            self._layout = Layout.EXPERT_MAJOR if self._ll_expert_major else Layout.RANK_MAJOR
        else:
            self._layout = Layout.FLAT
        # send_only=0 runs each dispatch/combine as a complete SEND|RECV operation.
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
            # Bounds which token counts are MEASURED. Equal to _LL_BUFFER_CAP under v0.2 (the
            # combine recv fence shipped — see the constants above); the two names stay separate
            # so a future defect can clamp the ladder without moving the footprint.
            return _LL_LADDER_CAP
        return None

    # ---- helpers -----------------------------------------------------------------------------

    @staticmethod
    def _t(x):
        """Wrap a torch tensor as an ``nccl.ep.Tensor`` (torch passthrough; the library reads
        the device pointer/shape/dtype internally and anchors the torch buffer's lifetime)."""
        return Tensor(x)

    def _window_t(self, x):
        """Wrap a view into the active NCCL-registered receive window."""
        return Tensor(
            x,
            window=self._recv_window,
            window_offset=x.data_ptr() - self._recv_x.data_ptr(),
        )

    def _stream(self):
        """Raw handle of torch's current CUDA stream — NCCL EP runs on the same stream torch
        times, so its work is captured by the harness's CUDA-event timing."""
        return torch.cuda.current_stream().cuda_stream

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
        # checkpoint). Set NCCL_GIN_TYPE in the launcher env if a
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
        # Sized from the BUFFER cap, not from the measured ladder, so clamping the ladder
        # around the combine race does not also shrink the transport footprint -- which drives
        # recv-slot memory traffic and would change what the remaining rungs measure.
        self.max_dispatch = _LL_BUFFER_CAP if self._ll else spec.max_tokens_per_rank
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
            zero_copy=ZeroCopyMode.ON if self.zero_copy else ZeroCopyMode.OFF,
            **(
                {"alloc": self._routing_window_alloc()}
                if not self._ll and os.environ.get("COLLX_DIAG_ROUTING_WINDOW", "1") == "1"
                else {}
            ),
        )
        self._ep_group = nccl_ep.Group.create(self._comm, config)
        if not self._ll:
            st = getattr(self, "_rw_state", None)
            tag = "none" if st is None else f"w{st['registered']}-rc{st['rc']}-n{st['matches']}-{st['rt']}"
            self.kernel_generation = f"{self.kernel_generation}-diagrw-{tag}-cta{os.environ.get('NCCL_CTA_POLICY', 'd')}"

        dev = self.device
        if self._ll_expert_major:
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
            self._recv_count_t = self._t(self._recv_count)
        elif self._ll:
            # RANK_MAJOR receive: [source rank, source slot, hidden].
            self._recv_x = nccl_core.torch.empty(
                (self.world_size, self.max_dispatch, hidden), dtype=torch.bfloat16, device=dev
            )
            # Per-source-rank received-token counts, written during dispatch.
            self._recv_count = torch.empty(
                (self.world_size,), dtype=torch.int32, device=dev
            )
            self._recv_w = torch.empty(
                (self.world_size, self.max_dispatch, self.args.topk),
                dtype=torch.float32, device=dev,
            )
            self._recv_idx = torch.empty(
                (self.world_size, self.max_dispatch, self.args.topk),
                dtype=torch.int32, device=dev,
            )
            self._recv_count_t = self._t(self._recv_count)
            self._recv_w_t = self._t(self._recv_w)
            self._recv_idx_t = self._t(self._recv_idx)
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
            self._recv_x = nccl_core.torch.empty(
                (rows, hidden), dtype=torch.bfloat16, device=dev
            )
            self._recv_w = torch.empty((rows, self.args.topk), dtype=torch.float32, device=dev)
            self._recv_idx = torch.empty((rows, self.args.topk), dtype=torch.int64, device=dev)
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

        if self.zero_copy:
            self._recv_window = self._comm.register_window(
                self._recv_x, flags=nccl_core.WindowFlag.COLL_SYMMETRIC
            )
            if not self._recv_window.is_valid:
                self._ep_group.destroy()
                self._ep_group = None
                raise RuntimeError("NCCL-EP zero-copy receive window registration failed")
            self._recv_x_t = self._window_t(self._recv_x)
        else:
            self._recv_window = None
            self._recv_x_t = self._t(self._recv_x)

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
        if self._ll_expert_major:
            # Expert-major applies the gate in its combine kernel, not on dispatch. Wrapped once
            # per handle: `time_us` charges the wrapper's host work to the window.
            h.combine_weights_t = self._t(p.topk_weights)
        else:
            # HT carries weights on dispatch; LL rank-major transports them with dispatch too.
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
                self._bind_ht_recv_count(h)
            self._bound = h
        else:
            h.handle = self._handle
            self._rebind(h)
        p._nccl = h
        return h

    def _routing_window_alloc(self):
        """DIAG: register the HT routing map as a symmetric NCCL window so ncclAllGather can
        take a symmetric (proxy-free) kernel under graph capture."""
        import ctypes
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            cudart = None
        from nccl.ep.allocator import AllocConfig, AllocFn, FreeFn

        lib = ctypes.CDLL("libnccl.so.2")
        lib.ncclMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.ncclCommWindowRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ]
        lsa = int(self.args.scale_up_domain)
        nodes = max(1, self.world_size // lsa)
        row = -(-(min(lsa, self.world_size) * self.num_local_experts) // 8) * nodes
        target = self.world_size * self.max_dispatch * row
        comm = self._comm.ptr
        st = self._rw_state = {"registered": 0, "rc": None, "matches": 0, "keep": set(), "target": target,
                                  "rt": "cudart" if cudart else "ncclmem"}

        @AllocFn
        def alloc(out, size, ctx):
            if size == target:
                st["matches"] += 1
                p = ctypes.c_void_p()
                rc = lib.ncclMemAlloc(ctypes.byref(p), size)
                if rc == 0:
                    win = ctypes.c_void_p()
                    rc = lib.ncclCommWindowRegister(ctypes.c_void_p(comm), p, size, ctypes.byref(win), 1)
                    if rc == 0:
                        st["registered"] += 1
                        st["keep"].add(p.value)
                st["rc"] = rc
                if rc == 0:
                    out[0] = p.value
                    return 0
            if cudart is None:
                p = ctypes.c_void_p()
                rc = lib.ncclMemAlloc(ctypes.byref(p), size)
                out[0] = p.value
                return 0 if rc == 0 else 2
            err, ptr = cudart.cudaMalloc(size)
            out[0] = int(ptr)
            return int(err)

        @FreeFn
        def free(ptr, ctx):
            if ptr in st["keep"]:
                return 0  # registered for the process lifetime (diag)
            if cudart is None:
                return 0
            (err,) = cudart.cudaFree(ptr)
            return int(err)

        self._rw_fns = (alloc, free)
        return AllocConfig(
            alloc_fn=ctypes.cast(alloc, ctypes.c_void_p).value,
            free_fn=ctypes.cast(free, ctypes.c_void_p).value,
        )

    def _bind_ht_recv_count(self, h):
        """Read HT's received-token count and bind the combine input to the full receive plane.

        The FLAT contract (ep_enums.h, NCCL_EP_LAYOUT_FLAT) gives combine the SAME
        `[num_recv_slots, hidden]` shape as the dispatch output, static at the group's
        `max_recv_tokens_per_rank` -- "Required under CUDA Graph capture"; sizing it to the
        received count is only valid for a group created with `max_recv_tokens_per_rank =
        NCCL_EP_AUTO`, which this one is not. An earlier revision sliced it to the count, which
        broke that contract in both regimes. The slice existed to dodge a whole-plane staging
        copy (~470-1295us on prefill); zero-copy HT elides that staging, so the full plane costs
        nothing. The count is still read here (untimed) for `recv_tokens` and the oracle.
        """
        h.count = int(h.recv_total.item())
        h.combine_in_t = self._recv_x_t

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
            self._bind_ht_recv_count(h)
        self._bound = h

    # ---- transport contract ------------------------------------------------------------------

    def dispatch(self, p):
        h = self._ensure_handle(p)
        stream = self._stream()
        if not self._ll:
            # Charge the per-step routing collective to the window. `ncclEpUpdateHandle` is
            # documented as a "per-step collective: prepare the handle for the given top-k
            # routing decisions", and production routing changes every MoE layer — a serving
            # step pays this update before every HT dispatch, at the handle's full token
            # capacity, exactly as here. Excluding it (as NVIDIA's ep_bench does) made HT
            # dispatch the one window that omitted its routing/layout work while deepep-v2,
            # uccl-ep, MoRI and FlashInfer all carry theirs per call. No sync or counter
            # read here: the bound problem's counters are deterministic and already read
            # (_bind_ht_recv_count) in the untimed rebind.
            h.handle.update(h.topk_idx_t, layout_info=h.layout_info, stream=stream)
        if self._ll_expert_major:
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
        elif self._ll:
            # LL RANK_MAJOR returns one plane per source rank.
            h.handle.dispatch(
                DispatchInputs(tokens=h.in_tokens_t, topk_weights=h.in_weights_t),
                DispatchOutputs(
                    tokens=self._recv_x_t,
                    topk_weights=self._recv_w_t,
                    topk_idx=self._recv_idx_t,
                ),
                layout_info=LayoutInfo(src_rank_counters=self._recv_count_t),
                config=self._dispatch_cfg,
                stream=stream,
            )
            h.recv_x = self._recv_x
            h.recv_count = self._recv_count
            h.recv_w = self._recv_w
            h.recv_idx = self._recv_idx
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
        return h

    def stage(self, p, h):
        # BF16 combine input is the received buffer itself; HT needs received rows only.
        # Still an nccl.ep tensor wrapper, not a torch tensor; shared code passes it through.
        h.combine_input = self._recv_x_t if self._ll else h.combine_in_t

    def combine(self, p, h):
        stream = self._stream()
        if self._ll_expert_major:
            # Weighted combine: the kernel multiplies each expert contribution by the source
            # token's gate before the FP32 accumulation.
            h.handle.combine(
                CombineInputs(tokens=h.combine_input),
                CombineOutputs(tokens=h.out_t, topk_weights=h.combine_weights_t),
                config=self._combine_cfg,
                stream=stream,
            )
            return h.out
        # HT and LL rank-major use an unweighted rank-sum combine.
        h.handle.combine(
            CombineInputs(tokens=h.combine_input),
            CombineOutputs(tokens=h.out_t),
            config=self._combine_cfg,
            stream=stream,
        )
        return h.out

    def recv_tokens(self, h):
        if self._ll:
            return int(h.recv_count.sum().item())
        return int(h.count)

    # ---- correctness-oracle views ------------------------------------------------------------

    def _ll_inspect_dispatch(self, p, h):
        """Flat valid-slot view over the RANK_MAJOR inference-framework receive model."""
        recv_bf16 = h.recv_x  # [source rank, source slot, hidden] BF16
        slots = recv_bf16.shape[1]
        counts = h.recv_count.to(torch.int64)
        valid = torch.arange(slots, device=recv_bf16.device).unsqueeze(0) < counts.unsqueeze(1)
        source_rank, source_slot = valid.nonzero(as_tuple=True)
        h.source_rank = source_rank
        h.source_slot = source_slot
        local_idx = h.recv_idx[source_rank, source_slot].to(torch.int64)
        valid_idx = local_idx >= 0
        expert_ids = torch.where(
            valid_idx, local_idx + self.rank * self.num_local_experts, local_idx
        )
        weights = h.recv_w[source_rank, source_slot].to(torch.float32).masked_fill(~valid_idx, 0)
        return types.SimpleNamespace(
            payload=recv_bf16[source_rank, source_slot],
            expert_ids=expert_ids,
            weights=weights,
            local_expert_counts=torch.bincount(
                local_idx[valid_idx], minlength=self.num_local_experts
            ),
        )

    def _ll_em_inspect_dispatch(self, p, h):
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
        if self._ll_expert_major:
            return self._ll_em_inspect_dispatch(p, h)
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
        """Scatter pre-reduced oracle rows into the LL combine plane."""
        combine_buf = self._recv_x
        combine_buf.zero_()
        combine_buf[h.source_rank, h.source_slot] = transformed.to(combine_buf.dtype)
        stream = self._stream()
        h.handle.combine(
            CombineInputs(
                tokens=self._window_t(combine_buf) if self.zero_copy else self._t(combine_buf)
            ),
            CombineOutputs(tokens=h.out_t),
            config=self._combine_cfg,
            stream=stream,
        )
        return h.out[: p.T]

    def _ll_em_combine_transformed(self, p, h, transformed):
        """Scatter the oracle-transformed rows back into a zeroed EXPERT_MAJOR combine buffer at
        the (expert, slot) coordinates inspect read them from, then run the weighted combine;
        the kernel applies p.topk_weights, so the staged transform is unweighted."""
        combine_buf = self._combine_scratch
        combine_buf.zero_()
        combine_buf[h.slot_expert, h.slot_j] = transformed.to(combine_buf.dtype)
        h.handle.combine(
            CombineInputs(tokens=self._t(combine_buf)),
            CombineOutputs(tokens=h.out_t, topk_weights=h.combine_weights_t),
            config=self._combine_cfg,
            stream=self._stream(),
        )
        return h.out[: p.T]

    def combine_transformed(self, p, h, transformed):
        if self._ll_expert_major:
            return self._ll_em_combine_transformed(p, h, transformed)
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
        # Fence the write across ranks. `_recv_x` is the zero-copy window peers access directly,
        # and nothing orders this rank's local write against a peer's combine touching it: after
        # graph replay shifted rank timing, the combine read a peer's half-written input
        # (combine_values failed, dispatch checks clean, EP16 only; h100/h200 3/3 runs). With this
        # fence the same cells passed 4/4 (runs 36231927003..36231931954). The timed path writes
        # nothing between dispatch and combine, so it needs no fence; this is the oracle's write.
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        stream = self._stream()
        h.handle.combine(
            # Same sliced input the timed path uses, so the two cannot diverge in shape.
            CombineInputs(tokens=h.combine_in_t),
            CombineOutputs(tokens=h.out_t),
            config=self._combine_cfg,
            stream=stream,
        )
        return h.out

    def finalize(self, rc):
        """Clean teardown: NCCL EP's Device-API objects tear down without MoRI's post-
        shmem_finalize assertion, so we destroy the handles/group/comm and the torch PG in
        order rather than hard-exiting."""
        try:
            dist.barrier()
            self._destroy_handles()
            if self._recv_window is not None:
                self._recv_window.close()
                self._recv_window = None
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
