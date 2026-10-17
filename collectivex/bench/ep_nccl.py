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

Engine contract: vLLM and SGLang expose no NCCL EP selector; TensorRT-LLM does. Its `NcclEP`
communication strategy (tensorrt_llm/_torch/moe/fused_moe/communication/nccl_ep.py at main
a6142c920), auto-selected after its NVLink strategies or forced with
TRTLLM_FORCE_COMM_METHOD=NCCL_EP, runs LL rank-major only and never NCCL EP's internal
DS_FP8E3M4 quantization: DeepSeek-FP8 activations cross it in BF16. LL rank-major BF16 rows run
that configuration: its group capacity, its staged dispatch under CUDA graph capture and its
window-backed zero-copy dispatch eager. HT, expert-major LL and DS-FP8 are modes no engine runs;
they stay library (`candidate`) rows.

FP8 uses NCCL EP's native DS_FP8E3M4 recipe, for low-latency decode only. No engine selects it,
so the matrix runs it only when named (sweep_matrix.OFF_PATH_PRECISIONS). Received E4M3 values
are dequantized into a BF16 staging plane, so the BF16 combine and correctness oracle remain the
common comparison contract. Normal mode remains BF16-only.

Communicator bootstrap: NCCL EP forms its OWN NCCL communicator (separate from PyTorch's
process group) via ``Communicator.init(nranks, rank, unique_id)``. Upstream broadcasts the
unique id with MPI; CollectiveX has no MPI, so rank 0 generates the id and we broadcast its
bytes over the already-initialized torch process group (see ``_bootstrap_comm``).

Python bindings are split across two wheels since v0.2: ``nccl-extensions`` owns ``nccl.ep``
(the libnccl_ep.so JIT runtime + Cython bindings) and ``nccl4py`` provides ``nccl.core``
(Communicator/UniqueId). The API surface used here is verified against the published
nccl-extensions wheel and driven exactly as upstream's ep_test.py drives it.
"""
from __future__ import annotations

import os
import re
import sys
import types
from importlib import metadata

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
_FP8_BLOCK_SIZE = 128

# SM budgets a default GroupConfig resolves to, as ncclEpCreateGroup does at nccl-extensions
# 1ee7136 (nccl_ep.cc:2219-2276): HT takes NCCL_EP_HT_DFLT_NUM_SMS (device/ht_ep_configs.cuh),
# LL every SM on the device; then the shared env override, then the per-operation one, each
# ignored outside [1, device SMs]. The group reads these variables once, at creation.
_HT_DEFAULT_NUM_SMS = 16
_SM_SHARED_ENV = "NCCL_EP_COMM_SMS"
_SM_OPERATION_ENV = {"dispatch": "NCCL_EP_DISPATCH_SMS", "combine": "NCCL_EP_COMBINE_SMS"}


def _resolve_num_sms(default: int, device_sms: int, env: dict) -> dict:
    """Per-operation SM budget the group runs, given the environment it was created under."""
    budgets = {}
    for operation, variable in _SM_OPERATION_ENV.items():
        budget = default
        for name in (_SM_SHARED_ENV, variable):
            # The library parses each with strtoul, so unset, empty and non-numeric all read as 0.
            digits = re.match(r"\s*\+?(\d+)", env.get(name) or "")
            requested = int(digits.group(1)) if digits else 0
            if 1 <= requested <= device_sms:
                budget = requested
        budgets[operation] = budget
    return budgets


def _blockwise_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Untimed oracle reference for native DS per-128-channel quantization."""
    if x.dim() != 2 or x.size(1) % _FP8_BLOCK_SIZE:
        raise ValueError(
            "NCCL EP FP8 requires a 2D hidden dimension divisible by "
            f"{_FP8_BLOCK_SIZE}, got {tuple(x.shape)}"
        )
    rows, hidden = x.shape
    blocks = x.view(rows, -1, _FP8_BLOCK_SIZE)
    amax = blocks.abs().float().amax(dim=2).clamp(1e-4)
    values = (blocks * (448.0 / amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return values.view(rows, hidden), (amax / 448.0).view(rows, -1)


def _blockwise_cast_back_into(
    values: torch.Tensor, scales: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    hidden = values.shape[-1]
    value_blocks = values.to(torch.float32).view(*values.shape[:-1], -1, _FP8_BLOCK_SIZE)
    scale_blocks = scales.to(torch.float32).view(*scales.shape, 1)
    out.copy_((value_blocks * scale_blocks).view(*values.shape[:-1], hidden).to(torch.bfloat16))
    return out


class NCCLEPBackend(EPBackend):
    name = "nccl-ep"
    # Set per row in __init__: "production" for the one configuration an engine selects
    # (TensorRT-LLM's NcclEP, LL rank-major with BF16 transport; communication_factory.py:254-278
    # and 438-459), "candidate" for the library modes no engine runs (HT, expert-major LL, DS-FP8).
    maturity = "candidate"
    # One library, two algorithms selected by args.mode. kernel_generation and the combine
    # semantics are switched to their LL values in __init__ (mirrors ep_deepep_v2), and
    # create_buffer appends what changes a row without changing its code path: "-cap<N>", the
    # group's token capacity (HT's per-step update and LL's rank-major routing reset are sized by
    # it, not by T), then the installed nccl-extensions version with its nightly date.
    #   normal      -> HT / FLAT layout / unweighted-rank-sum combine.
    #   low-latency -> LL / RANK_MAJOR layout / TensorRT-LLM's NcclEP contract.
    # "-routed" marks the generation whose timed HT dispatch charges the per-step
    # ncclEpUpdateHandle (see dispatch()); "v02" marks the nccl-extensions v0.2 kernels.
    # "-static" marks the combine input bound to the full static receive plane (see
    # `_bind_ht_recv_count`).
    # Graphed HT decode appends "-gum1": its communicator runs graph usage mode 1 (`_comm_config`),
    # which changes nothing outside a capture.
    kernel_generation = "nccl-ep-v02-ht-routed-zc-static"
    SUPPORTED_MODES = ("normal", "low-latency")
    SUPPORTED_PRECISIONS = ("bf16", "fp8")
    CUDA_GRAPH_MODES = ("normal", "low-latency")
    zero_copy = True
    _ll_expert_major = False

    @property
    def library_version(self) -> str:
        """Installed wheel identity, including the nightly date, for result provenance."""
        return metadata.version("nccl-extensions")

    def implementation_details(self):
        """The group's token capacity, whether dispatch writes a registered window (HT's combine
        reads one too; LL's combine always stages), the SM budgets the group resolved, and the
        engine whose configuration the row runs (None for a library row)."""
        device_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        default = device_sms if self._ll else _HT_DEFAULT_NUM_SMS
        return {
            "capacity_tokens_per_rank": self.max_dispatch,
            "zero_copy": bool(self.zero_copy),
            "num_sms": _resolve_num_sms(default, device_sms, self._sm_env),
            "engine_contract": self.engine_contract,
        }

    @property
    def cuda_graph_supported(self) -> bool:
        # HT replays at decode only, as a captured decode step runs it (engines run prefill
        # uncaptured); see methodology, CUDA Graph Replay.
        if self.mode == "normal" and getattr(self.args, "phase", None) != "decode":
            return False
        return super().cuda_graph_supported

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
        self._fp8 = getattr(self, "precision", "bf16") == "fp8"
        if self._fp8 and (not self._ll or args.phase != "decode"):
            raise ValueError("NCCL EP FP8 is enabled for low-latency decode only")
        self.stage_device_work = self._fp8
        # LL layout: rank-major (TensorRT-LLM's NCCL EP contract, the default) or expert-major
        # (DeepEP LL's contract, as vLLM/SGLang decode consume it). Both are native LL layouts.
        layout = os.environ.get("COLLX_NCCL_LL_LAYOUT", "rank-major")
        if layout not in ("rank-major", "expert-major"):
            raise ValueError(f"COLLX_NCCL_LL_LAYOUT must be rank-major or expert-major, got {layout!r}")
        self._ll_expert_major = self._ll and layout == "expert-major"
        # The one configuration an engine runs (module docstring); every other row is the library's.
        self.engine_contract = (
            "tensorrt-llm" if self._ll and not self._ll_expert_major and not self._fp8 else None
        )
        self.maturity = "production" if self.engine_contract else "candidate"
        # HT needs its windows in both regimes (`_bind_ht_recv_count`). LL rank-major follows
        # TensorRT-LLM: eager, dispatch writes into a registered NCCL window, but under CUDA graph
        # capture it writes into a plain context-owned tensor, which the library stages
        # (nccl_ep.py:305-327). So graphed rows run the staged dispatch and eager rows
        # (COLLX_CUDA_GRAPH=0) the window. Either way the library zero-copies only NVLink-only,
        # rank-major NONE/FWD dispatch (nccl_ep.cc:4589-4600): scale-out, expert-major and
        # DS_FP8E3M4, which generates its scales internally, always stage.
        self.zero_copy = not self._ll or not (
            self._internode or self._ll_expert_major or self._fp8 or self.cuda_graph_enabled
        )
        if self._ll_expert_major:
            # Weighted source-side combine over a per-expert padded receive: deepep-v2 LL's
            # contract, so this is the like-for-like row against the DeepEP-API backends.
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
        elif self.cuda_graph_enabled:
            # Only graphed HT captures a host NCCL collective (the routing ncclAllGather), so
            # only its rows change with the communicator's graph usage mode; eager rows, decode
            # under COLLX_CUDA_GRAPH=0 included, run the same communicator to no effect.
            self.kernel_generation = f"{type(self).kernel_generation}-gum1"
        if self._fp8:
            self.kernel_generation = f"{self.kernel_generation}-fp8-ds-e3m4"
            self.dispatch_dtype = "fp8-e4m3fn"
            self.dispatch_value_bytes = 1
            self.dispatch_scale_bytes_per_copy = (args.hidden // _FP8_BLOCK_SIZE) * 4
            self._dequant_into = self.fused_quantize(_blockwise_cast_back_into)
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
        dispatch_kwargs = {"send_only": 0, "round_scales": 0}
        if self._fp8:
            dispatch_kwargs["quantization_recipe"] = nccl_ep.DispatchQuantizationRecipe.DS_FP8E3M4
        self._dispatch_cfg = DispatchConfig(**dispatch_kwargs)
        self._combine_cfg = CombineConfig(send_only=0)
        self._comm = None
        self._ep_group = None
        # Exactly ONE handle per group, rebound per problem shape — see _ensure_handle.
        self._handle = None
        self._bound = None

    def buffer_cap(self, args):
        # Override EPBackend's shared 256-token LL default: the group is sized from the ladder.
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
            nranks=self.world_size, rank=self.rank, unique_id=uid, config=self._comm_config()
        )

    @staticmethod
    def _comm_config():
        """Graph usage mode 1: one graph at a time on this communicator, never concurrent with
        uncaptured work on it -- how the harness and a captured decode step both use it.

        The default (2, "mixing") wraps every captured collective in an external event wait and
        an event record so graph and uncaptured work can interleave (NCCL strongstream.cc). Traced
        on h100 HT decode EP8, that left ~14us idle before each routing ncclAllGather and made the
        graphed pair period 1.14x eager; mode 0/1 removes the gap (1.03x at T=1, 1.00x at T=256).
        """
        return nccl_core.NCCLConfig(graph_usage_mode=1)

    # ---- buffer construction -----------------------------------------------------------------

    def create_buffer(self, spec):
        """Bootstrap the communicator, create the EP group sized from the ladder maximum, and
        allocate the persistent receive/combine buffers reused across every ladder shape."""
        # Unit tests construct a deliberately partial adapter without __init__.
        self._fp8 = getattr(self, "_fp8", False)
        # Token capacity is the largest step the ladder runs. LL follows TensorRT-LLM, which sizes
        # its one NcclEP group to min(max_num_tokens, moe_max_num_tokens) (nccl_ep.py:98-103) and
        # rejects any step above it (124-129): the smallest such engine serving every rung has
        # max_num_tokens = max(ladder). HT has no engine to follow and keeps the same rule.
        self.max_dispatch = spec.max_tokens_per_rank
        hidden = self.args.hidden
        self._bootstrap_comm()
        # max_recv_tokens_per_rank: HT requires >0 and >= max_dispatch; LL auto-derives when 0.
        # world*max_dispatch is the recv-slot budget (every peer sends all its tokens here).
        max_recv = self.max_dispatch * self.world_size
        if self._fp8 and hidden % _FP8_BLOCK_SIZE:
            raise ValueError("NCCL EP DS FP8 requires hidden divisible by 128")
        config = GroupConfig(
            algorithm=self._algorithm,
            num_experts=self.args.experts,
            max_dispatch_tokens_per_rank=self.max_dispatch,
            max_recv_tokens_per_rank=max_recv,
            max_token_bytes=hidden * 2,  # BF16 input/combine bounds FP8 payload plus scales
            # ON fails any dispatch that lacks the window, so "-zc" cannot outlive the path it
            # names. OFF, which equals TensorRT-LLM's default AUTO at this pin, stages a dispatch
            # whose output carries no window.
            zero_copy=ZeroCopyMode.ON if self.zero_copy else ZeroCopyMode.OFF,
        )
        self._ep_group = nccl_ep.Group.create(self._comm, config)
        # The group read its SM-budget overrides just now (see _resolve_num_sms).
        self._sm_env = {
            name: os.environ.get(name) for name in (_SM_SHARED_ENV, *_SM_OPERATION_ENV.values())
        }
        self.kernel_generation = f"{self.kernel_generation}-cap{self.max_dispatch}-{self.library_version}"

        dev = self.device
        if self._ll_expert_major:
            # EXPERT_MAJOR recv: [num_local_experts, max_dispatch*num_ranks, hidden].
            slots = self.max_dispatch * self.world_size
            self._recv_x = nccl_core.torch.empty(
                (self.num_local_experts, slots, hidden),
                dtype=torch.float8_e4m3fn if self._fp8 else torch.bfloat16, device=dev
            )
            # Per-local-expert received-token counts, written by NCCL EP during dispatch.
            self._recv_count = torch.empty(
                (self.num_local_experts,), dtype=torch.int32, device=dev
            )
            self._combine_scratch = nccl_core.torch.empty(
                (self.num_local_experts, slots, hidden), dtype=torch.bfloat16, device=dev
            )
            if self._fp8:
                self._recv_scales = nccl_core.torch.empty(
                    (self.num_local_experts, slots, hidden // _FP8_BLOCK_SIZE),
                    dtype=torch.float32, device=dev,
                )
            self._recv_count_t = self._t(self._recv_count)
        elif self._ll:
            # RANK_MAJOR receive: [source rank, source slot, hidden].
            self._recv_x = nccl_core.torch.empty(
                (self.world_size, self.max_dispatch, hidden),
                dtype=torch.float8_e4m3fn if self._fp8 else torch.bfloat16, device=dev
            )
            self._combine_scratch = nccl_core.torch.empty(
                (self.world_size, self.max_dispatch, hidden), dtype=torch.bfloat16, device=dev
            )
            if self._fp8:
                self._recv_scales = nccl_core.torch.empty(
                    (self.world_size, self.max_dispatch, hidden // _FP8_BLOCK_SIZE),
                    dtype=torch.float32, device=dev,
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
            if self._fp8:
                self._recv_scales_t = self._t(self._recv_scales)
                self._combine_x_t = self._t(self._combine_scratch)

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
        rebinding IS the intended lifecycle. This also matches the other backends, which
        each allocate one object sized to the ladder maximum and vary the token count per call.

        Both create_handle and update are collective, and HT additionally performs the metadata
        exchange that fixes the received-token count, so they must run in the same order on
        every rank. They only ever run on a shape CHANGE, and every timed component is preceded
        by an untimed warm() on its own problem — so the collective always lands in warm (or in
        the oracle passes), never inside a timed window. Re-entering with the already-bound
        problem returns immediately without a collective or a sync.
        """
        self._fp8 = getattr(self, "_fp8", False)
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
            h.dispatch_inputs = DispatchInputs(tokens=h.in_tokens_t)
        else:
            # HT carries weights on dispatch; LL rank-major transports them with dispatch too.
            h.in_weights_t = self._t(p.topk_weights)
            h.dispatch_inputs = DispatchInputs(tokens=h.in_tokens_t, topk_weights=h.in_weights_t)
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
        if self._fp8:
            if self._ll_expert_major:
                h.dispatch_outputs = DispatchOutputs(tokens=self._recv_x_t, scales=self._recv_scales_t)
                h.dispatch_layout_info = LayoutInfo(expert_counters=self._recv_count_t)
            else:
                h.dispatch_outputs = DispatchOutputs(
                    tokens=self._recv_x_t, topk_weights=self._recv_w_t,
                    topk_idx=self._recv_idx_t, scales=self._recv_scales_t,
                )
                h.dispatch_layout_info = LayoutInfo(src_rank_counters=self._recv_count_t)
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

    def _bind_ht_recv_count(self, h):
        """Read HT's received-token count and bind the combine input to the full receive plane.

        FLAT combine takes the dispatch output's static `[num_recv_slots, hidden]` shape
        (ep_enums.h; a count-sized slice needs an NCCL_EP_AUTO group), and zero-copy leaves no
        staging copy for a slice to save. The count is read here, untimed, for the oracle.
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

    def semantic_payload(self, x):
        if not self._fp8:
            return x
        values, scales = _blockwise_cast_to_fp8(x)
        out = torch.empty_like(x, dtype=torch.bfloat16)
        return _blockwise_cast_back_into(values, scales, out)

    def dispatch(self, p):
        h = self._ensure_handle(p)
        stream = self._stream()
        if not self._ll:
            # Charge the per-step routing collective to the window. `ncclEpUpdateHandle` is
            # documented as a "per-step collective: prepare the handle for the given top-k
            # routing decisions", so a caller pays it before every HT dispatch whose routing
            # changed, and excluding it (as NVIDIA's ep_bench does) made HT dispatch the one
            # window that omitted its routing/layout work while deepep-v2, uccl-ep, MoRI and
            # FlashInfer all carry theirs per call. No engine runs HT (TensorRT-LLM's NcclEP is
            # LL-only), so this is a library row, and the update's HT work is sized by the
            # handle's capacity, not by T: capacity-wide memsets, a routing-map conversion, an
            # ncclAllGather of capacity rows and a metadata scan over them (nccl_ep.cc:4057-4137,
            # 4281). Every rung therefore carries a term set by the ladder's top, which
            # "-cap<N>" records. No sync or counter read here: the bound problem's counters are
            # deterministic and already read (_bind_ht_recv_count) in the untimed rebind.
            h.handle.update(h.topk_idx_t, layout_info=h.layout_info, stream=stream)
        if self._ll_expert_major:
            # LL EXPERT_MAJOR: tokens in, 3D per-expert padded tokens out, per-expert recv
            # counts written into expert_counters. No weights on the dispatch (the gate is
            # applied by the combine kernel at the source).
            if self._fp8:
                dispatch_outputs = h.dispatch_outputs
                layout_info = h.dispatch_layout_info
            else:
                dispatch_outputs = DispatchOutputs(tokens=self._recv_x_t)
                layout_info = LayoutInfo(expert_counters=self._recv_count_t)
            h.handle.dispatch(
                h.dispatch_inputs, dispatch_outputs, layout_info=layout_info,
                config=self._dispatch_cfg,
                stream=stream,
            )
            h.recv_x = self._recv_x
            h.recv_count = self._recv_count
        elif self._ll:
            # LL RANK_MAJOR returns one plane per source rank.
            if self._fp8:
                dispatch_outputs = h.dispatch_outputs
                layout_info = h.dispatch_layout_info
            else:
                dispatch_outputs = DispatchOutputs(
                    tokens=self._recv_x_t, topk_weights=self._recv_w_t, topk_idx=self._recv_idx_t
                )
                layout_info = LayoutInfo(src_rank_counters=self._recv_count_t)
            h.handle.dispatch(
                h.dispatch_inputs, dispatch_outputs, layout_info=layout_info,
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
                h.dispatch_inputs,
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
        if self._fp8:
            self._dequant_into(self._recv_x, self._recv_scales, self._combine_scratch)
            h.combine_input = self._combine_x_t
        else:
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
        recv_bf16 = h.recv_x
        slots = recv_bf16.shape[1]
        counts = h.recv_count.to(torch.int64)
        valid = torch.arange(slots, device=recv_bf16.device).unsqueeze(0) < counts.unsqueeze(1)
        source_rank, source_slot = valid.nonzero(as_tuple=True)
        h.source_rank = source_rank
        h.source_slot = source_slot
        payload = recv_bf16[source_rank, source_slot]
        if self._fp8:
            payload = _blockwise_cast_back_into(
                payload, self._recv_scales[source_rank, source_slot], torch.empty_like(payload, dtype=torch.bfloat16)
            )
        return self._local_id_view(
            payload, h.recv_idx[source_rank, source_slot],
            h.recv_w[source_rank, source_slot], self.num_local_experts,
        )

    def inspect_dispatch(self, p, h):
        if self._ll_expert_major:
            # EXPERT_MAJOR is deepep-v2 LL's padded [E, S, hidden] receive.
            payload = h.recv_x
            if self._fp8:
                self._dequant_into(h.recv_x, self._recv_scales, self._combine_scratch)
                payload = self._combine_scratch
            return self._expert_major_view(h, payload, h.recv_count)
        if self._ll:
            return self._ll_inspect_dispatch(p, h)
        # HT FLAT normal recv: front-packed to recv_total_counter, one row per received token.
        # recv_idx holds this rank's LOCAL expert indices [0, experts_per_rank) front-packed per
        # row (valid entries first, non-local padded to -1) with recv_w aligned to them — NOT the
        # global top-k. (Verified on h100 EP8: rank-1 token recv_idx=[2,17,-1..] for global experts
        # 34,49; rank 0 looks global only because its local range starts at 0.) So rebase the valid
        # locals to the GLOBAL ids the oracle compares, exactly as ep_uccl/ep_deepep_v2 normal
        # do. The oracle sorts each row over the top-k axis and sums the per-expert transforms,
        # so token order is free and no per-(token,expert) expansion is needed.
        count = int(h.count)
        return self._local_id_view(
            h.recv_x[:count], h.recv_idx[:count], h.recv_w[:count], self.experts_per_rank
        )

    def _ll_combine_transformed(self, p, h, transformed):
        """Scatter pre-reduced oracle rows into the LL combine plane."""
        combine_buf = self._combine_scratch if self._fp8 else self._recv_x
        combine_buf.zero_()
        combine_buf[h.source_rank, h.source_slot] = transformed.to(combine_buf.dtype)
        stream = self._stream()
        h.handle.combine(
            CombineInputs(
                tokens=self._combine_x_t if self._fp8 else
                (self._window_t(combine_buf) if self.zero_copy else self._t(combine_buf))
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
        # `_recv_x` is the zero-copy window peers read directly: fence this write across ranks
        # or a peer's combine can read it half-written (EP16). The timed path writes nothing here.
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        stream = self._stream()
        h.handle.combine(
            # Same input the timed path uses, so the two cannot diverge in shape.
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
