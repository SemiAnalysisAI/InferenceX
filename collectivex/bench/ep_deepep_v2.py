#!/usr/bin/env python3
"""DeepEP PR #605 adapter with the exact upstream PR #630 and #640 fixes."""

from __future__ import annotations

import inspect
import os
import sys
import types
from importlib import metadata
from pathlib import Path

import torch
import torch.distributed as dist
from ep_backend import EPBackend
from ep_legacy_ll import LegacyBufferLL

try:
    import deep_ep
    from deep_ep import ElasticBuffer  # type: ignore
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: DeepEP V2 import failed: {exc!r}", file=sys.stderr)
    raise


# The source pin in runtime/common.sh is upstream main, which carries #630 and #640. This
# adapter does not check the wheel's commit tag, only that the loaded deep_ep exposes
# ElasticBuffer. The pin is mirrored here for `library_version`, since the wheel reports
# 2.1.0+local (the build patches the tree, so setup.py finds no clean git hash).
_DEEPEP_COMMIT = "01dc3aaac82068020353dce2c302e38153c0bfaa"

# vLLM's ElasticBuffer flags: VLLM_DEEPEP_V2_PREFER_OVERLAP and
# VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION both default to 0 (vllm/envs.py) and reach the buffer
# through DeepEPV2All2AllManager._make_all2all_kwargs. With overlap off DeepEP sizes its kernels at
# max(model, 64) SMs, and hybrid mode at most four channels per SM. With a single reduction a
# hybrid forwarder relays each scale-up rank's row unreduced, so the only reduction is the
# source's FP32 sum over unique ranks (ep_oracle.RankFP32), at any scope.
_PREFER_OVERLAP_WITH_COMPUTE = False
_ALLOW_MULTIPLE_REDUCTION = False

# Low-latency receive sizing, deliberately two numbers: _LL_BUFFER_CAP sizes the pre-allocated
# receive (and so the transport footprint and fp8 dequant volume), _LL_LADDER_CAP bounds which
# token counts are measured. Equal today, but kept separate so the ladder can be clamped around a
# kernel defect at one rung without moving the footprint (see `create_buffer`).
_LL_BUFFER_CAP = 256
_LL_LADDER_CAP = 256
assert _LL_LADDER_CAP <= _LL_BUFFER_CAP <= 511, (
    "the LL receive cap must fit NVSHMEM_QP_DEPTH=1024 ((cap + 1) * 2 <= 1024 => cap <= 511) "
    "and the measured ladder must fit inside the buffer"
)


def _fp8_cast_helpers():
    """The pinned per-token FP8 cast pair (blockwise e4m3fn, per-128-block FP32 scale).

    Imported lazily so a BF16-only run never depends on deep_ep.utils.math, and so
    the quantization the oracle models is byte-identical to what dispatch transports.
    """
    from deep_ep.utils.math import per_token_cast_to_fp8, per_token_cast_back
    return per_token_cast_to_fp8, per_token_cast_back


def _jit_cache_directory(
    args,
    world_size: int,
    max_tokens: int,
    allow_hybrid_mode: bool,
    realized: dict[str, int | bool],
    use_fp8: bool,
) -> str:
    values = (
        args.runner, world_size, args.hidden, args.topk, args.experts,
        args.experts, max_tokens,
        int(allow_hybrid_mode), realized["num_allocated_qps"], realized["num_sms"],
        int(use_fp8),
    )
    return "jit-" + "-".join(str(value) for value in values)


# GIN/GDAKI allocates num_allocated_qps device QPs per peer rank on the local NIC
# (contexts x world_size QPs, before NCCL's own connection QPs). Upstream's hybrid
# default (129, or 65 with fast RDMA atomics), which vLLM leaves in place, exhausts the
# per-NIC QP budget at EP16: construction dies in ncclDevCommCreate with ibv_create_qp
# ENOMEM once NCCL's regular QPs land on top (identical on H200 bare-metal and B200 pods; on
# CX-7 the budget sits between 784 and 1040 QPs — 49x16 initializes, 65x16 does
# not). Spending a fixed ~512-QP budget keeps every EP size inside that limit
# with headroom. Only EP16 reaches this: the hybrid path needs world > scale_up_domain,
# so EP8 passes 0 and takes upstream's non-hybrid default of 17, and EP32 is not in the
# sweep. EP16 resolves to 33 (33 and 49 verified on the failing H200 pair). An explicit
# value also skips upstream's rank-local ibstat probe, which is not guaranteed to resolve
# identically across ranks. The allocation is fixed at construction and does not depend on
# the SM count: the 64-SM hybrid kernels ask for 64 x 16 + 1 QPs (get_theoretical_num_qps)
# and are capped here, as the fewer-SM kernels were.
_GIN_QP_BUDGET = 512


def _hybrid_num_allocated_qps(world_size: int) -> int:
    return max(9, 1 + _GIN_QP_BUDGET // world_size)


def _configure_gin_mode(args, world_size: int) -> bool:
    scale_up_domain = int(args.scale_up_domain)
    allow_hybrid_mode = world_size > scale_up_domain
    if allow_hybrid_mode:
        os.environ.pop("EP_DISABLE_GIN", None)
    else:
        os.environ["EP_DISABLE_GIN"] = "1"
    return allow_hybrid_mode


def _require_runtime() -> None:
    """Capability check only: the loaded deep_ep must expose ElasticBuffer (still
    catches the b300 image-bundled deep_ep 1.2.1 shadowing the from-source build,
    which lacks the class)."""
    if not inspect.isclass(ElasticBuffer) or ElasticBuffer.__name__ != "ElasticBuffer":
        raise RuntimeError("invalid DeepEP V2 runtime: deep_ep.ElasticBuffer is absent")


class DeepEPV2Backend(LegacyBufferLL, EPBackend):
    name = "deepep-v2"
    maturity = "production"  # vLLM --all2all-backend deepep_v2; SGLang --moe-a2a-backend deepep
    # Two kernel families under one adapter, selected by mode:
    #   normal      -> PR #605 ElasticBuffer (LSA vs hybrid GIN are transport paths, not
    #                  kernel families); rank-deduplicated unweighted-rank-sum combine.
    #   low-latency -> the legacy deep_ep.Buffer IBGDA decode kernels
    #                  (low_latency_dispatch/combine); per-expert padded layout with a
    #                  source-side weighted-kernel-sum combine (ep_legacy_ll). kernel_generation
    #                  and the combine semantics are switched to their LL values in __init__.
    # Normal rows run vLLM's flags (overlap off, single reduction) without the host sync in both
    # phases ("-vllm-nosync").
    kernel_generation = "v2-elastic-buffer-vllm-nosync"
    SUPPORTED_MODES = ("normal", "low-latency")
    SUPPORTED_PRECISIONS = ("bf16", "fp8")
    # Legacy decode kernels are graph compatible; ElasticBuffer normal replays at decode only.
    CUDA_GRAPH_MODES = ("normal", "low-latency")
    # vLLM's low-latency send/receive split (see LegacyBufferLL).
    ll_recv_hook = True
    # The num_sms the timed dispatch passes: 0 hands the count to DeepEP's model, as vLLM does;
    # create_buffer overrides it only on EFA.
    _dispatch_num_sms = 0

    @staticmethod
    def init_process_group(dist, rank, world_size, device):
        # PR #605 reuses PyTorch's NCCL communicator through `_comm_ptr`; device_id forms it
        # eagerly, before ElasticBuffer construction.
        dist.init_process_group("nccl", device_id=device)

    def __init__(self, args, rank, world_size, local_rank, device):
        # Mode picks the kernel family (normal ElasticBuffer vs low-latency legacy
        # Buffer); base SUPPORTED_MODES enforces the allowed set.
        super().__init__(args, rank, world_size, local_rank, device)
        self.group = dist.group.WORLD
        self._fp8 = self.precision == "fp8"
        # FP8 dispatch dequantizes the received (e4m3fn, per-128-block FP32 scale) payload
        # back to the BF16 combine sends, which is real device work and so a separately-
        # timed component. (Normal mode quantizes inside the timed dispatch; low-latency casts
        # inside the dispatch kernel — either way the dequant lands in stage().)
        self.stage_device_work = self._fp8
        if self._fp8:
            # Resolve the pinned cast pair once (lazily, so a BF16-only run never imports
            # deep_ep.utils.math) so the timed stage() does no module lookup in the
            # measured region. Low-latency keeps the eager quantize, whose bits its in-kernel
            # quantise matches.
            to_fp8, self._cast_back = _fp8_cast_helpers()
            self._enable_fp8("fp8-e4m3fn", to_fp8, self._cast_back)
        # Normal mode runs ElasticBuffer as vLLM does with CUDA graphs on (its default): the
        # prepare/finalize keys on the model-level flag (use_cudagraph = not enforce_eager), so every
        # batch, prefill as well as decode, dispatches without a host sync or expand, its receive
        # bounded at the next power of two of the batch. No dispatch here waits on the host
        # (host_synced_dispatch stays False).
        if self.mode == "normal":
            self.combine_reduction = "domain-fp32" if _ALLOW_MULTIPLE_REDUCTION else "rank-fp32"
        if (os.environ.get("EP_WIN_RELAXED_ORDERING") == "1" and self.mode == "normal"
                and world_size > int(args.scale_up_domain)):
            self.kernel_generation += "-relaxed-ordering"  # its own series; see methodology.md
        if self.mode == "low-latency":
            self._enable_ll("legacy-buffer-ll")  # the legacy Buffer IBGDA decode kernels

    @property
    def cuda_graph_supported(self) -> bool:
        # Normal replays at decode, as vLLM captures it; prefill stays eager, because vLLM runs
        # large prefill batches uncaptured (same no-sync path, outside the graph).
        if self.mode == "normal" and getattr(self.args, "phase", None) != "decode":
            return False
        return super().cuda_graph_supported

    @property
    def library_version(self) -> str:
        """The installed wheel at the pinned source commit, e.g. 2.1.0+local@01dc3aa..."""
        try:
            wheel = metadata.version("deep_ep")
        except metadata.PackageNotFoundError:
            wheel = getattr(deep_ep, "__version__", "unknown")
        return f"{wheel}@{_DEEPEP_COMMIT}"

    def _dispatch_capacity(self, tokens):
        """Per-call `num_max_tokens_per_rank`: without the host sync DeepEP receives that many
        rows per rank, so pass the next power of two of the batch, as vLLM does (deepep_v2.py).
        """
        return min(self.max_tokens, 1 << max(0, int(tokens) - 1).bit_length())

    def buffer_cap(self, args):
        if self.mode == "low-latency":
            # LL pre-allocates a fixed [num_local_experts, cap * num_ranks, hidden] receive
            # buffer, so the cap is a hard per-rank dispatch-slot bound; the harness clamps the
            # ladder to it and records any dropped point in the artifact. Clamping here is the
            # containment lever if a kernel defect reds the top rung (check the DeepEP pin first).
            return _LL_LADDER_CAP
        return None

    def create_buffer(self, spec):
        args, world_size = self.args, self.world_size
        self.max_tokens = spec.max_tokens_per_rank
        if self.mode == "low-latency":
            # Size the LL buffer from the fixed cap, not from the clamped ladder: the receive
            # footprint sets both the transport's memory traffic and the fp8 dequant volume
            # (`_ll_recv_bf16` converts the whole padded receive), so following the ladder would
            # shift every retained rung and break comparability with the published series.
            if spec.max_tokens_per_rank > _LL_BUFFER_CAP:
                raise RuntimeError(
                    f"low-latency ladder maximum {spec.max_tokens_per_rank} exceeds the LL "
                    f"buffer cap {_LL_BUFFER_CAP}"
                )
            self.max_tokens = _LL_BUFFER_CAP
            self._create_ll_buffer(spec)
            return
        _require_runtime()
        jit_root = Path(os.environ["EP_JIT_CACHE_DIR"])
        # Hybrid exactly where vLLM's VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1 autoselects it: DeepEP
        # goes hybrid only when the NVLink (LSA) team is smaller than the world, so passing False
        # inside one domain selects the same direct kernels while skipping the GIN set-up.
        allow_hybrid_mode = _configure_gin_mode(args, world_size)
        self.buffer = ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=self.max_tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            use_fp8_dispatch=self._fp8,  # FP8 sizes the buffer for the (e4m3fn, scale) tuple.
            deterministic=False,
            allow_hybrid_mode=allow_hybrid_mode,
            allow_multiple_reduction=_ALLOW_MULTIPLE_REDUCTION,
            prefer_overlap_with_compute=_PREFER_OVERLAP_WITH_COMPUTE,
            num_gpu_timeout_secs=100,
            explicitly_destroy=True,
            # 0 is upstream's use-the-default sentinel; only hybrid (GIN) mode
            # needs the explicit budget-derived allocation.
            num_allocated_qps=(
                _hybrid_num_allocated_qps(world_size) if allow_hybrid_mode else 0
            ),
        )
        # The hybrid choice above matches vLLM's autoselect only if the registry's scale-up domain
        # is the LSA team NCCL realized; fail closed rather than run other kernels (or hang, GIN off).
        lsa_ranks, expected = int(self.buffer.num_nvlink_ranks), min(world_size, int(args.scale_up_domain))
        if lsa_ranks != expected:
            raise RuntimeError(
                f"NCCL LSA team has {lsa_ranks} ranks, but scale_up_domain={args.scale_up_domain} at "
                f"EP{world_size} expects {expected}"
            )
        # vLLM passes neither num_sms nor num_qps, so each dispatch resolves them from DeepEP's
        # model, get_theoretical_num_sms(num_experts, num_topk), and combine reuses the handle's
        # count; num_qps follows from the SM count. The call below is the dispatch's own (same
        # cached arguments), so the recorded count is the one the kernels run with. The model
        # reads the scale-out rate from ibstat, which cannot see EFA: there the hybrid branch would
        # divide by zero, so the registry's per-GPU rate (COLLX_RDMA_GBS, set on EFA pools only)
        # stands in and the result is passed explicitly. With overlap off the model's own count
        # (10-26 SMs on these pools) is raised to 64 either way.
        rdma_gbs = float(os.environ.get("COLLX_RDMA_GBS") or 0)
        if rdma_gbs:
            self.num_sms = int(self.buffer.get_theoretical_num_sms(
                int(args.experts), int(args.topk), rdma_gbs=rdma_gbs))
            self._dispatch_num_sms = self.num_sms
        else:
            self.num_sms = int(self.buffer.get_theoretical_num_sms(int(args.experts), int(args.topk)))
            self._dispatch_num_sms = 0
        self.num_qps = int(self.buffer.get_theoretical_num_qps(self.num_sms))
        realized = {
            "num_sms": self.num_sms,
            "num_qps": self.num_qps,
            "num_allocated_qps": int(self.buffer.num_allocated_qps),
            "allow_hybrid_mode": allow_hybrid_mode,
            "lsa_team_ranks": lsa_ranks,
            "sm_model_rdma_gbs": rdma_gbs or None,
        }
        self._realized = realized
        jit_cache_directory = _jit_cache_directory(
            args,
            world_size,
            self.max_tokens,
            allow_hybrid_mode,
            realized,
            self._fp8,
        )
        os.environ["EP_JIT_CACHE_DIR"] = str(jit_root / jit_cache_directory)

    def implementation_details(self):
        # Every row here runs vLLM's configuration; see the module constants and LegacyBufferLL.
        details = {"engine_contract": "vllm", **super().implementation_details()}
        if self.mode == "low-latency":
            details.update({
                "allow_nvlink_for_low_latency_mode": True,
                "allow_mnnvl": getattr(self, "_ll_allow_mnnvl", None),
            })
            return details
        details.update(getattr(self, "_realized", {}))
        details.update({
            "prefer_overlap_with_compute": _PREFER_OVERLAP_WITH_COMPUTE,
            "allow_multiple_reduction": _ALLOW_MULTIPLE_REDUCTION,
            "do_cpu_sync": False,
            "do_expand": False,
            "receive_capacity": "next-pow2-of-tokens",
            "buffer_tokens_per_rank": getattr(self, "max_tokens", None),
            "zero_copy": False,
        })
        return details

    def _create_ll_buffer(self, spec):
        """Construct the legacy low-latency deep_ep.Buffer (IBGDA decode kernels).

        Distinct from the ElasticBuffer path: LL always allocates the NVSHMEM RDMA
        buffer and forces IBGDA even for single-node EP8, so there is no NVLink-only
        fallback. `allow_nvlink_for_low_latency_mode` lets NVLink carry intranode
        traffic alongside IBGDA; it does not remove the IBGDA requirement.
        """
        args, world_size = self.args, self.world_size
        assert args.experts % world_size == 0, (
            "low-latency EP requires num_experts divisible by the EP size"
        )
        self.num_local_experts = args.experts // world_size
        if not hasattr(deep_ep.Buffer, "low_latency_dispatch"):
            raise RuntimeError(
                "invalid DeepEP LL runtime: deep_ep.Buffer.low_latency_dispatch is absent"
            )
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            self.max_tokens, args.hidden, world_size, args.experts
        )
        kwargs = {}
        # On an MNNVL rack the scale-up fabric is NVLink across trays, but the legacy Buffer
        # defaults `allow_mnnvl=False` and a False there self-sets NVSHMEM_DISABLE_MNNVL, so
        # leaving it unset runs the low-latency kernels over IBGDA on exactly the systems whose
        # fast path is MNNVL. vLLM leaves it off unless VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL is set,
        # which its env doc says to turn on "for better latency on GB200 like system", so it is
        # set on MNNVL pools only. Keyed on the reported topology, not the SKU name.
        self._ll_allow_mnnvl = str(getattr(args, "scale_up_transport", "")) == "mnnvl"
        if self._ll_allow_mnnvl:
            if "allow_mnnvl" in inspect.signature(deep_ep.Buffer.__init__).parameters:
                kwargs["allow_mnnvl"] = True
            else:
                raise RuntimeError(
                    "MNNVL scale-up needs deep_ep.Buffer(allow_mnnvl=...); this wheel lacks it, "
                    "so the low-latency path would silently run over IBGDA"
                )
        # vLLM's DeepEPLLAll2AllManager also reserves a 1 GiB NVLink buffer (num_nvl_bytes); the
        # low-latency kernels never touch it (they reach peers through the NVSHMEM heap), so it
        # is left at 0 here.
        self.buffer = deep_ep.Buffer(
            self.group,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            # LL requires the QP-per-rank count to equal the number of local experts.
            num_qps_per_rank=self.num_local_experts,
            allow_nvlink_for_low_latency_mode=True,
            explicitly_destroy=True,
            **kwargs,
        )

    def _topk_idx_dtype(self):
        # DeepEP V2's kernels key routing indices on deep_ep.topk_idx_t, not int64.
        return deep_ep.topk_idx_t

    def dispatch(self, p):
        if self.mode == "low-latency":
            return self._ll_dispatch(p)
        # Quantise here, not in make_problem: production runs one fused bf16->fp8 kernel per
        # forward pass immediately before this collective, so the timed window must contain it.
        dispatch_x = self._quant(p.dispatch_x) if self._fp8 else p.dispatch_x
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = self.buffer.dispatch(
            dispatch_x,
            topk_idx=p.topk_idx,
            topk_weights=p.topk_weights,
            num_experts=self.args.experts,
            num_max_tokens_per_rank=self._dispatch_capacity(p.T),
            expert_alignment=1,
            num_sms=self._dispatch_num_sms,  # 0 (DeepEP's model) except on EFA; see create_buffer
            async_with_compute_stream=False,
            do_handle_copy=True,
            do_cpu_sync=False,
            do_expand=False,
        )
        return types.SimpleNamespace(
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            handle=handle,
        )

    def stage(self, p, h):
        if self.mode == "low-latency":
            # The timed combine sends the padded per-expert receive back as BF16 (dequant
            # under FP8). Value correctness is exercised by the oracle's combine_transformed
            # path; this only has to move the right shape for timing.
            h.combine_input = self._ll_recv_bf16(h.recv_x)
            return
        if self._fp8:
            # Dequantize the received (fp8, scale) tuple to the BF16 combine sends. Without the
            # host sync that is the whole worst-case plane: combine requires its input to match
            # the handle's capacity x ranks rows, of which it reads only the valid prefix.
            h.combine_input = self._cast_back(h.recv_x[0], h.recv_x[1])
        else:
            # BF16: the received buffer is already the semantic payload to combine.
            h.combine_input = h.recv_x

    def combine(self, p, h):
        if self.mode == "low-latency":
            # The kernel applies topk_weights internally (weighted).
            return self._ll_combine(p, h, h.combine_input)
        # SM and QP counts default as in vLLM: the dispatch handle's SMs, DeepEP's QP model.
        combined_x, _, _ = self.buffer.combine(
            h.combine_input,
            handle=h.handle,
            async_with_compute_stream=False,
        )
        return combined_x

    def inspect_dispatch(self, p, h):
        if self.mode == "low-latency":
            return self._ll_inspect_dispatch(p, h)
        count = self.recv_tokens(h)
        if self._fp8:
            # Dequantize the sliced (fp8, scale) recv tuple so the payload the oracle
            # inspects is BF16 [count, hidden] (source-ID decode + bit-exact compare).
            payload = self._cast_back(h.recv_x[0][:count], h.recv_x[1][:count])
        else:
            payload = h.recv_x[:count]
        return self._local_id_view(
            payload, h.recv_topk_idx[:count], h.recv_topk_weights[:count],
            self.args.experts // self.world_size,
        )

    def combine_transformed(self, p, h, transformed):
        if self.mode == "low-latency":
            return self._ll_combine_transformed(p, h, transformed)
        # Combine always sends BF16. Under FP8, recv_x is an (fp8, scale) tuple, so the
        # BF16 combine buffer is shaped from the fp8 payload rather than zeros_like it.
        if self._fp8:
            combine_input = torch.zeros_like(h.recv_x[0], dtype=torch.bfloat16)
        else:
            combine_input = torch.zeros_like(h.recv_x)
        combine_input[: transformed.shape[0]].copy_(transformed.to(combine_input.dtype))
        combined, _, _ = self.buffer.combine(
            combine_input,
            handle=h.handle,
            async_with_compute_stream=False,
        )
        return combined

    def recv_tokens(self, h):
        if self.mode == "low-latency":
            return int(h.recv_count.sum().item())
        return int(h.handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
