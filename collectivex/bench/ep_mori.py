#!/usr/bin/env python3
"""CollectiveX MoRI adapter: dispatch/combine over mori.ops, configured as the engines run it."""
from __future__ import annotations

import os
import sys
import types
from importlib import metadata
from typing import NamedTuple

# MoRI reads the symmetric-heap size when the heap is created (at shmem init,
# once a process group exists — see create_buffer). The pinned upstream
# inter-node benchmark uses 6 GiB for its InterNodeV1 staging and signal
# buffers. STATIC_HEAP is the shipped mode on every path, so this is 6 GiB
# really allocated (uncached) per rank against a worst-case EP8 need near 2 GiB.
os.environ["MORI_SHMEM_HEAP_SIZE"] = "6G"

import torch
import torch.distributed as dist

from ep_backend import EPBackend

try:
    import mori  # type: ignore
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: mori import failed: {exc!r}", file=sys.stderr)
    raise

# vLLM (block-quantized models) and SGLang quantize FP8 dispatch per 1x128 with FP32 scales.
_FP8_BLOCK = 128


class _Launch(NamedTuple):
    """One engine's EpDispatchCombineConfig launch fields for a cell."""

    kernel: str
    generation: str
    contract: str
    block_num: int
    rdma_block_num: int
    warp_num_per_block: int
    num_qp_per_pe: int


# vLLM's MoriAll2AllManager: one node runs IntraNode under both mori_* selectors; multi-node
# mori_high_throughput runs InterNodeV1 and mori_low_latency InterNodeV1LL, each with the arch's
# tuple; num_qp_per_pe is MoRI's default 1.
_VLLM_INTRANODE = _Launch("IntraNode", "intranode-vllm", "vllm", 80, 0, 16, 1)
_VLLM_INTERNODE_V1 = {
    "gfx950": _Launch("InterNodeV1", "inter-node-v1-vllm", "vllm", 64, 32, 8, 1),
    "gfx942": _Launch("InterNodeV1", "inter-node-v1-vllm", "vllm", 32, 16, 16, 1),
}
_VLLM_INTERNODE_V1LL = {
    "gfx950": _Launch("InterNodeV1LL", "inter-node-v1ll-vllm", "vllm", 64, 32, 8, 1),
    "gfx942": _Launch("InterNodeV1LL", "inter-node-v1ll-vllm", "vllm", 32, 16, 16, 1),
}


def _gfx_arch(device) -> str:
    """The realized CDNA arch; vLLM's MoRI integration supports gfx942 and gfx950 only."""
    arch = torch.cuda.get_device_properties(device).gcnArchName
    for name in ("gfx950", "gfx942"):
        if arch.startswith(name):
            return name
    raise RuntimeError(f"MoRI supports gfx942 and gfx950, not {arch!r}")


def _env_enabled(name: str) -> bool:
    """MoRI's env::IsEnvVarEnabled: unset, empty, 0, false, off and no are off; anything else is on."""
    value = os.environ.get(name, "")
    return bool(value) and value.lower() not in ("0", "false", "off", "no")


def _mori_version() -> str:
    """Installed MoRI build; setuptools-scm stamps an untagged build with its commit."""
    version = getattr(mori, "__version__", None)
    if version and version != "unknown":
        return str(version)
    try:
        return metadata.version("amd_mori")
    except metadata.PackageNotFoundError:
        return "unknown"


def _blockwise_codec(fp8_dtype, fp8_max):
    """(quantize, dequantize) for per-1x128 e4m3 with FP32 scales [m, n // 128]."""

    def quantize(x):
        m, n = x.shape
        blocks = x.view(m, -1, _FP8_BLOCK)
        amax = blocks.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
        values = (blocks * (fp8_max / amax.unsqueeze(2))).to(fp8_dtype).view(m, n)
        return values, (amax / fp8_max).view(m, -1)

    def dequantize(values, scales):
        m, n = values.shape
        blocks = values.view(m, -1, _FP8_BLOCK).float()
        return (blocks * scales.view(m, -1, 1)).view(m, n).to(torch.bfloat16)

    return quantize, dequantize


class MoRIBackend(EPBackend):
    name = "mori"
    maturity = "production"  # vLLM --all2all-backend mori_*; SGLang --moe-a2a-backend mori
    SUPPORTED_MODES = ("normal", "low-latency")
    SUPPORTED_PRECISIONS = ("bf16", "fp8")
    # Kernels launch with host-built args only (reset on-device since ROCm/mori#86); `stage`
    # slices by the per-rung `recv_tokens`, fixed for a rung, so it is safe to capture.
    CUDA_GRAPH_MODES = ("normal", "low-latency")
    requires_fresh_pair = True
    # InterNodeV1LL takes the caller-quantized tensor vLLM's prepare() builds, as IntraNode does.
    ll_quantizes_in_kernel = False

    @property
    def cuda_graph_supported(self) -> bool:
        # Decode only: vLLM captures at most 512 tokens by default, so prefill batches run eager.
        return super().cuda_graph_supported and getattr(self.args, "phase", None) == "decode"

    @property
    def library_version(self) -> str:
        return _mori_version()

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
        self._fp8 = self.precision == "fp8"
        self.experts_per_rank = args.experts // world_size
        arch = _gfx_arch(device)
        scale_out = args.scope == "scale-out"
        if self.mode == "low-latency":
            # On one node vLLM's mori_low_latency runs IntraNode, the normal decode row's kernel.
            if not scale_out:
                raise RuntimeError(
                    "MoRI low-latency is scale-out only: on one node vLLM's mori_low_latency runs "
                    "IntraNode, which the normal decode row measures"
                )
            self.launch = _VLLM_INTERNODE_V1LL[arch]
        else:
            self.launch = _VLLM_INTERNODE_V1[arch] if scale_out else _VLLM_INTRANODE
        # vLLM never sets MORI_ENABLE_SDMA (MoRI's default is off); a caller's setting is recorded.
        self._sdma = _env_enabled("MORI_ENABLE_SDMA")
        self.kernel_generation = self.launch.generation + ("-sdma" if self._sdma else "")
        kernel_enum = getattr(mori.ops, "EpDispatchCombineKernelType", None)
        if kernel_enum is None or not hasattr(kernel_enum, self.launch.kernel):
            raise RuntimeError(f"this MoRI image lacks EpDispatchCombineKernelType.{self.launch.kernel}")
        self._kernel_type = getattr(kernel_enum, self.launch.kernel)
        self._inter_node = self.launch.kernel in ("InterNodeV1", "InterNodeV1LL")
        self._fp8_dtype = None
        if self._fp8:
            if args.hidden % _FP8_BLOCK:
                raise ValueError(f"MoRI FP8 dispatch needs hidden divisible by {_FP8_BLOCK}, got {args.hidden}")
            # vLLM's current_platform.fp8_dtype(): AMD's e4m3fnuz on gfx94x, OCP e4m3fn on gfx950.
            self._fp8_dtype, fp8_max, label = (
                (torch.float8_e4m3fn, 448.0, "fp8-e4m3fn") if arch == "gfx950"
                else (torch.float8_e4m3fnuz, 240.0, "fp8-e4m3fnuz")
            )
            self._enable_fp8(label, *_blockwise_codec(self._fp8_dtype, fp8_max))
        # External-input kernels consume the dispatch output directly, so stage has no copy of its
        # own; under FP8 it dequantizes the received payload with its scales.
        self.stage_device_work = self._fp8
        self._library_config = {}

    def create_buffer(self, spec):
        args, world_size, rank, launch = self.args, self.world_size, self.rank, self.launch
        # vLLM passes min(8, EP size), which names this placement's node size on 8-GPU nodes only.
        gpu_per_node = min(8, world_size)
        if gpu_per_node != min(int(args.gpus_per_node), world_size):
            raise RuntimeError(
                f"MoRI gpu_per_node {gpu_per_node} does not match {args.gpus_per_node} GPUs per node"
            )

        world_group = torch.distributed.group.WORLD
        torch._C._distributed_c10d._register_process_group("default", world_group)
        # Every path, scale-out EP16 included, runs MoRI's default STATIC_HEAP: one
        # contiguous uncached allocation registered as a single MR. VMM_HEAP corrupts EP16
        # combine: ROCm 7.2's clr leaves VMM allocations requested uncached in the cached pool,
        # so cross-node partials land in stale lines (ROCm/mori#610). Nothing is set here so the
        # mode stays whatever MoRI ships as default; the guard below fails closed if that changes.
        heap_mode = os.environ.get("MORI_SHMEM_MODE", "STATIC_HEAP").upper()
        if self._inter_node and heap_mode != "STATIC_HEAP":
            raise RuntimeError(
                "MoRI scale-out requires STATIC_HEAP; VMM_HEAP corrupts inter-node combine on "
                "ROCm 7.2 (ROCm/mori#610)"
            )
        mori.shmem.shmem_torch_process_group_init("default")
        # The op clamps num_qp_per_pe to the QPs shmem created (MORI_NUM_QP_PER_PE, default 4).
        shmem_qps = int(mori.shmem.shmem_num_qp_per_pe())
        if shmem_qps < launch.num_qp_per_pe:
            raise RuntimeError(
                f"MoRI shmem created {shmem_qps} QPs per PE; {launch.num_qp_per_pe} required"
            )

        # MoRI preallocates one communicator buffer for the case's entire ladder, at least the
        # low-latency cap (see `buffer_cap`). Normal mode still takes the larger ladder maximum.
        self._cap = max(self.LL_LADDER_CAP, spec.max_tokens_per_rank)
        scale_dim = args.hidden // _FP8_BLOCK if self._fp8 else 0
        scale_type_size = torch.float32.itemsize if self._fp8 else 0
        # The engines' call: the launch tuple lives in the config and dispatch/combine pass no
        # per-call override. The external input buffer (MoRI's default) feeds combine from the
        # caller's tensor; quant_type "none" keeps combine BF16, so max_token_type_size covers BF16.
        self.config = mori.ops.EpDispatchCombineConfig(
            data_type=self._fp8_dtype if self._fp8 else torch.bfloat16,
            rank=rank,
            world_size=world_size,
            hidden_dim=args.hidden,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            max_token_type_size=torch.bfloat16.itemsize,
            max_num_inp_token_per_rank=self._cap,
            num_experts_per_rank=self.experts_per_rank,
            num_experts_per_token=args.topk,
            warp_num_per_block=launch.warp_num_per_block,
            block_num=launch.block_num,
            use_external_inp_buf=True,
            kernel_type=self._kernel_type,
            gpu_per_node=gpu_per_node,
            rdma_block_num=launch.rdma_block_num,
            num_qp_per_pe=launch.num_qp_per_pe,
            quant_type="none",
        )
        # MANUAL is MoRI's default and neither engine overrides it; AUTO would replace the config's
        # tuple with token-dependent tuning rules from the image.
        os.environ["MORI_EP_LAUNCH_CONFIG_MODE"] = "MANUAL"
        self.op = mori.ops.EpDispatchCombineOp(self.config)
        if getattr(self.op, "launch_config_mode", None) != "MANUAL":
            raise RuntimeError("MoRI explicit launch configuration was not applied")
        # Read back the handle's realized config (MoRI's get_handle_info snapshot, in its key names).
        requested = {
            "block_num": launch.block_num,
            "rdma_block_num": launch.rdma_block_num,
            "warp_num_per_block": launch.warp_num_per_block,
            "gpu_per_node": gpu_per_node,
            "scale_dim": scale_dim,
            "scale_type_size": scale_type_size,
            "use_external_inp_buffer": True,
        }
        info = getattr(self.op, "_handle_info", None)
        realized = {key: info.get(key) for key in requested} if isinstance(info, dict) else dict(requested)
        drift = {key: (value, realized[key]) for key, value in requested.items() if realized[key] != value}
        if drift:
            raise RuntimeError(f"MoRI realized a different launch/topology configuration: {drift}")
        self._library_config = {
            "contract": launch.contract,
            "kernel": launch.kernel,
            **realized,
            "realized_from": "handle" if isinstance(info, dict) else "request",
            # Shmem created at least this many (checked above), so the op keeps the request.
            "num_qp_per_pe": launch.num_qp_per_pe,
            "shmem_num_qp_per_pe": shmem_qps,
            "quant_type": "none",
            "sdma": self._sdma,
            "heap_mode": heap_mode,
            "launch_config_mode": "MANUAL",
        }

    def implementation_details(self):
        return dict(self._library_config)

    def _topk_idx_dtype(self):
        return torch.int32

    def make_problem(self, T, idx, weights, x):
        problem = super().make_problem(T, idx, weights, x)
        problem.indices, problem.weights = problem.topk_idx, problem.topk_weights
        return problem

    def dispatch(self, p):
        # Quantize inside dispatch, where vLLM's prepare() pays it, then make vLLM's call.
        x, scales = self._quant(p.dispatch_x) if self._fp8 else (p.dispatch_x, None)
        output, weights, out_scales, indices, recv_num = self.op.dispatch(x, p.weights, scales, p.indices)
        return types.SimpleNamespace(
            dispatch_output=output,
            dispatch_scales=out_scales,
            dispatch_weights=weights,
            dispatch_indices=indices,
            recv_num=recv_num[0],
            combine_input=None,
        )

    def _received(self, h, rows):
        """BF16 view of the first `rows` received tokens: the payload, or its blockwise dequant."""
        if not self._fp8:
            return h.dispatch_output[:rows]
        if h.dispatch_scales is None:
            raise RuntimeError("MoRI FP8 dispatch returned no scales")
        return self._fp8_codec[1](h.dispatch_output[:rows], h.dispatch_scales[:rows])

    def stage(self, p, h):
        rows = getattr(p, "recv_tokens", None)
        if not isinstance(rows, int) or rows < 0 or rows > h.dispatch_output.size(0):
            raise RuntimeError("MoRI receive count was not validated before staging")
        # The kernel's staging loop is bounded by `tokenIdx < totalRecvTokenNum` over
        # `args.inpTokenBuf`, not by the buffer, so only the filled rows need converting.
        h.combine_input = self._received(h, rows) if self._fp8 else h.dispatch_output

    def combine(self, p, h):
        # `indices` must be this rank's own [T, topk] routing -- the tensor handed to
        # dispatch() -- never dispatch()'s returned recv-slot-layout indices. The inter-node
        # combines key tokenIndices by this rank's own token id (the interNodeDispSendMap
        # key), so recv-layout indices make its node predicate answer from an unrelated
        # token's routing and silently drop or pollute cross-node partials (ROCm/mori#475;
        # upstream now rejects the mismatched row count, ROCm/mori#546). The intranode
        # kernels reach the gather through dispatch-side metadata instead and do not
        # consume the routing content, which is why EP8 was always clean.
        combined, _weights = self.op.combine(h.combine_input, None, p.indices)
        return combined[:p.T]

    def inspect_dispatch(self, p, h):
        count = self.recv_tokens(h)
        if h.dispatch_weights is None:
            raise RuntimeError("MoRI dispatch did not expose gate weights")
        received = (h.dispatch_output, h.dispatch_indices, h.dispatch_weights)
        if self._fp8 and h.dispatch_scales is not None:
            received += (h.dispatch_scales,)
        if count < 0 or any(tensor.ndim == 0 or count > tensor.size(0) for tensor in received):
            raise RuntimeError("MoRI receive count exceeds dispatch metadata")
        return self._global_id_view(
            self._received(h, count),
            h.dispatch_indices[:count].to(torch.int64),
            h.dispatch_weights[:count].to(torch.float32),
            self.experts_per_rank,
        )

    def combine_transformed(self, p, h, transformed):
        h.combine_input = transformed.to(torch.bfloat16)
        rows = getattr(p, "recv_tokens", None)
        if not isinstance(rows, int) or rows < 0 or rows > h.combine_input.size(0):
            raise RuntimeError("MoRI receive count was not validated before transformed combine")
        return self.combine(p, h)

    def recv_tokens(self, h):
        return int(h.recv_num.item())

    def finalize(self, rc):
        try:
            dist.barrier()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc if 0 <= rc <= 255 else 1)
