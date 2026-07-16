#!/usr/bin/env python3
"""CollectiveX MoRI adapter: native BF16 dispatch/combine over mori.ops."""
from __future__ import annotations

import os
import sys
import types

# MoRI reads the symmetric-heap size when the heap is created (at shmem init,
# once a process group exists — see create_buffer). The pinned upstream
# inter-node benchmark uses 6 GiB for its InterNodeV1 staging and signal
# buffers; under VMM_HEAP this is a virtual reservation backed on demand.
os.environ["MORI_SHMEM_HEAP_SIZE"] = "6G"

import torch
import torch.distributed as dist

from ep_backend import EPBackend

try:
    import mori  # type: ignore
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: mori import failed: {exc!r}", file=sys.stderr)
    raise


def _project_local_metadata(torch_module, raw_expert_ids, raw_weights, rank, experts_per_rank):
    local_start = rank * experts_per_rank
    local = (raw_expert_ids >= local_start) & (
        raw_expert_ids < local_start + experts_per_rank
    )
    expert_ids = torch_module.where(
        local, raw_expert_ids, torch_module.full_like(raw_expert_ids, -1)
    )
    weights = torch_module.where(local, raw_weights, torch_module.zeros_like(raw_weights))
    return expert_ids, weights, raw_expert_ids[local] - local_start


class MoRIBackend(EPBackend):
    name = "mori"
    SUPPORTED_PRECISIONS = ("bf16", "fp8")
    combine_needs_redispatch = True
    dispatch_needs_combine_cleanup = True

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
        self._fp8 = self.precision == "fp8"
        # MoRI's FP8 wire format is the SKU's arch fact, not a scheduled axis: gfx942
        # (MI300X/MI325X) uses OCP-unsigned-zero e4m3fnuz, gfx950 (MI355X) uses OCP
        # e4m3fn. Read from the realized device so it never has to be plumbed through
        # argv. FP8 dispatch is caller-prequantized: MoRI's dispatch kernel keys purely
        # on the passed tensor dtype, so handing it an e4m3 tensor selects the FP8
        # dispatch kernel with no in-kernel cast. Combine stays genuinely BF16 (quant_type
        # "none" -> EpCombineIntraNodeKernel_bf16_nop2p); "fp8_direct_cast" would instead
        # pick the _fp8cast combine that compresses the BF16 combine wire to FP8.
        self._fp8_dtype = None
        if self._fp8:
            arch = torch.cuda.get_device_properties(device).gcnArchName
            if arch.startswith("gfx950"):
                self._fp8_dtype = torch.float8_e4m3fn
                self.dispatch_dtype = "fp8-e4m3fn"
            elif arch.startswith("gfx942"):
                self._fp8_dtype = torch.float8_e4m3fnuz
                self.dispatch_dtype = "fp8-e4m3fnuz"
            else:
                raise RuntimeError(f"MoRI FP8 dispatch unsupported on arch {arch!r}")
            self.dispatch_value_bytes = 1
            self.dispatch_scale_bytes_per_copy = 0  # plain e4m3 cast: no scale payload
        self.ep_size = world_size
        self.experts_per_rank = args.experts // self.ep_size
        gpus_per_node = int(args.gpus_per_node)
        scale_out = args.scope == "scale-out"

        # The kernel is a pinned function of the cell, not an operator choice,
        # and no cell runs a low-latency-family kernel (mirroring the
        # normal-mode-only NVIDIA backends): scale-up uses the direct IntraNode
        # kernel on every CDNA SKU (mori's default; `kernel_type` kwarg
        # omitted); scale-out EP16 uses InterNodeV1, whose required enum member
        # is an image-lineage check.
        # (kernel, generation label, (block_num, rdma_block_num, dispatch_warps, combine_warps))
        kernel_name, self.kernel_generation, blocks = (
            ("InterNodeV1", "inter-node-v1", (96, 64, 8, 8)) if scale_out
            else ("IntraNode", "intranode", (80, 0, 16, 8))
        )
        self._kernel_type = None
        if kernel_name != "IntraNode":
            kernel_enum = getattr(mori.ops, "EpDispatchCombineKernelType", None)
            if kernel_enum is None or not hasattr(kernel_enum, kernel_name):
                raise RuntimeError(
                    f"this MoRI image lacks EpDispatchCombineKernelType.{kernel_name}"
                )
            self._kernel_type = getattr(kernel_enum, kernel_name)
        self._inter_node = kernel_name == "InterNodeV1"
        self.num_qps = 1
        self.block_num, self.rdma_block_num, self.dispatch_warps, self.combine_warps = blocks
        self._external_input = self._inter_node
        # Registered-input MoRI copies expert output into a device-side symmetric buffer. External
        # input kernels consume the dispatch output directly, so their stage is not applicable.
        # Under FP8, stage also dequantizes the received fp8 payload to BF16 (device work) on
        # either path, so it is a timed component regardless of the input-buffer mode.
        self.stage_device_work = self._fp8 or not self._external_input
        # Stash the __init__-only locals the moved create_buffer body reads back.
        self._gpus_per_node = gpus_per_node

    def create_buffer(self, spec):
        args, world_size, rank = self.args, self.world_size, self.rank
        gpus_per_node = self._gpus_per_node

        world_group = torch.distributed.group.WORLD
        torch._C._distributed_c10d._register_process_group("default", world_group)
        # Scale-out EP16 registers the symmetric heap over the AMD AI NIC. The
        # default STATIC_HEAP registers it as one contiguous MR; on the Ionic
        # stack that registration fails during InterNodeV1 init (an EINVAL that
        # is a firmware command failure, not an MR-size violation — the NIC
        # advertises multi-GiB max_mr_size). VMM_HEAP backs the same reservation
        # with on-demand 64 MiB DMA-BUF chunks, the supported inter-node path
        # (MoRI PR #155, validated on MI355X + AI NIC). Read at heap init below,
        # so it must precede shmem_torch_process_group_init. Scale-up EP8 is
        # intranode (no RDMA registration) and keeps the default heap.
        if self._inter_node:
            os.environ["MORI_SHMEM_MODE"] = "VMM_HEAP"
        mori.shmem.shmem_torch_process_group_init("default")
        realized_qps = int(mori.shmem.shmem_num_qp_per_pe())
        if realized_qps < self.num_qps:
            raise RuntimeError(
                f"MoRI realized {realized_qps} QPs per PE; {self.num_qps} required"
            )

        # MoRI preallocates one communicator buffer for the case's entire ladder.
        self._cap = max(512, spec.max_tokens_per_rank)
        # quant_type stays "none" for both precisions: dispatch precision is carried by
        # the passed tensor dtype (caller-prequantized e4m3 under FP8, BF16 otherwise),
        # and "none" keeps combine a genuine BF16 send. data_type is deprecated upstream
        # (kernel launch dtype is inferred from the runtime tensor) so it is left BF16.
        # max_token_type_size sizes the shared token buffer and must cover the widest
        # element moved across BOTH directions: FP8 dispatch is 1 byte, the BF16 combine
        # input copied into the registered buffer is 2, so size for BF16 unconditionally.
        token_type_size = torch.tensor([], dtype=torch.bfloat16).element_size()
        config_kwargs = {
            "data_type": torch.bfloat16,
            "rank": rank,
            "world_size": world_size,
            "hidden_dim": args.hidden,
            "scale_dim": 0,
            "scale_type_size": 1,
            "max_token_type_size": token_type_size,
            "max_num_inp_token_per_rank": self._cap,
            "num_experts_per_rank": self.experts_per_rank,
            "num_experts_per_token": args.topk,
            "use_external_inp_buf": self._external_input,
            "quant_type": "none",
        }
        if self._kernel_type is not None:
            config_kwargs["kernel_type"] = self._kernel_type
        if self._inter_node:
            config_kwargs.update({
                "block_num": self.block_num,
                "warp_num_per_block": self.dispatch_warps,
                "gpu_per_node": gpus_per_node,
                "rdma_block_num": self.rdma_block_num,
                "num_qp_per_pe": self.num_qps,
            })
        self.config = mori.ops.EpDispatchCombineConfig(**config_kwargs)
        expected_config = {
            "data_type": torch.bfloat16,
            "scale_dim": 0,
            "scale_type_size": 1,
            "use_external_inp_buf": self._external_input,
            "quant_type": config_kwargs["quant_type"],
        }
        if self._inter_node:
            expected_config.update({
                "block_num": self.block_num,
                "warp_num_per_block": self.dispatch_warps,
                "gpu_per_node": 8,
                "rdma_block_num": 64,
                "num_qp_per_pe": 1,
            })
        if any(getattr(self.config, key, None) != value for key, value in expected_config.items()):
            raise RuntimeError("MoRI requested launch/topology configuration was not realized")
        # The newer pinned MoRI revision can otherwise replace explicit values
        # with token-dependent tuning rules from the image.
        os.environ["MORI_EP_LAUNCH_CONFIG_MODE"] = "MANUAL"
        self.op = mori.ops.EpDispatchCombineOp(self.config)
        if getattr(self.op, "launch_config_mode", None) != "MANUAL":
            raise RuntimeError("MoRI explicit launch configuration was not applied")

    def semantic_payload(self, x):
        if not self._fp8:
            return x
        return x.to(self._fp8_dtype).to(torch.bfloat16)

    def _encode_dispatch(self, x):
        if not self._fp8:
            return x, None
        quantized = x.to(self._fp8_dtype)
        return quantized, quantized.to(torch.bfloat16)

    def make_problem(self, T, idx, weights, x):
        indices = idx.to(torch.int32)
        gate_weights = weights.to(torch.float32)
        dispatch_x, oracle_semantic = self._encode_dispatch(x)
        problem = types.SimpleNamespace(
            T=T,
            x=x,
            dispatch_x=dispatch_x,
            topk_idx=indices,
            topk_weights=gate_weights,
            indices=indices,
            weights=gate_weights,
            scales=torch.empty((T, 0), dtype=torch.uint8, device=self.device),
        )
        if oracle_semantic is not None:
            problem.oracle_x = oracle_semantic
        return problem

    def dispatch(self, p):
        dispatch_output, dispatch_weights, _scales, dispatch_indices, recv_num = (
            self.op.dispatch(
                p.dispatch_x,
                p.weights,
                p.scales,
                p.indices,
                block_num=self.block_num,
                rdma_block_num=self.rdma_block_num,
                warp_per_block=self.dispatch_warps,
            )
        )
        return types.SimpleNamespace(
            dispatch_output=dispatch_output,
            dispatch_weights=dispatch_weights,
            dispatch_indices=dispatch_indices,
            recv_num=recv_num[0],
            combine_input=None,
        )

    def stage(self, p, h):
        rows = getattr(p, "recv_tokens", None)
        if not isinstance(rows, int) or rows < 0 or rows > h.dispatch_output.size(0):
            raise RuntimeError("MoRI receive count was not validated before staging")
        # FP8: dispatch delivered an e4m3 payload; dequantize it to the BF16 combine sends.
        h.combine_input = (
            h.dispatch_output.to(torch.bfloat16) if self._fp8 else h.dispatch_output
        )
        if self._external_input:
            return None
        buffer = self.op.get_registered_combine_input_buffer(
            torch.bfloat16, hidden_dim=h.combine_input.size(1)
        )
        buffer[:rows, :].copy_(h.combine_input[:rows, :])
        h.combine_input = buffer

    def combine(self, p, h):
        combined, _weights = self.op.combine(
            h.combine_input,
            None,
            h.dispatch_indices,
            block_num=self.block_num,
            rdma_block_num=self.rdma_block_num,
            warp_per_block=self.combine_warps,
        )
        return combined[:p.T]

    def inspect_dispatch(self, p, h):
        count = self.recv_tokens(h)
        if h.dispatch_weights is None:
            raise RuntimeError("MoRI dispatch did not expose gate weights")
        if count < 0 or any(
            tensor.ndim == 0 or count > tensor.size(0)
            for tensor in (h.dispatch_output, h.dispatch_indices, h.dispatch_weights)
        ):
            raise RuntimeError("MoRI receive count exceeds dispatch metadata")
        raw_expert_ids = h.dispatch_indices[:count].to(torch.int64)
        expert_ids, weights, local_expert_ids = _project_local_metadata(
            torch,
            raw_expert_ids,
            h.dispatch_weights[:count].to(torch.float32),
            self.rank,
            self.experts_per_rank,
        )
        # FP8: the oracle compares a BF16 payload, so dequantize the received fp8 slice.
        payload = h.dispatch_output[:count]
        if self._fp8:
            payload = payload.to(torch.bfloat16)
        return types.SimpleNamespace(
            payload=payload,
            expert_ids=expert_ids,
            weights=weights,
            local_expert_counts=torch.bincount(
                local_expert_ids, minlength=self.experts_per_rank
            ),
        )

    def combine_transformed(self, p, h, transformed):
        h.combine_input = transformed.to(torch.bfloat16)
        rows = getattr(p, "recv_tokens", None)
        if not isinstance(rows, int) or rows < 0 or rows > h.combine_input.size(0):
            raise RuntimeError("MoRI receive count was not validated before transformed combine")
        if not self._external_input:
            buffer = self.op.get_registered_combine_input_buffer(
                torch.bfloat16, hidden_dim=h.combine_input.size(1)
            )
            buffer[:rows, :].copy_(h.combine_input[:rows, :])
            h.combine_input = buffer
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
