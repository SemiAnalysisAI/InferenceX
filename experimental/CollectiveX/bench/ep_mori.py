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
    combine_needs_redispatch = True
    dispatch_needs_combine_cleanup = True

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
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
        self.stage_device_work = not self._external_input
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
        # BF16 communication path: no FP8 dispatch dtype, no scale payload, no combine quantization.
        dispatch_dtype = torch.bfloat16
        config_kwargs = {
            "data_type": dispatch_dtype,
            "rank": rank,
            "world_size": world_size,
            "hidden_dim": args.hidden,
            "scale_dim": 0,
            "scale_type_size": 1,
            "max_token_type_size": torch.tensor([], dtype=dispatch_dtype).element_size(),
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
            "data_type": dispatch_dtype,
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

    def make_problem(self, T, idx, weights, x):
        indices = idx.to(torch.int32)
        gate_weights = weights.to(torch.float32)
        return types.SimpleNamespace(
            T=T,
            x=x,
            dispatch_x=x,
            topk_idx=indices,
            topk_weights=gate_weights,
            indices=indices,
            weights=gate_weights,
            scales=torch.empty((T, 0), dtype=torch.uint8, device=self.device),
        )

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
        h.combine_input = h.dispatch_output
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
        return types.SimpleNamespace(
            payload=h.dispatch_output[:count],
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
