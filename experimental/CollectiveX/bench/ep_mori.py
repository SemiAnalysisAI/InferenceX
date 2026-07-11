#!/usr/bin/env python3
"""CollectiveX MoRI adapter: native BF16 dispatch/combine over mori.ops."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import types

# MoRI registers the whole symmetric heap at import time. The pinned upstream
# inter-node benchmark uses 6 GiB for its InterNodeV1 staging and signal buffers.
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


def _mori_source_commit() -> str:
    module_path = Path(mori.__file__).resolve()
    for root in module_path.parents:
        head = root / ".git" / "HEAD"
        if not head.is_symlink() and head.is_file() and head.stat().st_size <= 128:
            value = head.read_text(encoding="ascii").strip()
            if re.fullmatch(r"[0-9a-f]{40}", value):
                return value
            raise RuntimeError("MoRI image source is not pinned to a detached commit")
    raise RuntimeError("MoRI image source revision is unavailable")


class MoRIBackend(EPBackend):
    name = "mori"
    combine_needs_redispatch = True
    dispatch_needs_combine_cleanup = True

    def __init__(self, args, rank, world_size, local_rank, device):
        super().__init__(args, rank, world_size, local_rank, device)
        # The runner pins the AMD GPU architecture MoRI is built against. This is a
        # hardware-lineage contract, independent of the (fixed BF16) communication path.
        runner = str(getattr(args, "runner", ""))
        if runner.startswith("mi355x"):
            expected_arch = "gfx950"
        elif runner.startswith(("mi300x", "mi325x")):
            expected_arch = "gfx942"
        else:
            raise RuntimeError(
                f"MoRI has no pinned GPU architecture for runner {runner!r}"
            )

        self.ep_size = world_size
        self.experts_per_rank = args.experts // self.ep_size
        device_properties = torch.cuda.get_device_properties(device)
        realized_arch = str(getattr(device_properties, "gcnArchName", "")).split(":", 1)[0]
        if realized_arch != expected_arch:
            raise RuntimeError(
                f"MoRI runner {runner!r} realized architecture {realized_arch!r}, "
                f"expected {expected_arch!r}"
            )
        gpus_per_node = int(args.gpus_per_node or world_size)
        scale_up_domain = int(args.scale_up_domain or gpus_per_node)
        scale_out = world_size > scale_up_domain
        if (
            gpus_per_node <= 0
            or scale_up_domain <= 0
            or world_size % gpus_per_node
            or world_size % scale_up_domain
        ):
            raise RuntimeError("MoRI placement is not divisible into complete domains")
        if scale_out != (args.scope == "scale-out"):
            raise RuntimeError("MoRI requested scope differs from the EP topology")
        if not scale_out and (
            world_size != 8
            or gpus_per_node != 8
            or scale_up_domain != 8
            or args.scale_up_transport != "xgmi"
            or args.scale_out_transport
            or args.transport != "xgmi"
        ):
            raise RuntimeError("MoRI scale-up is pinned to EP8 over one XGMI domain")
        if scale_out and (
            world_size != 16
            or gpus_per_node != 8
            or scale_up_domain != 8
            or args.scale_up_transport != "xgmi"
            or args.scale_out_transport != "rdma"
            or args.transport != "xgmi-rdma"
        ):
            raise RuntimeError(
                "MoRI InterNodeV1 is pinned to EP16 over two 8-GPU XGMI/RDMA nodes"
            )
        self.block_num = 80
        self.rdma_block_num = 0
        self.num_qps = 1
        self.dispatch_warps = 16
        self.combine_warps = 8

        # MI355X uses the direct intranode kernel. MI300X/MI325X use MoRI's split
        # AsyncLL send/receive kernel as their normal-mode XGMI transport.
        kernel_request = os.environ.get("COLLX_MORI_KERNEL_TYPE", "intranode").strip().lower()
        self._kernel_type = None
        self._async_ll = False
        self._inter_node = False
        if kernel_request in ("asyncll", "async_ll", "async-ll"):
            if scale_out:
                raise RuntimeError("MoRI EP16 must use InterNodeV1, not AsyncLL")
            kernel_enum = getattr(mori.ops, "EpDispatchCombineKernelType", None)
            if kernel_enum is None or not hasattr(kernel_enum, "AsyncLL"):
                raise RuntimeError(
                    "COLLX_MORI_KERNEL_TYPE=asyncll requires "
                    "EpDispatchCombineKernelType.AsyncLL"
                )
            self._kernel_type = kernel_enum.AsyncLL
            self._async_ll = True
            self.block_num = 64
            self.dispatch_warps = self.combine_warps = 8
        elif kernel_request in ("internode-v1", "internode_v1", "internodev1"):
            if not scale_out:
                raise RuntimeError("MoRI InterNodeV1 is valid only for scale-out EP16")
            kernel_enum = getattr(mori.ops, "EpDispatchCombineKernelType", None)
            if kernel_enum is None or not hasattr(kernel_enum, "InterNodeV1"):
                raise RuntimeError(
                    "COLLX_MORI_KERNEL_TYPE=internode-v1 requires "
                    "EpDispatchCombineKernelType.InterNodeV1"
                )
            self._kernel_type = kernel_enum.InterNodeV1
            self._inter_node = True
            self.block_num = 96
            self.rdma_block_num = 64
            self.dispatch_warps = self.combine_warps = 8
        elif kernel_request not in ("intranode", "intra_node", "intra-node", ""):
            raise RuntimeError(
                f"unknown COLLX_MORI_KERNEL_TYPE={kernel_request!r} "
                "(expected intranode|asyncll|internode-v1)"
            )
        elif scale_out:
            raise RuntimeError("MoRI scale-out EP16 requires COLLX_MORI_KERNEL_TYPE=internode-v1")
        self.kernel_generation = (
            "inter-node-v1" if self._inter_node
            else "async-ll" if self._async_ll
            else "intranode"
        )
        self._external_input = self._async_ll or self._inter_node
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
        mori.shmem.shmem_torch_process_group_init("default")
        realized_qps = int(mori.shmem.shmem_num_qp_per_pe())
        if realized_qps < self.num_qps:
            raise RuntimeError(
                f"MoRI realized {realized_qps} QPs per PE; {self.num_qps} required"
            )

        self._cap = self.buffer_cap(args)
        assert spec.max_tokens_per_rank <= self._cap
        # BF16 communication path: no FP8 dispatch dtype, no scale payload, no combine quantization.
        dispatch_dtype = torch.bfloat16
        config_kwargs = {
            "data_type": dispatch_dtype,
            "rank": rank,
            "world_size": world_size,
            "hidden_dim": args.hidden,
            "scale_dim": 0,
            "scale_type_size": 1,
            "max_token_type_size": (
                torch.tensor([], dtype=torch.bfloat16).element_size()
                if self._inter_node
                else torch.tensor([], dtype=torch.float32).element_size()
            ),
            "max_num_inp_token_per_rank": max(512, self._cap),
            "num_experts_per_rank": self.experts_per_rank,
            "num_experts_per_token": args.topk,
            "use_external_inp_buf": self._external_input,
            "quant_type": "none",
        }
        if self._kernel_type is not None:
            config_kwargs["kernel_type"] = self._kernel_type
        if self._async_ll:
            config_kwargs["max_total_recv_tokens"] = 0
        if self._async_ll or self._inter_node:
            config_kwargs["block_num"] = self.block_num
            config_kwargs["warp_num_per_block"] = self.dispatch_warps
        if self._inter_node:
            config_kwargs.update({
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
        if self._async_ll or self._inter_node:
            expected_config.update({
                "block_num": self.block_num,
                "warp_num_per_block": self.dispatch_warps,
            })
        if self._inter_node:
            expected_config.update({
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

        expected_mori_commit = os.environ.get("MORI_COMMIT")
        mori_commit = _mori_source_commit()
        if expected_mori_commit and mori_commit != expected_mori_commit:
            raise RuntimeError("MoRI image source revision differs from the pinned revision")

    def buffer_cap(self, args):
        # Rows reserved per rank for a uniformly routed dispatch.
        return 512

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
        if self._async_ll:
            self.op.dispatch_recv(warp_per_block=self.dispatch_warps)
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
        combine_indices = p.indices if self._async_ll else h.dispatch_indices
        combined, _weights = self.op.combine(
            h.combine_input,
            None,
            combine_indices,
            block_num=self.block_num,
            rdma_block_num=self.rdma_block_num,
            warp_per_block=self.combine_warps,
        )
        if self._async_ll:
            self.op.combine_recv(warp_per_block=self.combine_warps)
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
