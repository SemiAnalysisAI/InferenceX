#!/usr/bin/env python3
"""CollectiveX MoRI adapter for native V1 dispatch/combine precision profiles."""
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
import ep_precision

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


class MoRIBackend:
    name = "mori"
    stage_device_work = False
    combine_needs_redispatch = True
    combine_weight_semantics = "unweighted-rank-sum"

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = "normal"
        runner = str(getattr(args, "runner", ""))
        if runner.startswith("mi355x"):
            fp8_format = "fp8-e4m3fn"
            supported_profiles = {
                "d-bf16.c-bf16",
                "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16",
            }
            if world_size == 8:
                supported_profiles.update({
                    "d-bf16.c-fp8-e4m3fn-direct-cast-noscale",
                    "d-fp8-e4m3fn-b128-f32-prequantized.c-fp8-e4m3fn-direct-cast-noscale",
                })
        elif runner.startswith("mi325x"):
            fp8_format = "fp8-e4m3fnuz"
            supported_profiles = {
                "d-bf16.c-bf16",
                "d-fp8-e4m3fnuz-b128-f32-prequantized.c-bf16",
            }
            if world_size == 8:
                supported_profiles.update({
                    "d-bf16.c-fp8-e4m3fnuz-direct-cast-noscale",
                    "d-fp8-e4m3fnuz-b128-f32-prequantized.c-fp8-e4m3fnuz-direct-cast-noscale",
                })
        else:
            raise ep_precision.PrecisionError(
                f"MoRI precision contract has no pinned FP8 format for runner {runner!r}"
            )
        self.precision_profile_id, self.communication_precision = (
            ep_precision.resolve_precision(
                args,
                backend=self.name,
                mode=self.mode,
                supported_profiles=supported_profiles,
            )
        )
        self._fp8_dispatch = ep_precision.is_low_precision_dispatch(
            self.communication_precision
        )
        self._direct_cast_combine = ep_precision.uses_direct_cast_combine(
            self.communication_precision
        )
        if self._fp8_dispatch and ep_precision.communication_format(
            self.communication_precision, "dispatch"
        ) != fp8_format:
            raise ep_precision.PrecisionError(
                "MoRI dispatch FP8 format differs from the pinned GPU architecture"
            )
        if self._direct_cast_combine:
            quant_enum = getattr(mori.ops, "EpDispatchCombineQuantType", None)
            if quant_enum is None or not hasattr(quant_enum, "Fp8DirectCast"):
                raise ep_precision.PrecisionError(
                    "pinned MoRI API omits EpDispatchCombineQuantType.Fp8DirectCast"
                )

        self.ep_size = world_size
        self.experts_per_rank = args.experts // self.ep_size
        device_properties = torch.cuda.get_device_properties(device)
        device_cus = device_properties.multi_processor_count
        realized_arch = str(getattr(device_properties, "gcnArchName", "")).split(":", 1)[0]
        expected_arch = "gfx950" if runner.startswith("mi355x") else "gfx942"
        if realized_arch != expected_arch:
            raise ep_precision.PrecisionError(
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
        self.block_num = self._block_target = 80
        self.rdma_block_num = 0
        self.num_qps = 1
        self._block_floored = False
        self._tuned_source = "default-80"
        self.dispatch_warps = 16
        self.combine_warps = 8

        # MI355X uses the direct intranode kernel. MI325X uses MoRI's split
        # AsyncLL send/receive kernel as its normal-mode XGMI transport.
        kernel_request = os.environ.get("CX_MORI_KERNEL_TYPE", "intranode").strip().lower()
        self._kernel_type = None
        self._kernel_type_label = "IntraNode"
        self._async_ll = False
        self._inter_node = False
        if kernel_request in ("asyncll", "async_ll", "async-ll"):
            if scale_out:
                raise RuntimeError("MoRI EP16 must use InterNodeV1, not AsyncLL")
            kernel_enum = getattr(mori.ops, "EpDispatchCombineKernelType", None)
            if kernel_enum is None or not hasattr(kernel_enum, "AsyncLL"):
                raise RuntimeError(
                    "CX_MORI_KERNEL_TYPE=asyncll requires "
                    "EpDispatchCombineKernelType.AsyncLL"
                )
            self._kernel_type = kernel_enum.AsyncLL
            self._kernel_type_label = "AsyncLL"
            self._async_ll = True
            self.block_num = self._block_target = 64
            self.dispatch_warps = self.combine_warps = 8
            self._tuned_source = "upstream-asyncll-64x8-external-input"
        elif kernel_request in ("internode-v1", "internode_v1", "internodev1"):
            if not scale_out:
                raise RuntimeError("MoRI InterNodeV1 is valid only for scale-out EP16")
            kernel_enum = getattr(mori.ops, "EpDispatchCombineKernelType", None)
            if kernel_enum is None or not hasattr(kernel_enum, "InterNodeV1"):
                raise RuntimeError(
                    "CX_MORI_KERNEL_TYPE=internode-v1 requires "
                    "EpDispatchCombineKernelType.InterNodeV1"
                )
            self._kernel_type = kernel_enum.InterNodeV1
            self._kernel_type_label = "InterNodeV1"
            self._inter_node = True
            self.block_num = self._block_target = 96
            self.rdma_block_num = 64
            self.dispatch_warps = self.combine_warps = 8
            self._tuned_source = "upstream-internode-v1-96-64x8-qps1"
        elif kernel_request not in ("intranode", "intra_node", "intra-node", ""):
            raise RuntimeError(
                f"unknown CX_MORI_KERNEL_TYPE={kernel_request!r} "
                "(expected intranode|asyncll|internode-v1)"
            )
        elif scale_out:
            raise RuntimeError("MoRI scale-out EP16 requires CX_MORI_KERNEL_TYPE=internode-v1")
        self.kernel_generation = (
            "inter-node-v1" if self._inter_node
            else "async-ll" if self._async_ll
            else "intranode"
        )
        self._external_input = (
            self._async_ll or self._inter_node or self._direct_cast_combine
        )
        # Registered-input MoRI copies expert output into a device-side symmetric buffer. External
        # input kernels consume the dispatch output directly, so their stage is not applicable.
        self.stage_device_work = self._fp8_dispatch or not self._external_input

        world_group = torch.distributed.group.WORLD
        torch._C._distributed_c10d._register_process_group("default", world_group)
        mori.shmem.shmem_torch_process_group_init("default")
        realized_qps = int(mori.shmem.shmem_num_qp_per_pe())
        if realized_qps < self.num_qps:
            raise RuntimeError(
                f"MoRI realized {realized_qps} QPs per PE; {self.num_qps} required"
            )

        self._cap = self.buffer_cap(args)
        dispatch_dtype = (
            getattr(
                torch,
                "float8_e4m3fn"
                if fp8_format == "fp8-e4m3fn"
                else "float8_e4m3fnuz",
                None,
            )
            if self._fp8_dispatch
            else torch.bfloat16
        )
        if dispatch_dtype is None:
            raise ep_precision.PrecisionError(
                f"active torch build does not expose {fp8_format}"
            )
        scale_dim = args.hidden // 128 if self._fp8_dispatch else 0
        if self._fp8_dispatch and args.hidden % 128:
            raise ep_precision.PrecisionError(
                "MoRI native FP8 dispatch requires hidden divisible by 128"
            )
        config_kwargs = {
            "data_type": dispatch_dtype,
            "rank": rank,
            "world_size": world_size,
            "hidden_dim": args.hidden,
            "scale_dim": scale_dim,
            "scale_type_size": 4 if self._fp8_dispatch else 1,
            "max_token_type_size": (
                torch.tensor([], dtype=torch.bfloat16).element_size()
                if self._inter_node
                else torch.tensor([], dtype=torch.float32).element_size()
            ),
            "max_num_inp_token_per_rank": max(512, self._cap),
            "num_experts_per_rank": self.experts_per_rank,
            "num_experts_per_token": args.topk,
            "use_external_inp_buf": self._external_input,
            "quant_type": (
                "fp8_direct_cast" if self._direct_cast_combine else "none"
            ),
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
            "scale_dim": scale_dim,
            "scale_type_size": 4 if self._fp8_dispatch else 1,
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
            raise RuntimeError("MoRI image source revision differs from canonical provenance")
        self.backend_provenance = {
            "mori_commit": mori_commit,
            "api": (
                "mori.ops.EpDispatchCombineOp/external-input"
                if self._external_input
                else "mori.ops.EpDispatchCombineOp/registered-input"
            ),
            "mode": "normal",
            "dispatch_dtype": ep_precision.communication_format(
                self.communication_precision, "dispatch"
            ),
            "combine_dtype": ep_precision.communication_format(
                self.communication_precision, "combine"
            ),
            "kernel_type": self._kernel_type_label,
            "enable_sdma": os.environ.get("MORI_ENABLE_SDMA"),
            "heap_size": os.environ.get("MORI_SHMEM_HEAP_SIZE"),
            "max_num_inp_token_per_rank": max(512, self._cap),
            "max_total_recv_tokens": config_kwargs.get("max_total_recv_tokens"),
            "gpus_per_node": gpus_per_node,
            "rdma_block_num": self.rdma_block_num,
            "use_external_inp_buf": self._external_input,
            "num_qps": self.num_qps,
            "resource_mode": "fixed-profile",
            "block_num": self.block_num,
            "block_num_target": self._block_target,
            "block_num_floored": self._block_floored,
            "dispatch_warps": self.dispatch_warps,
            "combine_warps": self.combine_warps,
            "device_cus": device_cus,
            "sm_fraction": None if self._async_ll else self.block_num / device_cus,
            "tuned_source": self._tuned_source,
        }

    def buffer_cap(self, args):
        return 512

    def make_problem(self, T, idx, weights, x):
        encoding = ep_precision.encode_dispatch(
            torch, x, self.communication_precision
        )
        indices = idx.to(torch.int32)
        gate_weights = weights.to(torch.float32)
        return types.SimpleNamespace(
            T=T,
            x=x,
            dispatch_x=encoding.native_input[0] if self._fp8_dispatch else x,
            oracle_x=encoding.semantic,
            dispatch_precision_evidence=encoding.evidence,
            topk_idx=indices,
            topk_weights=gate_weights,
            indices=indices,
            weights=gate_weights,
            scales=(
                encoding.scales
                if encoding.scales is not None
                else torch.empty((T, 0), dtype=torch.uint8, device=self.device)
            ),
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
            dispatch_scales=_scales,
            dispatch_indices=dispatch_indices,
            recv_num=recv_num[0],
            combine_input=None,
        )

    def stage(self, p, h):
        rows = getattr(p, "recv_tokens", None)
        if not isinstance(rows, int) or rows < 0 or rows > h.dispatch_output.size(0):
            raise RuntimeError("MoRI receive count was not validated before staging")
        h.combine_input = self._semantic_recv(h, rows)
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
            payload=self._semantic_recv(h, count)[:count],
            encoded_payload=h.dispatch_output[:count],
            scales=(
                h.dispatch_scales[:count]
                if h.dispatch_scales is not None
                else None
            ),
            expert_ids=expert_ids,
            weights=weights,
            local_expert_counts=torch.bincount(
                local_expert_ids, minlength=self.experts_per_rank
            ),
            ordering_contract="mori-global-topk-masked-v1",
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

    def _semantic_recv(self, h, rows):
        if not self._fp8_dispatch:
            return h.dispatch_output
        if not hasattr(h, "recv_semantic"):
            if h.dispatch_scales is None:
                raise ep_precision.PrecisionError(
                    "MoRI FP8 dispatch did not return scaling factors"
                )
            semantic = torch.empty(
                h.dispatch_output.shape,
                dtype=torch.bfloat16,
                device=h.dispatch_output.device,
            )
            semantic[:rows].copy_(ep_precision.dequantize_dispatch(
                torch,
                h.dispatch_output[:rows],
                h.dispatch_scales[:rows],
                self.communication_precision["dispatch"],
            ))
            h.recv_semantic = semantic
            h.recv_semantic_rows = rows
        elif h.recv_semantic_rows != rows:
            raise RuntimeError("MoRI receive count changed for one dispatch handle")
        return h.recv_semantic

    def oracle_dispatch_payload(self, payload):
        return ep_precision.encode_dispatch(
            torch, payload, self.communication_precision
        ).semantic

    def precision_evidence(self, problem, view=None):
        return ep_precision.precision_evidence(
            torch,
            profile_id=self.precision_profile_id,
            profile=self.communication_precision,
            problem=problem,
            view=view,
        )

    def finalize(self, rc):
        try:
            dist.barrier()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc if 0 <= rc <= 255 else 1)
