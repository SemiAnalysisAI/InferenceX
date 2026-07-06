#!/usr/bin/env python3
"""CollectiveX DeepEP adapter for native V1 dispatch/combine precision profiles."""
from __future__ import annotations

import inspect
import os
import sys
import types

import torch
import torch.distributed as dist
import contracts
import ep_precision

try:
    import deep_ep
    from deep_ep import Buffer  # type: ignore
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: deep_ep import failed: {exc!r}", file=sys.stderr)
    raise


def _deepep_version() -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version("deep_ep")
    except Exception:
        return getattr(deep_ep, "__version__", "unknown")


def _mnnvl_buffer_configuration() -> tuple[dict[str, bool], str]:
    """Resolve the explicit DeepEP MNNVL API contract."""
    requested_value = os.environ.get("CX_ALLOW_MNNVL")
    if requested_value not in {None, "", "0", "1"}:
        raise RuntimeError("CX_ALLOW_MNNVL must be unset, 0, or 1")
    requested = requested_value == "1"
    if not requested:
        return contracts.resolve_deepep_mnnvl(
            requested=False, signature_parameters=(),
            deepep_commit=os.environ.get("DEEPEP_COMMIT"),
        )
    try:
        parameters = inspect.signature(Buffer.__init__).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cannot inspect DeepEP Buffer MNNVL API") from exc
    try:
        return contracts.resolve_deepep_mnnvl(
            requested=True, signature_parameters=parameters,
            deepep_commit=os.environ.get("DEEPEP_COMMIT"),
        )
    except contracts.ContractError as exc:
        raise RuntimeError(str(exc)) from exc


def _normal_buffer_sizes(hidden: int, world_size: int) -> tuple[int, int]:
    """Apply DeepEP's dispatch/combine buffer sizing contract for this EP world."""
    hidden_bytes = hidden * torch.tensor([], dtype=torch.bfloat16).element_size()
    configs = (Buffer.get_dispatch_config(world_size), Buffer.get_combine_config(world_size))
    num_nvl_bytes = max(
        int(config.get_nvl_buffer_size_hint(hidden_bytes, world_size)) for config in configs
    )
    num_rdma_bytes = max(
        int(config.get_rdma_buffer_size_hint(hidden_bytes, world_size)) for config in configs
    )
    if num_nvl_bytes <= 0 or num_rdma_bytes < 0:
        raise RuntimeError("DeepEP returned invalid normal-mode buffer size hints")
    return num_nvl_bytes, num_rdma_bytes


class DeepEPBackend:
    name = "deepep"
    stage_device_work = False
    combine_needs_redispatch = False
    # DeepEP reduces activations and top-k weights independently. The activation
    # tensor must therefore carry the complete local weighted expert sum.
    combine_weight_semantics = "unweighted-rank-sum"
    oracle_layout = "token-rank"
    payload_unit = "token-rank"

    def __init__(self, args, rank, world_size, local_rank, device):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = getattr(args, "mode", "normal")
        if self.mode not in {"normal", "low-latency"}:
            raise ValueError(f"unsupported DeepEP mode {self.mode!r}")
        supported_profiles = {
            "normal": {
                "d-bf16.c-bf16",
                "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16",
            },
            "low-latency": {
                "d-bf16.c-bf16",
                "d-fp8-e4m3fn-b128-f32-fused.c-bf16",
                "d-bf16.c-logfmt10-dynamic64",
                "d-fp8-e4m3fn-b128-f32-fused.c-logfmt10-dynamic64",
            },
        }
        self.precision_profile_id, self.communication_precision = (
            ep_precision.resolve_precision(
                args,
                backend=self.name,
                mode=self.mode,
                supported_profiles=supported_profiles[self.mode],
            )
        )
        self._fp8_dispatch = ep_precision.is_low_precision_dispatch(
            self.communication_precision
        )
        self._use_logfmt = ep_precision.uses_logfmt_combine(
            self.communication_precision
        )
        self.stage_device_work = self._fp8_dispatch

        self.group = dist.group.WORLD
        device_sms = torch.cuda.get_device_properties(device).multi_processor_count
        mnnvl_kwargs, mnnvl_comm = _mnnvl_buffer_configuration()
        if self.mode == "low-latency":
            ep_precision.require_keyword(
                Buffer.low_latency_dispatch,
                "use_fp8",
                api="deep_ep.Buffer.low_latency_dispatch",
            )
            ep_precision.require_keyword(
                Buffer.low_latency_combine,
                "use_logfmt",
                api="deep_ep.Buffer.low_latency_combine",
            )
            if args.phase != "decode":
                raise ValueError("DeepEP low-latency mode only supports the decode ladder")
            if args.experts % world_size:
                raise ValueError("DeepEP low-latency experts must divide the EP group")
            self.combine_needs_redispatch = True
            self.combine_weight_semantics = "gate-weighted-sum"
            self.oracle_layout = "expert-packed"
            self.payload_unit = "token-expert"
            self.max_tokens_per_rank = 128
            num_qps_per_rank = args.experts // world_size
            num_rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
                self.max_tokens_per_rank, args.hidden, world_size, args.experts
            )
            self.buffer = Buffer(
                self.group,
                num_nvl_bytes=0,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_qps_per_rank,
                allow_nvlink_for_low_latency_mode=True,
                explicitly_destroy=True,
                **mnnvl_kwargs,
            )
            self.buffer.clean_low_latency_buffer(
                self.max_tokens_per_rank, args.hidden, args.experts
            )
            resource_provenance = {
                "requested_num_sms": None,
                "num_sms": None,
                "sm_fraction": None,
                "tuned_source": "deepep-low-latency-fixed-kernel",
                "num_max_tokens_per_rank": self.max_tokens_per_rank,
                "num_nvl_bytes": 0,
                "num_rdma_bytes": num_rdma_bytes,
                "num_qps_per_rank": num_qps_per_rank,
            }
        else:
            ep_precision.require_keyword(
                Buffer.dispatch,
                "async_finish",
                api="deep_ep.Buffer.dispatch",
            )
            ep_precision.require_keyword(
                Buffer.combine,
                "async_finish",
                api="deep_ep.Buffer.combine",
            )
            num_nvl_bytes, num_rdma_bytes = _normal_buffer_sizes(args.hidden, world_size)
            if world_size > args.scale_up_domain and num_rdma_bytes == 0:
                raise RuntimeError("DeepEP scale-out configuration returned no RDMA buffer")
            self.buffer = Buffer(
                self.group, num_nvl_bytes, num_rdma_bytes, **mnnvl_kwargs
            )
            num_sms = int(getattr(Buffer, "num_sms", args.num_sms))
            try:
                Buffer.set_num_sms(num_sms)
            except Exception as exc:  # pragma: no cover - version dependent
                raise RuntimeError(
                    f"DeepEP did not apply requested num_sms={num_sms}: {exc!r}"
                ) from exc
            applied_num_sms = int(getattr(Buffer, "num_sms", num_sms))
            if applied_num_sms != num_sms:
                raise RuntimeError(
                    f"DeepEP num_sms mismatch: requested={num_sms} applied={applied_num_sms}"
                )
            resource_provenance = {
                "requested_num_sms": num_sms,
                "num_sms": applied_num_sms,
                "sm_fraction": applied_num_sms / device_sms,
                "tuned_source": "deepep-default-num_sms",
                "num_nvl_bytes": num_nvl_bytes,
                "num_rdma_bytes": num_rdma_bytes,
            }
        version = _deepep_version()
        self.backend_provenance = {
            "deepep_version": version,
            "deepep_commit": os.environ.get("DEEPEP_COMMIT") or f"pkg-{version}",
            "backend_lineage": "deepep-v1",
            "mode": self.mode,
            "dispatch_dtype": ep_precision.communication_format(
                self.communication_precision, "dispatch"
            ),
            "combine_dtype": ep_precision.communication_format(
                self.communication_precision, "combine"
            ),
            "resource_mode": "fixed-profile",
            "device_sms": device_sms,
            "allow_mnnvl": bool(mnnvl_kwargs),
            "mnnvl_comm": mnnvl_comm,
            "nvshmem_ibgda_nic_handler": os.environ.get(
                "NVSHMEM_IBGDA_NIC_HANDLER", "not-active"
            ),
            **resource_provenance,
        }

    def buffer_cap(self, args):
        return self.max_tokens_per_rank if self.mode == "low-latency" else None

    def make_problem(self, T, idx, weights, x):
        encoding = ep_precision.encode_dispatch(
            torch, x, self.communication_precision
        )
        return types.SimpleNamespace(
            T=T,
            x=x,
            dispatch_x=encoding.native_input,
            oracle_x=encoding.semantic,
            dispatch_precision_evidence=encoding.evidence,
            topk_idx=idx.to(torch.int64),
            topk_weights=weights.to(torch.float32),
        )

    def dispatch(self, p):
        if self.mode == "low-latency":
            recv_x, recv_counts, handle, _, _ = self.buffer.low_latency_dispatch(
                p.x,
                p.topk_idx,
                self.max_tokens_per_rank,
                self.args.experts,
                use_fp8=self._fp8_dispatch,  # BF16 control realizes use_fp8=False.
                async_finish=False,
                return_recv_hook=False,
            )
            return types.SimpleNamespace(
                recv_x=recv_x,
                recv_counts=recv_counts,
                handle=handle,
            )
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = self.buffer.get_dispatch_layout(p.topk_idx, self.args.experts)
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, _ = self.buffer.dispatch(
            p.dispatch_x,
            topk_idx=p.topk_idx,
            topk_weights=p.topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            async_finish=False,
        )
        return types.SimpleNamespace(
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            recv_counts=recv_counts,
            handle=handle,
        )

    def stage(self, p, h):
        h.combine_input = self._semantic_recv(h, p)

    def combine(self, p, h):
        if self.mode == "low-latency":
            combined_x, _, _ = self.buffer.low_latency_combine(
                h.combine_input,
                p.topk_idx,
                p.topk_weights,
                h.handle,
                use_logfmt=self._use_logfmt,
                async_finish=False,
                return_recv_hook=False,
            )
            return combined_x
        combined_x, _, _ = self.buffer.combine(
            h.combine_input, h.handle, async_finish=False
        )
        return combined_x

    def inspect_dispatch(self, p, h):
        valid = h.recv_topk_idx >= 0
        expert_ids = torch.where(
            valid,
            h.recv_topk_idx + self.rank * (self.args.experts // self.world_size),
            h.recv_topk_idx,
        )
        return types.SimpleNamespace(
            payload=self._semantic_recv(h, p),
            encoded_payload=self._encoded_recv(h),
            scales=self._recv_scales(h),
            expert_ids=expert_ids,
            weights=h.recv_topk_weights.masked_fill(~valid, 0),
            local_expert_counts=torch.tensor(h.recv_counts, device=self.device, dtype=torch.int64),
            ordering_contract="source-rank-major-stable-v1",
        )

    def inspect_expert_dispatch(self, p, h):
        if self.mode != "low-latency":
            raise RuntimeError("expert-packed inspection requires low-latency mode")
        p.recv_counts = tuple(int(value) for value in h.recv_counts.tolist())
        return types.SimpleNamespace(
            payload=self._semantic_recv(h, p),
            encoded_payload=self._encoded_recv(h),
            scales=self._recv_scales(h),
            local_expert_counts=h.recv_counts,
            source_info=h.handle[0],
            layout_range=h.handle[1],
        )

    def combine_transformed(self, p, h, transformed):
        if self.mode == "low-latency":
            packed = torch.zeros(
                self._encoded_recv(h).shape,
                dtype=torch.bfloat16,
                device=self._encoded_recv(h).device,
            )
            packed[h.oracle_local_expert_slots, h.oracle_packed_positions] = transformed.to(
                packed.dtype
            )
            combined, _, _ = self.buffer.low_latency_combine(
                packed,
                p.topk_idx,
                p.topk_weights,
                h.handle,
                use_logfmt=self._use_logfmt,
                async_finish=False,
                return_recv_hook=False,
            )
            return combined
        semantic = self._semantic_recv(h, p)
        combined, _, _ = self.buffer.combine(
            transformed.to(semantic.dtype), h.handle, async_finish=False
        )
        return combined

    def recv_tokens(self, h):
        if self.mode == "low-latency":
            return int(h.recv_counts.to(torch.int64).sum().item())
        return int(self._encoded_recv(h).shape[0])

    def _encoded_recv(self, h):
        return h.recv_x[0] if isinstance(h.recv_x, tuple) else h.recv_x

    def _recv_scales(self, h):
        return h.recv_x[1] if isinstance(h.recv_x, tuple) else None

    def _semantic_recv(self, h, problem=None):
        if not self._fp8_dispatch:
            return h.recv_x
        if not hasattr(h, "recv_semantic"):
            if self.mode == "low-latency":
                counts = getattr(problem, "recv_counts", None)
                if counts is None:
                    counts = tuple(int(value) for value in h.recv_counts.tolist())
                    if problem is not None:
                        problem.recv_counts = counts
                workspace = getattr(self, "_ll_semantic_workspace", None)
                if workspace is None:
                    encoded = self._encoded_recv(h)
                    workspace = torch.empty(
                        encoded.shape, dtype=torch.bfloat16, device=encoded.device
                    )
                    self._ll_semantic_workspace = workspace
                h.recv_semantic = ep_precision.dequantize_expert_prefixes(
                    torch,
                    self._encoded_recv(h),
                    self._recv_scales(h),
                    self.communication_precision["dispatch"],
                    counts,
                    workspace,
                )
            else:
                h.recv_semantic = ep_precision.dequantize_dispatch(
                    torch,
                    self._encoded_recv(h),
                    self._recv_scales(h),
                    self.communication_precision["dispatch"],
                )
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
            if self.mode == "low-latency":
                self.buffer.destroy()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
