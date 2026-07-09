#!/usr/bin/env python3
"""CollectiveX DeepEP adapter: native BF16 dispatch/combine over deep_ep."""
from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.distributed as dist
import ep_provenance
from ep_deepep_family import DeepEPFamilyBackend

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
        return ep_provenance.resolve_deepep_mnnvl(
            requested=False, signature_parameters=(),
            deepep_commit=os.environ.get("DEEPEP_COMMIT"),
        )
    try:
        parameters = inspect.signature(Buffer.__init__).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError("cannot inspect DeepEP Buffer MNNVL API") from exc
    try:
        return ep_provenance.resolve_deepep_mnnvl(
            requested=True, signature_parameters=parameters,
            deepep_commit=os.environ.get("DEEPEP_COMMIT"),
        )
    except ep_provenance.ContractError as exc:
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


class DeepEPBackend(DeepEPFamilyBackend):
    # Mode handling and dispatch/combine live in DeepEPFamilyBackend; only the native
    # deep_ep buffer construction and teardown live here.
    name = "deepep"

    def create_buffer(self, spec):
        # Local aliases keep the moved buffer-construction body byte-verbatim.
        args, world_size, device = self.args, self.world_size, self.device
        device_sms = torch.cuda.get_device_properties(device).multi_processor_count
        mnnvl_kwargs, mnnvl_comm = _mnnvl_buffer_configuration()
        if self.mode == "low-latency":
            assert spec.max_tokens_per_rank <= 128
            ep_provenance.require_keyword(
                Buffer.low_latency_dispatch,
                "use_fp8",
                api="deep_ep.Buffer.low_latency_dispatch",
            )
            ep_provenance.require_keyword(
                Buffer.low_latency_combine,
                "use_logfmt",
                api="deep_ep.Buffer.low_latency_combine",
            )
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
            ep_provenance.require_keyword(
                Buffer.dispatch,
                "async_finish",
                api="deep_ep.Buffer.dispatch",
            )
            ep_provenance.require_keyword(
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
            "dispatch_dtype": "bf16",
            "combine_dtype": "bf16",
            "resource_mode": "fixed-profile",
            "device_sms": device_sms,
            "allow_mnnvl": bool(mnnvl_kwargs),
            "mnnvl_comm": mnnvl_comm,
            "nvshmem_ibgda_nic_handler": os.environ.get(
                "NVSHMEM_IBGDA_NIC_HANDLER", "not-active"
            ),
            **resource_provenance,
        }

    def finalize(self, rc):
        try:
            dist.barrier()
            if self.mode == "low-latency":
                self.buffer.destroy()
            dist.destroy_process_group()
        except Exception:
            pass
        return rc
