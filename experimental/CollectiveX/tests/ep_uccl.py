#!/usr/bin/env python3
"""CollectiveX UCCL adapter for native V1 dispatch/combine precision profiles."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import sys
import types

import torch
import torch.distributed as dist
import contracts
import ep_precision

try:
    import uccl
    import uccl_deepep
    from uccl_deepep import Buffer  # type: ignore
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: uccl.ep import failed: {exc!r}", file=sys.stderr)
    raise


def _uccl_version() -> str:
    try:
        return metadata.version("uccl")
    except Exception:
        return getattr(uccl, "__version__", "unknown")


def _uccl_dependency_versions() -> dict[str, str]:
    versions = {
        package: metadata.version(package)
        for package in contracts.UCCL_DEPENDENCY_VERSIONS
    }
    if versions != contracts.UCCL_DEPENDENCY_VERSIONS:
        raise RuntimeError(
            "UCCL runtime dependency versions differ from the v1 contract"
        )
    return versions


def _is_uccl_runtime_payload(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(path.parts)
        and path.parts[0] in {"uccl", "uccl.libs"}
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def _python_dependency_evidence(package: str, version: str) -> dict[str, str]:
    distribution = metadata.distribution(package)
    runtime_files = []
    for entry in distribution.files or ():
        logical = PurePosixPath(entry.as_posix())
        path = Path(distribution.locate_file(entry))
        if (
            logical.parts
            and logical.parts[0] == package
            and "__pycache__" not in logical.parts
            and logical.suffix != ".pyc"
            and path.is_file()
        ):
            runtime_files.append((entry.as_posix(), path))
    return contracts.content_manifest_evidence(
        role=f"{package}-distribution",
        name=f"{package}-{version}",
        files=runtime_files,
    )


def _loaded_libcudart_evidence(
    version: str, maps_path: Path = Path("/proc/self/maps")
) -> dict[str, str]:
    distribution = metadata.distribution("nvidia-cuda-runtime-cu12")
    candidates = {
        Path(distribution.locate_file(entry)).resolve()
        for entry in distribution.files or ()
        if PurePosixPath(entry.as_posix()).name.startswith("libcudart.so")
        and Path(distribution.locate_file(entry)).is_file()
    }
    candidate_names = {path.name for path in candidates}
    if not candidates or not candidate_names:
        raise RuntimeError("pinned CUDA runtime distribution has no libcudart payload")

    loaded: set[Path] = set()
    try:
        mappings = maps_path.read_text().splitlines()
    except OSError as exc:
        raise RuntimeError("cannot inspect mapped UCCL runtime libraries") from exc
    for mapping in mappings:
        columns = mapping.split(maxsplit=5)
        if len(columns) != 6:
            continue
        raw_path = columns[5]
        deleted = raw_path.endswith(" (deleted)")
        if deleted:
            raw_path = raw_path.removesuffix(" (deleted)")
        mapped = Path(raw_path)
        if mapped.name not in candidate_names:
            continue
        if deleted or not mapped.is_file():
            raise RuntimeError(
                "mapped libcudart is unavailable for content verification"
            )
        resolved = mapped.resolve()
        if resolved not in candidates:
            raise RuntimeError(
                "mapped libcudart is not owned by the pinned CUDA runtime package"
            )
        loaded.add(resolved)
    if len(loaded) != 1:
        raise RuntimeError(
            "expected exactly one mapped libcudart from the pinned CUDA runtime"
        )
    return contracts.content_manifest_evidence(
        role="cuda-runtime",
        name=f"nvidia-cuda-runtime-cu12-{version}",
        files=[("libcudart.so", loaded.pop())],
    )


def _uccl_build_evidence(
    version: str, dependency_versions: dict[str, str]
) -> list[dict[str, str]]:
    distribution = metadata.distribution("uccl")
    distribution_files = [
        (entry.as_posix(), distribution.locate_file(entry))
        for entry in distribution.files or ()
        if _is_uccl_runtime_payload(entry.as_posix())
        and Path(distribution.locate_file(entry)).is_file()
    ]
    wrapper_root = Path(uccl_deepep.__file__).resolve().parent
    wrapper_files = [
        (path.relative_to(wrapper_root).as_posix(), path)
        for path in wrapper_root.rglob("*.py")
        if path.is_file()
    ]
    return [
        contracts.content_manifest_evidence(
            role="uccl-distribution",
            name=f"uccl-{version}",
            files=distribution_files,
        ),
        contracts.content_manifest_evidence(
            role="uccl-wrapper",
            name="uccl-deepep-wrapper",
            files=wrapper_files,
        ),
        _python_dependency_evidence("intervaltree", dependency_versions["intervaltree"]),
        _python_dependency_evidence(
            "sortedcontainers", dependency_versions["sortedcontainers"]
        ),
        _loaded_libcudart_evidence(dependency_versions["nvidia-cuda-runtime-cu12"]),
    ]


def _require_cross_rank_equal(value, label: str) -> None:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    canonical = {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in gathered}
    if len(canonical) != 1:
        raise RuntimeError(f"UCCL {label} differs across ranks")


def _normal_buffer_sizes(hidden: int, world_size: int) -> tuple[int, int]:
    """Apply the wrapped DeepEP dispatch/combine sizing contract for this EP world."""
    hidden_bytes = hidden * torch.tensor([], dtype=torch.bfloat16).element_size()
    configs = (Buffer.get_dispatch_config(world_size), Buffer.get_combine_config(world_size))
    num_nvl_bytes = max(
        int(config.get_nvl_buffer_size_hint(hidden_bytes, world_size)) for config in configs
    )
    num_rdma_bytes = max(
        int(config.get_rdma_buffer_size_hint(hidden_bytes, world_size)) for config in configs
    )
    if num_nvl_bytes <= 0 or num_rdma_bytes < 0:
        raise RuntimeError("UCCL returned invalid normal-mode buffer size hints")
    return num_nvl_bytes, num_rdma_bytes


class UCCLBackend:
    name = "uccl"
    stage_device_work = False
    combine_needs_redispatch = False
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
            raise ValueError(f"unsupported UCCL mode {self.mode!r}")
        supported_profiles = {
            "normal": {
                "d-bf16.c-bf16",
                "d-fp8-e4m3fn-b128-f32-prequantized.c-bf16",
            },
            "low-latency": {
                "d-bf16.c-bf16",
                "d-fp8-e4m3fn-b128-f32-fused.c-bf16",
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
        if self.mode == "low-latency":
            ep_precision.require_keyword(
                Buffer.low_latency_dispatch,
                "use_fp8",
                api="uccl_deepep.Buffer.low_latency_dispatch",
            )
            ep_precision.require_keyword(
                Buffer.low_latency_combine,
                "use_logfmt",
                api="uccl_deepep.Buffer.low_latency_combine",
            )
            if args.phase != "decode":
                raise ValueError("UCCL low-latency mode only supports the decode ladder")
            if args.experts % world_size:
                raise ValueError("UCCL low-latency experts must divide the EP group")
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
            )
            self.buffer.clean_low_latency_buffer(
                self.max_tokens_per_rank, args.hidden, args.experts
            )
            resource_provenance = {
                "requested_num_sms": None,
                "num_sms": None,
                "sm_fraction": None,
                "tuned_source": "uccl-low-latency-fixed-kernel",
                "num_max_tokens_per_rank": self.max_tokens_per_rank,
                "num_nvl_bytes": 0,
                "num_rdma_bytes": num_rdma_bytes,
            }
        else:
            ep_precision.require_keyword(
                Buffer.dispatch,
                "async_finish",
                api="uccl_deepep.Buffer.dispatch",
            )
            ep_precision.require_keyword(
                Buffer.combine,
                "async_finish",
                api="uccl_deepep.Buffer.combine",
            )
            num_nvl_bytes, num_rdma_bytes = _normal_buffer_sizes(args.hidden, world_size)
            if world_size > args.scale_up_domain and num_rdma_bytes == 0:
                raise RuntimeError("UCCL scale-out configuration returned no RDMA buffer")
            self.buffer = Buffer(self.group, num_nvl_bytes, num_rdma_bytes)
            num_sms = int(getattr(Buffer, "num_sms", args.num_sms))
            try:
                Buffer.set_num_sms(num_sms)
            except Exception as exc:  # pragma: no cover - version dependent
                raise RuntimeError(
                    f"UCCL did not apply requested num_sms={num_sms}: {exc!r}"
                ) from exc
            applied_num_sms = int(getattr(Buffer, "num_sms", num_sms))
            if applied_num_sms != num_sms:
                raise RuntimeError(
                    f"UCCL num_sms mismatch: requested={num_sms} applied={applied_num_sms}"
                )
            resource_provenance = {
                "requested_num_sms": num_sms,
                "num_sms": applied_num_sms,
                "sm_fraction": applied_num_sms / device_sms,
                "tuned_source": "uccl-default-num_sms",
                "num_nvl_bytes": num_nvl_bytes,
                "num_rdma_bytes": num_rdma_bytes,
            }
        version = _uccl_version()
        dependency_versions = _uccl_dependency_versions()
        loaded_libraries = _uccl_build_evidence(version, dependency_versions)
        _require_cross_rank_equal(loaded_libraries, "installed content identities")
        self.backend_provenance = {
            "uccl_version": version,
            "uccl_commit": os.environ.get("UCCL_COMMIT") or f"pkg-{version}",
            "uccl_wrapper_commit": os.environ.get("UCCL_WRAPPER_COMMIT"),
            "backend_lineage": "uccl",
            "uccl_dependency_versions": dependency_versions,
            "loaded_libraries": loaded_libraries,
            "mode": self.mode,
            "dispatch_dtype": ep_precision.communication_format(
                self.communication_precision, "dispatch"
            ),
            "combine_dtype": ep_precision.communication_format(
                self.communication_precision, "combine"
            ),
            "resource_mode": "fixed-profile",
            "device_sms": device_sms,
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
        # UCCL's proxy teardown can crash after results are written; preserve the real rc.
        try:
            dist.barrier()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc if 0 <= rc <= 255 else 1)
