#!/usr/bin/env python3
"""DeepEP PR #605 adapter with the exact upstream PR #630 and #640 fixes."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import ep_harness
import ep_provenance
from ep_backend import EPBackend

try:
    import deep_ep
    from deep_ep import ElasticBuffer  # type: ignore
except Exception as exc:  # pragma: no cover - requires the benchmark image
    print(f"ERROR: DeepEP V2 import failed: {exc!r}", file=sys.stderr)
    raise


DEEPEP_V2_PR = 605
DEEPEP_V2_FIX_PR = 630
DEEPEP_V2_NCCL_CHECK_FIX_PR = 640
DEEPEP_V2_NCCL_CHECK_COMMIT = "93d0564188f7a0a6288c6e316484861b0efa042e"
DEEPEP_V2_COMMIT = "fa8a9b16898204afd347c663b89e65ef87dc6ce6"
DEEPEP_V2_TREE = "29809e75c5874e6609dac4804e7b651d5226959f"
DEEPEP_V2_FMT_COMMIT = "a4c7e17133ee9cb6a2f45545f6e974dd3c393efa"
DEEPEP_V2_VERSION = "2.0.0"
DEEPEP_V2_DISTRIBUTIONS = frozenset({"2.0.0+fa8a9b1", "2.0.0+local"})
DEEPEP_V2_JIT_RANDOM_SEED = "collectivex-deepep-v2-fa8a9b1"
TORCH_VERSION = "2.10.0+cu130"
NCCL_VERSION = "2.30.4"
NVSHMEM_VERSION = "3.3.9"
DEEPEP_V2_JIT_KERNELS = ep_provenance.DEEPEP_V2_JIT_KERNELS


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _api_sha256() -> str:
    signatures = {
        "ElasticBuffer.__init__": str(inspect.signature(ElasticBuffer.__init__)),
        "ElasticBuffer.dispatch": str(inspect.signature(ElasticBuffer.dispatch)),
        "ElasticBuffer.combine": str(inspect.signature(ElasticBuffer.combine)),
    }
    return hashlib.sha256(
        json.dumps(signatures, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _loaded_library_paths() -> set[str]:
    extension = getattr(getattr(deep_ep, "_C", None), "__file__", None)
    if not extension or not os.path.isfile(extension):
        raise RuntimeError("DeepEP V2 extension library is not loaded")
    paths = {os.path.realpath(extension)}
    try:
        with open("/proc/self/maps", encoding="utf-8") as handle:
            for line in handle:
                path = line.rstrip().split()[-1]
                name = os.path.basename(path)
                if ("libnccl.so" in name or "libnvshmem_host.so" in name) and os.path.isfile(path):
                    paths.add(os.path.realpath(path))
    except OSError as exc:  # pragma: no cover - benchmark runtime is Linux
        raise RuntimeError("cannot inspect loaded communication libraries") from exc
    return paths


def _loaded_nccl_version() -> str:
    matches = [
        path for path in _loaded_library_paths()
        if "libnccl.so" in os.path.basename(path)
    ]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one loaded NCCL library")
    version = ctypes.c_int()
    if ctypes.CDLL(matches[0]).ncclGetVersion(ctypes.byref(version)) != 0:
        raise RuntimeError("loaded NCCL version query failed")
    return ep_harness.format_collective_version(version.value)


def _loaded_library_evidence() -> list[dict[str, str]]:
    """Return content identities, never private library paths."""
    paths = _loaded_library_paths()
    required = {
        "nccl": [path for path in paths if "libnccl.so" in os.path.basename(path)],
        "nvshmem": [path for path in paths if "libnvshmem_host.so" in os.path.basename(path)],
    }
    mismatches = [f"{name}={len(matches)}" for name, matches in required.items() if len(matches) != 1]
    if mismatches:
        raise RuntimeError("expected one loaded library for each dependency: " + ", ".join(mismatches))

    def role(path: str) -> str:
        name = os.path.basename(path)
        if "libnccl.so" in name:
            return "nccl"
        if "libnvshmem_host.so" in name:
            return "nvshmem"
        return "deepep-extension"

    def label(path: str) -> str:
        return "deep_ep._C" if role(path) == "deepep-extension" else os.path.basename(path)

    return sorted(
        ({"role": role(path), "name": label(path), "sha256": _sha256(path)} for path in paths),
        key=lambda item: (item["role"], item["name"], item["sha256"]),
    )


def _jit_artifact_evidence() -> list[dict[str, str]]:
    root = Path(os.environ["EP_JIT_CACHE_DIR"]) / "cache"
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("DeepEP V2 produced no JIT cache evidence")
    artifacts = []
    kernel_names = set()
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        match = re.fullmatch(r"kernel\.([A-Za-z0-9_+-]+)\.([0-9a-f]{32})", directory.name)
        if directory.is_symlink() or not directory.is_dir() or match is None:
            raise RuntimeError("DeepEP V2 JIT cache contains an invalid entry")
        if {path.name for path in directory.iterdir()} != {
            "kernel.cu", "kernel.cubin", "kernel.sass",
        }:
            raise RuntimeError("DeepEP V2 JIT kernel evidence is incomplete")
        source = directory / "kernel.cu"
        cubin = directory / "kernel.cubin"
        sass = directory / "kernel.sass"
        if any(path.is_symlink() or not path.is_file() for path in (source, cubin, sass)):
            raise RuntimeError("DeepEP V2 JIT evidence is not a regular file")
        if any(path.stat().st_size <= 0 for path in (source, cubin, sass)):
            raise RuntimeError("DeepEP V2 JIT evidence is empty")
        kernel_names.add(match.group(1))
        artifacts.append({
            "cache_key": directory.name,
            "source_sha256": _sha256(str(source)),
            "sass_sha256": _sha256(str(sass)),
            "cubin_sha256": _sha256(str(cubin)),
        })
    if (
        len(artifacts) != len(DEEPEP_V2_JIT_KERNELS)
        or kernel_names != DEEPEP_V2_JIT_KERNELS
    ):
        raise RuntimeError("DeepEP V2 JIT kernel set differs from the v1 contract")
    return sorted(artifacts, key=lambda item: item["cache_key"])


def _jit_cache_key(
    args,
    world_size: int,
    max_tokens: int,
    allow_hybrid_mode: bool,
    realized: dict[str, int | bool],
) -> str:
    """Key generated kernels by codegen inputs, not routing data or case identity."""
    payload = {
        "contract": "deepep-v2-jit-config-v3",
        "runner": args.runner,
        "world_size": world_size,
        "hidden": args.hidden,
        "topk": args.topk,
        "physical_experts": args.experts,
        "tuning_experts": getattr(args, "num_logical_experts", args.experts),
        "max_tokens": max_tokens,
        "dispatch_dtype": "bf16",
        "combine_dtype": "bf16",
        "input_layout": "bf16",
        "expert_alignment": 1,
        "do_cpu_sync": True,
        "cached_mode": False,
        "do_expand": False,
        "use_expanded_layout": False,
        "allow_hybrid_mode": allow_hybrid_mode,
        "allow_multiple_reduction": True,
        "prefer_overlap_with_compute": True,
        "deterministic": False,
        **realized,
    }
    return "jitcfg-v3-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_cross_rank_equal(value, label: str) -> None:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    canonical = {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in gathered}
    if len(canonical) != 1:
        raise RuntimeError(f"DeepEP V2 {label} differs across ranks")


# GIN/GDAKI allocates num_allocated_qps device QPs per peer rank on the local NIC
# (contexts x world_size QPs, before NCCL's own connection QPs). Upstream's hybrid
# default (129, or 65 with fast RDMA atomics) exhausts the per-NIC QP budget at
# EP16: construction dies in ncclDevCommCreate with ibv_create_qp ENOMEM once
# NCCL's regular QPs land on top (identical on H200 bare-metal and B200 pods; on
# CX-7 the budget sits between 784 and 1040 QPs — 49x16 initializes, 65x16 does
# not). Spending a fixed ~512-QP budget keeps every EP size inside that limit
# with headroom: EP8 resolves to 65 (the allocation CX-8 racks already run
# successfully), EP16 to 33 and EP32 to 17 (33 and 49 verified on the failing
# H200 pair). An explicit value also skips upstream's rank-local ibstat probe,
# which is not guaranteed to resolve identically across ranks.
_GIN_QP_BUDGET = 512


def _hybrid_num_allocated_qps(world_size: int) -> int:
    return max(9, 1 + _GIN_QP_BUDGET // world_size)


def _configure_gin_mode(args, world_size: int) -> bool:
    scale_up_domain = int(
        getattr(args, "scale_up_domain", None)
        or getattr(args, "gpus_per_node", None)
        or world_size
    )
    allow_hybrid_mode = world_size > scale_up_domain
    if allow_hybrid_mode:
        os.environ.pop("EP_DISABLE_GIN", None)
    else:
        os.environ["EP_DISABLE_GIN"] = "1"
    return allow_hybrid_mode


def _lsa_topology_is_valid(
    gin_enabled: bool,
    world_size: int,
    scale_up_domain: int,
    config: dict[str, int | bool],
) -> bool:
    if gin_enabled:
        domains = world_size // scale_up_domain
        return (
            world_size % scale_up_domain == 0
            and domains > 1
            and config["physical_rdma_ranks"] == domains
            and config["physical_nvlink_ranks"] == scale_up_domain
            and config["logical_scaleout_ranks"] == domains
            and config["logical_scaleup_ranks"] == scale_up_domain
            and config["is_scaleup_nvlink"] is True
        )
    return (
        config["physical_rdma_ranks"] == 1
        and config["physical_nvlink_ranks"] == world_size
        and config["logical_scaleout_ranks"] == 1
        and config["logical_scaleup_ranks"] == world_size
        and config["is_scaleup_nvlink"] is True
    )


def _require_runtime() -> tuple[str, str]:
    expected = {
        "DEEPEP_V2_PR": str(DEEPEP_V2_PR),
        "DEEPEP_V2_FIX_PR": str(DEEPEP_V2_FIX_PR),
        "DEEPEP_V2_NCCL_CHECK_FIX_PR": str(DEEPEP_V2_NCCL_CHECK_FIX_PR),
        "DEEPEP_V2_COMMIT": DEEPEP_V2_COMMIT,
        "DEEPEP_V2_TREE": DEEPEP_V2_TREE,
        "DEEPEP_V2_FMT_COMMIT": DEEPEP_V2_FMT_COMMIT,
        "DEEPEP_V2_NCCL_CHECK_COMMIT": DEEPEP_V2_NCCL_CHECK_COMMIT,
        "DEEPEP_V2_JIT_RANDOM_SEED": DEEPEP_V2_JIT_RANDOM_SEED,
        "EP_JIT_DUMP_SASS": "1",
    }
    mismatches = [
        f"{name}={os.environ.get(name)!r}, expected {value!r}"
        for name, value in expected.items()
        if os.environ.get(name) != value
    ]
    torch_version = str(torch.__version__)
    nccl_package_version = importlib.metadata.version("nvidia-nccl-cu13")
    nvshmem_package_version = importlib.metadata.version("nvidia-nvshmem-cu12")
    actual = {
        "deep_ep": str(getattr(deep_ep, "__version__", "")),
        "deep_ep distribution": importlib.metadata.version("deep_ep"),
        "torch": torch_version,
        "nvidia-nccl-cu13": nccl_package_version,
        "nvidia-nvshmem-cu12": nvshmem_package_version,
    }
    required = {
        "deep_ep": DEEPEP_V2_VERSION,
        "torch": TORCH_VERSION,
        "nvidia-nccl-cu13": NCCL_VERSION,
        "nvidia-nvshmem-cu12": NVSHMEM_VERSION,
    }
    mismatches.extend(
        f"{name}={actual[name]!r}, expected {value!r}"
        for name, value in required.items()
        if actual[name] != value
    )
    if actual["deep_ep distribution"] not in DEEPEP_V2_DISTRIBUTIONS:
        mismatches.append(
            "deep_ep distribution="
            f"{actual['deep_ep distribution']!r}, expected one of "
            f"{sorted(DEEPEP_V2_DISTRIBUTIONS)!r}"
        )
    if not inspect.isclass(ElasticBuffer) or ElasticBuffer.__name__ != "ElasticBuffer":
        mismatches.append("deep_ep.ElasticBuffer is absent")
    if os.environ.get("EP_SUPPRESS_NCCL_CHECK"):
        mismatches.append("EP_SUPPRESS_NCCL_CHECK must be unset")
    nccl_runtime_version = _loaded_nccl_version()
    if nccl_runtime_version != NCCL_VERSION:
        mismatches.append(
            f"loaded NCCL={nccl_runtime_version!r}, expected {NCCL_VERSION!r}"
        )
    if mismatches:
        raise RuntimeError("invalid DeepEP V2 runtime: " + "; ".join(mismatches))
    return torch_version, nccl_runtime_version


class DeepEPV2Backend(EPBackend):
    name = "deepep-v2"
    stage_device_work = False
    combine_needs_redispatch = False
    combine_weight_semantics = "unweighted-rank-sum"

    def __init__(self, args, rank, world_size, local_rank, device):
        # deepep-v2 is normal-mode only; base SUPPORTED_MODES=("normal",) enforces it.
        super().__init__(args, rank, world_size, local_rank, device)
        ep_provenance.require_keyword(
            ElasticBuffer.__init__,
            "use_fp8_dispatch",
            api="deep_ep.ElasticBuffer.__init__",
        )
        self.group = dist.group.WORLD
        self._deferred_jit_snapshot = None

    def create_buffer(self, spec):
        # Local aliases keep the moved buffer-construction body byte-verbatim. The cap
        # equals the value the deleted __init__ ladder-peek computed
        # (token_ladder(.., None) + conditioning), so the jit_cache_key stays stable.
        args, world_size, device = self.args, self.world_size, self.device
        self.max_tokens = spec.max_tokens_per_rank
        torch_version, nccl_runtime_version = _require_runtime()
        jit_root = Path(os.environ["EP_JIT_CACHE_DIR"])
        scale_up_domain = int(
            getattr(args, "scale_up_domain", None)
            or getattr(args, "gpus_per_node", None)
            or world_size
        )
        allow_hybrid_mode = _configure_gin_mode(args, world_size)
        gin_enabled = allow_hybrid_mode
        communication_backend = "nccl-gin" if gin_enabled else "nccl-device-lsa"
        self.buffer = ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=self.max_tokens,
            hidden=args.hidden,
            num_topk=args.topk,
            use_fp8_dispatch=False,  # BF16 communication path.
            deterministic=False,
            allow_hybrid_mode=allow_hybrid_mode,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=True,
            num_gpu_timeout_secs=100,
            explicitly_destroy=True,
            # 0 is upstream's use-the-default sentinel; only hybrid (GIN) mode
            # needs the explicit budget-derived allocation.
            num_allocated_qps=(
                _hybrid_num_allocated_qps(world_size) if allow_hybrid_mode else 0
            ),
        )
        tuning_num_experts = int(getattr(args, "num_logical_experts", args.experts))
        self.num_sms = int(
            self.buffer.get_theoretical_num_sms(tuning_num_experts, args.topk)
        )
        self.num_qps = int(self.buffer.get_theoretical_num_qps(self.num_sms))
        properties = torch.cuda.get_device_properties(device)
        device_sms = int(properties.multi_processor_count)
        jit_config = {
            "num_sms": self.num_sms,
            "num_qps": self.num_qps,
            "allocated_qps": int(self.buffer.num_allocated_qps),
            "logical_scaleout_ranks": int(self.buffer.num_scaleout_ranks),
            "logical_scaleup_ranks": int(self.buffer.num_scaleup_ranks),
            "physical_rdma_ranks": int(self.buffer.num_rdma_ranks),
            "physical_nvlink_ranks": int(self.buffer.num_nvlink_ranks),
            "is_scaleup_nvlink": self.buffer.num_scaleup_ranks == self.buffer.num_nvlink_ranks,
            "device_arch_major": int(properties.major),
            "device_arch_minor": int(properties.minor),
            "device_sms": device_sms,
            "device_smem_bytes": int(properties.shared_memory_per_block_optin),
            "gpu_timeout_cycles": 100 * int(properties.clock_rate) * 1000,
        }
        _require_cross_rank_equal(jit_config, "JIT configuration")
        if not _lsa_topology_is_valid(
            gin_enabled, world_size, scale_up_domain, jit_config
        ):
            raise RuntimeError("DeepEP V2 realized communication domains differ from topology")
        self.jit_cache_key = _jit_cache_key(
            args,
            world_size,
            self.max_tokens,
            allow_hybrid_mode,
            jit_config,
        )
        os.environ["EP_JIT_CACHE_DIR"] = str(jit_root / self.jit_cache_key)
        realized_config = {
            "jit_cache_key": self.jit_cache_key,
            "num_max_tokens_per_rank": self.max_tokens,
            **jit_config,
        }
        _require_cross_rank_equal(realized_config, "realized tuning/topology")
        comm = getattr(self.buffer, "nccl_comm_handle", None)
        communicator = (
            "deepep-managed" if getattr(comm, "managed", True) else "pytorch-reused"
        )

        loaded_libraries = _loaded_library_evidence()
        _require_cross_rank_equal(loaded_libraries, "loaded libraries")
        self.backend_provenance = {
            "deepep_version": DEEPEP_V2_VERSION,
            "deepep_distribution_version": importlib.metadata.version("deep_ep"),
            "deepep_commit": DEEPEP_V2_COMMIT,
            "deepep_tree": DEEPEP_V2_TREE,
            "deepep_pr": DEEPEP_V2_PR,
            "deepep_fix_pr": DEEPEP_V2_FIX_PR,
            "deepep_nccl_check_fix_pr": DEEPEP_V2_NCCL_CHECK_FIX_PR,
            "deepep_nccl_check_commit": DEEPEP_V2_NCCL_CHECK_COMMIT,
            "fmt_commit": DEEPEP_V2_FMT_COMMIT,
            "api": "deep_ep.ElasticBuffer",
            "api_signature_sha256": _api_sha256(),
            "communication_backend": communication_backend,
            "gin_enabled": gin_enabled,
            "nccl_communicator": communicator,
            "torch_version": torch_version,
            "torch_git_version": str(torch.version.git_version),
            "cuda_version": str(torch.version.cuda),
            "nccl_package_version": importlib.metadata.version("nvidia-nccl-cu13"),
            "nccl_version": nccl_runtime_version,
            "nvshmem_package_version": importlib.metadata.version("nvidia-nvshmem-cu12"),
            "loaded_libraries": loaded_libraries,
            "jit_cache_key": self.jit_cache_key,
            "jit_cubins": [],
            "jit_random_seed": DEEPEP_V2_JIT_RANDOM_SEED,
            "num_experts": int(args.experts),
            "mode": "normal",
            "dispatch_dtype": "bf16",
            "combine_dtype": "bf16",
            "deterministic": False,
            "resource_mode": "fixed-profile",
            "requested_num_sms": self.num_sms,
            "tuning_num_experts": tuning_num_experts,
            "num_sms": self.num_sms,
            "num_qps": self.num_qps,
            "allocated_qps": int(self.buffer.num_allocated_qps),
            "device_sms": device_sms,
            "sm_fraction": self.num_sms / device_sms,
            "tuned_source": "deepep-v2-analytical-sm-qp-logical-experts-v1",
            "num_max_tokens_per_rank": self.max_tokens,
            "allow_hybrid_mode": bool(self.buffer.allow_hybrid_mode),
            "allow_multiple_reduction": bool(self.buffer.allow_multiple_reduction),
            "prefer_overlap_with_compute": bool(
                self.buffer.prefer_overlap_with_compute
            ),
            "logical_scaleout_ranks": int(self.buffer.num_scaleout_ranks),
            "logical_scaleup_ranks": int(self.buffer.num_scaleup_ranks),
            "physical_rdma_ranks": int(self.buffer.num_rdma_ranks),
            "physical_nvlink_ranks": int(self.buffer.num_nvlink_ranks),
        }

    def _topk_idx_dtype(self):
        # DeepEP V2's kernels key routing indices on deep_ep.topk_idx_t, not int64.
        return deep_ep.topk_idx_t

    def dispatch(self, p):
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = self.buffer.dispatch(
            p.dispatch_x,
            topk_idx=p.topk_idx,
            topk_weights=p.topk_weights,
            num_experts=self.args.experts,
            num_max_tokens_per_rank=self.max_tokens,
            expert_alignment=1,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
            async_with_compute_stream=False,
            do_handle_copy=True,
            do_cpu_sync=True,
            do_expand=False,
        )
        return types.SimpleNamespace(
            recv_x=recv_x,
            recv_topk_idx=recv_topk_idx,
            recv_topk_weights=recv_topk_weights,
            handle=handle,
        )

    def stage(self, p, h):
        # BF16: the received buffer is already the semantic payload to combine.
        h.combine_input = h.recv_x

    def combine(self, p, h):
        combined_x, _, _ = self.buffer.combine(
            h.combine_input,
            handle=h.handle,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
            async_with_compute_stream=False,
        )
        return combined_x

    def capture_deferred_provenance(self):
        # destroy() uses this same barrier. Materialize its JIT kernel before hashing the
        # implementation so the first and later routing cases see identical evidence.
        self.buffer.barrier(use_comm_stream=True, with_cpu_sync=True)
        torch.cuda.synchronize()
        jit_cubins = _jit_artifact_evidence()
        _require_cross_rank_equal(jit_cubins, "JIT CUBINs")
        if (
            self._deferred_jit_snapshot is not None
            and jit_cubins != self._deferred_jit_snapshot
        ):
            raise RuntimeError("DeepEP V2 JIT CUBIN set changed after measurement")
        self._deferred_jit_snapshot = jit_cubins
        self.backend_provenance["jit_cubins"] = jit_cubins

    def inspect_dispatch(self, p, h):
        count = self.recv_tokens(h)
        local_idx = h.recv_topk_idx[:count]
        valid = local_idx >= 0
        expert_ids = torch.where(
            valid,
            local_idx + self.rank * (self.args.experts // self.world_size),
            local_idx,
        )
        local = local_idx[valid].to(torch.int64)
        return types.SimpleNamespace(
            payload=h.recv_x[:count],
            expert_ids=expert_ids,
            weights=h.recv_topk_weights[:count].masked_fill(~valid, 0),
            local_expert_counts=torch.bincount(
                local, minlength=self.args.experts // self.world_size
            ),
            ordering_contract="elastic-source-metadata-v1",
        )

    def combine_transformed(self, p, h, transformed):
        combine_input = torch.zeros_like(h.recv_x)
        combine_input[: transformed.shape[0]].copy_(transformed.to(combine_input.dtype))
        combined, _, _ = self.buffer.combine(
            combine_input,
            handle=h.handle,
            num_sms=self.num_sms,
            num_qps=self.num_qps,
            async_with_compute_stream=False,
        )
        return combined

    def recv_tokens(self, h):
        return int(h.handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())

    def finalize(self, rc):
        try:
            dist.barrier()
            self.buffer.destroy()
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            return 1
        return rc
