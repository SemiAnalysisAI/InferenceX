#!/usr/bin/env python3
"""Strict native attempt contracts and metric validation for CollectiveX v1."""
from __future__ import annotations

import argparse
import datetime as dt
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable

import artifact_safety
import capability
import identity

TESTS = Path(__file__).resolve().parent / "tests"
sys.path.insert(0, str(TESTS))
import eplb as eplb_contract  # noqa: E402
import workload as workload_contract  # noqa: E402

RAW_FORMAT = "collectivex.ep.v1"
SAMPLES_FORMAT = "collectivex.samples.v1"
TERMINAL_FORMAT = "collectivex.terminal.v1"
TERMINAL_CASE_FIELDS = {
    "backend", "canonical", "eplb", "ep", "experts", "gpus_per_node", "hidden",
    "ladder", "mode", "nodes", "phase", "required_publication", "routing",
    "samples_per_point", "scale_out_transport", "scale_up_domain", "scale_up_transport",
    "scope", "suite", "timing", "topk", "topology_class", "transport",
    "warmup_semantics", "workload",
}
ALLOCATION_FACTOR_FIELDS = {
    "artifact", "execution_id", "job", "repo", "run_attempt", "run_id", "runner",
    "source_sha", "qualification_index",
}
GIT_RUN_FIELDS = {
    "artifact", "job", "qualification_index", "ref", "repo", "run_attempt", "run_id",
    "source_sha",
}
PRE_EXECUTION_FAILURE_REASONS = {
    "setup": "launcher-setup-failed",
    "repository-stage": "repository-staging-failed",
    "registry-verification": "container-registry-verification-failed",
    "scheduler-allocation": "scheduler-allocation-failed",
    "container-import": "container-image-preparation-failed",
    "container-hash": "container-image-identity-failed",
    "container-launch": "container-runtime-launch-failed",
    "backend-setup": "backend-setup-failed",
    "artifact-collection": "artifact-collection-failed",
}
RUNTIME_FAILURE_REASONS = {
    **PRE_EXECUTION_FAILURE_REASONS,
    "runtime-identity": "runtime-identity-mismatch",
    "timeout": "execution-timeout",
    "deadlock": "execution-deadlock",
    "execution": "distributed-command-failed",
}
POST_EMIT_FAILURE_REASONS = {
    mode: "post-emit-distributed-command-failed"
    for mode in ("runtime-identity", "timeout", "deadlock", "execution")
}
CAPABILITY_FAILURE_REASONS = frozenset({
    "backend-platform-unsupported",
    "backend-token-capacity",
    "precision-profile-unsupported",
})
RETURN_CODE_FAILURE_MODES = {
    5: "runtime-identity",
    124: "timeout",
    137: "timeout",
}
PERCENTILES = ("p50", "p90", "p95", "p99")
V1_CONDITIONING_LADDERS = {
    "decode": (1, 2, 4, 8, 16, 32, 64, 128),
    "prefill": (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
}
V1_CONDITIONING_ROUNDS_PER_SHAPE = 8
DEEPEP_V2_JIT_KERNELS = frozenset({
    "barrier", "combine", "combine_reduce_epilogue", "dispatch",
    "dispatch_copy_epilogue",
})
DEEPEP_V2_V1_PROVENANCE = {
    "deepep_version": "2.0.0",
    "deepep_distribution_version": "2.0.0+fa8a9b1",
    "deepep_commit": "fa8a9b16898204afd347c663b89e65ef87dc6ce6",
    "deepep_tree": "29809e75c5874e6609dac4804e7b651d5226959f",
    "deepep_pr": 605,
    "deepep_fix_pr": 630,
    "deepep_nccl_check_fix_pr": 640,
    "deepep_nccl_check_commit": "93d0564188f7a0a6288c6e316484861b0efa042e",
    "fmt_commit": "a4c7e17133ee9cb6a2f45545f6e974dd3c393efa",
    "torch_version": "2.10.0+cu130",
    "nccl_package_version": "2.30.4",
    "nccl_version": "2.30.4",
    "nvshmem_package_version": "3.3.9",
}
DEEPEP_V2_DISTRIBUTION_VERSIONS = frozenset({
    "2.0.0+fa8a9b1", "2.0.0+local",
})
UCCL_DEPENDENCY_VERSIONS = {
    "intervaltree": "3.1.0",
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "sortedcontainers": "2.4.0",
}
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}
REQUIRED_BACKEND_PROVENANCE = {
    "deepep": (
        "deepep_version", "deepep_commit", "backend_lineage", "allow_mnnvl",
        "mnnvl_comm", "mode", "num_nvl_bytes", "num_rdma_bytes",
    ),
    "deepep-v2": (
        *DEEPEP_V2_V1_PROVENANCE, "api_signature_sha256", "loaded_libraries",
        "jit_cubins", "jit_random_seed", "deterministic", "num_experts",
        "tuning_num_experts", "allow_hybrid_mode", "gin_enabled",
        "communication_backend",
    ),
    "deepep-hybrid": (
        "deepep_commit", "deepep_tree", "branch", "backend_lineage",
        "loaded_libraries", "realized_config", "jit_kernel_keys", "jit_shared_objects",
    ),
    "uccl": (
        "uccl_version", "uccl_commit", "uccl_wrapper_commit", "backend_lineage",
        "loaded_libraries", "uccl_dependency_versions", "mode", "num_nvl_bytes",
        "num_rdma_bytes",
    ),
    "mori": ("mori_commit",),
    "nccl-ep": ("nccl_version", "collective_library", "backend_lineage"),
}
PROVENANCE_KEYS = {
    "allocated_qps", "allow_hybrid_mode", "allow_mnnvl", "allow_multiple_reduction",
    "api", "api_signature_sha256", "backend", "backend_lineage", "block_num",
    "block_num_floored", "block_num_target", "branch", "collective_library",
    "combine_dtype", "combine_warps", "communication_backend", "cuda_version",
    "deepep_commit", "deepep_distribution_version", "deepep_fix_pr",
    "deepep_nccl_check_commit", "deepep_nccl_check_fix_pr", "deepep_pr", "deepep_tree",
    "deepep_version", "deterministic", "device_cus",
    "device_sms", "dispatch_dtype", "dispatch_warps", "enable_sdma", "fmt_commit",
    "gin_enabled",
    "gpus_per_node", "heap_size",
    "impl", "jit_cache_key", "jit_cubins", "jit_kernel_keys", "jit_random_seed",
    "jit_shared_objects", "kernel_type",
    "loaded_libraries", "local_experts",
    "logical_scaleout_ranks",
    "logical_scaleup_ranks", "mapping_variant", "max_num_inp_token_per_rank",
    "max_num_tokens", "max_total_recv_tokens", "mnnvl_comm", "mode", "mori_commit",
    "nccl_communicator", "nccl_package_version", "nccl_version", "num_experts",
    "nvshmem_package_version",
    "num_max_tokens_per_rank", "num_nvl_bytes", "num_qps", "num_qps_per_rank",
    "num_rdma_bytes", "num_sms", "path",
    "physical_nvlink_ranks", "physical_rdma_ranks", "prefer_overlap_with_compute",
    "rdma_block_num",
    "realized_config", "reference_semantics", "requested_num_sms", "resource_mode", "routing_factor",
    "routing_metadata", "sm_fraction", "top_k",
    "torch_git_version", "torch_version", "transport", "trtllm", "tuned_source",
    "tuning_num_experts",
    "uccl_commit", "uccl_dependency_versions", "uccl_version", "uccl_wrapper_commit",
    "use_external_inp_buf",
    "workspace",
}


class ContractError(ValueError):
    """A document differs from the native v1 contract."""


def scheduled_case_profile(case: dict[str, Any], path: str = "case") -> dict[str, Any]:
    """Resolve an explicit scheduled mode to its immutable measurement profile."""
    try:
        return identity.profile_for_case(case)
    except identity.IdentityError as exc:
        raise ContractError(f"{path}: {exc}") from exc


def _scheduled_case(value: Any, path: str) -> dict[str, Any]:
    """Validate baseline or explicit-precision scheduled case fields."""
    fields = set(TERMINAL_CASE_FIELDS)
    if isinstance(value, dict) and "precision_profile" in value:
        fields.add("precision_profile")
    return _keys(value, fields, path)


def resolve_deepep_mnnvl(
    *, requested: bool, signature_parameters: Iterable[str], deepep_commit: str | None
) -> tuple[dict[str, bool], str]:
    """Resolve one explicit DeepEP MNNVL API mode without signature fallbacks."""
    if not requested:
        return {}, "not-requested"
    if "allow_mnnvl" in set(signature_parameters):
        return {"allow_mnnvl": True}, "explicit-allow-mnnvl"
    raise ContractError(
        f"requested DeepEP MNNVL is unsupported by commit {deepep_commit or 'unknown'}"
    )


def collective_kernel_generation(collective_library: Any) -> str:
    """Return the public NCCL/RCCL implementation lineage."""
    if collective_library not in {"nccl", "rccl"}:
        raise ContractError("reference collective library must be nccl or rccl")
    return collective_library


def project_resource_profile(provenance: dict[str, Any]) -> dict[str, Any]:
    """Project backend provenance into the canonical cross-backend resource vocabulary."""
    device_units = provenance.get("device_sms") or provenance.get("device_cus")
    if provenance.get("num_sms") is not None:
        kind, configured = "sm", provenance["num_sms"]
    elif (
        provenance.get("block_num") is not None
        and provenance.get("kernel_type") != "AsyncLL"
    ):
        kind, configured = "cu_block", provenance["block_num"]
    else:
        kind, configured = None, None
    achieved = configured / device_units if configured and device_units else None
    fixed = "fixed-kernel" in str(provenance.get("tuned_source", ""))
    source = str(provenance.get("tuned_source", ""))
    num_nvl_bytes = provenance.get("num_nvl_bytes")
    num_rdma_bytes = provenance.get("num_rdma_bytes")
    persistent_bytes = (
        (num_nvl_bytes or 0) + (num_rdma_bytes or 0)
        if num_nvl_bytes is not None or num_rdma_bytes is not None
        else provenance.get("heap_size")
    )
    return {
        "achieved_fraction": round(achieved, 4) if achieved else None,
        "comm_units_kind": kind,
        "configured_units": configured,
        "conformance_class": (
            "not-applicable" if fixed else "backend-default" if "default" in source
            else "pinned-upstream"
        ),
        "device_units": device_units,
        "fixed_kernel": fixed,
        "nonconforming": False,
        "pareto_eligible": False,
        "persistent_bytes": persistent_bytes,
        "qps_per_rank": provenance.get("num_qps_per_rank"),
        "requested_fraction": None,
        "resource_class": "fixed-kernel" if fixed else "fixed-profile",
        "target_achieved_within_tol": None,
        "tolerance": 0.10,
        "tuned_source": provenance.get("tuned_source"),
        "warps_combine": provenance.get("combine_warps"),
        "warps_dispatch": provenance.get("dispatch_warps"),
    }


def backend_version(provenance: dict[str, Any]) -> str | None:
    """Return the canonical public backend version from implementation provenance."""
    for field in (
        "deepep_version", "uccl_version", "nccl_version",
        "mori_commit", "deepep_commit",
    ):
        value = provenance.get(field)
        if value is not None and str(value).strip():
            return str(value)[:160]
    return None


def public_series_config(
    *, kernel_generation: Any, provenance: dict[str, Any],
    resource_profile: dict[str, Any], resource_mode: Any, device_product: Any,
) -> dict[str, Any]:
    """Project raw implementation facts into the exact public configuration fields."""
    generation = None if kernel_generation == "n-a" else kernel_generation
    profile = "profile-" + _sha256_json(resource_profile)[:16]
    return {
        "backend": {
            "generation": generation,
            "version": backend_version(provenance),
        },
        "resource": {
            "mode": resource_mode,
            "profile": profile,
            "comm_units_kind": resource_profile.get("comm_units_kind"),
            "configured_units": resource_profile.get("configured_units"),
        },
        "system": {"label": str(device_product)[:160]},
    }


def public_series_config_sha256(config: dict[str, Any]) -> str:
    """Commit the canonical public configuration projection into series identity."""
    return _sha256_json(config)


SOURCE_BUILT_LIBRARY_ROLES = frozenset({
    "deepep-extension", "deepep-hybrid-extension",
})


def series_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Project stable semantic build identity while retaining raw binaries in private evidence."""
    projected = {
        key: value for key, value in provenance.items()
        if key not in {"jit_cache_key", "jit_shared_objects", "path", "sm_fraction"}
    }
    libraries = provenance.get("loaded_libraries")
    if isinstance(libraries, list):
        projected["loaded_libraries"] = [
            {
                "name": item.get("name"),
                "role": item.get("role"),
                "source_tree": provenance.get("deepep_tree"),
            }
            if isinstance(item, dict) and item.get("role") in SOURCE_BUILT_LIBRARY_ROLES
            else item
            for item in libraries
        ]
    jit_cubins = provenance.get("jit_cubins")
    if isinstance(jit_cubins, list):
        projected["jit_cubins"] = [
            {
                "cache_key": item.get("cache_key"),
                "sass_sha256": item.get("sass_sha256"),
                "source_sha256": item.get("source_sha256"),
            }
            if isinstance(item, dict)
            else item
            for item in jit_cubins
        ]
    return projected


def routing_implementation_control_sha256(implementation: dict[str, Any]) -> str:
    """Bind routing cohorts to the same static build/generator and non-treatment configuration."""
    provenance = implementation.get("provenance")
    if not isinstance(provenance, dict):
        raise ContractError("implementation provenance is unavailable")
    semantic = series_provenance(provenance)
    treatment_fields = {
        "jit_cache_key", "jit_cubins", "jit_kernel_keys", "jit_shared_objects",
        "local_experts", "num_experts", "path", "realized_config", "sm_fraction",
    }
    return _sha256_json({
        "kernel_generation": implementation.get("kernel_generation"),
        "name": implementation.get("name"),
        "provenance": {
            key: value for key, value in semantic.items()
            if key not in treatment_fields
        },
        "resource_profile": implementation.get("resource_profile"),
    })


def _resolved_provenance_value(field: str, value: Any) -> bool:
    if value is None or isinstance(value, (dict, list, tuple, set)) and not value:
        return False
    text = str(value).strip().lower()
    if not text or text in {"unknown", "none", "null", "n/a", "?", "capture-failed"}:
        return False
    if "capture-failed" in text:
        return False
    if field.endswith("_commit") and (
        text in {"main", "hybrid-ep", "uccl", "pkg-uccl"}
        or text.endswith(("-unknown", "-none", "-main", "-hybrid-ep"))
    ):
        return False
    return True


def _content_evidence_is_valid(value: Any, required_roles: set[str]) -> bool:
    if not isinstance(value, list) or not value:
        return False
    records: set[tuple[str, str]] = set()
    roles: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "role", "sha256"}:
            return False
        name, role, digest = item["name"], item["role"], item["sha256"]
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,159}", name)
            or not isinstance(role, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", role)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or (role, name) in records
        ):
            return False
        records.add((role, name))
        roles.add(role)
    return required_roles <= roles


def _deepep_v2_jit_cubins_are_valid(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != len(DEEPEP_V2_JIT_KERNELS):
        return False
    cache_keys = []
    kernel_names = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "cache_key", "cubin_sha256", "sass_sha256", "source_sha256",
        }:
            return False
        cache_key = item["cache_key"]
        match = (
            re.fullmatch(r"kernel\.([A-Za-z0-9_+-]+)\.[0-9a-f]{32}", cache_key)
            if isinstance(cache_key, str)
            else None
        )
        if (
            match is None
            or any(
                not isinstance(item[field], str)
                or not re.fullmatch(r"[0-9a-f]{64}", item[field])
                for field in ("cubin_sha256", "sass_sha256", "source_sha256")
            )
        ):
            return False
        cache_keys.append(cache_key)
        kernel_names.add(match.group(1))
    return (
        cache_keys == sorted(set(cache_keys))
        and kernel_names == DEEPEP_V2_JIT_KERNELS
    )


HYBRID_REALIZED_CONFIG_FIELDS = {
    "hidden_dim", "max_num_of_tokens_per_rank", "num_of_experts_per_rank",
    "num_of_ranks_per_node", "num_of_nodes", "pad_multiple",
    "num_of_tokens_per_chunk_preprocessing_api",
    "num_of_threads_per_block_preprocessing_api", "num_of_blocks_preprocessing_api",
    "num_of_blocks_permute", "num_of_blocks_unpermute", "token_data_type",
    "num_of_stages_dispatch_api", "num_of_stages_permute_block_dispatch_api",
    "num_of_in_flight_s2g_dispatch_api",
    "num_of_in_flight_s2g_permute_block_dispatch_api",
    "num_of_additional_in_flight_s2g_dispatch_api",
    "num_of_tokens_per_chunk_dispatch_api", "num_of_blocks_dispatch_api",
    "forward_dispatch_api", "device_side_sync_dispatch_api",
    "num_of_stages_g2s_combine_api", "num_of_stages_s2g_combine_api",
    "num_of_tokens_per_chunk_combine_api", "num_of_tokens_per_group_combine_api",
    "num_of_blocks_combine_api", "num_of_additional_in_flight_s2g_combine_api",
    "backward_combine_api", "device_side_sync_combine_api",
}
HYBRID_REALIZED_BOOL_FIELDS = {
    "forward_dispatch_api", "device_side_sync_dispatch_api", "backward_combine_api",
    "device_side_sync_combine_api",
}


def _hybrid_realized_config_is_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != HYBRID_REALIZED_CONFIG_FIELDS:
        return False
    for field, field_value in value.items():
        if field in HYBRID_REALIZED_BOOL_FIELDS:
            if type(field_value) is not bool:
                return False
        elif field == "token_data_type":
            if field_value not in {"UINT8", "UINT16"}:
                return False
        elif type(field_value) is not int or field_value < 0:
            return False
    return all(value[field] > 0 for field in (
        "hidden_dim", "max_num_of_tokens_per_rank", "num_of_experts_per_rank",
        "num_of_ranks_per_node", "num_of_nodes",
    ))


def hybrid_communication_domains(ep_size: int, scale_up_domain: int) -> tuple[int, int]:
    """Return active ranks per fabric domain and the number of such domains."""
    if type(ep_size) is not int or type(scale_up_domain) is not int:
        raise ContractError("hybrid communication topology must be integral")
    if ep_size <= 0 or scale_up_domain <= 0:
        raise ContractError("hybrid communication topology must be positive")
    domain_ranks = min(ep_size, scale_up_domain)
    if ep_size % domain_ranks:
        raise ContractError("hybrid EP size does not divide into communication domains")
    return domain_ranks, ep_size // domain_ranks


def _hybrid_kernel_keys_are_valid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and len(set(value)) == 3
        and value == sorted(value)
        and all(
            isinstance(key, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,511}", key)
            for key in value
        )
    )


def _hybrid_jit_evidence_is_valid(value: Any, kernel_keys: Any) -> bool:
    if not _hybrid_kernel_keys_are_valid(kernel_keys) or not isinstance(value, list):
        return False
    if len(value) != len(kernel_keys):
        return False
    rank_sets = []
    for expected_key, item in zip(kernel_keys, value):
        if not isinstance(item, dict) or set(item) != {"kernel_key", "rank_artifacts"}:
            return False
        rank_artifacts = item["rank_artifacts"]
        if item["kernel_key"] != expected_key or not isinstance(rank_artifacts, list):
            return False
        ranks = []
        for artifact in rank_artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {"bytes", "rank", "sha256"}:
                return False
            rank, digest, size = artifact["rank"], artifact["sha256"], artifact["bytes"]
            if (
                type(rank) is not int
                or rank < 0
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(size) is not int
                or size <= 0
            ):
                return False
            ranks.append(rank)
        if not ranks or ranks != list(range(len(ranks))):
            return False
        rank_sets.append(ranks)
    return all(ranks == rank_sets[0] for ranks in rank_sets)


def backend_provenance_issues(backend: str, provenance: dict[str, Any]) -> list[str]:
    unknown = [
        field for field, value in provenance.items()
        if isinstance(value, str) and value.strip().lower() == "unknown"
    ]
    unresolved = [
        field for field in REQUIRED_BACKEND_PROVENANCE.get(backend, ())
        if not _resolved_provenance_value(field, provenance.get(field))
    ]
    if backend == "deepep":
        mode = provenance.get("mnnvl_comm")
        allow = provenance.get("allow_mnnvl")
        valid_modes = {
            "not-requested": False,
            "explicit-allow-mnnvl": True,
        }
        if type(allow) is not bool or valid_modes.get(mode) is not allow:
            unresolved.append("mnnvl_comm")
        if provenance.get("backend_lineage") != "deepep-v1":
            unresolved.append("backend_lineage")
    if backend in {"deepep", "uccl"}:
        mode = provenance.get("mode")
        num_nvl_bytes = provenance.get("num_nvl_bytes")
        num_rdma_bytes = provenance.get("num_rdma_bytes")
        if mode not in {"normal", "low-latency"}:
            unresolved.append("mode")
        if type(num_nvl_bytes) is not int or num_nvl_bytes < 0:
            unresolved.append("num_nvl_bytes")
        if type(num_rdma_bytes) is not int or num_rdma_bytes < 0:
            unresolved.append("num_rdma_bytes")
        if mode == "normal" and (type(num_nvl_bytes) is not int or num_nvl_bytes <= 0):
            unresolved.append("num_nvl_bytes")
        if mode == "low-latency":
            if num_nvl_bytes != 0:
                unresolved.append("num_nvl_bytes")
            if type(num_rdma_bytes) is not int or num_rdma_bytes <= 0:
                unresolved.append("num_rdma_bytes")
            if (
                type(provenance.get("num_max_tokens_per_rank")) is not int
                or provenance["num_max_tokens_per_rank"] <= 0
            ):
                unresolved.append("num_max_tokens_per_rank")
            if backend == "deepep" and (
                type(provenance.get("num_qps_per_rank")) is not int
                or provenance["num_qps_per_rank"] <= 0
            ):
                unresolved.append("num_qps_per_rank")
    if backend == "deepep-v2":
        for field in ("num_experts", "tuning_num_experts"):
            if type(provenance.get(field)) is not int or provenance[field] <= 0:
                unresolved.append(field)
        if not _deepep_v2_jit_cubins_are_valid(provenance.get("jit_cubins")):
            unresolved.append("jit_cubins")
        if provenance.get("jit_random_seed") != "collectivex-deepep-v2-fa8a9b1":
            unresolved.append("jit_random_seed")
        unresolved.extend(
            field for field, expected in DEEPEP_V2_V1_PROVENANCE.items()
            if field != "deepep_distribution_version"
            and provenance.get(field) != expected
        )
        if provenance.get("deepep_distribution_version") not in (
            DEEPEP_V2_DISTRIBUTION_VERSIONS
        ):
            unresolved.append("deepep_distribution_version")
        policy = (
            provenance.get("allow_hybrid_mode"),
            provenance.get("gin_enabled"),
            provenance.get("communication_backend"),
        )
        if policy not in {
            (False, False, "nccl-device-lsa"),
            (True, True, "nccl-gin"),
        }:
            unresolved.extend(
                ("allow_hybrid_mode", "gin_enabled", "communication_backend")
            )
    content_roles = {
        "deepep-v2": {"deepep-extension", "nccl", "nvshmem"},
        "deepep-hybrid": {"deepep-extension", "deepep-hybrid-extension"},
        "uccl": {
            "uccl-distribution", "uccl-wrapper", "intervaltree-distribution",
            "sortedcontainers-distribution", "cuda-runtime",
        },
    }.get(backend)
    if content_roles is not None and not _content_evidence_is_valid(
        provenance.get("loaded_libraries"), content_roles
    ):
        unresolved.append("loaded_libraries")
    if backend in {"deepep-v2", "deepep-hybrid"} and not re.fullmatch(
        r"[0-9a-f]{40}", str(provenance.get("deepep_tree", ""))
    ):
        unresolved.append("deepep_tree")
    if backend == "deepep-hybrid" and provenance.get("backend_lineage") != "deepep-hybrid":
        unresolved.append("backend_lineage")
    if backend == "deepep-hybrid":
        if not _hybrid_realized_config_is_valid(provenance.get("realized_config")):
            unresolved.append("realized_config")
        if not _hybrid_kernel_keys_are_valid(provenance.get("jit_kernel_keys")):
            unresolved.append("jit_kernel_keys")
        if not _hybrid_jit_evidence_is_valid(
            provenance.get("jit_shared_objects"), provenance.get("jit_kernel_keys")
        ):
            unresolved.append("jit_shared_objects")
    if backend == "uccl" and provenance.get("backend_lineage") != "uccl":
        unresolved.append("backend_lineage")
    if backend == "uccl" and provenance.get("uccl_dependency_versions") != (
        UCCL_DEPENDENCY_VERSIONS
    ):
        unresolved.append("uccl_dependency_versions")
    if backend == "nccl-ep":
        collective = provenance.get("collective_library")
        if collective not in {"nccl", "rccl"}:
            unresolved.append("collective_library")
        if provenance.get("backend_lineage") != collective:
            unresolved.append("backend_lineage")
    if backend == "mori" and provenance.get("kernel_type") == "InterNodeV1":
        expected = {
            "block_num": 96,
            "rdma_block_num": 64,
            "dispatch_warps": 8,
            "combine_warps": 8,
            "num_qps": 1,
            "use_external_inp_buf": True,
            "gpus_per_node": 8,
        }
        unresolved.extend(
            field for field, value in expected.items()
            if provenance.get(field) != value
        )
    for field, minimum in (
        ("num_nvl_bytes", 0), ("num_rdma_bytes", 0),
        ("num_qps_per_rank", 1),
    ):
        if field in provenance and (
            type(provenance[field]) is not int or provenance[field] < minimum
        ):
            unresolved.append(field)
    if "rdma_block_num" in provenance and (
        type(provenance["rdma_block_num"]) is not int
        or provenance["rdma_block_num"] < 0
    ):
        unresolved.append("rdma_block_num")
    if "use_external_inp_buf" in provenance and type(
        provenance["use_external_inp_buf"]
    ) is not bool:
        unresolved.append("use_external_inp_buf")
    return sorted(set(unknown + unresolved))


def provenance_complete(
    provenance: dict[str, Any], backend: str, git_run: dict[str, Any] | None,
    *, allocation_stratum_sha256: Any, image_digest: Any, image_verified: Any,
    squash_sha256: Any,
) -> bool:
    image = str(image_digest or "")
    squash = str(squash_sha256 or "")
    allocation_stratum = str(allocation_stratum_sha256 or "")
    return (
        not backend_provenance_issues(backend, provenance)
        and bool(re.fullmatch(r"[0-9a-f]{64}", allocation_stratum))
        and image_verified is True
        and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", image))
        and bool(re.fullmatch(r"[0-9a-f]{64}", squash))
        and isinstance(git_run, dict)
        and all(git_run.get(field) for field in GIT_RUN_FIELDS)
    )


def strict_load(path: str | os.PathLike[str]) -> Any:
    """Load JSON while rejecting duplicate keys and non-finite constants."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise ContractError(f"non-finite JSON number {value}")

    try:
        with open(path) as handle:
            return json.load(handle, object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON {path}: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical finite JSON bytes for checksums and immutable artifacts."""
    _finite_tree(value)
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc


def content_manifest_evidence(
    *, role: str, name: str, files: Iterable[tuple[str, str | os.PathLike[str]]]
) -> dict[str, str]:
    """Hash a labeled file set without exposing any host path in provenance."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", role):
        raise ContractError("content evidence role is invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,159}", name):
        raise ContractError("content evidence name is invalid")
    manifest: list[dict[str, Any]] = []
    labels: set[str] = set()
    for label, raw_path in files:
        logical = PurePosixPath(label)
        if (
            not label
            or logical.is_absolute()
            or ".." in logical.parts
            or label in labels
            or any(ord(character) < 0x20 or ord(character) > 0x7E for character in label)
        ):
            raise ContractError("content evidence label is invalid or duplicated")
        path = Path(raw_path)
        if not path.is_file():
            raise ContractError("content evidence source is not a file")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        labels.add(label)
        manifest.append({"bytes": size, "label": label, "sha256": digest.hexdigest()})
    if not manifest:
        raise ContractError("content evidence cannot be empty")
    digest = hashlib.sha256(
        canonical_json_bytes(sorted(manifest, key=lambda item: item["label"]))
    ).hexdigest()
    return {"name": name, "role": role, "sha256": digest}


def _obj(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{path} must be an object")
    return value


def _keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    obj = _obj(value, path)
    actual = set(obj)
    if actual != expected:
        raise ContractError(
            f"{path} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return obj


def _text(value: Any, path: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ContractError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{path} must be an integer >= {minimum}")
    return value


def validate_conditioning_contract(value: Any, phase: str) -> dict[str, Any]:
    """Validate the exact phase-specific v1 conditioning schedule."""
    if phase not in V1_CONDITIONING_LADDERS:
        raise ContractError("raw conditioning phase is invalid")
    conditioning = _keys(
        value, {"contract", "ladder", "roundtrips_per_shape"},
        "raw.measurement.conditioning",
    )
    ladder = conditioning["ladder"]
    if (
        conditioning["contract"] != identity.V1_CASE_PROFILE["conditioning_contract"]
        or type(ladder) is not list
        or any(type(point) is not int for point in ladder)
        or ladder != list(V1_CONDITIONING_LADDERS[phase])
        or _integer(
            conditioning["roundtrips_per_shape"],
            "raw.measurement.conditioning.roundtrips_per_shape",
            minimum=1,
        ) != V1_CONDITIONING_ROUNDS_PER_SHAPE
    ):
        raise ContractError(f"raw {phase} conditioning contract differs")
    return conditioning


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{path} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ContractError(f"{path} must be >= {minimum}")
    return result


def _finite_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")


def _typed(value: Any, kind: str, path: str) -> str:
    if not identity.is_typed_id(value, kind):
        raise ContractError(f"{path} is not a {kind} ID")
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _precision_byte_provenance(
    axis: dict[str, Any], logical_copies: int, hidden: int
) -> dict[str, Any]:
    bits_per_value = {
        "bf16": 16,
        "fp8-e4m3fn": 8,
        "fp8-e4m3fnuz": 8,
        "logfmt10": 10,
    }.get(axis["communication_format"])
    if bits_per_value is None:
        raise ContractError("unknown communication precision format")
    scale_size = {None: 0, "f32": 4, "implicit-logfmt10": 0}.get(axis["scale_dtype"])
    if scale_size is None:
        raise ContractError("unknown communication scale dtype")
    group_size = axis["scale_group_size"]
    groups = math.ceil(hidden / group_size) if group_size is not None else 0
    activation = logical_copies * math.ceil(hidden * bits_per_value / 8)
    scales = logical_copies * groups * scale_size
    return {
        "accounting_contract": "activation-data-plus-scales-v1",
        "activation_data_bytes": activation,
        "scale_bytes": scales,
        "total_logical_bytes": activation + scales,
    }


@lru_cache(maxsize=None)
def _expected_eplb_calibration(
    routing: str,
    hidden: int,
    topk: int,
    logical_experts: int,
    physical_experts: int,
    ep_size: int,
    seed: int,
    reference_tokens_per_rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    member, checksums, indices, _ = workload_contract.canonical_eplb_calibration_member(
        routing,
        hidden,
        topk,
        logical_experts,
        ep_size,
        reference_tokens_per_rank,
        seed,
    )
    load = [0] * logical_experts
    for row in indices:
        for expert in row:
            load[expert] += 1
    plan = eplb_contract.build_plan(load, physical_experts, ep_size)
    descriptor = {
        "calibration_token_offset": workload_contract.EPLB_CALIBRATION_TOKEN_OFFSET,
        "calibration_trace_sha256": checksums["trace"],
        "calibration_window": workload_contract.EPLB_CALIBRATION_WINDOW,
        "calibration_workload_id": member,
    }
    return plan, descriptor


@lru_cache(maxsize=None)
def _expected_eplb_plan(
    routing: str,
    topk: int,
    logical_experts: int,
    physical_experts: int,
    ep_size: int,
    seed: int,
    reference_tokens_per_rank: int,
    hidden: int = 7168,
) -> dict[str, Any]:
    """Compatibility wrapper returning the disjoint calibration plan."""
    plan, _ = _expected_eplb_calibration(
        routing,
        hidden,
        topk,
        logical_experts,
        physical_experts,
        ep_size,
        seed,
        reference_tokens_per_rank,
    )
    return plan


@lru_cache(maxsize=None)
def _expected_canonical_trace(
    routing: str,
    hidden: int,
    topk: int,
    logical_experts: int,
    physical_experts: int,
    ep_size: int,
    tokens_per_rank: int,
    seed: int,
    eplb_enabled: bool,
    reference_tokens_per_rank: int,
) -> tuple[str, dict[str, str], str, list[list[int]], list[list[float]]]:
    member, checksums, indices, weights = workload_contract.canonical_member(
        routing,
        hidden,
        topk,
        logical_experts,
        ep_size,
        tokens_per_rank,
        seed,
    )
    if eplb_enabled:
        plan = _expected_eplb_plan(
            routing,
            topk,
            logical_experts,
            physical_experts,
            ep_size,
            seed,
            reference_tokens_per_rank,
            hidden,
        )
        indices = eplb_contract.remap_rows(indices, plan)
    routing_hash = workload_contract.trace_checksums(indices, weights)["trace"]
    return member, checksums, routing_hash, indices, weights


def _coefficient_of_variation(values: list[int]) -> float:
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5 / mean


def _expected_routing_summary(
    indices: list[list[int]],
    weights: list[list[float]],
    *,
    physical_experts: int,
    ep_size: int,
    tokens_per_rank: int,
    gpus_per_node: int,
    scale_up_domain: int,
) -> dict[str, Any]:
    """Recompute every published routing/load statistic without torch."""
    experts_per_rank = physical_experts // ep_size
    expert_load = [0] * physical_experts
    assignment_load = [0] * ep_size
    payload_load = [0] * ep_size
    fanouts: list[int] = []
    local = same_node = same_domain = copies = 0
    for token, row in enumerate(indices):
        destinations = {expert // experts_per_rank for expert in row}
        source = token // tokens_per_rank
        fanouts.append(len(destinations))
        for expert in row:
            expert_load[expert] += 1
            assignment_load[expert // experts_per_rank] += 1
        for destination in destinations:
            payload_load[destination] += 1
            copies += 1
            local += destination == source
            same_node += destination // gpus_per_node == source // gpus_per_node
            same_domain += destination // scale_up_domain == source // scale_up_domain
    fanout_histogram = [fanouts.count(value) for value in range(1, ep_size + 1)]
    expert_mean = sum(expert_load) / len(expert_load)
    return {
        "empty_expert_count": expert_load.count(0),
        "empty_rank_count": payload_load.count(0),
        "expert_assignment_rank_cv": _coefficient_of_variation(assignment_load),
        "expert_assignments_per_rank": assignment_load,
        "expert_load_cv": _coefficient_of_variation(expert_load),
        "expert_load_max": max(expert_load),
        "expert_load_mean": expert_mean,
        "expert_load_min": min(expert_load),
        "fanout_histogram": fanout_histogram,
        "fanout_max": max(fanouts),
        "fanout_mean": sum(fanouts) / len(fanouts),
        "fanout_min": min(fanouts),
        "hash": workload_contract.trace_checksums(indices, weights)["trace"],
        "hotspot_ratio": max(expert_load) / expert_mean if expert_mean else 0.0,
        "locality": {
            "placement": "packed",
            "local_rank_fraction": local / copies,
            "same_node_fraction": same_node / copies,
            "same_scaleup_domain_fraction": same_domain / copies,
            "cross_node_fraction": 1 - same_node / copies,
            "cross_domain_fraction": 1 - same_domain / copies,
            "gpus_per_node": gpus_per_node,
            "scale_up_domain": scale_up_domain,
            "copies": copies,
        },
        "payload_copies_per_rank": payload_load,
        "payload_rank_cv": _coefficient_of_variation(payload_load),
        "routed_copies": copies,
        "source_token_stats": {
            "min": tokens_per_rank,
            "mean": float(tokens_per_rank),
            "max": tokens_per_rank,
            "cv": 0.0,
            "empty_ranks": 0,
            "total": tokens_per_rank * ep_size,
            "ranks": ep_size,
        },
    }


def _expected_histogram(samples: list[float], bins: int = 40) -> dict[str, Any]:
    low, high = min(samples), max(samples)
    if high <= low:
        return {"n": len(samples), "min": low, "max": high, "bins": bins, "counts": [len(samples)]}
    counts = [0] * bins
    span = high - low
    for sample in samples:
        index = min(bins - 1, int((sample - low) / span * bins))
        counts[index] += 1
    return {
        "n": len(samples),
        "min": round(low, 3),
        "max": round(high, 3),
        "bins": bins,
        "counts": counts,
    }


def _expected_anomalies(
    tokens: int, components: dict[str, Any]
) -> list[dict[str, Any]]:
    dispatch = components["dispatch"]["percentiles_us"]
    stage = components["stage"]["percentiles_us"]
    combine = components["combine"]["percentiles_us"]
    roundtrip = components["roundtrip"]["percentiles_us"]
    isolated = components["isolated_sum"]["percentiles_us"]
    anomalies: list[dict[str, Any]] = []
    if isolated is not None and roundtrip["p99"] > 3.0 * isolated["p99"]:
        anomalies.append({
            "type": "roundtrip_gt_isolated_sum",
            "T": tokens,
            "roundtrip_p99": round(roundtrip["p99"], 2),
            "isolated_sum_p99": round(isolated["p99"], 2),
            "ratio": round(roundtrip["p99"] / isolated["p99"], 2),
            "threshold": 3.0,
        })
    floor = (
        max(dispatch["p50"], combine["p50"], stage["p50"] if stage is not None else 0.0)
        if dispatch and combine else None
    )
    if floor and roundtrip["p50"] < 0.95 * floor:
        anomalies.append({
            "type": "roundtrip_lt_component_floor",
            "T": tokens,
            "roundtrip_p50": round(roundtrip["p50"], 2),
            "component_floor_p50": round(floor, 2),
        })
    return anomalies


def _validate_canonical_workload(
    workload: dict[str, Any],
    scheduled_case: dict[str, Any],
    rows: list[dict[str, Any]],
    eplb: dict[str, Any],
) -> None:
    """Bind every canonical member and measured routing hash to its scheduled token row."""
    profile = identity.profile_for_case(scheduled_case)
    if eplb["enabled"]:
        plan = _expected_eplb_plan(
            scheduled_case["routing"],
            scheduled_case["topk"],
            scheduled_case["experts"],
            eplb["num_physical_experts"],
            scheduled_case["ep"],
            profile["seed"],
            profile["eplb_reference_tokens_per_rank"],
            scheduled_case["hidden"],
        )
        if eplb["mapping_hash"] != eplb_contract.mapping_hash(plan):
            raise ContractError("raw EPLB mapping differs from the frozen canonical plan")

    expected: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        member, checksums, routing_hash, _, _ = _expected_canonical_trace(
            scheduled_case["routing"],
            scheduled_case["hidden"],
            scheduled_case["topk"],
            scheduled_case["experts"],
            eplb["num_physical_experts"],
            scheduled_case["ep"],
            row["tokens_per_rank"],
            profile["seed"],
            eplb["enabled"],
            profile["eplb_reference_tokens_per_rank"],
        )
        if row["routing"]["hash"] != routing_hash:
            raise ContractError(
                f"raw.measurement.rows[{index}].routing.hash differs from its canonical member"
            )
        expected[member] = checksums
    if (
        len(expected) != len(rows)
        or workload["members"] != sorted(expected)
        or workload["manifest_checksums"] != expected
    ):
        raise ContractError("raw canonical member set/checksums differ from scheduled rows")
    expected_workload_id = identity.workload_id({
        "members": [
            {"checksums": expected[member], "workload_id": member}
            for member in sorted(expected)
        ]
    })
    if workload["workload_id"] != expected_workload_id:
        raise ContractError("raw composite workload identity differs from scheduled rows")


def _nearest_rank(samples: list[float], q: int) -> float:
    ordered = sorted(samples)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))]


def _close(observed: Any, expected: float, path: str, tolerance: float = 1e-6) -> None:
    value = _number(observed, path)
    if not math.isclose(value, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ContractError(f"{path}={value} differs from recomputed {expected}")


def _equivalent(
    observed: Any, expected: Any, path: str, *, tolerance: float = 1e-6
) -> None:
    """Compare a recomputed JSON subtree while allowing only float roundoff."""
    if isinstance(expected, dict):
        value = _keys(observed, set(expected), path)
        for key, child in expected.items():
            _equivalent(value[key], child, f"{path}.{key}", tolerance=tolerance)
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ContractError(f"{path} differs from recomputed evidence")
        for index, child in enumerate(expected):
            _equivalent(observed[index], child, f"{path}[{index}]", tolerance=tolerance)
        return
    if isinstance(expected, float):
        _close(observed, expected, path, tolerance)
        return
    if type(observed) is not type(expected) or observed != expected:
        raise ContractError(f"{path} differs from recomputed evidence")


def _schema_equal(left: Any, right: Any) -> bool:
    """JSON Schema equality: booleans are distinct from numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _schema_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _schema_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _schema_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ContractError("native artifact schema contains a non-local reference")
    value: Any = root
    for part in reference[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or part not in value:
            raise ContractError("native artifact schema contains a broken reference")
        value = value[part]
    if not isinstance(value, dict):
        raise ContractError("native artifact schema reference is not an object")
    return value


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return type(value) is bool
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    if expected == "integer":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and float(value).is_integer()
        )
    raise ContractError(f"native artifact schema uses unsupported type {expected!r}")


def _validate_schema_value(
    value: Any, schema: dict[str, Any], root: dict[str, Any], path: str
) -> None:
    """Validate the bounded JSON Schema subset used by native artifact contracts."""
    if "$ref" in schema:
        _validate_schema_value(value, _schema_ref(root, schema["$ref"]), root, path)
        return
    if "oneOf" in schema:
        matches = 0
        for candidate in schema["oneOf"]:
            try:
                _validate_schema_value(value, candidate, root, path)
            except ContractError:
                continue
            matches += 1
        if matches != 1:
            raise ContractError(f"{path} must match exactly one native schema alternative")
        return
    expected_type = schema.get("type")
    if expected_type is not None and not _schema_type_matches(value, expected_type):
        raise ContractError(f"{path} is not a schema {expected_type}")
    if "const" in schema and not _schema_equal(value, schema["const"]):
        raise ContractError(f"{path} differs from its schema constant")
    if "enum" in schema and not any(_schema_equal(value, item) for item in schema["enum"]):
        raise ContractError(f"{path} is outside its schema enum")

    if isinstance(value, dict):
        required = set(schema.get("required", ()))
        properties = schema.get("properties", {})
        missing = required - set(value)
        if missing:
            raise ContractError(f"{path} lacks schema fields {sorted(missing)}")
        additional = schema.get("additionalProperties", True)
        extra = set(value) - set(properties)
        if additional is False and extra:
            raise ContractError(f"{path} has extra schema fields {sorted(extra)}")
        for key, item in value.items():
            if key in properties:
                _validate_schema_value(item, properties[key], root, f"{path}.{key}")
            elif isinstance(additional, dict):
                _validate_schema_value(item, additional, root, f"{path}.{key}")
        property_names = schema.get("propertyNames")
        if property_names is not None:
            for key in value:
                _validate_schema_value(key, property_names, root, f"{path}.<key>")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ContractError(f"{path} has too few schema items")
        maximum = schema.get("maxItems")
        if maximum is not None and len(value) > maximum:
            raise ContractError(f"{path} has too many schema items")
        if schema.get("uniqueItems") and any(
            _schema_equal(item, prior)
            for index, item in enumerate(value)
            for prior in value[:index]
        ):
            raise ContractError(f"{path} schema items are not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema_value(item, schema["items"], root, f"{path}[{index}]")

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ContractError(f"{path} is shorter than its schema minimum")
        maximum = schema.get("maxLength")
        if maximum is not None and len(value) > maximum:
            raise ContractError(f"{path} is longer than its schema maximum")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ContractError(f"{path} does not match its schema pattern")
        if schema.get("format") == "date-time":
            try:
                parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ContractError(f"{path} is not a schema date-time") from exc
            if parsed.tzinfo is None:
                raise ContractError(f"{path} schema date-time lacks a timezone")

    if (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    ):
        if "minimum" in schema and value < schema["minimum"]:
            raise ContractError(f"{path} is below its schema minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ContractError(f"{path} is above its schema maximum")


def _validate_native_schema(name: str, value: Any) -> None:
    schema = _SCHEMA_CACHE.get(name)
    if schema is None:
        loaded = strict_load(SCHEMA_DIR / name)
        if not isinstance(loaded, dict):
            raise ContractError(f"native artifact schema {name} is not an object")
        schema = loaded
        _SCHEMA_CACHE[name] = schema
    _validate_schema_value(value, schema, schema, "$")


def validate_samples_document(document: Any) -> dict[str, Any]:
    _validate_native_schema("samples-v1.schema.json", document)
    doc = _keys(
        document,
        {"allocation_id", "attempt_id", "case_id", "format", "points",
         "qualification_index", "sampling", "schema_version", "series_id"},
        "samples",
    )
    if doc["format"] != SAMPLES_FORMAT or doc["schema_version"] != 1:
        raise ContractError("samples format/schema differs from v1")
    for field, kind in (
        ("allocation_id", "allocation"), ("attempt_id", "attempt"),
        ("case_id", "case"), ("series_id", "series"),
    ):
        _typed(doc[field], kind, f"samples.{field}")
    qualification_index = _integer(
        doc["qualification_index"], "samples.qualification_index", minimum=1
    )
    if qualification_index > 3:
        raise ContractError("samples.qualification_index must be in 1..3")
    sampling = _keys(
        doc["sampling"], {"iterations_per_trial", "reduction", "trials"}, "samples.sampling"
    )
    if (
        _integer(sampling["iterations_per_trial"], "samples.sampling.iterations_per_trial", minimum=1) != 8
        or _integer(sampling["trials"], "samples.sampling.trials", minimum=1) != 64
        or sampling["reduction"] != identity.V1_CASE_PROFILE["rank_reduction"]
    ):
        raise ContractError("samples must use the fixed 8x64 cross-rank-max contract")
    points = doc["points"]
    if not isinstance(points, list) or not points:
        raise ContractError("samples.points must be non-empty")
    seen = set()
    for index, point_value in enumerate(points):
        path = f"samples.points[{index}]"
        point = _keys(
            point_value,
            {"components", "evidence_id", "point_id", "sample_sha256", "tokens_per_rank"},
            path,
        )
        tokens = _integer(point["tokens_per_rank"], f"{path}.tokens_per_rank", minimum=1)
        if tokens in seen:
            raise ContractError(f"duplicate sample token point {tokens}")
        seen.add(tokens)
        _typed(point["point_id"], "point", f"{path}.point_id")
        _typed(point["evidence_id"], "evidence", f"{path}.evidence_id")
        components = _keys(
            point["components"], {"combine", "dispatch", "roundtrip", "stage"},
            f"{path}.components",
        )
        for name, component_value in components.items():
            component = _keys(
                component_value, {"availability", "sample_count", "trials"},
                f"{path}.components.{name}",
            )
            availability = component["availability"]
            count = _integer(component["sample_count"], f"{path}.components.{name}.sample_count")
            trials = component["trials"]
            if availability == "unavailable":
                if count != 0 or trials is not None or name == "roundtrip":
                    raise ContractError(f"{path}.components.{name} has invalid unavailability")
                continue
            if availability != "measured" or not isinstance(trials, list) or len(trials) != 64:
                raise ContractError(f"{path}.components.{name} must contain 64 measured trials")
            if any(not isinstance(trial, list) or len(trial) != 8 for trial in trials):
                raise ContractError(f"{path}.components.{name} trials must each contain 8 samples")
            flattened = [
                _number(sample, f"{path}.components.{name}.trials", minimum=0.0)
                for trial in trials for sample in trial
            ]
            if count != 512 or len(flattened) != 512:
                raise ContractError(f"{path}.components.{name} must contain 512 samples")
        sample_base = {"components": components, "tokens_per_rank": tokens}
        if point["sample_sha256"] != _sha256_json(sample_base):
            raise ContractError(f"{path}.sample_sha256 differs")
    return doc


def _validate_component(
    component_value: Any,
    sample_component: dict[str, Any] | None,
    path: str,
    *,
    derived: bool = False,
) -> None:
    component = _keys(
        component_value, {"availability", "origin", "percentiles_us", "sample_count"}, path
    )
    availability = component["availability"]
    if availability == "unavailable":
        if component != {
            "availability": "unavailable", "origin": None,
            "percentiles_us": None, "sample_count": 0,
        }:
            raise ContractError(f"{path} has invalid unavailable representation")
        if sample_component and sample_component["availability"] != "unavailable":
            raise ContractError(f"{path} disagrees with samples")
        return
    expected_availability = "derived" if derived else "measured"
    expected_origin = "derived-percentile-sum" if derived else "measured"
    if availability != expected_availability or component["origin"] != expected_origin:
        raise ContractError(f"{path} has invalid availability/origin")
    percentiles = _keys(component["percentiles_us"], set(PERCENTILES), f"{path}.percentiles_us")
    if derived:
        if component["sample_count"] != 0:
            raise ContractError(f"{path}.sample_count must be zero for a derived value")
        return
    if sample_component is None or sample_component["availability"] != "measured":
        raise ContractError(f"{path} lacks measured sample evidence")
    flattened = [sample for trial in sample_component["trials"] for sample in trial]
    if component["sample_count"] != len(flattened):
        raise ContractError(f"{path}.sample_count differs from exact samples")
    for name, percentile in zip(PERCENTILES, (50, 90, 95, 99), strict=True):
        _close(percentiles[name], _nearest_rank(flattened, percentile), f"{path}.{name}")


def _validate_oracle(
    value: Any,
    path: str,
    profile: dict[str, Any] | None = None,
    communication_precision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = profile or identity.V1_NORMAL_CASE_PROFILE
    communication_precision = communication_precision or identity.precision_profile(
        identity.V1_CONTROL_PRECISION_PROFILE
    )
    tolerance = identity.combine_oracle_tolerances(communication_precision)
    oracle = _keys(
        value,
        {"atol", "checks", "combine_weight_semantics", "contract", "dispatch_sha256",
         "max_absolute_error", "max_elementwise_relative_error", "max_relative_error",
         "max_weight_error", "order_sha256", "ordering_contract", "passed", "receive_count",
         "rtol"},
        path,
    )
    if oracle["contract"] != profile["oracle_contract"]:
        raise ContractError(f"{path}.contract differs")
    checks = _keys(
        oracle["checks"],
        {"combine_values", "counts", "metadata", "multiplicity", "payload", "source_set",
         "weights"},
        f"{path}.checks",
    )
    if any(type(value) is not bool for value in checks.values()):
        raise ContractError(f"{path}.checks must be boolean")
    if type(oracle["passed"]) is not bool:
        raise ContractError(f"{path}.passed must be boolean")
    _integer(oracle["receive_count"], f"{path}.receive_count")
    _text(oracle["ordering_contract"], f"{path}.ordering_contract")
    expected_weight_semantics = (
        "gate-weighted-sum"
        if profile["combine_semantics"] == "gate-weighted"
        else "unweighted-rank-sum"
    )
    if oracle["combine_weight_semantics"] != expected_weight_semantics:
        raise ContractError(f"{path}.combine_weight_semantics differs from v1")
    _close(oracle["rtol"], tolerance["rtol"], f"{path}.rtol")
    _close(oracle["atol"], tolerance["atol"], f"{path}.atol")
    for field in ("dispatch_sha256", "order_sha256"):
        digest = oracle[field]
        if digest is not None and (
            not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ContractError(f"{path}.{field} is not a SHA-256 digest")
    for field in (
        "max_absolute_error", "max_elementwise_relative_error", "max_relative_error",
        "max_weight_error",
    ):
        if oracle[field] is not None:
            _number(oracle[field], f"{path}.{field}", minimum=0.0)
    expected_pass = (
        all(checks.values())
        and oracle["max_relative_error"] is not None
        and oracle["max_relative_error"] < tolerance["rtol"]
    )
    if oracle["passed"] != expected_pass:
        raise ContractError(f"{path}.passed differs from its evidence")
    return oracle


def _validate_precision_evidence(
    value: Any, profile_id: str, communication_precision: dict[str, Any], path: str
) -> dict[str, Any]:
    precision = _keys(value, {"combine", "dispatch", "passed", "profile_id"}, path)
    if precision["profile_id"] != profile_id or type(precision["passed"]) is not bool:
        raise ContractError(f"{path} profile/outcome differs")
    for direction in ("dispatch", "combine"):
        axis_path = f"{path}.{direction}"
        axis = _keys(
            precision[direction],
            {"dequantized_semantics", "encoded_payload_valid", "max_abs_error",
             "max_rel_error", "passed", "saturation_count", "saturation_rate",
             "scales_finite", "scales_positive"},
            axis_path,
        )
        for field in ("dequantized_semantics", "encoded_payload_valid", "passed"):
            if type(axis[field]) is not bool:
                raise ContractError(f"{axis_path}.{field} must be boolean")
        scale_dtype = communication_precision[direction]["scale_dtype"]
        expects_scales = scale_dtype not in (None, "implicit-logfmt10")
        for field in ("scales_finite", "scales_positive"):
            if expects_scales:
                if type(axis[field]) is not bool:
                    raise ContractError(f"{axis_path}.{field} must be boolean")
            elif axis[field] is not None:
                raise ContractError(f"{axis_path}.{field} must be null without scales")
        saturation_count = _integer(
            axis["saturation_count"], f"{axis_path}.saturation_count"
        )
        saturation_rate = _number(
            axis["saturation_rate"], f"{axis_path}.saturation_rate", minimum=0.0
        )
        if saturation_rate > 1.0:
            raise ContractError(f"{axis_path}.saturation_rate must be <= 1")
        _number(axis["max_abs_error"], f"{axis_path}.max_abs_error", minimum=0.0)
        _number(axis["max_rel_error"], f"{axis_path}.max_rel_error", minimum=0.0)
        expected_pass = (
            axis["encoded_payload_valid"]
            and axis["dequantized_semantics"]
            and (not expects_scales or (axis["scales_finite"] and axis["scales_positive"]))
            and saturation_count == 0
        )
        if axis["passed"] != bool(expected_pass):
            raise ContractError(f"{axis_path}.passed differs from its evidence")
    expected_pass = precision["dispatch"]["passed"] and precision["combine"]["passed"]
    if precision["passed"] != expected_pass:
        raise ContractError(f"{path}.passed differs from direction evidence")
    return precision


def validate_raw_document(document: Any, samples_document: Any) -> dict[str, Any]:
    """Validate identities, exact samples, formulas, privacy, and the native raw shape."""
    _validate_native_schema("raw-case-v1.schema.json", document)
    doc = _keys(
        document,
        {"case", "format", "generated_at", "identity", "implementation", "measurement",
         "outcome", "provenance", "record_type", "runtime_fingerprint", "sample_artifact",
         "schema_version", "topology", "workload"},
        "raw",
    )
    _finite_tree(doc)
    if doc["format"] != RAW_FORMAT or doc["schema_version"] != 1 or doc["record_type"] != "case-attempt":
        raise ContractError("raw format/schema/record type differs from v1")
    _text(doc["generated_at"], "raw.generated_at")
    identifiers = _keys(
        doc["identity"],
        {"allocation_factors", "allocation_id", "attempt_id", "attempt_ordinal", "case_factors",
         "case_id", "series_factors", "series_id"},
        "raw.identity",
    )
    for field, kind in (
        ("allocation_id", "allocation"), ("attempt_id", "attempt"),
        ("case_id", "case"), ("series_id", "series"),
    ):
        _typed(identifiers[field], kind, f"raw.identity.{field}")
    ordinal = _integer(identifiers["attempt_ordinal"], "raw.identity.attempt_ordinal", minimum=1)
    allocation_factors = _keys(
        identifiers["allocation_factors"], ALLOCATION_FACTOR_FIELDS,
        "raw.identity.allocation_factors",
    )
    qualification_index = _integer(
        allocation_factors["qualification_index"],
        "raw.identity.allocation_factors.qualification_index",
        minimum=1,
    )
    if qualification_index > 3:
        raise ContractError("raw qualification index must be in 1..3")
    case_factors = _keys(
        identifiers["case_factors"], {"case", "profile", "sku"},
        "raw.identity.case_factors",
    )
    scheduled_case = _scheduled_case(
        case_factors["case"], "raw.identity.case_factors.case"
    )
    profile = scheduled_case_profile(scheduled_case, "raw.identity.case_factors.case")
    if case_factors["profile"] != profile:
        raise ContractError("raw case profile differs from CollectiveX v1")
    _text(case_factors["sku"], "raw.identity.case_factors.sku")
    series_factors = _keys(
        identifiers["series_factors"],
        {"backend", "case_id", "image_digest", "implementation_contract_sha256",
         "public_config_sha256", "routing_control_sha256",
         "runtime_fingerprint_sha256", "source_sha", "squash_sha256", "workload_id"},
        "raw.identity.series_factors",
    )
    if identity.allocation_id(identifiers["allocation_factors"]) != identifiers["allocation_id"]:
        raise ContractError("allocation identity differs")
    if identity.digest("case", identifiers["case_factors"]) != identifiers["case_id"]:
        raise ContractError("case identity differs")
    if identity.series_id(identifiers["series_factors"]) != identifiers["series_id"]:
        raise ContractError("series identity differs")
    if identity.attempt_id(
        allocation=identifiers["allocation_id"], case=identifiers["case_id"], ordinal=ordinal
    ) != identifiers["attempt_id"]:
        raise ContractError("attempt identity differs")

    samples = validate_samples_document(samples_document)
    for field in ("allocation_id", "attempt_id", "case_id", "series_id"):
        if samples[field] != identifiers[field]:
            raise ContractError(f"samples.{field} differs from raw identity")
    if samples["qualification_index"] != qualification_index:
        raise ContractError("samples qualification index differs from raw allocation")
    sample_by_token = {point["tokens_per_rank"]: point for point in samples["points"]}

    case = _keys(
        doc["case"],
        {"attempt_ordinal", "backend", "eplb", "ep_size", "mode", "phase",
         "required_publication", "resource_mode", "runner", "shape", "suite", "workload_name"},
        "raw.case",
    )
    ep_size = _integer(case["ep_size"], "raw.case.ep_size", minimum=1)
    if case["attempt_ordinal"] != ordinal:
        raise ContractError("case attempt ordinal differs")
    for field in ("backend", "mode", "phase", "required_publication", "resource_mode", "runner",
                  "suite", "workload_name"):
        _text(case[field], f"raw.case.{field}")
    shape = _keys(
        case["shape"],
        {"activation_profile", "combine_precision", "dispatch_precision", "eplb", "experts",
         "experts_per_rank", "hidden", "kernel_gen", "num_logical_experts",
         "precision_profile", "routing", "topk"},
        "raw.case.shape",
    )
    hidden = _integer(shape["hidden"], "raw.case.shape.hidden", minimum=1)
    topk = _integer(shape["topk"], "raw.case.shape.topk", minimum=1)
    physical_experts = _integer(
        shape["experts"], "raw.case.shape.experts", minimum=1
    )
    logical_experts = _integer(
        shape["num_logical_experts"],
        "raw.case.shape.num_logical_experts",
        minimum=1,
    )
    experts_per_rank = _integer(
        shape["experts_per_rank"], "raw.case.shape.experts_per_rank", minimum=1
    )
    precision_profile_id = scheduled_case.get(
        "precision_profile", identity.V1_CONTROL_PRECISION_PROFILE
    )
    communication_precision = identity.precision_profile(precision_profile_id)
    if (
        shape["precision_profile"] != precision_profile_id
        or shape["dispatch_precision"] != communication_precision["dispatch"]
        or shape["combine_precision"] != communication_precision["combine"]
    ):
        raise ContractError("raw communication precision differs from scheduled case")
    eplb = _keys(
        case["eplb"],
        {"calibration_token_offset", "calibration_trace_sha256", "calibration_window",
         "calibration_workload_id", "enabled", "imbalance_after", "imbalance_before",
         "mapping_hash", "max_replicas", "num_logical_experts", "num_physical_experts",
         "num_redundant", "planner", "reference_tokens_per_rank", "replicated_experts"},
        "raw.case.eplb",
    )
    if not isinstance(eplb["enabled"], bool):
        raise ContractError("raw.case.eplb.enabled must be boolean")
    expected_redundant = (
        profile["eplb_redundant_experts"] if eplb["enabled"] else 0
    )
    expected_physical = eplb_contract.physical_count(
        scheduled_case["experts"], expected_redundant, ep_size
    )
    if (
        shape["eplb"] != eplb["enabled"]
        or logical_experts != scheduled_case["experts"]
        or physical_experts != expected_physical
        or experts_per_rank * ep_size != physical_experts
        or eplb["num_logical_experts"] != logical_experts
        or eplb["num_physical_experts"] != physical_experts
        or eplb["num_redundant"] != expected_redundant
    ):
        raise ContractError("raw EPLB/shape dimensions differ from the frozen profile")
    if eplb["enabled"]:
        expected_plan, calibration_descriptor = _expected_eplb_calibration(
            scheduled_case["routing"],
            hidden,
            topk,
            logical_experts,
            physical_experts,
            ep_size,
            profile["seed"],
            profile["eplb_reference_tokens_per_rank"],
        )
        expected_eplb = {
            **calibration_descriptor,
            "enabled": True,
            "imbalance_after": expected_plan["imbalance_after"],
            "imbalance_before": expected_plan["imbalance_before"],
            "mapping_hash": eplb_contract.mapping_hash(expected_plan),
            "max_replicas": expected_plan["max_replicas"],
            "num_logical_experts": logical_experts,
            "num_physical_experts": physical_experts,
            "num_redundant": expected_redundant,
            "planner": profile["eplb_planner"],
            "reference_tokens_per_rank": profile[
                "eplb_reference_tokens_per_rank"
            ],
            "replicated_experts": expected_plan["replicated_experts"],
        }
    else:
        expected_eplb = {
            "calibration_token_offset": None,
            "calibration_trace_sha256": None,
            "calibration_window": None,
            "calibration_workload_id": None,
            "enabled": False,
            "imbalance_after": None,
            "imbalance_before": None,
            "mapping_hash": None,
            "max_replicas": None,
            "num_logical_experts": logical_experts,
            "num_physical_experts": physical_experts,
            "num_redundant": 0,
            "planner": None,
            "reference_tokens_per_rank": None,
            "replicated_experts": 0,
        }
    _equivalent(eplb, expected_eplb, "raw.case.eplb", tolerance=1e-9)
    if case_factors["sku"] != case["runner"]:
        raise ContractError("raw case runner differs from case identity")

    workload = _keys(
        doc["workload"],
        {"activation_generator", "activation_identity", "activation_profile",
         "cross_rank_consistent", "manifest_checksums", "members", "routing_generator", "source",
         "trace_hashes", "trace_signature", "workload_id"},
        "raw.workload",
    )
    if workload["source"] not in {"canonical-serialized", "seeded-runtime"}:
        raise ContractError("raw workload source is invalid")
    if workload["source"] == "canonical-serialized":
        _typed(workload["workload_id"], "workload", "raw.workload.workload_id")
        members = workload["members"]
        checksums = workload["manifest_checksums"]
        if (
            not isinstance(members, list)
            or not members
            or members != sorted(set(members))
            or not all(identity.is_typed_id(member, "workload") for member in members)
            or not isinstance(checksums, dict)
            or set(checksums) != set(members)
        ):
            raise ContractError("raw canonical workload members/checksums are invalid")
        for member, values in checksums.items():
            if (
                not isinstance(values, dict)
                or set(values) != {"topk_idx", "topk_weights", "trace"}
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in values.values())
            ):
                raise ContractError(f"raw canonical workload checksums differ for {member}")
        expected_workload_id = identity.workload_id({
            "members": [
                {"checksums": checksums[member], "workload_id": member}
                for member in members
            ]
        })
        if workload["workload_id"] != expected_workload_id:
            raise ContractError("raw composite workload identity differs from its members")
    elif any(workload[field] is not None for field in ("members", "manifest_checksums", "workload_id")):
        raise ContractError("raw seeded workload cannot claim serialized members")
    if workload["cross_rank_consistent"] is not True:
        raise ContractError("raw workload is not consistent across ranks")

    measurement = _keys(
        doc["measurement"],
        {"component_order_contract", "conditioning", "contract", "execution_order_sha256",
         "qualification_index", "rows", "sampling", "source_allocation"},
        "raw.measurement",
    )
    if measurement["qualification_index"] != qualification_index:
        raise ContractError("raw measurement qualification index differs from allocation")
    if not isinstance(measurement["execution_order_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", measurement["execution_order_sha256"]
    ):
        raise ContractError("raw measurement execution order digest is invalid")
    validate_conditioning_contract(measurement["conditioning"], case["phase"])
    sampling = _keys(
        measurement["sampling"],
        {"contract", "iterations_per_trial", "percentile_method", "reduction",
         "samples_per_component", "trials", "warmup_iterations", "warmup_semantics"},
        "raw.measurement.sampling",
    )
    expected_sampling = {
        "contract": profile["sampling_contract"], "iterations_per_trial": 8,
        "percentile_method": profile["percentile_method"],
        "reduction": profile["rank_reduction"],
        "samples_per_component": 512, "trials": 64, "warmup_iterations": 32,
        "warmup_semantics": "full-roundtrip-before-each-component-trial-point-v1",
    }
    if sampling != expected_sampling:
        raise ContractError("raw sampling contract differs from fixed-512-v1")
    if (
        case["mode"] != profile["mode"]
        or case["resource_mode"] != profile["resource_mode"]
        or measurement["contract"] != profile["contract"]
        or measurement["component_order_contract"] != profile["component_order_contract"]
        or measurement["source_allocation"] != "even"
        or shape["activation_profile"] != profile["activation_profile"]
        or workload["activation_generator"] != profile["activation_generator"]
        or workload["activation_profile"] != profile["activation_profile"]
        or workload["routing_generator"] != profile["routing_generator"]
    ):
        raise ContractError("raw case differs from the frozen v1 profile")
    expected_activation = hashlib.sha256(
        (
            f"counter|seed={profile['seed']}|hidden={hidden}|"
            f"gen={profile['activation_generator']}"
        ).encode()
    ).hexdigest()
    if workload["activation_identity"] != expected_activation:
        raise ContractError("raw activation identity differs from the frozen seed/profile")
    rows = measurement["rows"]
    if not isinstance(rows, list) or not rows:
        raise ContractError("raw.measurement.rows must be non-empty")
    seen_points = set()
    row_tokens = []
    recomputed_anomalies = 0
    for index, row_value in enumerate(rows):
        path = f"raw.measurement.rows[{index}]"
        row = _keys(
            row_value,
            {"anomalies", "byte_provenance", "components", "correctness", "evidence_id",
             "global_tokens", "point_id", "receive", "routing",
             "sample_histograms", "sample_sha256", "token_rate_at_latency_percentile",
             "tokens_per_rank"},
            path,
        )
        tokens = _integer(row["tokens_per_rank"], f"{path}.tokens_per_rank", minimum=1)
        row_tokens.append(tokens)
        if tokens in seen_points or tokens not in sample_by_token:
            raise ContractError(f"{path} token point is duplicate or missing samples")
        seen_points.add(tokens)
        if row["global_tokens"] != tokens * ep_size:
            raise ContractError(f"{path}.global_tokens formula differs")
        sample_point = sample_by_token[tokens]
        expected_point = identity.point_id(series=identifiers["series_id"], tokens_per_rank=tokens)
        if row["point_id"] != expected_point or sample_point["point_id"] != expected_point:
            raise ContractError(f"{path}.point_id differs")
        expected_evidence = identity.evidence_id(
            point=expected_point, allocation=identifiers["allocation_id"],
            attempt=identifiers["attempt_id"], sample_sha256=sample_point["sample_sha256"],
        )
        if row["evidence_id"] != expected_evidence or sample_point["evidence_id"] != expected_evidence:
            raise ContractError(f"{path}.evidence_id differs")
        if row["sample_sha256"] != sample_point["sample_sha256"]:
            raise ContractError(f"{path}.sample_sha256 differs")
        components = _keys(
            row["components"], {"combine", "dispatch", "isolated_sum", "roundtrip", "stage"},
            f"{path}.components",
        )
        for name in ("combine", "dispatch", "roundtrip", "stage"):
            _validate_component(
                components[name], sample_point["components"][name], f"{path}.components.{name}"
            )
        _validate_component(
            components["isolated_sum"], None, f"{path}.components.isolated_sum", derived=True
        )
        expected_stage_availability = (
            "measured"
            if communication_precision["dispatch"]["communication_format"] != "bf16"
            or (case["backend"] == "mori" and shape["kernel_gen"] == "intranode")
            else "unavailable"
        )
        if components["stage"]["availability"] != expected_stage_availability:
            raise ContractError(f"{path}.components.stage differs from adapter device work")
        _, _, _, expected_indices, expected_weights = _expected_canonical_trace(
            scheduled_case["routing"],
            hidden,
            topk,
            logical_experts,
            physical_experts,
            ep_size,
            tokens,
            profile["seed"],
            eplb["enabled"],
            profile["eplb_reference_tokens_per_rank"],
        )
        expected_routing = _expected_routing_summary(
            expected_indices,
            expected_weights,
            physical_experts=physical_experts,
            ep_size=ep_size,
            tokens_per_rank=tokens,
            gpus_per_node=scheduled_case["gpus_per_node"],
            scale_up_domain=scheduled_case["scale_up_domain"],
        )
        _equivalent(
            row["routing"], expected_routing, f"{path}.routing", tolerance=1e-5
        )
        expected_payload_counts = (
            expected_routing["expert_assignments_per_rank"]
            if profile["payload_unit"] == "token-expert"
            else expected_routing["payload_copies_per_rank"]
        )
        throughput = _keys(
            row["token_rate_at_latency_percentile"], set(PERCENTILES),
            f"{path}.token_rate_at_latency_percentile",
        )
        for percentile in PERCENTILES:
            latency = components["roundtrip"]["percentiles_us"][percentile]
            if latency <= 0:
                raise ContractError(f"{path} roundtrip latency must be positive")
            _close(
                throughput[percentile], row["global_tokens"] / (latency * 1e-6),
                f"{path}.token_rate_at_latency_percentile.{percentile}", 1e-9,
            )
        correctness = _keys(
            row["correctness"],
            {"contract", "max_relative_error", "passed", "precision", "rank_evidence", "scope"},
            f"{path}.correctness",
        )
        if (
            correctness["contract"] != profile["oracle_contract"]
            or correctness["scope"] != profile["correctness_scope"]
            or type(correctness["passed"]) is not bool
        ):
            raise ContractError(f"{path}.correctness contract differs")
        precision_evidence = _validate_precision_evidence(
            correctness["precision"], precision_profile_id, communication_precision,
            f"{path}.correctness.precision",
        )
        _number(
            correctness["max_relative_error"],
            f"{path}.correctness.max_relative_error",
            minimum=0.0,
        )
        rank_evidence = correctness["rank_evidence"]
        if not isinstance(rank_evidence, list) or len(rank_evidence) != ep_size:
            raise ContractError(f"{path}.correctness.rank_evidence must cover every rank")
        ranks = set()
        observed_max_error = 0.0
        evidence_passed = True
        for evidence_index, evidence_value in enumerate(rank_evidence):
            evidence_path = f"{path}.correctness.rank_evidence[{evidence_index}]"
            evidence = _keys(
                evidence_value,
                {"input_unchanged", "order_stable", "post_timing", "pre_timing", "rank"},
                evidence_path,
            )
            evidence_rank = _integer(evidence["rank"], f"{evidence_path}.rank")
            if evidence_rank >= ep_size:
                raise ContractError(f"{evidence_path}.rank is outside the EP group")
            ranks.add(evidence_rank)
            if type(evidence["input_unchanged"]) is not bool or type(evidence["order_stable"]) is not bool:
                raise ContractError(f"{evidence_path} stability fields must be boolean")
            pre = _validate_oracle(
                evidence["pre_timing"], f"{evidence_path}.pre_timing", profile,
                communication_precision,
            )
            post = _validate_oracle(
                evidence["post_timing"], f"{evidence_path}.post_timing", profile,
                communication_precision,
            )
            if (
                pre["receive_count"] != expected_payload_counts[evidence_rank]
                or post["receive_count"] != expected_payload_counts[evidence_rank]
            ):
                raise ContractError(
                    f"{evidence_path}.receive_count differs from canonical routing"
                )
            expected_stability = all(
                pre[field] == post[field]
                for field in ("ordering_contract", "order_sha256", "dispatch_sha256")
            )
            if evidence["order_stable"] != expected_stability:
                raise ContractError(f"{evidence_path}.order_stable differs from the evidence")
            errors = [
                oracle["max_relative_error"]
                for oracle in (pre, post)
                if oracle["max_relative_error"] is not None
            ]
            observed_max_error = max([observed_max_error, *errors])
            evidence_passed = evidence_passed and all(
                (evidence["input_unchanged"], evidence["order_stable"], pre["passed"], post["passed"])
            )
        evidence_passed = evidence_passed and precision_evidence["passed"]
        if ranks != set(range(ep_size)) or correctness["passed"] != evidence_passed:
            raise ContractError(f"{path}.correctness rank coverage or outcome differs")
        _close(
            correctness["max_relative_error"], observed_max_error,
            f"{path}.correctness.max_relative_error",
        )
        if components["dispatch"]["availability"] == "measured":
            for percentile in PERCENTILES:
                expected = (
                    components["dispatch"]["percentiles_us"][percentile]
                    + (
                        components["stage"]["percentiles_us"][percentile]
                        if components["stage"]["availability"] == "measured"
                        else 0.0
                    )
                    + components["combine"]["percentiles_us"][percentile]
                )
                _close(
                    components["isolated_sum"]["percentiles_us"][percentile], expected,
                    f"{path}.components.isolated_sum.{percentile}",
                )
        logical_copies = (
            sum(expected_routing["expert_assignments_per_rank"])
            if profile["payload_unit"] == "token-expert"
            else expected_routing["routed_copies"]
        )
        dispatch_bytes = _precision_byte_provenance(
            communication_precision["dispatch"], logical_copies, hidden
        )
        combine_bytes = _precision_byte_provenance(
            communication_precision["combine"], logical_copies, hidden
        )
        stage_bytes = {
            "accounting_contract": "activation-data-plus-scales-v1",
            "activation_data_bytes": 0,
            "scale_bytes": 0,
            "total_logical_bytes": 0,
        }
        roundtrip_bytes = {
            "accounting_contract": "activation-data-plus-scales-v1",
            **{
                field: dispatch_bytes[field] + combine_bytes[field]
                for field in (
                    "activation_data_bytes", "scale_bytes", "total_logical_bytes"
                )
            },
        }
        expected_byte_provenance = {
            "combine": combine_bytes,
            "dispatch": dispatch_bytes,
            "roundtrip": roundtrip_bytes,
            "stage": stage_bytes,
        }
        _equivalent(
            row["byte_provenance"], expected_byte_provenance, f"{path}.byte_provenance"
        )

        max_receive = max(expected_payload_counts)
        expected_receive = {
            "max": max_receive,
            "mean": sum(expected_payload_counts) / ep_size,
            "min": min(expected_payload_counts),
            "total": sum(expected_payload_counts),
        }
        _equivalent(row["receive"], expected_receive, f"{path}.receive")
        expected_histograms = {
            name: (
                _expected_histogram([
                    sample
                    for trial in sample_point["components"][name]["trials"]
                    for sample in trial
                ])
                if sample_point["components"][name]["availability"] == "measured"
                else None
            )
            for name in ("dispatch", "stage", "combine", "roundtrip")
        }
        _equivalent(
            row["sample_histograms"], expected_histograms, f"{path}.sample_histograms"
        )
        expected_anomalies = _expected_anomalies(tokens, components)
        _equivalent(row["anomalies"], expected_anomalies, f"{path}.anomalies")
        recomputed_anomalies += len(expected_anomalies)
    if seen_points != set(sample_by_token):
        raise ContractError("raw rows and sample points differ")
    if row_tokens != sorted(row_tokens):
        raise ContractError("raw rows must follow the scheduled token ladder")
    expected_trace_hashes = sorted(row["routing"]["hash"] for row in rows)
    if workload["trace_hashes"] != expected_trace_hashes:
        raise ContractError("raw workload trace hashes differ from measured rows")
    expected_trace_signature = hashlib.sha256(
        "|".join(expected_trace_hashes).encode()
    ).hexdigest()
    if workload["trace_signature"] != expected_trace_signature:
        raise ContractError("raw workload trace signature differs from measured rows")

    implementation = _keys(
        doc["implementation"], {"kernel_generation", "name", "provenance", "resource_profile"},
        "raw.implementation",
    )
    if (
        implementation["name"] != case["backend"]
        or implementation["kernel_generation"] != shape["kernel_gen"]
    ):
        raise ContractError("raw implementation identity differs from the case")
    provenance_fields = _obj(implementation["provenance"], "raw.implementation.provenance")
    unknown = set(provenance_fields) - PROVENANCE_KEYS
    if unknown:
        raise ContractError(f"raw implementation provenance has unknown fields {sorted(unknown)}")
    if (
        implementation["name"] == "deepep-v2"
        and provenance_fields.get("deterministic") is not False
    ):
        raise ContractError("DeepEP V2 deterministic mode differs from the v1 kernel contract")
    if implementation["name"] == "deepep-v2" and (
        _integer(
            provenance_fields.get("tuning_num_experts"),
            "raw.implementation.provenance.tuning_num_experts",
            minimum=1,
        ) != logical_experts
        or _integer(
            provenance_fields.get("num_experts"),
            "raw.implementation.provenance.num_experts",
            minimum=1,
        ) != physical_experts
    ):
        raise ContractError("DeepEP V2 expert-count provenance differs from the case")
    if implementation["name"] == "deepep-hybrid":
        realized_config = provenance_fields.get("realized_config")
        jit_kernel_keys = provenance_fields.get("jit_kernel_keys")
        jit_shared_objects = provenance_fields.get("jit_shared_objects")
        domain_ranks, communication_domains = hybrid_communication_domains(
            ep_size, scheduled_case["scale_up_domain"]
        )
        if (
            not _hybrid_realized_config_is_valid(realized_config)
            or not _hybrid_jit_evidence_is_valid(jit_shared_objects, jit_kernel_keys)
            or realized_config["hidden_dim"] != shape["hidden"]
            or realized_config["num_of_experts_per_rank"] * ep_size != physical_experts
            or realized_config["num_of_ranks_per_node"] != domain_ranks
            or realized_config["num_of_nodes"] != communication_domains
            or realized_config["token_data_type"] != "UINT16"
            or any(
                len(artifact["rank_artifacts"]) != ep_size
                for artifact in jit_shared_objects
            )
        ):
            raise ContractError("DeepEP Hybrid realized config/JIT evidence differs from the case")
    if implementation["name"] == "nccl-ep" and implementation["kernel_generation"] != (
        collective_kernel_generation(provenance_fields.get("collective_library"))
    ):
        raise ContractError("NCCL/RCCL kernel generation differs from collective lineage")
    resource_profile = _obj(
        implementation["resource_profile"], "raw.implementation.resource_profile"
    )
    expected_resource_profile = project_resource_profile(provenance_fields)
    if resource_profile != expected_resource_profile:
        raise ContractError("raw resource profile differs from implementation provenance")
    topology = _keys(
        doc["topology"],
        {"device_count", "device_product", "gpus_per_node", "nodes", "placement",
         "realized_placement", "scale_out_transport", "scale_up_domain",
         "scale_up_transport", "scope", "topology_class", "transport", "world_size"},
        "raw.topology",
    )
    for field in ("device_count", "gpus_per_node", "nodes", "scale_up_domain", "world_size"):
        _integer(topology[field], f"raw.topology.{field}", minimum=1)
    for field in ("scale_up_transport", "scope", "topology_class", "transport"):
        _text(topology[field], f"raw.topology.{field}")
    if topology["scale_out_transport"] is not None:
        _text(topology["scale_out_transport"], "raw.topology.scale_out_transport")
    realized = _keys(
        topology["realized_placement"],
        {"gpus_per_node", "nodes", "ranks_per_node", "unique_local_ranks", "valid"},
        "raw.topology.realized_placement",
    )
    if realized != {
        "gpus_per_node": topology["gpus_per_node"],
        "nodes": topology["nodes"],
        "ranks_per_node": topology["gpus_per_node"],
        "unique_local_ranks": True,
        "valid": True,
    }:
        raise ContractError("raw realized placement differs from requested topology")
    if (
        topology["world_size"] != ep_size
        or topology["nodes"] * topology["gpus_per_node"] != ep_size
        or topology["device_count"] != topology["gpus_per_node"]
        or topology["placement"] != profile["placement"]
        or (
            topology["scope"] == "scale-up"
            and (
                ep_size > topology["scale_up_domain"]
                or topology["scale_out_transport"] is not None
            )
        )
        or (
            topology["scope"] == "scale-out"
            and (
                ep_size <= topology["scale_up_domain"]
                or ep_size % topology["scale_up_domain"] != 0
                or topology["scale_out_transport"] is None
            )
        )
        or topology["scope"] not in {"scale-up", "scale-out"}
    ):
        raise ContractError("raw topology dimensions differ from the case")
    if implementation["name"] == "deepep-v2":
        scale_out = scheduled_case["scope"] == "scale-out"
        expected_policy = (
            (True, True, "nccl-gin")
            if scale_out
            else (False, False, "nccl-device-lsa")
        )
        if (
            provenance_fields.get("allow_hybrid_mode"),
            provenance_fields.get("gin_enabled"),
            provenance_fields.get("communication_backend"),
        ) != expected_policy:
            raise ContractError("DeepEP V2 communication policy differs from the v1 contract")
        lsa_topology = tuple(
            _integer(
                provenance_fields.get(field),
                f"raw.implementation.provenance.{field}",
                minimum=1,
            )
            for field in (
                "physical_rdma_ranks", "physical_nvlink_ranks",
                "logical_scaleout_ranks", "logical_scaleup_ranks",
            )
        )
        domains = ep_size // scheduled_case["scale_up_domain"]
        expected_v2_topology = (
            (
                domains,
                scheduled_case["scale_up_domain"],
                domains,
                scheduled_case["scale_up_domain"],
            )
            if scale_out
            else (1, ep_size, 1, ep_size)
        )
        if lsa_topology != expected_v2_topology:
            raise ContractError("DeepEP V2 realized communication domains differ from topology")
    runtime = _keys(
        doc["runtime_fingerprint"],
        {"accelerator_runtime", "collective_library", "device", "driver_version", "framework",
         "machine", "python_version", "vendor"},
        "raw.runtime_fingerprint",
    )
    for field in ("machine", "python_version", "vendor"):
        _text(runtime[field], f"raw.runtime_fingerprint.{field}")
    runtime_device = _keys(
        runtime["device"], {"arch", "compute_units", "memory_bytes", "product", "warp_size"},
        "raw.runtime_fingerprint.device",
    )
    if topology["device_product"] != runtime_device["product"]:
        raise ContractError("raw topology and runtime device products differ")
    platform = capability.PLATFORMS.get(case["runner"])
    if platform is not None:
        identity_issues = capability.runtime_identity_issues(
            case["runner"], vendor=runtime["vendor"], arch=runtime_device["arch"],
            machine=runtime["machine"], device_name=runtime_device["product"],
            device_count=topology["device_count"], world_size=topology["world_size"],
        )
        registered_topology = capability.topology_for(case["runner"], ep_size)
        if identity_issues or (
            registered_topology is None
            or topology["gpus_per_node"] != platform["gpus_per_node"]
            or topology["scale_up_domain"] != platform["scale_up_domain"]
            or any(
                topology[field] != registered_topology[field]
                for field in (
                    "nodes", "scope", "scale_up_transport", "scale_out_transport",
                    "topology_class", "transport",
                )
            )
        ):
            raise ContractError(
                "raw runtime/topology differs from the scheduled SKU: "
                + "; ".join(identity_issues)
            )
    raw_provenance = _keys(
        doc["provenance"],
        {"allocation_stratum_sha256", "command", "distributed_launcher", "git_run",
         "image", "redaction"},
        "raw.provenance",
    )
    allocation_stratum = raw_provenance["allocation_stratum_sha256"]
    if workload["source"] == "canonical-serialized" and not (
        isinstance(allocation_stratum, str)
        and re.fullmatch(r"[0-9a-f]{64}", allocation_stratum)
    ):
        raise ContractError("canonical raw evidence is missing its private allocation stratum")
    image = _keys(
        raw_provenance["image"],
        {"arch", "digest", "digest_verified", "reference", "squash_sha256"},
        "raw.provenance.image",
    )
    if (
        image["digest_verified"] is not True
        or not isinstance(image["digest"], str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", image["digest"])
    ):
        raise ContractError("raw image digest was not registry-verified")
    if raw_provenance["redaction"] != "sanitized-v1":
        raise ContractError("raw provenance redaction contract differs")
    git_run = raw_provenance["git_run"]
    if git_run is not None:
        git_run = _keys(git_run, GIT_RUN_FIELDS, "raw.provenance.git_run")
        if git_run["qualification_index"] != qualification_index:
            raise ContractError("raw git run qualification index differs from allocation")
    expected_provenance_complete = provenance_complete(
        provenance_fields,
        case["backend"],
        git_run,
        allocation_stratum_sha256=allocation_stratum,
        image_digest=image["digest"],
        image_verified=image["digest_verified"],
        squash_sha256=image["squash_sha256"],
    )

    actual_scheduled_case = {
        "backend": case["backend"],
        "canonical": workload["source"] == "canonical-serialized",
        "eplb": eplb["enabled"],
        "ep": ep_size,
        "experts": shape["num_logical_experts"],
        "gpus_per_node": topology["gpus_per_node"],
        "hidden": hidden,
        "ladder": " ".join(map(str, row_tokens)),
        "mode": case["mode"],
        "nodes": topology["nodes"],
        "phase": case["phase"],
        "required_publication": case["required_publication"],
        "routing": shape["routing"],
        "samples_per_point": sampling["samples_per_component"],
        "scale_out_transport": topology["scale_out_transport"],
        "scale_up_domain": topology["scale_up_domain"],
        "scale_up_transport": topology["scale_up_transport"],
        "scope": topology["scope"],
        "suite": case["suite"],
        "timing": (
            f"{sampling['iterations_per_trial']}:{sampling['trials']}:"
            f"{sampling['warmup_iterations']}"
        ),
        "topk": shape["topk"],
        "topology_class": topology["topology_class"],
        "transport": topology["transport"],
        "warmup_semantics": sampling["warmup_semantics"],
        "workload": case["workload_name"],
    }
    if "precision_profile" in scheduled_case:
        actual_scheduled_case["precision_profile"] = shape["precision_profile"]
    if scheduled_case != actual_scheduled_case:
        mismatches = sorted(
            field for field in scheduled_case
            if scheduled_case[field] != actual_scheduled_case[field]
        )
        raise ContractError(f"raw data differs from scheduled case fields {mismatches}")

    if workload["source"] == "canonical-serialized":
        _validate_canonical_workload(workload, scheduled_case, rows, eplb)

    expected_series = {
        "backend": case["backend"],
        "case_id": identifiers["case_id"],
        "image_digest": image["digest"],
        "implementation_contract_sha256": _sha256_json({
            "kernel_generation": implementation["kernel_generation"],
            "name": implementation["name"],
            "provenance": series_provenance(provenance_fields),
            "resource_profile": resource_profile,
        }),
        "public_config_sha256": public_series_config_sha256(public_series_config(
            kernel_generation=implementation["kernel_generation"],
            provenance=provenance_fields,
            resource_profile=resource_profile,
            resource_mode=case["resource_mode"],
            device_product=topology["device_product"],
        )),
        "routing_control_sha256": routing_implementation_control_sha256(implementation),
        "runtime_fingerprint_sha256": _sha256_json(runtime),
        "source_sha": git_run["source_sha"] if git_run is not None else None,
        "squash_sha256": image["squash_sha256"],
        "workload_id": workload["workload_id"] or workload["trace_signature"],
    }
    if series_factors != expected_series:
        raise ContractError("raw series factors differ from measured implementation/runtime")
    expected_allocation = {
        "artifact": git_run["artifact"] if git_run is not None else None,
        "execution_id": allocation_factors["execution_id"],
        "job": git_run["job"] if git_run is not None else None,
        "qualification_index": qualification_index,
        "repo": git_run["repo"] if git_run is not None else None,
        "run_attempt": git_run["run_attempt"] if git_run is not None else None,
        "run_id": git_run["run_id"] if git_run is not None else None,
        "runner": case["runner"],
        "source_sha": git_run["source_sha"] if git_run is not None else None,
    }
    if allocation_factors != expected_allocation:
        raise ContractError("raw allocation factors differ from provenance")
    artifact = _keys(doc["sample_artifact"], {"bytes", "format", "path", "sha256"}, "raw.sample_artifact")
    if artifact["format"] != SAMPLES_FORMAT or Path(artifact["path"]).name != artifact["path"]:
        raise ContractError("raw.sample_artifact format/path is invalid")
    if not isinstance(artifact["sha256"], str) or len(artifact["sha256"]) != 64:
        raise ContractError("raw.sample_artifact.sha256 is invalid")
    _integer(artifact["bytes"], "raw.sample_artifact.bytes", minimum=1)
    outcome = _keys(doc["outcome"], {"publication_status", "reasons", "status", "validity"}, "raw.outcome")
    if outcome["status"] not in {"success", "invalid"} or outcome["publication_status"] not in {"diagnostic", "invalid"}:
        raise ContractError("raw outcome status is invalid")
    if not isinstance(outcome["reasons"], list) or not all(isinstance(x, str) for x in outcome["reasons"]):
        raise ContractError("raw outcome reasons must be strings")
    validity = _keys(
        outcome["validity"],
        {"anomaly_free", "execution_status", "measurement_conformance", "provenance_complete",
         "resource_conformance", "sampling_conformance", "semantic_correctness",
         "workload_identity", "workload_source"},
        "raw.outcome.validity",
    )
    correctness_passed = all(row["correctness"]["passed"] for row in rows)
    workload_consistent = workload["cross_rank_consistent"] is True
    expected_status = "success" if correctness_passed and workload_consistent else "invalid"
    expected_publication = "diagnostic" if expected_status == "success" else "invalid"
    if (
        outcome["status"] != expected_status
        or outcome["publication_status"] != expected_publication
        or bool(outcome["reasons"]) == (expected_status == "success")
        or validity["execution_status"] != "complete"
        or validity["semantic_correctness"] != ("pass" if correctness_passed else "fail")
        or validity["workload_identity"] != (
            "consistent-across-ranks" if workload_consistent else "inconsistent"
        )
        or validity["workload_source"] != workload["source"]
        or validity["measurement_conformance"] != "conformant"
        or validity["sampling_conformance"] != "conformant"
        or validity["resource_conformance"] != resource_profile["conformance_class"]
        or validity["anomaly_free"] != (recomputed_anomalies == 0)
        or validity["provenance_complete"] is not expected_provenance_complete
    ):
        raise ContractError("raw outcome differs from its measurement evidence")
    artifact_safety.assert_publication_safe([doc])
    return doc


def make_terminal_document(
    *,
    allocation_factors: dict[str, Any],
    attempt_ordinal: int,
    case: dict[str, Any],
    case_factors: dict[str, Any],
    control_sha256: str | None,
    failure_mode: str,
    generated_at: str,
    git_run: dict[str, Any] | None,
    reason: str,
    return_code: int,
    source: str,
    status: str,
    expected_case_id: str | None = None,
) -> dict[str, Any]:
    """Build and self-validate one attributable non-success attempt."""
    case_id = identity.digest("case", case_factors)
    if expected_case_id is not None and expected_case_id != case_id:
        raise ContractError(
            f"scheduled case ID differs from terminal factors: {expected_case_id} != {case_id}"
        )
    allocation_id = identity.allocation_id(allocation_factors)
    attempt_id = identity.attempt_id(
        allocation=allocation_id, case=case_id, ordinal=attempt_ordinal
    )
    document = {
        "format": TERMINAL_FORMAT,
        "schema_version": 1,
        "record_type": "terminal-outcome",
        "generated_at": generated_at,
        "identity": {
            "allocation_factors": allocation_factors,
            "allocation_id": allocation_id,
            "attempt_id": attempt_id,
            "attempt_ordinal": attempt_ordinal,
            "case_factors": case_factors,
            "case_id": case_id,
        },
        "case": case,
        "provenance": {
            "git_run": git_run,
            "control_sha256": control_sha256,
            "redaction": "sanitized-v1",
            "source": source,
        },
        "outcome": {
            "status": status,
            "failure_mode": failure_mode,
            "reason": reason,
            "return_code": return_code,
        },
    }
    return validate_terminal_document(document)


def validate_terminal_document(document: Any) -> dict[str, Any]:
    _validate_native_schema("terminal-outcome-v1.schema.json", document)
    doc = _keys(
        document,
        {"case", "format", "generated_at", "identity", "outcome", "provenance", "record_type",
         "schema_version"},
        "terminal",
    )
    if doc["format"] != TERMINAL_FORMAT or doc["schema_version"] != 1 or doc["record_type"] != "terminal-outcome":
        raise ContractError("terminal format/schema/record type differs from v1")
    ids = _keys(doc["identity"], {
        "allocation_factors", "allocation_id", "attempt_id", "attempt_ordinal",
        "case_factors", "case_id",
    }, "terminal.identity")
    for field, kind in (("allocation_id", "allocation"), ("attempt_id", "attempt"), ("case_id", "case")):
        _typed(ids[field], kind, f"terminal.identity.{field}")
    ordinal = _integer(ids["attempt_ordinal"], "terminal.identity.attempt_ordinal", minimum=1)
    case = _scheduled_case(doc["case"], "terminal.case")
    factors = _keys(ids["case_factors"], {"case", "profile", "sku"}, "terminal.identity.case_factors")
    if factors["case"] != case or factors["profile"] != scheduled_case_profile(
        case, "terminal.case"
    ):
        raise ContractError("terminal case factors differ from the scheduled case/profile")
    _text(factors["sku"], "terminal.identity.case_factors.sku")
    allocation = _keys(
        ids["allocation_factors"], ALLOCATION_FACTOR_FIELDS,
        "terminal.identity.allocation_factors",
    )
    qualification_index = _integer(
        allocation["qualification_index"],
        "terminal.identity.allocation_factors.qualification_index",
        minimum=1,
    )
    if qualification_index > 3:
        raise ContractError("terminal qualification index must be in 1..3")
    expected_case = identity.digest("case", factors)
    expected_allocation = identity.allocation_id(allocation)
    expected_attempt = identity.attempt_id(
        allocation=expected_allocation, case=expected_case, ordinal=ordinal
    )
    if (ids["case_id"], ids["allocation_id"], ids["attempt_id"]) != (
        expected_case, expected_allocation, expected_attempt
    ):
        raise ContractError("terminal typed identities do not match their factors")
    provenance = _keys(
        doc["provenance"], {"git_run", "control_sha256", "redaction", "source"},
        "terminal.provenance",
    )
    git_run = provenance["git_run"]
    if git_run is not None:
        git_run = _keys(git_run, GIT_RUN_FIELDS, "terminal.provenance.git_run")
        if git_run["qualification_index"] != qualification_index:
            raise ContractError(
                "terminal git run qualification index differs from allocation"
            )
    control = provenance["control_sha256"]
    if control is not None and (
        not isinstance(control, str) or len(control) != 64
        or any(char not in "0123456789abcdef" for char in control)
    ):
        raise ContractError("terminal control_sha256 is invalid")
    if provenance["redaction"] != "sanitized-v1":
        raise ContractError("terminal redaction contract differs")
    source = _text(provenance["source"], "terminal.provenance.source")
    outcome = _keys(
        doc["outcome"], {"failure_mode", "reason", "return_code", "status"}, "terminal.outcome"
    )
    if outcome["status"] not in {"failed", "invalid", "unsupported"}:
        raise ContractError("terminal outcome status is invalid")
    failure_mode = _text(outcome["failure_mode"], "terminal.outcome.failure_mode")
    reason = _text(outcome["reason"], "terminal.outcome.reason")
    _integer(outcome["return_code"], "terminal.outcome.return_code")
    if source == "runtime-emitter":
        expected_runner = factors["sku"]
        expected_reason = RUNTIME_FAILURE_REASONS.get(failure_mode)
        valid_outcome = outcome["status"] == "failed" and reason == expected_reason
    elif source == "post-emit-command":
        expected_runner = factors["sku"]
        expected_reason = POST_EMIT_FAILURE_REASONS.get(failure_mode)
        valid_outcome = outcome["status"] == "failed" and reason == expected_reason
    elif source == "matrix-capability-resolver":
        expected_runner = "capability-resolver"
        valid_outcome = (
            outcome["status"] == "unsupported"
            and failure_mode == "capability"
            and reason in CAPABILITY_FAILURE_REASONS
        )
    else:
        raise ContractError("terminal provenance source is not registered")
    if not valid_outcome:
        raise ContractError("terminal source and outcome are not registered")
    expected_allocation = {
        "artifact": git_run["artifact"] if git_run is not None else None,
        "execution_id": allocation["execution_id"],
        "job": git_run["job"] if git_run is not None else None,
        "qualification_index": qualification_index,
        "repo": git_run["repo"] if git_run is not None else None,
        "run_attempt": git_run["run_attempt"] if git_run is not None else None,
        "run_id": git_run["run_id"] if git_run is not None else None,
        "runner": expected_runner,
        "source_sha": git_run["source_sha"] if git_run is not None else None,
    }
    if allocation != expected_allocation:
        raise ContractError("terminal allocation factors differ from provenance or source")
    artifact_safety.assert_publication_safe([doc])
    return doc


def load_raw_attempt(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = strict_load(path)
    artifact = _obj(document, "raw").get("sample_artifact")
    artifact = _obj(artifact, "raw.sample_artifact")
    sample_path = Path(path).with_name(_text(artifact.get("path"), "raw.sample_artifact.path"))
    payload = sample_path.read_bytes()
    if len(payload) != artifact.get("bytes") or hashlib.sha256(payload).hexdigest() != artifact.get("sha256"):
        raise ContractError("sample artifact bytes or digest differ")
    samples = strict_load(sample_path)
    return validate_raw_document(document, samples)


def load_attempt(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Fully validate and return one native raw or terminal attempt."""
    document = strict_load(path)
    if isinstance(document, dict) and document.get("format") == RAW_FORMAT:
        return load_raw_attempt(path)
    if isinstance(document, dict) and document.get("format") == TERMINAL_FORMAT:
        return validate_terminal_document(document)
    raise ContractError("unknown native attempt format")


def quarantine_invalid_attempt(path: str | os.PathLike[str]) -> bool:
    """Move an invalid attempt and its basename-safe sample outside JSON upload globs."""
    destination = Path(path)
    if not destination.is_file():
        return False
    try:
        load_attempt(destination)
        return False
    except (ContractError, OSError, ValueError):
        try:
            document = json.loads(destination.read_bytes())
        except (OSError, json.JSONDecodeError):
            document = {}
        artifact = document.get("sample_artifact") if isinstance(document, dict) else None
        sample_name = artifact.get("path") if isinstance(artifact, dict) else None
        if isinstance(sample_name, str) and Path(sample_name).name == sample_name:
            sample_path = destination.with_name(sample_name)
            if sample_path.is_file():
                os.replace(sample_path, sample_path.with_name(sample_path.name + ".quarantine"))
        os.replace(destination, destination.with_name(destination.name + ".quarantine"))
        return True


def normalize_attempt(document: dict[str, Any]) -> dict[str, Any]:
    """Return the publisher-facing projection after native validation."""
    if document.get("format") == RAW_FORMAT:
        ids = document["identity"]
        return {
            "allocation_id": ids["allocation_id"],
            "attempt_id": ids["attempt_id"],
            "case": document["case"],
            "case_id": ids["case_id"],
            "generated_at": document["generated_at"],
            "outcome": document["outcome"],
            "points": document["measurement"]["rows"],
            "runtime_fingerprint": document["runtime_fingerprint"],
            "series_id": ids["series_id"],
        }
    if document.get("format") == TERMINAL_FORMAT:
        ids = document["identity"]
        return {
            "allocation_id": ids["allocation_id"],
            "attempt_id": ids["attempt_id"],
            "case": document["case"],
            "case_id": ids["case_id"],
            "generated_at": document["generated_at"],
            "outcome": document["outcome"],
            "points": [],
            "runtime_fingerprint": None,
            "series_id": None,
        }
    raise ContractError("unknown attempt format")


def _env_integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _terminal_case_from_environment(backend: str, phase: str) -> dict[str, Any]:
    ep = _env_integer("CX_EP", _env_integer("CX_NGPUS", 1))
    gpus_per_node = _env_integer("CX_GPUS_PER_NODE", ep)
    ladder = os.environ.get("CX_TOKENS_LADDER", "") or (
        "1 2 4 8 16 32 64 128"
        if phase == "decode"
        else "128 256 512 1024 2048 4096"
    )
    case = {
        "suite": os.environ.get("CX_SUITE") or "manual",
        "workload": os.environ.get("CX_WORKLOAD_NAME") or "manual",
        "required_publication": os.environ.get("CX_REQUIRED_PUBLICATION") or "diagnostic",
        "backend": backend,
        "mode": os.environ.get("CX_MODE", "normal"),
        "routing": os.environ.get("CX_ROUTING", "uniform"),
        "phase": phase,
        "ep": ep,
        "eplb": _env_enabled("CX_EPLB"),
        "hidden": _env_integer("CX_HIDDEN", 7168),
        "topk": _env_integer("CX_TOPK", 8),
        "experts": _env_integer("CX_EXPERTS", 256),
        "samples_per_point": _env_integer("CX_SAMPLES_PER_POINT", 512),
        "warmup_semantics": os.environ.get(
            "CX_WARMUP_SEMANTICS",
            "full-roundtrip-before-each-component-trial-point-v1",
        ),
        "ladder": ladder,
        "timing": (
            f'{_env_integer("CX_ITERS", 8)}:{_env_integer("CX_TRIALS", 64)}:'
            f'{_env_integer("CX_WARMUP", 32)}'
        ),
        "canonical": _env_enabled("CX_CANONICAL"),
        "nodes": _env_integer("CX_NODES", _env_integer("SLURM_NNODES", 1)),
        "gpus_per_node": gpus_per_node,
        "scale_up_domain": _env_integer("CX_SCALE_UP_DOMAIN", gpus_per_node),
        "scope": os.environ.get("CX_SCOPE", "scale-up"),
        "topology_class": os.environ.get("CX_TOPO", "manual"),
        "transport": os.environ.get("CX_TRANSPORT", "unknown"),
        "scale_up_transport": os.environ.get("CX_SCALE_UP_TRANSPORT", "unknown"),
        "scale_out_transport": os.environ.get("CX_SCALE_OUT_TRANSPORT") or None,
    }
    precision_profile = os.environ.get("CX_PRECISION_PROFILE") or None
    if precision_profile is not None:
        case["precision_profile"] = precision_profile
    return case


def _git_run_from_environment() -> dict[str, Any] | None:
    def value(name: str) -> str | None:
        return os.environ.get(name) or None

    git_run = {
        "run_id": value("GITHUB_RUN_ID"),
        "run_attempt": value("GITHUB_RUN_ATTEMPT"),
        "ref": value("GITHUB_REF_NAME") or value("GITHUB_REF"),
        "source_sha": value("COLLECTIVEX_SOURCE_SHA") or value("GITHUB_SHA"),
        "repo": value("GITHUB_REPOSITORY"),
        "job": value("GITHUB_JOB"),
        "artifact": value("COLLECTIVEX_ARTIFACT_NAME"),
    }
    if not any(item is not None for item in git_run.values()):
        return None
    git_run["qualification_index"] = _env_integer("CX_QUALIFICATION_INDEX", 1)
    return git_run


def _allocation_factors_from_environment(
    runner: str, git_run: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "artifact": git_run["artifact"] if git_run is not None else None,
        "execution_id": os.environ.get("COLLECTIVEX_EXECUTION_ID") or None,
        "job": git_run["job"] if git_run is not None else None,
        "qualification_index": _env_integer("CX_QUALIFICATION_INDEX", 1),
        "repo": git_run["repo"] if git_run is not None else None,
        "run_attempt": git_run["run_attempt"] if git_run is not None else None,
        "run_id": git_run["run_id"] if git_run is not None else None,
        "runner": runner,
        "source_sha": git_run["source_sha"] if git_run is not None else None,
    }


def make_terminal_from_environment(
    *, backend: str, phase: str, return_code: int, failure_mode: str | None = None
) -> dict[str, Any]:
    """Build a terminal document from the same exported case coordinates as run_ep."""
    mode = failure_mode or RETURN_CODE_FAILURE_MODES.get(return_code, "execution")
    reason = RUNTIME_FAILURE_REASONS.get(mode)
    if reason is None:
        raise ContractError("runtime failure mode is not registered")
    runner = os.environ.get("CX_RUNNER", "")
    case = _terminal_case_from_environment(backend, phase)
    case_factors = {
        "case": case,
        "profile": scheduled_case_profile(case, "runtime case"),
        "sku": runner,
    }
    git_run = _git_run_from_environment()
    control = os.environ.get("COLLECTIVEX_CONTROL_SHA256") or None
    return make_terminal_document(
        allocation_factors=_allocation_factors_from_environment(runner, git_run),
        attempt_ordinal=_env_integer("CX_ATTEMPT_ID", 1),
        case=case,
        case_factors=case_factors,
        control_sha256=control,
        failure_mode=mode,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        git_run=git_run,
        reason=reason,
        return_code=return_code,
        source="runtime-emitter",
        status="failed",
        expected_case_id=os.environ.get("CX_CASE_ID") or None,
    )


def _write_document(path: str | os.PathLike[str], document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def demote_raw_attempt(path: str | os.PathLike[str], return_code: int) -> dict[str, Any]:
    """Replace a rank-zero raw result when the distributed command later fails."""
    destination = Path(path)
    raw = strict_load(destination)
    if not isinstance(raw, dict) or raw.get("format") != RAW_FORMAT:
        raise ContractError("only a native raw attempt can be demoted")
    ids = _obj(raw.get("identity"), "raw.identity")
    required = {
        "allocation_factors", "allocation_id", "attempt_id", "attempt_ordinal",
        "case_factors", "case_id",
    }
    if not required.issubset(ids):
        raise ContractError("raw identity lacks terminal factors")
    mode = RETURN_CODE_FAILURE_MODES.get(return_code, "execution")
    git_run = _obj(raw.get("provenance"), "raw.provenance").get("git_run")
    if git_run is not None:
        git_run = _keys(git_run, GIT_RUN_FIELDS, "raw.provenance.git_run")
    terminal = make_terminal_document(
        allocation_factors=ids["allocation_factors"],
        attempt_ordinal=ids["attempt_ordinal"],
        case=ids["case_factors"]["case"],
        case_factors=ids["case_factors"],
        control_sha256=os.environ.get("COLLECTIVEX_CONTROL_SHA256") or None,
        failure_mode=mode,
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        git_run=git_run,
        reason=POST_EMIT_FAILURE_REASONS[mode],
        return_code=return_code,
        source="post-emit-command",
        status="failed",
        expected_case_id=ids["case_id"],
    )
    artifact = raw.get("sample_artifact") or {}
    sample_name = artifact.get("path")
    if isinstance(sample_name, str) and Path(sample_name).name == sample_name:
        destination.with_name(sample_name).unlink(missing_ok=True)
    _write_document(destination, terminal)
    return terminal


def validate_attempt_paths(paths: list[str]) -> int:
    """Fully validate a result directory's attempts and paired sample artifacts."""
    if not paths or len(paths) != len(set(paths)):
        raise ContractError("validate-many requires unique result paths")
    sample_paths: set[Path] = set()
    referenced_samples: set[Path] = set()
    attempt_count = 0
    for raw_path in paths:
        path = Path(raw_path).resolve()
        document = strict_load(path)
        if isinstance(document, dict) and document.get("format") == RAW_FORMAT:
            document = load_raw_attempt(path)
            referenced_samples.add(path.with_name(document["sample_artifact"]["path"]))
            attempt_count += 1
        elif isinstance(document, dict) and document.get("format") == TERMINAL_FORMAT:
            validate_terminal_document(document)
            attempt_count += 1
        elif isinstance(document, dict) and document.get("format") == SAMPLES_FORMAT:
            validate_samples_document(document)
            sample_paths.add(path)
        else:
            raise ContractError(f"unknown result artifact {path.name}")
    if sample_paths != referenced_samples:
        raise ContractError("sample artifacts are missing, orphaned, or outside the validated set")
    if attempt_count == 0:
        raise ContractError("result set contains no native attempts")
    return attempt_count


def validate_delivery(
    paths: list[str], source_path: str, *, disposition: str | None = None
) -> int:
    """Reconcile a shard or matrix disposition with its complete native attempt set."""
    source_file = Path(source_path).resolve()
    source = strict_load(source_file)
    if isinstance(source, dict) and source.get("format") == "collectivex.matrix.v1":
        if disposition is None:
            raise ContractError("matrix delivery validation requires a disposition")
        wrappers = [
            item for item in source.get("requested_cases", [])
            if isinstance(item, dict) and item.get("disposition") == disposition
        ]
        expected = {
            item["case"]["case_id"]: (item["sku"], item["case"])
            for item in wrappers
        }
        expected_count = len(wrappers)
        require_one_allocation = disposition == "unsupported"
    elif isinstance(source, dict) and isinstance(source.get("cases"), list):
        expected = {
            case["case_id"]: (source.get("sku"), case)
            for case in source["cases"]
        }
        expected_count = len(source["cases"])
        require_one_allocation = True
    else:
        raise ContractError("delivery source is not a matrix or shard control")
    if not expected or len(expected) != expected_count:
        raise ContractError("delivery source has empty or duplicate case coverage")

    validate_attempt_paths(paths)
    attempts = []
    for raw_path in paths:
        document = strict_load(raw_path)
        if isinstance(document, dict) and document.get("format") in {RAW_FORMAT, TERMINAL_FORMAT}:
            attempts.append(load_attempt(raw_path))
    by_case: dict[str, list[dict[str, Any]]] = {}
    attempt_ids = set()
    allocation_ids = set()
    source_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()
    for document in attempts:
        ids = document["identity"]
        case_id = ids["case_id"]
        if case_id not in expected or ids["attempt_id"] in attempt_ids:
            raise ContractError("delivery contains an extra case or duplicate attempt")
        attempt_ids.add(ids["attempt_id"])
        allocation_ids.add(ids["allocation_id"])
        sku, scheduled = expected[case_id]
        scheduled_case = {key: value for key, value in scheduled.items() if key != "case_id"}
        if ids["case_factors"] != {
            "case": scheduled_case,
            "profile": scheduled_case_profile(scheduled_case, "delivery case"),
            "sku": sku,
        }:
            raise ContractError("delivery attempt differs from its scheduled case")
        factors = ids["allocation_factors"]
        expected_environment = {
            "artifact": os.environ.get("COLLECTIVEX_ARTIFACT_NAME"),
            "execution_id": os.environ.get("COLLECTIVEX_EXECUTION_ID"),
            "job": os.environ.get("GITHUB_JOB"),
            "repo": os.environ.get("GITHUB_REPOSITORY"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA") or os.environ.get("GITHUB_SHA"),
        }
        expected_runner = (
            "capability-resolver"
            if document["format"] == TERMINAL_FORMAT
            and document["provenance"]["source"] == "matrix-capability-resolver"
            else sku
        )
        if any(
            value is not None and factors[field] != value
            for field, value in expected_environment.items()
        ) or factors["runner"] != expected_runner:
            raise ContractError("delivery allocation factors differ from the workflow")
        if document["format"] == TERMINAL_FORMAT:
            control = document["provenance"]["control_sha256"]
            if control != source_sha256:
                raise ContractError("terminal outcome does not reference its exact control document")
        by_case.setdefault(case_id, []).append(document)
    if set(by_case) != set(expected):
        raise ContractError("delivery case coverage is incomplete")
    for case_id, documents in by_case.items():
        ordinals = sorted(document["identity"]["attempt_ordinal"] for document in documents)
        if ordinals != list(range(1, len(ordinals) + 1)):
            raise ContractError(f"delivery attempt ordinals are not contiguous for {case_id}")
    if require_one_allocation and len(allocation_ids) != 1:
        raise ContractError("one shard must use exactly one allocation identity")
    return len(attempts)


def main() -> int:
    parser = argparse.ArgumentParser(description="CollectiveX native attempt contracts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("path")
    probe.add_argument("--status", choices=("success", "invalid"))
    emit = subparsers.add_parser("emit-terminal")
    emit.add_argument("--out", required=True)
    emit.add_argument("--backend", required=True)
    emit.add_argument("--phase", required=True, choices=("decode", "prefill"))
    emit.add_argument("--return-code", required=True, type=int)
    emit.add_argument("--failure-mode")
    demote = subparsers.add_parser("demote")
    demote.add_argument("path")
    demote.add_argument("--return-code", required=True, type=int)
    validate_many = subparsers.add_parser("validate-many")
    validate_many.add_argument("paths", nargs="+")
    quarantine = subparsers.add_parser("quarantine-invalid")
    quarantine.add_argument("path")
    delivery = subparsers.add_parser("validate-delivery")
    delivery.add_argument("--source", required=True)
    delivery.add_argument("--disposition")
    delivery.add_argument("paths", nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "probe":
            document = load_attempt(args.path)
            if args.status is None:
                return 0
            if document.get("format") != RAW_FORMAT:
                return 1
            outcome = document["outcome"]
            validity = outcome.get("validity")
            return int(
                not (
                    isinstance(validity, dict)
                    and validity.get("execution_status") == "complete"
                    and outcome.get("status") == args.status
                )
            )
        if args.command == "emit-terminal":
            document = make_terminal_from_environment(
                backend=args.backend,
                phase=args.phase,
                return_code=args.return_code,
                failure_mode=args.failure_mode,
            )
            _write_document(args.out, document)
            print(f"preserved terminal outcome ({document['outcome']['failure_mode']})")
            return 0
        if args.command == "validate-many":
            print(f"validated {validate_attempt_paths(args.paths)} native attempts")
            return 0
        if args.command == "quarantine-invalid":
            quarantine_invalid_attempt(args.path)
            return 0
        if args.command == "validate-delivery":
            print(
                f"validated {validate_delivery(args.paths, args.source, disposition=args.disposition)} "
                "delivery attempts"
            )
            return 0
        demote_raw_attempt(args.path, args.return_code)
        return 0
    except (ContractError, identity.IdentityError, OSError, ValueError) as exc:
        print(f"terminal contract error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
