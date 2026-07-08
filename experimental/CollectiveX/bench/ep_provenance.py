#!/usr/bin/env python3
"""Backend implementation-provenance evidence and resource projection.

The EP benchmark emitter builds and self-checks the implementation-provenance
evidence it embeds in each raw attempt document: which backend libraries were
loaded, that a deepep-v2 attempt carries internally consistent JIT cubins, that
a hybrid attempt realized its kernels, and so on. Those are cross-field facts a
JSON Schema cannot express, so they live in Python — but they are an emitter
concern, not part of the neutral cross-file validation boundary. They are kept
here, beside the executable bench modules that call them, so ``contracts.py``
holds only the neutral document/delivery validation.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from contracts import (
    ContractError,
    canonical_json_bytes,
    DEEPEP_V2_DISTRIBUTION_VERSIONS,
    DEEPEP_V2_JIT_KERNELS,
    DEEPEP_V2_V1_PROVENANCE,
    GIT_RUN_FIELDS,
    REQUIRED_BACKEND_PROVENANCE,
    UCCL_DEPENDENCY_VERSIONS,
)


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
        if provenance.get("nvshmem_ibgda_nic_handler") not in {
            "cpu", "gpu", "not-active",
        }:
            unresolved.append("nvshmem_ibgda_nic_handler")
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
    *, image_digest: Any, image_verified: Any, squash_sha256: Any,
) -> bool:
    """Return whether backend provenance and run identity are fully resolved."""
    image = str(image_digest or "")
    squash = str(squash_sha256 or "")
    return (
        not backend_provenance_issues(backend, provenance)
        and image_verified is True
        and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", image))
        and bool(re.fullmatch(r"[0-9a-f]{64}", squash))
        and isinstance(git_run, dict)
        and all(git_run.get(field) for field in GIT_RUN_FIELDS)
    )


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
