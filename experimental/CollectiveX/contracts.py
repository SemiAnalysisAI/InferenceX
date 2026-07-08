#!/usr/bin/env python3
"""Result contracts for CollectiveX.

JSON Schema (``schemas/*.schema.json``, Draft 2020-12) owns document shape and
field constraints. The Python here covers only the facts JSON Schema cannot
express: typed-identity recomputation, detached-sample path/hash joins, one
terminal result per scheduled case, complete shard delivery, and the artifact
privacy boundary. The public surface is intentionally small:

    strict_json_load()   - duplicate/non-finite-rejecting JSON reader
    validate_result()     - validate one emitted result file (raw+samples or terminal)
    make_terminal_result() - build and self-validate one non-success outcome
    validate_delivery()    - reconcile a shard/matrix with its complete attempt set

The backend-provenance and resource-projection helpers stay here so the bench
emitter can call them; they move to the bench layer with the executable modules.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Iterable

from jsonschema import Draft202012Validator

import identity

RAW_FORMAT = "collectivex.ep.v1"
SAMPLES_FORMAT = "collectivex.samples.v1"
TERMINAL_FORMAT = "collectivex.terminal.v1"
TERMINAL_CASE_FIELDS = {
    "backend", "canonical", "eplb", "ep", "experts", "gpus_per_node", "hidden",
    "ladder", "mode", "nodes", "phase", "routing",
    "samples_per_point", "scale_out_transport", "scale_up_domain", "scale_up_transport",
    "scope", "suite", "timing", "topk", "topology_class", "transport",
    "warmup_semantics", "workload",
}
ALLOCATION_FACTOR_FIELDS = {
    "artifact", "execution_id", "job", "repo", "run_attempt", "run_id", "runner",
    "source_sha",
}
GIT_RUN_FIELDS = {
    "artifact", "job", "ref", "repo", "run_attempt", "run_id", "source_sha",
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
})
RETURN_CODE_FAILURE_MODES = {
    5: "runtime-identity",
    124: "timeout",
    137: "timeout",
}
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
REQUIRED_BACKEND_PROVENANCE = {
    "deepep": (
        "deepep_version", "deepep_commit", "backend_lineage", "allow_mnnvl",
        "mnnvl_comm", "mode", "num_nvl_bytes", "num_rdma_bytes",
        "nvshmem_ibgda_nic_handler",
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


class ContractError(ValueError):
    """A document differs from the CollectiveX result contract."""


# ---------------------------------------------------------------------------
# Artifact privacy boundary (kept narrow; enforced in delivery + terminal).
# ---------------------------------------------------------------------------
SENSITIVE_FIELDS = frozenset({
    "environment", "env", "host", "hostname", "uuid", "gpu_uuid", "device_uuid",
    "pci_bus_id", "ip_address", "ip_addresses", "master_addr", "ssh", "ssh_target",
    "nodelist", "node_list", "nic_guid", "ib_guid", "topology_matrix", "rdma_devices",
    "user", "username", "password", "passwd", "secret", "token", "access_token",
    "api_token", "auth_token", "api_key", "private_key", "credential", "credentials",
    "address", "addresses", "ip", "ips",
})
SENSITIVE_FIELDS_COMPACT = frozenset(item.replace("_", "") for item in SENSITIVE_FIELDS)
SENSITIVE_FIELD_SUFFIXES = (
    "_host", "_hostname", "_address", "_addresses", "_path", "_paths", "_ip", "_ips",
    "_password", "_passwd", "_secret", "_token", "_credential", "_credentials",
    "_uuid", "_guid", "_bus_id",
)
SENSITIVE_VALUE_PATTERNS = (
    ("private-path", re.compile(
        r"(?<![A-Za-z0-9_.-])/(?:home|mnt|workspace|root|users|tmp|data|it-share|lustre|raid|nvme_home|scratch|gpfs|fsx)(?:/|$)",
        re.I,
    )),
    ("ipv4-address", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    ("pci-address", re.compile(r"\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\b", re.I)),
    ("hardware-address", re.compile(
        r"\b(?:[0-9a-f]{2}[:-]){5}(?:[0-9a-f]{2})\b|"
        r"\b(?:[0-9a-f]{2}:){7}(?:[0-9a-f]{2})\b|\b0x[0-9a-f]{16}\b",
        re.I,
    )),
    ("uuid", re.compile(
        r"\b(?:GPU-|MIG-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.I,
    )),
    ("ssh-target", re.compile(r"(?:ssh://|\bssh\s+[^\s/@]+@[^\s/]+)", re.I)),
    ("host-identifier", re.compile(
        r"\b(?:host(?:name)?|master[_-]?(?:addr|address)|node[_-]?list)\s*(?:=|:)\s*[^\s,;]+",
        re.I,
    )),
    ("private-hostname", re.compile(
        r"\b(?:[a-z0-9-]+\.)+(?:cluster|corp|internal|lan|local)\b|"
        r"\b(?:compute|gpu|head|login|node|worker)[-_]?[0-9][a-z0-9_.-]*\b|"
        r"\bdgx-[a-z0-9-]+-[0-9]+\b|\bip-(?:[0-9]{1,3}-){3}[0-9]{1,3}\b",
        re.I,
    )),
    ("secret-token", re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"glpat-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
        r"(?:AKIA|ASIA)[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|"
        r"(?:sk-(?:proj|svcacct)-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|"
        r"sk_(?:live|test)_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})|"
        r"npm_[A-Za-z0-9]{20,}|"
        r"pypi-[A-Za-z0-9_-]{20,}|dckr_pat_[A-Za-z0-9_-]{20,}|"
        r"Bearer\s+[A-Za-z0-9._~+/-]{16,}|Basic\s+[A-Za-z0-9+/=]{16,}|"
        r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|"
        r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----)",
        re.I,
    )),
    ("secret-assignment", re.compile(
        r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"password|passwd|secret|accountkey)\s*(?:=|:)\s*[\"']?"
        r"[A-Za-z0-9+/_=.~-]{8,}",
        re.I,
    )),
)
IPV6_CANDIDATE = re.compile(
    r"(?<![0-9A-Za-z])\[?([0-9A-Fa-f:]{2,}(?:%[0-9A-Za-z_.-]+)?)\]?"
)
CONTEXTUAL_VALUE_RULES = frozenset({"ssh-target", "host-identifier", "private-hostname"})


class ArtifactSafetyError(ContractError):
    """A document contains data that cannot cross the public boundary."""


def _normalized_field(value: object) -> str:
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(value).strip())
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return normalized.lower().replace("-", "_")


def _sensitive_value_rule(value: str, *, contextual: bool = True) -> str | None:
    matched = next(
        (
            name for name, pattern in SENSITIVE_VALUE_PATTERNS
            if (contextual or name not in CONTEXTUAL_VALUE_RULES) and pattern.search(value)
        ),
        None,
    )
    if matched:
        return matched
    for candidate in IPV6_CANDIDATE.findall(value):
        try:
            address = candidate.split("%", 1)[0]
            if ipaddress.ip_address(address).version == 6:
                return "ipv6-address"
        except ValueError:
            continue
    return None


def assert_publication_safe(docs: list[dict]) -> None:
    """Reject private infrastructure fields and value shapes at the artifact boundary."""
    def walk(value, doc_index: int, parent_field: str | None = None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                field = _normalized_field(key)
                compact = field.replace("_", "")
                if (
                    field in SENSITIVE_FIELDS
                    or compact in SENSITIVE_FIELDS_COMPACT
                    or field.endswith(SENSITIVE_FIELD_SUFFIXES)
                ):
                    raise ArtifactSafetyError(
                        f"artifact safety: doc[{doc_index}] contains forbidden private field"
                    )
                key_rule = _sensitive_value_rule(str(key))
                if key_rule:
                    raise ArtifactSafetyError(
                        f"artifact safety: doc[{doc_index}] contains forbidden {key_rule} key"
                    )
                walk(child, doc_index, field)
        elif isinstance(value, list):
            for child in value:
                walk(child, doc_index, parent_field)
        elif isinstance(value, str):
            rule = _sensitive_value_rule(value, contextual=parent_field != "ref")
            if rule:
                raise ArtifactSafetyError(
                    f"artifact safety: doc[{doc_index}] contains forbidden {rule} value"
                )

    for index, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise ArtifactSafetyError(f"artifact safety: doc[{index}] is not a JSON object")
        walk(doc, index)


# ---------------------------------------------------------------------------
# Backend emission helpers (called by the bench emitter, not by validation).
# ---------------------------------------------------------------------------
def scheduled_case_profile(case: dict[str, Any], path: str = "case") -> dict[str, Any]:
    """Resolve an explicit scheduled mode to its immutable measurement profile."""
    try:
        return identity.profile_for_case(case)
    except identity.IdentityError as exc:
        raise ContractError(f"{path}: {exc}") from exc


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


# ---------------------------------------------------------------------------
# JSON IO helpers.
# ---------------------------------------------------------------------------
def strict_json_load(path: str | os.PathLike[str]) -> Any:
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


# Compatibility alias for existing runtime/summary callers.
strict_load = strict_json_load


def _finite_tree(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")


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


# ---------------------------------------------------------------------------
# Schema validation core (JSON Schema owns document shape).
# ---------------------------------------------------------------------------
_VALIDATOR_CACHE: dict[str, Draft202012Validator] = {}


def _validator(schema_file: str) -> Draft202012Validator:
    validator = _VALIDATOR_CACHE.get(schema_file)
    if validator is None:
        try:
            schema = json.loads((SCHEMA_DIR / schema_file).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"schema {schema_file} is unavailable: {exc}") from exc
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        _VALIDATOR_CACHE[schema_file] = validator
    return validator


def _schema_validate(document: Any, schema_file: str, label: str) -> None:
    errors = sorted(
        _validator(schema_file).iter_errors(document),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ContractError(f"{label} schema error at {location}: {error.message}")


def validate_samples_document(document: Any) -> dict[str, Any]:
    """Validate one detached exact-samples document against its schema."""
    if not isinstance(document, dict):
        raise ContractError("samples document must be an object")
    _schema_validate(document, "samples-v1.schema.json", "samples")
    return document


def validate_raw_document(document: Any, samples: Any) -> dict[str, Any]:
    """Validate a raw measurement document and its paired detached samples.

    JSON Schema owns both document shapes. The cross-file checks here recompute
    the typed identities from their factors and join the detached samples to the
    measurement rows by point identity and sample digest.
    """
    if not isinstance(document, dict):
        raise ContractError("raw document must be an object")
    _schema_validate(document, "raw-case-v1.schema.json", "raw")
    _schema_validate(samples, "samples-v1.schema.json", "samples")
    ids = document["identity"]
    case_id = identity.digest("case", ids["case_factors"])
    allocation_id = identity.allocation_id(ids["allocation_factors"])
    attempt_id = identity.attempt_id(
        allocation=allocation_id, case=case_id, ordinal=ids["attempt_ordinal"]
    )
    if (case_id, allocation_id, attempt_id) != (
        ids["case_id"], ids["allocation_id"], ids["attempt_id"]
    ):
        raise ContractError("raw typed identities do not match their factors")
    series_id = ids["series_id"]
    for field in ("case_id", "series_id", "allocation_id", "attempt_id"):
        if samples.get(field) != ids[field]:
            raise ContractError(f"detached samples {field} differs from raw identity")
    rows = document["measurement"]["rows"]
    sample_points = {point["point_id"]: point for point in samples["points"]}
    if {row["point_id"] for row in rows} != set(sample_points):
        raise ContractError("raw rows and detached samples cover different points")
    success = document["outcome"]["status"] == "success"
    for row in rows:
        expected_point = identity.point_id(
            series=series_id, tokens_per_rank=row["tokens_per_rank"]
        )
        expected_evidence = identity.evidence_id(
            point=expected_point, allocation=allocation_id, attempt=attempt_id,
            sample_sha256=row["sample_sha256"],
        )
        if row["point_id"] != expected_point or row["evidence_id"] != expected_evidence:
            raise ContractError("raw point/evidence identity does not match its factors")
        point = sample_points[row["point_id"]]
        if (point["sample_sha256"], point["evidence_id"], point["tokens_per_rank"]) != (
            row["sample_sha256"], row["evidence_id"], row["tokens_per_rank"]
        ):
            raise ContractError("detached sample point differs from its raw row")
        if success and row["correctness"]["passed"] is not True:
            raise ContractError("successful raw case has a failed correctness check")
    return document


def load_raw_attempt(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a raw attempt, verify its detached sample bytes, and validate both."""
    document = strict_json_load(path)
    artifact = _obj(_obj(document, "raw").get("sample_artifact"), "raw.sample_artifact")
    sample_path = Path(path).with_name(_text(artifact.get("path"), "raw.sample_artifact.path"))
    payload = sample_path.read_bytes()
    if (
        len(payload) != artifact.get("bytes")
        or hashlib.sha256(payload).hexdigest() != artifact.get("sha256")
    ):
        raise ContractError("sample artifact bytes or digest differ")
    samples = strict_json_load(sample_path)
    return validate_raw_document(document, samples)


def load_attempt(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Fully validate and return one raw or terminal attempt."""
    document = strict_json_load(path)
    if isinstance(document, dict) and document.get("format") == RAW_FORMAT:
        return load_raw_attempt(path)
    if isinstance(document, dict) and document.get("format") == TERMINAL_FORMAT:
        return validate_terminal_document(document)
    raise ContractError("unknown result attempt format")


def validate_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate one emitted result file (raw+samples or terminal) and return it."""
    return load_attempt(path)


# ---------------------------------------------------------------------------
# Terminal outcome construction and validation.
# ---------------------------------------------------------------------------
def make_terminal_result(
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


# Compatibility alias for existing matrix/runtime callers.
make_terminal_document = make_terminal_result


def validate_terminal_document(document: Any) -> dict[str, Any]:
    """Validate a terminal outcome against its schema and cross-field semantics."""
    if not isinstance(document, dict):
        raise ContractError("terminal document must be an object")
    _schema_validate(document, "terminal-outcome-v1.schema.json", "terminal")
    ids = document["identity"]
    case = document["case"]
    factors = ids["case_factors"]
    if factors["case"] != case or factors["profile"] != scheduled_case_profile(
        case, "terminal.case"
    ):
        raise ContractError("terminal case factors differ from the scheduled case/profile")
    allocation = ids["allocation_factors"]
    ordinal = ids["attempt_ordinal"]
    expected_case = identity.digest("case", factors)
    expected_allocation = identity.allocation_id(allocation)
    expected_attempt = identity.attempt_id(
        allocation=expected_allocation, case=expected_case, ordinal=ordinal
    )
    if (ids["case_id"], ids["allocation_id"], ids["attempt_id"]) != (
        expected_case, expected_allocation, expected_attempt
    ):
        raise ContractError("terminal typed identities do not match their factors")
    provenance = document["provenance"]
    git_run = provenance["git_run"]
    outcome = document["outcome"]
    failure_mode = outcome["failure_mode"]
    source = provenance["source"]
    if source == "runtime-emitter":
        expected_runner = factors["sku"]
        valid_outcome = (
            outcome["status"] == "failed"
            and outcome["reason"] == RUNTIME_FAILURE_REASONS.get(failure_mode)
        )
    elif source == "post-emit-command":
        expected_runner = factors["sku"]
        valid_outcome = (
            outcome["status"] == "failed"
            and outcome["reason"] == POST_EMIT_FAILURE_REASONS.get(failure_mode)
        )
    elif source == "matrix-capability-resolver":
        expected_runner = "capability-resolver"
        valid_outcome = (
            outcome["status"] == "unsupported"
            and failure_mode == "capability"
            and outcome["reason"] in CAPABILITY_FAILURE_REASONS
        )
    else:
        raise ContractError("terminal provenance source is not registered")
    if not valid_outcome:
        raise ContractError("terminal source and outcome are not registered")
    expected_allocation_factors = {
        "artifact": git_run["artifact"] if git_run is not None else None,
        "execution_id": allocation["execution_id"],
        "job": git_run["job"] if git_run is not None else None,
        "repo": git_run["repo"] if git_run is not None else None,
        "run_attempt": git_run["run_attempt"] if git_run is not None else None,
        "run_id": git_run["run_id"] if git_run is not None else None,
        "runner": expected_runner,
        "source_sha": git_run["source_sha"] if git_run is not None else None,
    }
    if allocation != expected_allocation_factors:
        raise ContractError("terminal allocation factors differ from provenance or source")
    assert_publication_safe([document])
    return document


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


# ---------------------------------------------------------------------------
# Terminal construction from the runtime environment.
# ---------------------------------------------------------------------------
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
    return {
        "suite": os.environ.get("CX_SUITE") or "manual",
        "workload": os.environ.get("CX_WORKLOAD_NAME") or "manual",
        "backend": backend,
        "mode": os.environ.get("CX_MODE", "normal"),
        "routing": os.environ.get("CX_ROUTING", "uniform"),
        "phase": phase,
        "ep": ep,
        "eplb": False,
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
    return git_run


def _allocation_factors_from_environment(
    runner: str, git_run: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "artifact": git_run["artifact"] if git_run is not None else None,
        "execution_id": os.environ.get("COLLECTIVEX_EXECUTION_ID") or None,
        "job": git_run["job"] if git_run is not None else None,
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
    return make_terminal_result(
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
    raw = strict_json_load(destination)
    if not isinstance(raw, dict) or raw.get("format") != RAW_FORMAT:
        raise ContractError("only a raw attempt can be demoted")
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
    terminal = make_terminal_result(
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


# ---------------------------------------------------------------------------
# Shard / matrix delivery reconciliation.
# ---------------------------------------------------------------------------
def _validate_attempt_paths(paths: list[str]) -> int:
    """Fully validate a result directory's attempts and paired sample artifacts."""
    if not paths or len(paths) != len(set(paths)):
        raise ContractError("delivery requires unique result paths")
    sample_paths: set[Path] = set()
    referenced_samples: set[Path] = set()
    attempt_count = 0
    for raw_path in paths:
        path = Path(raw_path).resolve()
        document = strict_json_load(path)
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
        raise ContractError("result set contains no attempts")
    return attempt_count


def validate_delivery(
    paths: list[str], source_path: str, *, disposition: str | None = None
) -> int:
    """Reconcile a shard or matrix disposition with its complete attempt set."""
    source_file = Path(source_path).resolve()
    source = strict_json_load(source_file)
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

    _validate_attempt_paths(paths)
    attempts = []
    for raw_path in paths:
        document = strict_json_load(raw_path)
        if isinstance(document, dict) and document.get("format") in {RAW_FORMAT, TERMINAL_FORMAT}:
            attempts.append(load_attempt(raw_path))
    assert_publication_safe(attempts)
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
    parser = argparse.ArgumentParser(description="CollectiveX result contracts")
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
