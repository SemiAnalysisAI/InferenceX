#!/usr/bin/env python3
"""Backend implementation-provenance evidence and resource projection.

The EP benchmark emitter builds and self-checks the implementation-provenance
evidence it embeds in each raw attempt document: which backend libraries were
loaded, that a deepep-v2 attempt carries internally consistent JIT cubins, that
a hybrid attempt realized its kernels, and so on. Those are cross-field facts a
JSON Schema cannot express, so they live in Python here, beside the executable
bench modules that build and self-check them.
"""
from __future__ import annotations

import inspect
import json
import math
import re
from typing import Any

class ContractError(ValueError):
    """A provenance payload differs from the CollectiveX emitter contract."""


# Git run-identity fields an emitted attempt carries when produced under CI.
GIT_RUN_FIELDS = {
    "artifact", "job", "ref", "repo", "run_attempt", "run_id", "source_sha",
}


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


# Backend implementation-provenance constants: the required evidence fields per
# backend and the pinned native versions a self-check compares against. They
# describe executable code, so they live beside the emitter that consumes them
# rather than in the neutral document/delivery validator.
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
REQUIRED_BACKEND_PROVENANCE = {
    "deepep": (
        "deepep_version", "deepep_commit", "backend_lineage", "allow_mnnvl",
        "mnnvl_comm", "mode", "num_nvl_bytes", "num_rdma_bytes",
        "nvshmem_ibgda_nic_handler",
    ),
    "deepep-v2": (
        *DEEPEP_V2_V1_PROVENANCE, "jit_kernels", "jit_random_seed",
        "deterministic", "num_experts",
        "tuning_num_experts", "allow_hybrid_mode", "gin_enabled",
        "communication_backend",
    ),
    "deepep-hybrid": (
        "deepep_commit", "deepep_tree", "branch", "backend_lineage",
        "realized_config", "jit_kernel_keys",
    ),
    "mori": ("mori_commit",),
}


def require_keyword(callable_object, keyword: str, *, api: str) -> None:
    """Fail closed when a pinned native API does not expose a required control.

    A runtime-API contract guard: it verifies the loaded kernel build still accepts
    the keyword the adapter drives (e.g. the BF16 ``use_fp8=False`` control), so a
    version mismatch surfaces here instead of as an opaque call-site TypeError.
    """
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError) as exc:
        raise ContractError(f"cannot inspect required native API {api}") from exc
    if keyword not in parameters:
        raise ContractError(f"required native API {api} omits {keyword!r}")


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
        text in {"main", "hybrid-ep"}
        or text.endswith(("-unknown", "-none", "-main", "-hybrid-ep"))
    ):
        return False
    return True


def _deepep_v2_jit_kernels_are_valid(value: Any) -> bool:
    return isinstance(value, list) and set(value) == DEEPEP_V2_JIT_KERNELS


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
    if backend == "deepep":
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
            if (
                type(provenance.get("num_qps_per_rank")) is not int
                or provenance["num_qps_per_rank"] <= 0
            ):
                unresolved.append("num_qps_per_rank")
    if backend == "deepep-v2":
        for field in ("num_experts", "tuning_num_experts"):
            if type(provenance.get(field)) is not int or provenance[field] <= 0:
                unresolved.append(field)
        if not _deepep_v2_jit_kernels_are_valid(provenance.get("jit_kernels")):
            unresolved.append("jit_kernels")
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
    *, image_reference: Any, squash_sha256: Any,
) -> bool:
    """Return whether backend provenance and run identity are fully resolved."""
    image = str(image_reference or "")
    squash = str(squash_sha256 or "")
    return (
        not backend_provenance_issues(backend, provenance)
        and bool(re.fullmatch(r"[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+", image))
        and bool(re.fullmatch(r"[0-9a-f]{64}", squash))
        and isinstance(git_run, dict)
        and all(git_run.get(field) for field in GIT_RUN_FIELDS)
    )
