#!/usr/bin/env python3
"""Focused end-to-end tests for the isolated CollectiveX publisher."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import contracts  # noqa: E402
import identity  # noqa: E402
import publisher  # noqa: E402
import summarize  # noqa: E402
import sweep_matrix  # noqa: E402


RUN = {
    "repository": "SemiAnalysisAI/InferenceX",
    "run_id": "12345",
    "run_attempt": 1,
    "source_sha": "a" * 40,
}


def _unsupported_delivery(
    root: Path, ordinals: tuple[int, ...] = (1,), run: dict = RUN,
) -> tuple[Path, Path]:
    matrix = sweep_matrix.resolve_matrix(backends="all")
    wrapper = next(item for item in matrix["requested_cases"] if item["disposition"] == "unsupported")
    matrix = {
        "format": "collectivex.matrix.v1",
        "schema_version": 1,
        "requested_cases": [wrapper],
        "include": [],
    }
    case = {key: value for key, value in wrapper["case"].items() if key != "case_id"}
    artifact_name = f"cxunsupported-{run['run_id']}-{run['run_attempt']}"
    git_run = {
        "artifact": artifact_name,
        "job": "setup",
        "ref": "collectivex",
        "repo": run["repository"],
        "run_attempt": str(run["run_attempt"]),
        "run_id": run["run_id"],
        "source_sha": run["source_sha"],
    }
    allocation = {
        "artifact": artifact_name,
        "execution_id": f"{run['run_id']}_{run['run_attempt']}_unsupported",
        "job": "setup",
        "repo": run["repository"],
        "run_attempt": str(run["run_attempt"]),
        "run_id": run["run_id"],
        "runner": "capability-resolver",
        "source_sha": run["source_sha"],
    }
    matrix_path = root / "matrix.json"
    artifact = root / artifact_name
    artifact.mkdir()
    matrix_path.write_text(json.dumps(matrix))
    control_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    for ordinal in ordinals:
        terminal = contracts.make_terminal_document(
            allocation_factors=allocation, attempt_ordinal=ordinal, case=case,
            case_factors={"case": case, "profile": identity.V1_CASE_PROFILE,
                          "sku": wrapper["sku"]},
            control_sha256=control_sha256, failure_mode="capability",
            generated_at="2026-07-04T00:00:00Z", git_run=git_run,
            reason=wrapper["reason"], return_code=5, source="matrix-capability-resolver",
            status="unsupported", expected_case_id=wrapper["case"]["case_id"],
        )
        (artifact / f"unsupported-{ordinal}.json").write_text(json.dumps(terminal))
    return matrix_path, artifact


def _args(
    store: Path, matrix: Path, artifact: Path, run: dict = RUN
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        store_root=str(store),
        matrix=str(matrix),
        artifact=[str(artifact)],
        repository=run["repository"],
        run_id=run["run_id"],
        run_attempt=run["run_attempt"],
        source_sha=run["source_sha"],
    )


def _ids(seed: str) -> tuple[str, str, str, str, str, str]:
    case = identity.digest("case", {"seed": seed})
    allocation = identity.allocation_id({"seed": seed})
    attempt = identity.attempt_id(allocation=allocation, case=case, ordinal=1)
    series = identity.series_id({"seed": seed})
    point = identity.point_id(series=series, tokens_per_rank=8)
    evidence = identity.evidence_id(
        point=point, allocation=allocation, attempt=attempt, sample_sha256="b" * 64
    )
    return case, allocation, attempt, series, point, evidence


def _component(scale: float = 1.0) -> dict:
    latency = {"p50": 10.0 * scale, "p90": 12.0 * scale,
               "p95": 14.0 * scale, "p99": 20.0 * scale}
    logical_bytes = 100_000
    return {
        "origin": "measured",
        "latency_us": latency,
        "logical_bytes": logical_bytes,
        "logical_payload_rate_gbps_at_latency_percentile": {
            name: logical_bytes / (value * 1000.0) for name, value in latency.items()
        },
        "sample_count": 512,
    }


def _hybrid_provenance(ep_size: int = 1) -> dict:
    realized = {field: 1 for field in contracts.HYBRID_REALIZED_CONFIG_FIELDS}
    for field in contracts.HYBRID_REALIZED_BOOL_FIELDS:
        realized[field] = True
    realized.update({
        "num_of_experts_per_rank": 1,
        "num_of_nodes": 1,
        "num_of_ranks_per_node": ep_size,
        "token_data_type": "UINT16",
    })
    kernel_keys = ["combine-key", "dispatch-key", "preprocess-key"]
    return {
        "backend_lineage": "deepep-hybrid", "branch": "hybrid-ep",
        "deepep_commit": "a" * 40, "deepep_tree": "b" * 40,
        "device_sms": 1,
        "jit_kernel_keys": kernel_keys,
        "jit_shared_objects": [
            {
                "kernel_key": key,
                "rank_artifacts": [
                    {"bytes": 1, "rank": rank, "sha256": f"{index + 1:x}" * 64}
                    for rank in range(ep_size)
                ],
            }
            for index, key in enumerate(kernel_keys)
        ],
        "loaded_libraries": [
            {"name": "deep_ep_cpp", "role": "deepep-extension", "sha256": "4" * 64},
            {"name": "hybrid_ep_cpp", "role": "deepep-hybrid-extension", "sha256": "5" * 64},
        ],
        "realized_config": realized,
        "resource_mode": "tuned",
        "tuned_source": "deepep-hybrid-configurer-autotune-v1",
    }


def _native_fixture(backend: str = "nccl-ep") -> tuple[dict, dict]:
    def digest(value: object) -> str:
        return hashlib.sha256(contracts.canonical_json_bytes(value)).hexdigest()

    scheduled = {
        "backend": backend, "canonical": True, "eplb": False, "ep": 1,
        "experts": 1, "gpus_per_node": 1, "hidden": 1, "ladder": "1", "nodes": 1,
        "mode": "normal", "phase": "decode", "required_publication": "official",
        "routing": "uniform", "samples_per_point": 512,
        "scale_out_transport": None, "scale_up_domain": 1,
        "scale_up_transport": "nvlink", "scope": "scale-up", "suite": "ep-core-v1",
        "timing": "8:64:32", "topk": 1,
        "topology_class": "fixture", "transport": "nvlink",
        "warmup_semantics": "full-roundtrip-before-each-component-trial-point-v1",
        "workload": "deepseek-v3-v1",
    }
    case_factors = {
        "case": scheduled, "profile": identity.V1_NORMAL_CASE_PROFILE, "sku": "fixture"
    }
    case_id = identity.digest("case", case_factors)
    git_run = {
        "artifact": "cxshard-fixture-999-1", "job": "sweep", "ref": "collectivex",
        "repo": RUN["repository"], "run_attempt": "1", "run_id": "999",
        "source_sha": RUN["source_sha"],
    }
    allocation_factors = {
        "artifact": git_run["artifact"], "execution_id": "999_1_fixture",
        "job": git_run["job"], "repo": git_run["repo"], "run_attempt": "1",
        "run_id": "999", "runner": "fixture", "source_sha": git_run["source_sha"],
    }
    allocation_id = identity.allocation_id(allocation_factors)
    attempt_id = identity.attempt_id(allocation=allocation_id, case=case_id, ordinal=1)
    member_id, member_checksums, routing_hash, routing_rows, routing_weights = (
        contracts._expected_canonical_trace(
        "uniform", hidden=1, topk=1, logical_experts=1, physical_experts=1,
        ep_size=1, tokens_per_rank=1, seed=67, eplb_enabled=False,
        reference_tokens_per_rank=2048,
        )
    )
    workload_id = identity.workload_id({
        "members": [{"checksums": member_checksums, "workload_id": member_id}]
    })
    runtime = {
        "accelerator_runtime": {"kind": "cuda", "version": "13.0"},
        "collective_library": {"kind": "nccl", "version": "2.30.4"},
        "device": {
            "arch": "sm100", "compute_units": 1, "memory_bytes": 1,
            "product": "Fixture GPU", "warp_size": 32,
        },
        "driver_version": "1", "framework": {"kind": "torch", "version": "2.10.0"},
        "machine": "fixture", "python_version": "3.12", "vendor": "nvidia",
    }
    implementation_provenance = (
        {
            "backend": "nccl-ep", "backend_lineage": "nccl",
            "collective_library": "nccl", "nccl_version": "2.30.4",
            "reference_semantics": "fixture-v1",
        }
        if backend == "nccl-ep"
        else _hybrid_provenance()
    )
    kernel_generation = "nccl" if backend == "nccl-ep" else "hybrid"
    implementation = {
        "kernel_generation": kernel_generation,
        "name": backend,
        "provenance": implementation_provenance,
        "resource_profile": contracts.project_resource_profile(implementation_provenance),
    }
    public_config = contracts.public_series_config(
        kernel_generation=implementation["kernel_generation"],
        provenance=implementation_provenance,
        resource_profile=implementation["resource_profile"],
        resource_mode="tuned",
        device_product=runtime["device"]["product"],
    )
    series_factors = {
        "backend": backend, "case_id": case_id,
        "image_digest": "sha256:" + "d" * 64,
        "implementation_contract_sha256": digest({
            **implementation,
            "provenance": contracts.series_provenance(implementation_provenance),
        }),
        "public_config_sha256": contracts.public_series_config_sha256(public_config),
        "routing_control_sha256": contracts.routing_implementation_control_sha256(
            implementation
        ),
        "runtime_fingerprint_sha256": digest(runtime),
        "source_sha": RUN["source_sha"], "squash_sha256": "e" * 64,
        "workload_id": workload_id,
    }
    series_id = identity.series_id(series_factors)
    point_id = identity.point_id(series=series_id, tokens_per_rank=1)
    sample_components = {
        name: {
            "availability": "measured", "sample_count": 512,
            "trials": [[latency] * 8 for _ in range(64)],
        }
        for name, latency in (("combine", 20.0), ("dispatch", 10.0), ("roundtrip", 40.0))
    }
    sample_sha = digest({"components": sample_components, "tokens_per_rank": 1})
    evidence_id = identity.evidence_id(
        point=point_id, allocation=allocation_id, attempt=attempt_id,
        sample_sha256=sample_sha,
    )
    samples = {
        "allocation_id": allocation_id, "attempt_id": attempt_id, "case_id": case_id,
        "format": contracts.SAMPLES_FORMAT,
        "points": [{
            "components": sample_components, "evidence_id": evidence_id,
            "point_id": point_id, "sample_sha256": sample_sha, "tokens_per_rank": 1,
        }],
        "sampling": {
            "iterations_per_trial": 8, "reduction": "cross-rank-max-per-iteration",
            "trials": 64,
        },
        "schema_version": 1, "series_id": series_id,
    }
    sample_bytes = contracts.canonical_json_bytes(samples)
    oracle = {
        "atol": 0.02,
        "checks": {name: True for name in (
            "combine_values", "counts", "metadata", "multiplicity", "payload",
            "source_set", "weights",
        )},
        "combine_weight_semantics": "unweighted-rank-sum",
        "contract": "expert-specific-transform-v1", "dispatch_sha256": "1" * 64,
        "max_absolute_error": 0.0, "max_elementwise_relative_error": 0.0,
        "max_relative_error": 0.0, "max_weight_error": 0.0,
        "order_sha256": "2" * 64, "ordering_contract": "fixture-order-v1",
        "passed": True, "receive_count": 1, "rtol": 0.05,
    }
    def pct(value: float) -> dict[str, float]:
        return {name: value for name in ("p50", "p90", "p95", "p99")}

    def measured(value: float) -> dict:
        return {
            "availability": "measured", "origin": "measured",
            "percentiles_us": pct(value), "sample_count": 512,
        }
    row = {
        "anomalies": [],
        "components": {
            "combine": measured(20.0), "dispatch": measured(10.0),
            "isolated_sum": {
                "availability": "derived", "origin": "derived-percentile-sum",
                "percentiles_us": pct(30.0), "sample_count": 0,
            },
            "roundtrip": measured(40.0),
        },
        "correctness": {
            "contract": "expert-specific-transform-v1", "max_relative_error": 0.0,
            "passed": True,
            "rank_evidence": [{
                "input_unchanged": True, "order_stable": True,
                "post_timing": copy.deepcopy(oracle), "pre_timing": copy.deepcopy(oracle),
                "rank": 0,
            }],
            "scope": "dispatch-metadata-and-transformed-combine",
        },
        "evidence_id": evidence_id, "global_tokens": 1,
        "logical_bytes": {"combine": 2, "dispatch": 2, "roundtrip": 4},
        "point_id": point_id,
        "receive": {"max": 1, "mean": 1.0, "min": 1, "total": 1},
        "routing": contracts._expected_routing_summary(
            routing_rows,
            routing_weights,
            physical_experts=1,
            ep_size=1,
            tokens_per_rank=1,
            gpus_per_node=1,
            scale_up_domain=1,
        ),
        "sample_histograms": {
            name: contracts._expected_histogram([value] * 512)
            for name, value in (("combine", 20.0), ("dispatch", 10.0), ("roundtrip", 40.0))
        },
        "sample_sha256": sample_sha,
        "token_rate_at_latency_percentile": pct(25_000.0), "tokens_per_rank": 1,
    }
    raw = {
        "case": {
            "attempt_ordinal": 1, "backend": backend,
            "eplb": {
                "enabled": False, "imbalance_after": None, "imbalance_before": None,
                "mapping_hash": None, "max_replicas": None, "num_logical_experts": 1,
                "num_physical_experts": 1, "num_redundant": 0, "planner": None,
                "reference_tokens_per_rank": None, "replicated_experts": 0,
            },
            "ep_size": 1, "mode": "normal", "phase": "decode",
            "required_publication": "official", "resource_mode": "tuned", "runner": "fixture",
            "shape": {
                "activation_profile": "canonical-counter-source-v3", "dispatch_dtype": "bf16",
                "eplb": False, "experts": 1, "experts_per_rank": 1, "hidden": 1,
                "kernel_gen": kernel_generation, "num_logical_experts": 1,
                "quant": {
                    "combine_accum_dtype": "fp32", "combine_input_dtype": "bf16",
                    "combine_output_dtype": "bf16", "combine_quant_mode": "none",
                    "scale_layout": None,
                },
                "routing": "uniform", "topk": 1,
            },
            "suite": "ep-core-v1", "workload_name": "deepseek-v3-v1",
        },
        "format": contracts.RAW_FORMAT, "generated_at": "2026-07-04T00:00:00Z",
        "identity": {
            "allocation_factors": allocation_factors, "allocation_id": allocation_id,
            "attempt_id": attempt_id, "attempt_ordinal": 1, "case_factors": case_factors,
            "case_id": case_id, "series_factors": series_factors, "series_id": series_id,
        },
        "implementation": implementation,
        "measurement": {
            "component_order_contract": "roundtrip-dispatch-activation-only-combine-v2",
            "conditioning": {
                "contract": "fixed-phase-ramp-8-roundtrips-v1",
                "ladder": [1, 2, 4, 8, 16, 32, 64, 128],
                "roundtrips_per_shape": 8,
            },
            "contract": "layout-and-dispatch-v1",
            "rows": [row],
            "sampling": {
                "contract": "fixed-512-v1", "iterations_per_trial": 8,
                "percentile_method": "nearest-rank",
                "reduction": "cross-rank-max-per-iteration", "samples_per_component": 512,
                "trials": 64, "warmup_iterations": 32,
                "warmup_semantics": "full-roundtrip-before-each-component-trial-point-v1",
            },
            "source_allocation": "even",
        },
        "outcome": {
            "publication_status": "diagnostic", "reasons": [], "status": "success",
            "validity": {
                "anomaly_free": True, "execution_status": "complete",
                "measurement_conformance": "conformant", "provenance_complete": True,
                "resource_conformance": implementation["resource_profile"]["conformance_class"],
                "sampling_conformance": "conformant",
                "semantic_correctness": "pass",
                "workload_identity": "consistent-across-ranks",
                "workload_source": "canonical-serialized",
            },
        },
        "provenance": {
            "command": "run_ep", "distributed_launcher": "torchrun", "git_run": git_run,
            "image": {
                "arch": "amd64", "digest": "sha256:" + "d" * 64,
                "digest_verified": True, "reference": "fixture:1", "squash_sha256": "e" * 64,
            },
            "redaction": "sanitized-v1",
        },
        "record_type": "case-attempt",
        "runtime_fingerprint": runtime,
        "sample_artifact": {
            "bytes": len(sample_bytes), "format": contracts.SAMPLES_FORMAT,
            "path": "samples.json", "sha256": hashlib.sha256(sample_bytes).hexdigest(),
        },
        "schema_version": 1,
        "topology": {
            "device_count": 1, "device_product": "Fixture GPU", "gpus_per_node": 1,
            "nodes": 1, "placement": "packed",
            "realized_placement": {
                "gpus_per_node": 1, "nodes": 1, "ranks_per_node": 1,
                "unique_local_ranks": True, "valid": True,
            },
            "scale_out_transport": None, "scale_up_domain": 1,
            "scale_up_transport": "nvlink", "scope": "scale-up",
            "topology_class": "fixture", "transport": "nvlink",
            "world_size": 1,
        },
        "workload": {
            "activation_generator": "collectivex-activation-counter-v3",
            "activation_identity": hashlib.sha256(
                b"counter|seed=67|hidden=1|gen=collectivex-activation-counter-v3"
            ).hexdigest(),
            "activation_profile": "canonical-counter-source-v3", "cross_rank_consistent": True,
            "manifest_checksums": {member_id: member_checksums}, "members": [member_id],
            "routing_generator": "collectivex-routing-counter-v3", "source": "canonical-serialized",
            "trace_hashes": [routing_hash],
            "trace_signature": hashlib.sha256(routing_hash.encode()).hexdigest(),
            "workload_id": workload_id,
        },
    }
    return raw, samples


def _series(seed: str, backend: str, *, decision_grade: bool = False) -> tuple[dict, dict]:
    case, allocation, attempt, series_id, point_id, evidence = _ids(seed)
    allocations = [identity.allocation_id({"seed": seed, "run": run}) for run in range(3)]
    eligibility = publisher._eligibility_record(
        allocations if decision_grade else [allocation],
        complete=decision_grade,
        correct=True,
        measured=True,
        stable_ordering=True,
        p50_ratio=1.01 if decision_grade else None,
        p99_ratio=1.02 if decision_grade else None,
    )
    component = _component(1.0 if backend == "deepep" else 1.2)
    item = {
        "series_id": series_id,
        "label": f"H100 / {backend}",
        "status": "decision-grade" if decision_grade else "diagnostic",
        "case_ids": [case],
        "allocation_ids": allocations if decision_grade else [allocation],
        "model": "deepseek-v3-v1",
        "suite": "ep-core-v1",
        "mode": "normal",
        "phase": "decode",
        "publication_tier": "official",
        "backend": {
                    "id": backend, "label": publisher.BACKEND_LABELS[backend],
                    "role": "reference" if backend == "nccl-ep" else "library",
                    "generation": "nccl" if backend == "nccl-ep" else None,
                    "version": "1.0"},
        "build": {
            "implementation_contract_sha256": hashlib.sha256(backend.encode()).hexdigest(),
            "public_config_sha256": "0" * 64,
            "routing_control_sha256": hashlib.sha256(backend.encode()).hexdigest(),
            "runtime_fingerprint_sha256": "3" * 64,
            "image_digest": "sha256:" + "1" * 64,
            "source_sha": "a" * 40,
            "squash_sha256": "2" * 64,
        },
        "system": {
            "sku": "h100-dgxc", "label": "NVIDIA H100", "vendor": "nvidia",
            "topology_class": "h100-nvlink-island", "transport": "nvlink",
            "scale_up_transport": "nvlink", "scale_out_transport": None,
            "scope": "scale-up", "nodes": 1, "gpus_per_node": 8,
            "scale_up_domain": 8,
            "world_size": 8, "ep_size": 8, "placement": "packed",
        },
        "workload": {
            "workload_id": identity.workload_id({"shape": "shared"}),
            "hidden": 7168, "top_k": 8, "experts": 256,
            "routing": "uniform", "eplb": False,
            "dispatch_dtype": "bf16", "combine_dtype": "bf16",
            "activation_profile": "canonical-counter-source-v3",
        },
        "eplb": {
            "enabled": False, "planner": None, "mapping_sha256": None,
            "logical_experts": 256, "physical_experts": 256,
            "redundant_experts": 0, "reference_tokens_per_rank": None,
            "replicated_experts": 0, "max_replicas": None,
            "imbalance_before": None, "imbalance_after": None,
        },
        "resource": {"mode": "tuned", "profile": "profile-1", "comm_units_kind": "sm", "configured_units": 24},
        "measurement": {
            "contract": "layout-and-dispatch-v1",
            "component_order_contract": "roundtrip-dispatch-activation-only-combine-v2",
            "combine_semantics": "activation-only", "payload_unit": "token-rank",
            "sampling_contract": "fixed-512-v1",
            "iters": 8, "trials": 64, "warmups": 32, "samples_per_component": 512,
            "headline_component": "roundtrip", "headline_percentile": "p99",
        },
        "points": [{
            "point_id": point_id, "tokens_per_rank": 8, "global_tokens": 64,
            "correct": True,
            "routing": {
                "fanout_mean": 4.0, "recv_tokens_max": 64,
                "expert_load_cv": 0.5, "payload_rank_cv": 0.25,
                "hotspot_ratio": 2.0, "empty_expert_count": 0,
                "empty_rank_count": 0, "routed_copies": 256,
            },
            "components": {"dispatch": None, "combine": None,
                           "roundtrip": component, "isolated_sum": None},
            "roundtrip_token_rate_at_latency_percentile": {
                name: 64 / (latency * 1e-6)
                for name, latency in component["latency_us"].items()
            },
            "evidence_ids": [evidence],
        }],
        "eligibility": eligibility,
    }
    item["build"]["public_config_sha256"] = contracts.public_series_config_sha256(
        publisher._public_series_config(item)
    )
    case = identity.digest("case", publisher._public_case_factors(item))
    item["case_ids"] = [case]
    build = item["build"]
    series_id = identity.series_id({
        "backend": item["backend"]["id"],
        "case_id": case,
        "image_digest": build["image_digest"],
        "implementation_contract_sha256": build["implementation_contract_sha256"],
        "public_config_sha256": build["public_config_sha256"],
        "routing_control_sha256": build["routing_control_sha256"],
        "runtime_fingerprint_sha256": build["runtime_fingerprint_sha256"],
        "source_sha": build["source_sha"],
        "squash_sha256": build["squash_sha256"],
        "workload_id": item["workload"]["workload_id"],
    })
    item["series_id"] = series_id
    point_id = identity.point_id(series=series_id, tokens_per_rank=8)
    item["points"][0]["point_id"] = point_id
    attempt = identity.attempt_id(allocation=allocation, case=case, ordinal=1)
    evidence = identity.evidence_id(
        point=point_id, allocation=allocation, attempt=attempt,
        sample_sha256=hashlib.sha256(seed.encode()).hexdigest(),
    )
    item["points"][0]["evidence_ids"] = [evidence]
    runs = {
        str(run): {8: {
            "latency_us": {
                statistic: component["latency_us"][statistic] * (1 + run / 100)
                for statistic in ("p50", "p99")
            },
            "logical_payload_rate_gbps_at_latency_percentile": {
                statistic: component["logical_payload_rate_gbps_at_latency_percentile"][statistic] / (1 + run / 100)
                for statistic in ("p50", "p99")
            },
        }}
        for run in range(3)
    }
    internal = {"run_metrics": runs}
    return item, internal


def _dataset() -> dict:
    item, _ = _series("one", "deepep")
    case = item["case_ids"][0]
    allocation = item["allocation_ids"][0]
    attempt = identity.attempt_id(allocation=allocation, case=case, ordinal=1)
    evidence = item["points"][0]["evidence_ids"][0]
    return {
        "format": "collectivex.public.v1", "schema_version": 1,
        "generated_at": "2026-07-04T00:00:00Z", "source_bundle_ids": ["c" * 64],
        "promotion": {
            "status": "diagnostic", "reason": None, "matrix_id": "d" * 64,
            "allocation_ids": [allocation], "required_allocations": 3,
            "requested_cases": 1, "terminal_cases": 1,
            "policy": "collectivex-decision-grade-v1",
        },
        "coverage": [{
            "case_id": case, "label": "case", "required": True, "sku": "h100-dgxc",
            "backend": "deepep", "mode": "normal", "phase": "decode",
            "topology": publisher._coverage_topology(item["system"]),
            "disposition": "runnable",
            "selected_attempt_id": attempt,
            "outcome": "success", "failure_mode": None, "reason": None,
            "attempt_ids": [attempt],
        }],
        "attempts": [{
            "attempt_id": attempt,
            "evidence": [{"evidence_id": evidence,
                          "point_id": item["points"][0]["point_id"]}],
            "case_id": case,
            "allocation_id": allocation, "run_id": "1", "run_attempt": 1,
            "attempt_index": 1,
            "selected": True, "outcome": "success", "failure_mode": None, "reason": None,
            "series_id": item["series_id"],
            "completed_at": "2026-07-04T00:00:00Z",
        }],
        "series": [item], "cohorts": [], "rankings": [], "recommendations": [],
        "sensitivities": [],
    }


def _promoted_dataset() -> dict:
    specifications = (
        ("library-fast", "deepep", None, False),
        ("library-slow", "uccl", None, False),
        ("chip-peer", "deepep", "h200-dgxc", False),
        ("system-one", "nccl-ep", None, True),
        ("system-two", "nccl-ep", "h200-dgxc", True),
        ("routing-zipf", "deepep", None, False),
        ("routing-zipf-eplb", "deepep", None, False),
    )
    series = []
    internals = {}
    attempts = []
    coverage = []
    for seed, backend, peer_sku, reference in specifications:
        item, internal = _series(seed, backend, decision_grade=True)
        if peer_sku:
            platform = publisher.capability.PLATFORMS[peer_sku]
            item["system"].update({
                "sku": peer_sku,
                "label": f"NVIDIA {platform['product'].upper()}",
                "topology_class": platform["topology_class"],
                "transport": platform["transport"],
            })
        if reference:
            item["backend"]["role"] = "reference"
        if seed.startswith("routing-zipf"):
            item["suite"] = "ep-routing-v1"
            item["publication_tier"] = "comparable-experimental"
            item["workload"]["routing"] = "zipf"
        if seed == "routing-zipf-eplb":
            item["workload"]["eplb"] = True
            plan = contracts._expected_eplb_plan(
                "zipf", 8, 256, 288, item["system"]["ep_size"], 67, 2048
            )
            item["eplb"] = {
                "enabled": True, "planner": "greedy-rank-major-v1",
                "mapping_sha256": contracts.eplb_contract.mapping_hash(plan),
                "logical_experts": 256, "physical_experts": 288,
                "redundant_experts": 32, "reference_tokens_per_rank": 2048,
                "replicated_experts": plan["replicated_experts"],
                "max_replicas": plan["max_replicas"],
                "imbalance_before": plan["imbalance_before"],
                "imbalance_after": plan["imbalance_after"],
            }
            item["build"]["implementation_contract_sha256"] = "8" * 64
        case_id = identity.digest("case", publisher._public_case_factors(item))
        item["case_ids"] = [case_id]
        build = item["build"]
        build["public_config_sha256"] = contracts.public_series_config_sha256(
            publisher._public_series_config(item)
        )
        item["series_id"] = identity.series_id({
            "backend": item["backend"]["id"],
            "case_id": case_id,
            "image_digest": build["image_digest"],
            "implementation_contract_sha256": build["implementation_contract_sha256"],
            "public_config_sha256": build["public_config_sha256"],
            "routing_control_sha256": build["routing_control_sha256"],
            "runtime_fingerprint_sha256": build["runtime_fingerprint_sha256"],
            "source_sha": build["source_sha"],
            "squash_sha256": build["squash_sha256"],
            "workload_id": item["workload"]["workload_id"],
        })
        point = item["points"][0]
        point["point_id"] = identity.point_id(
            series=item["series_id"], tokens_per_rank=point["tokens_per_rank"]
        )
        case_attempts = []
        evidence_ids = []
        for run_id, allocation_id in enumerate(item["allocation_ids"], 1):
            attempt_id = identity.attempt_id(
                allocation=allocation_id, case=case_id, ordinal=1
            )
            evidence_id = identity.evidence_id(
                point=point["point_id"], allocation=allocation_id,
                attempt=attempt_id,
                sample_sha256=hashlib.sha256(f"{seed}-{run_id}".encode()).hexdigest(),
            )
            attempts.append({
                "attempt_id": attempt_id,
                "evidence": [{"evidence_id": evidence_id, "point_id": point["point_id"]}],
                "case_id": case_id, "allocation_id": allocation_id,
                "run_id": str(run_id), "run_attempt": 1,
                "attempt_index": 1, "selected": True,
                "outcome": "success", "failure_mode": None, "reason": None,
                "series_id": item["series_id"],
                "completed_at": "2026-07-04T00:00:00Z",
            })
            case_attempts.append(attempt_id)
            evidence_ids.append(evidence_id)
        point["evidence_ids"] = evidence_ids
        coverage.append({
            "case_id": case_id, "label": seed, "required": True,
            "sku": item["system"]["sku"], "backend": backend,
            "mode": item["mode"], "phase": item["phase"], "disposition": "runnable",
            "topology": publisher._coverage_topology(item["system"]),
            "selected_attempt_id": case_attempts[-1], "outcome": "success",
            "failure_mode": None, "reason": None, "attempt_ids": case_attempts,
        })
        series.append(item)
        internals[item["series_id"]] = internal

    unsupported_case = (
        "cxcase-v1-f5d0b1c21df4eea0f5b74da25a0b85da7388a0dd22c8d81564c4bfb4e4465a34"
    )
    unsupported_attempts = []
    for run_id in range(1, 4):
        allocation_id = identity.allocation_id(
            {"seed": "planned-unsupported", "run": run_id}
        )
        attempt_id = identity.attempt_id(
            allocation=allocation_id, case=unsupported_case, ordinal=1
        )
        attempts.append({
            "attempt_id": attempt_id, "evidence": [], "case_id": unsupported_case,
            "allocation_id": allocation_id, "run_id": str(run_id),
            "run_attempt": 1,
            "attempt_index": 1, "selected": True, "outcome": "unsupported",
            "failure_mode": "capability", "reason": "backend-platform-unsupported",
            "series_id": None, "completed_at": "2026-07-04T00:00:00Z",
        })
        unsupported_attempts.append(attempt_id)
    coverage.append({
        "case_id": unsupported_case, "label": "planned unsupported", "required": True,
        "sku": "mi355x", "backend": "deepep", "mode": "normal", "phase": "decode",
        "topology": {
            "ep_size": 8, "nodes": 1, "gpus_per_node": 8, "scale_up_domain": 8,
            "scope": "scale-up", "scale_up_transport": "xgmi",
            "scale_out_transport": None, "transport": "xgmi",
            "topology_class": "mi355x-xgmi",
        },
        "disposition": "unsupported", "selected_attempt_id": unsupported_attempts[-1],
        "outcome": "unsupported", "failure_mode": "capability",
        "reason": "backend-platform-unsupported", "attempt_ids": unsupported_attempts,
    })
    cohorts, rankings, recommendations, sensitivities = publisher.build_decisions(
        series, internals
    )
    return {
        "format": "collectivex.public.v1", "schema_version": 1,
        "generated_at": "2026-07-04T00:00:00Z",
        "source_bundle_ids": ["a" * 64, "b" * 64, "c" * 64],
        "promotion": {
            "status": "promoted", "reason": None,
            "matrix_id": publisher.CANONICAL_FULL_V1_MATRIX_SHA256,
            "allocation_ids": sorted({item["allocation_id"] for item in attempts}),
            "required_allocations": 3, "requested_cases": len(coverage),
            "terminal_cases": len(coverage), "policy": "collectivex-decision-grade-v1",
        },
        "coverage": sorted(coverage, key=lambda item: item["case_id"]),
        "attempts": sorted(attempts, key=lambda item: item["attempt_id"]),
        "series": sorted(series, key=lambda item: item["series_id"]),
        "cohorts": cohorts, "rankings": rankings,
        "recommendations": recommendations, "sensitivities": sensitivities,
    }


def _cohort_counts(dataset: dict) -> dict[str, int]:
    return {
        kind: sum(item["kind"] == kind for item in dataset["cohorts"])
        for kind in ("library", "system", "routing")
    }


class PublisherTest(unittest.TestCase):
    def test_terminal_allocation_and_source_status_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            path = next(artifact.glob("*.json"))
            terminal = contracts.strict_load(path)
            self.assertIs(contracts.validate_terminal_document(terminal), terminal)
            self.assertEqual(
                contracts.validate_delivery(
                    [str(path)], str(matrix), disposition="unsupported"
                ),
                1,
            )

            for control_sha256 in (None, "0" * 64):
                broken = copy.deepcopy(terminal)
                broken["provenance"]["control_sha256"] = control_sha256
                path.write_text(json.dumps(broken))
                with self.assertRaisesRegex(contracts.ContractError, "exact control document"):
                    contracts.validate_delivery(
                        [str(path)], str(matrix), disposition="unsupported"
                    )
            path.write_text(json.dumps(terminal))

            for field in (
                "artifact", "job", "repo", "run_attempt", "run_id", "source_sha", "runner"
            ):
                broken = copy.deepcopy(terminal)
                broken["identity"]["allocation_factors"][field] = f"forged-{field}"
                allocation_id = identity.allocation_id(
                    broken["identity"]["allocation_factors"]
                )
                broken["identity"]["allocation_id"] = allocation_id
                broken["identity"]["attempt_id"] = identity.attempt_id(
                    allocation=allocation_id,
                    case=broken["identity"]["case_id"],
                    ordinal=broken["identity"]["attempt_ordinal"],
                )
                with self.assertRaisesRegex(
                    contracts.ContractError, "allocation factors differ"
                ):
                    contracts.validate_terminal_document(broken)

            broken = copy.deepcopy(terminal)
            broken["outcome"]["status"] = "failed"
            with self.assertRaisesRegex(contracts.ContractError, "source and outcome"):
                contracts.validate_terminal_document(broken)
            broken = copy.deepcopy(terminal)
            broken["provenance"]["source"] = "runtime-emitter"
            with self.assertRaisesRegex(contracts.ContractError, "source and outcome"):
                contracts.validate_terminal_document(broken)

            for path_parts, replacement in (
                (("provenance", "source"), "unregistered-producer"),
                (("outcome", "failure_mode"), "unsupported-capability"),
                (("outcome", "reason"), "unregistered-capability"),
            ):
                with self.subTest(path=path_parts):
                    broken = copy.deepcopy(terminal)
                    broken[path_parts[0]][path_parts[1]] = replacement
                    with self.assertRaises(publisher.PublisherError):
                        publisher._schema("terminal-outcome-v1.schema.json", broken)
                    with self.assertRaises(contracts.ContractError):
                        contracts.validate_terminal_document(broken)

            runtime_allocation = copy.deepcopy(
                terminal["identity"]["allocation_factors"]
            )
            runtime_allocation["runner"] = terminal["identity"]["case_factors"]["sku"]
            runtime = contracts.make_terminal_document(
                allocation_factors=runtime_allocation,
                attempt_ordinal=1,
                case=terminal["case"],
                case_factors=terminal["identity"]["case_factors"],
                control_sha256=terminal["provenance"]["control_sha256"],
                failure_mode="setup",
                generated_at=terminal["generated_at"],
                git_run=terminal["provenance"]["git_run"],
                reason="launcher-setup-failed",
                return_code=1,
                source="runtime-emitter",
                status="failed",
                expected_case_id=terminal["identity"]["case_id"],
            )
            publisher._schema("terminal-outcome-v1.schema.json", runtime)
            broken = copy.deepcopy(runtime)
            broken["outcome"]["reason"] = "backend-setup-failed"
            with self.assertRaises(publisher.PublisherError):
                publisher._schema("terminal-outcome-v1.schema.json", broken)
            with self.assertRaises(contracts.ContractError):
                contracts.validate_terminal_document(broken)

    def test_post_emit_demotion_uses_closed_failure_taxonomy(self) -> None:
        raw, _ = _native_fixture()
        expected = {
            5: "runtime-identity",
            6: "execution",
            124: "timeout",
            137: "timeout",
            134: "execution",
            9: "execution",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for return_code, failure_mode in expected.items():
                with self.subTest(return_code=return_code):
                    path = root / f"attempt-{return_code}.json"
                    path.write_text(json.dumps(raw))
                    terminal = contracts.demote_raw_attempt(path, return_code)
                    self.assertEqual(
                        terminal["outcome"],
                        {
                            "failure_mode": failure_mode,
                            "reason": "post-emit-distributed-command-failed",
                            "return_code": return_code,
                            "status": "failed",
                        },
                    )
                    self.assertEqual(terminal["provenance"]["source"], "post-emit-command")
                    publisher._schema("terminal-outcome-v1.schema.json", terminal)

                    broken = copy.deepcopy(terminal)
                    broken["outcome"]["reason"] = "distributed-command-failed"
                    with self.assertRaises(publisher.PublisherError):
                        publisher._schema("terminal-outcome-v1.schema.json", broken)
                    with self.assertRaises(contracts.ContractError):
                        contracts.validate_terminal_document(broken)

    def test_artifact_safety_accepts_current_v1_fixtures(self) -> None:
        raw, samples = _native_fixture()
        publisher.artifact_safety.assert_publication_safe([
            sweep_matrix.resolve_matrix(backends="all"),
            raw,
            samples,
            _dataset(),
            _promoted_dataset(),
        ])

    def test_native_raw_and_sample_schema_match_semantic_validator(self) -> None:
        raw, samples = _native_fixture()
        publisher._schema("samples-v1.schema.json", samples)
        publisher._schema("raw-case-v1.schema.json", raw)
        self.assertIs(contracts.validate_raw_document(raw, samples), raw)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "samples.json").write_bytes(contracts.canonical_json_bytes(samples))
            (root / "raw.json").write_bytes(contracts.canonical_json_bytes(raw))
            self.assertEqual(contracts.load_raw_attempt(root / "raw.json"), raw)
        for target in ("raw", "samples"):
            broken_raw, broken_samples = copy.deepcopy((raw, samples))
            broken = broken_raw if target == "raw" else broken_samples
            broken["unexpected"] = True
            with self.assertRaises(publisher.PublisherError):
                publisher._schema(
                    "raw-case-v1.schema.json" if target == "raw" else "samples-v1.schema.json",
                    broken,
                )
            with self.assertRaises(contracts.ContractError):
                contracts.validate_raw_document(broken_raw, broken_samples)
        tampered = copy.deepcopy(raw)
        tampered["measurement"]["rows"][0]["token_rate_at_latency_percentile"]["p50"] *= 2
        with self.assertRaisesRegex(contracts.ContractError, "token_rate_at_latency_percentile"):
            contracts.validate_raw_document(tampered, samples)
        tampered = copy.deepcopy(raw)
        tampered["case"]["shape"]["hidden"] = 2
        with self.assertRaises(contracts.ContractError):
            contracts.validate_raw_document(tampered, samples)
        tampered = copy.deepcopy(raw)
        configured = tampered["implementation"]["resource_profile"]["configured_units"]
        tampered["implementation"]["resource_profile"]["configured_units"] = (
            1 if configured is None else configured + 1
        )
        with self.assertRaisesRegex(contracts.ContractError, "resource profile"):
            contracts.validate_raw_document(tampered, samples)
        tampered = copy.deepcopy(raw)
        oracle = tampered["measurement"]["rows"][0]["correctness"]["rank_evidence"][0]
        oracle["pre_timing"]["checks"]["combine_values"] = False
        with self.assertRaisesRegex(contracts.ContractError, "passed differs"):
            contracts.validate_raw_document(tampered, samples)

    def test_hybrid_raw_binds_realized_config_and_every_rank_artifact(self) -> None:
        raw, samples = _native_fixture("deepep-hybrid")
        publisher._schema("raw-case-v1.schema.json", raw)
        self.assertIs(contracts.validate_raw_document(raw, samples), raw)

        mutations = {
            "hidden_dim": lambda provenance: provenance["realized_config"].update(
                hidden_dim=2
            ),
            "experts_per_rank": lambda provenance: provenance["realized_config"].update(
                num_of_experts_per_rank=2
            ),
            "ranks_per_node": lambda provenance: provenance["realized_config"].update(
                num_of_ranks_per_node=2
            ),
            "num_nodes": lambda provenance: provenance["realized_config"].update(
                num_of_nodes=2
            ),
            "token_data_type": lambda provenance: provenance["realized_config"].update(
                token_data_type="UINT8"
            ),
            "rank_coverage": lambda provenance: [
                artifact["rank_artifacts"].append({
                    "bytes": 1, "rank": 1, "sha256": "9" * 64,
                })
                for artifact in provenance["jit_shared_objects"]
            ],
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(raw)
                mutate(changed["implementation"]["provenance"])
                with self.assertRaisesRegex(
                    contracts.ContractError,
                    "DeepEP Hybrid realized config/JIT evidence differs",
                ):
                    contracts.validate_raw_document(changed, samples)

    def test_native_contract_recomputes_routing_receive_histograms_and_anomalies(self) -> None:
        raw, samples = _native_fixture()

        tampered = copy.deepcopy(raw)
        changed = tampered["measurement"]["rows"][0]
        changed["routing"]["routed_copies"] *= 2
        for name in ("combine", "dispatch", "roundtrip"):
            changed["logical_bytes"][name] *= 2
        with self.assertRaisesRegex(contracts.ContractError, "routing.routed_copies"):
            contracts.validate_raw_document(tampered, samples)

        tampered = copy.deepcopy(raw)
        changed = tampered["measurement"]["rows"][0]
        changed["routing"]["payload_copies_per_rank"] = [2]
        changed["receive"] = {"max": 2, "mean": 2.0, "min": 2, "total": 2}
        with self.assertRaisesRegex(contracts.ContractError, "payload_copies_per_rank"):
            contracts.validate_raw_document(tampered, samples)

        tampered = copy.deepcopy(raw)
        tampered["measurement"]["rows"][0]["sample_histograms"]["roundtrip"][
            "counts"
        ] = [511]
        with self.assertRaisesRegex(contracts.ContractError, "sample_histograms"):
            contracts.validate_raw_document(tampered, samples)

        tampered = copy.deepcopy(raw)
        tampered["measurement"]["rows"][0]["anomalies"] = [{
            "type": "roundtrip_gt_isolated_sum",
            "T": 1,
            "roundtrip_p99": 40.0,
            "isolated_sum_p99": 30.0,
            "ratio": 1.33,
            "threshold": 3.0,
        }]
        tampered["outcome"]["validity"]["anomaly_free"] = False
        with self.assertRaisesRegex(contracts.ContractError, "anomalies"):
            contracts.validate_raw_document(tampered, samples)

        anomalous_raw, anomalous_samples = copy.deepcopy((raw, samples))
        sample_point = anomalous_samples["points"][0]
        sample_point["components"]["roundtrip"]["trials"] = [
            [100.0] * 8 for _ in range(64)
        ]
        sample_core = {
            "components": sample_point["components"],
            "tokens_per_rank": sample_point["tokens_per_rank"],
        }
        sample_sha = hashlib.sha256(
            contracts.canonical_json_bytes(sample_core)
        ).hexdigest()
        point_id = sample_point["point_id"]
        evidence_id = identity.evidence_id(
            point=point_id,
            allocation=anomalous_raw["identity"]["allocation_id"],
            attempt=anomalous_raw["identity"]["attempt_id"],
            sample_sha256=sample_sha,
        )
        sample_point.update({"sample_sha256": sample_sha, "evidence_id": evidence_id})
        changed = anomalous_raw["measurement"]["rows"][0]
        changed["sample_sha256"] = sample_sha
        changed["evidence_id"] = evidence_id
        changed["components"]["roundtrip"]["percentiles_us"] = {
            name: 100.0 for name in ("p50", "p90", "p95", "p99")
        }
        changed["token_rate_at_latency_percentile"] = {
            name: 10_000.0 for name in ("p50", "p90", "p95", "p99")
        }
        changed["sample_histograms"]["roundtrip"] = contracts._expected_histogram(
            [100.0] * 512
        )
        changed["anomalies"] = contracts._expected_anomalies(1, changed["components"])
        anomalous_raw["outcome"]["validity"]["anomaly_free"] = False
        sample_bytes = contracts.canonical_json_bytes(anomalous_samples)
        anomalous_raw["sample_artifact"].update({
            "bytes": len(sample_bytes),
            "sha256": hashlib.sha256(sample_bytes).hexdigest(),
        })
        self.assertIs(
            contracts.validate_raw_document(anomalous_raw, anomalous_samples),
            anomalous_raw,
        )
        changed["anomalies"] = []
        anomalous_raw["outcome"]["validity"]["anomaly_free"] = True
        with self.assertRaisesRegex(contracts.ContractError, "anomalies"):
            contracts.validate_raw_document(anomalous_raw, anomalous_samples)

    def test_native_contract_rejects_every_schema_only_nested_mutation(self) -> None:
        raw, samples = _native_fixture()
        self.assertIs(contracts.validate_raw_document(raw, samples), raw)

        def locate(document: object, path: tuple[object, ...]) -> object:
            value = document
            for part in path:
                value = value[part]  # type: ignore[index]
            return value

        def reject_raw(document: dict) -> None:
            with self.assertRaises(publisher.PublisherError):
                publisher._schema("raw-case-v1.schema.json", document)
            with self.assertRaises(contracts.ContractError):
                contracts.validate_raw_document(document, samples)

        required_fields = (
            (("measurement", "rows", 0, "receive"), "total"),
            (("measurement", "rows", 0, "routing"), "fanout_mean"),
            (("measurement", "rows", 0, "routing", "source_token_stats"), "ranks"),
            (("measurement", "rows", 0, "sample_histograms"), "roundtrip"),
            (("measurement", "rows", 0, "sample_histograms", "roundtrip"), "n"),
            (("runtime_fingerprint", "accelerator_runtime"), "kind"),
            (("runtime_fingerprint", "collective_library"), "kind"),
            (("runtime_fingerprint", "framework"), "kind"),
        )
        for path, required in required_fields:
            with self.subTest(path=path, mutation="missing"):
                broken = copy.deepcopy(raw)
                del locate(broken, path)[required]  # type: ignore[index]
                reject_raw(broken)
            with self.subTest(path=path, mutation="extra"):
                broken = copy.deepcopy(raw)
                locate(broken, path)["unexpected"] = None  # type: ignore[index]
                reject_raw(broken)

        invalid_values = (
            (("measurement", "rows", 0, "receive", "mean"), "one"),
            (("measurement", "rows", 0, "routing", "fanout_mean"), "one"),
            (("measurement", "rows", 0, "sample_histograms", "roundtrip", "bins"), 0),
            (("provenance", "image", "arch"), "AMD64"),
            (("runtime_fingerprint", "accelerator_runtime", "kind"), "rocm"),
        )
        for path, invalid in invalid_values:
            with self.subTest(path=path, mutation="value"):
                broken = copy.deepcopy(raw)
                parent = locate(broken, path[:-1])
                parent[path[-1]] = invalid  # type: ignore[index]
                reject_raw(broken)

        def reject_samples(document: dict) -> None:
            with self.assertRaises(publisher.PublisherError):
                publisher._schema("samples-v1.schema.json", document)
            with self.assertRaises(contracts.ContractError):
                contracts.validate_samples_document(document)

        for path, required in (
            (("points", 0), "evidence_id"),
            (("points", 0, "components"), "roundtrip"),
            (("points", 0, "components", "roundtrip"), "trials"),
            (("sampling",), "reduction"),
        ):
            with self.subTest(path=path, artifact="samples"):
                broken = copy.deepcopy(samples)
                del locate(broken, path)[required]  # type: ignore[index]
                reject_samples(broken)

    def test_terminal_contract_and_schema_reject_the_same_shape_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, artifact = _unsupported_delivery(Path(temporary).resolve())
            terminal = contracts.strict_load(next(artifact.glob("*.json")))
        publisher._schema("terminal-outcome-v1.schema.json", terminal)
        self.assertIs(contracts.validate_terminal_document(terminal), terminal)

        def reject(document: dict) -> None:
            with self.assertRaises(publisher.PublisherError):
                publisher._schema("terminal-outcome-v1.schema.json", document)
            with self.assertRaises(contracts.ContractError):
                contracts.validate_terminal_document(document)

        for path, invalid in (
            (("outcome", "failure_mode"), "Not Safe"),
            (("outcome", "reason"), "x" * 241),
            (("provenance", "source"), "Not Safe"),
            (("provenance", "git_run", "ref"), ""),
        ):
            with self.subTest(path=path):
                broken = copy.deepcopy(terminal)
                parent = broken
                for part in path[:-1]:
                    parent = parent[part]
                parent[path[-1]] = invalid
                reject(broken)

    def test_invalid_retry_is_quarantined_before_valid_retry_upload(self) -> None:
        raw, samples = _native_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sample_bytes = contracts.canonical_json_bytes(samples)
            bad = copy.deepcopy(raw)
            bad["sample_artifact"].update({
                "path": "a01.samples.json", "bytes": len(sample_bytes),
                "sha256": hashlib.sha256(sample_bytes).hexdigest(),
            })
            bad["measurement"]["rows"][0]["token_rate_at_latency_percentile"]["p50"] *= 2
            (root / "a01.samples.json").write_bytes(sample_bytes)
            (root / "a01.json").write_bytes(contracts.canonical_json_bytes(bad))
            self.assertTrue(contracts.quarantine_invalid_attempt(root / "a01.json"))
            valid = copy.deepcopy(raw)
            valid["sample_artifact"].update({
                "path": "a02.samples.json", "bytes": len(sample_bytes),
                "sha256": hashlib.sha256(sample_bytes).hexdigest(),
            })
            (root / "a02.samples.json").write_bytes(sample_bytes)
            (root / "a02.json").write_bytes(contracts.canonical_json_bytes(valid))
            paths = sorted(str(path) for path in root.glob("*.json"))
            self.assertEqual(contracts.validate_attempt_paths(paths), 1)
            self.assertTrue((root / "a01.json.quarantine").is_file())
            self.assertTrue((root / "a01.samples.json.quarantine").is_file())

    def test_ingest_archives_first_and_publishes_latest_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            self.assertEqual(len(summarize.load_results(str(artifact), None, None)), 1)
            result = publisher.ingest_command(_args(root / "store", matrix, artifact))
            store = publisher.Store(root / "store")
            pointer = store.verify_channel("latest-attempt")
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(pointer["dataset"]["sha256"], result["dataset_sha256"])
            self.assertTrue((store.incoming / result["incoming_id"] / "COMPLETE").is_file())
            self.assertTrue((store.bundles / result["bundle_id"] / "COMPLETE").is_file())
            self.assertFalse((store.channels / "dev-latest.json").exists())
            self.assertEqual(os.stat(store.private).st_mode & 0o777, 0o700)
            self.assertEqual(os.stat(store.public).st_mode & 0o777, 0o755)
            self.assertEqual(os.stat(store.bundles / result["bundle_id"]).st_mode & 0o777, 0o500)
            dataset_dir = store.datasets / result["dataset_sha256"]
            self.assertEqual(os.stat(dataset_dir).st_mode & 0o777, 0o555)
            self.assertEqual(os.stat(dataset_dir / "dataset.json").st_mode & 0o777, 0o444)

    def test_repeated_ingest_is_content_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            args = _args(root / "store", matrix, artifact)
            first = publisher.ingest_command(args)
            store = publisher.Store(root / "store")
            pointer_before = (store.channels / "latest-attempt.json").read_bytes()
            second = publisher.ingest_command(args)
            self.assertEqual(second, first)
            self.assertEqual(
                (store.channels / "latest-attempt.json").read_bytes(), pointer_before
            )
            self.assertEqual(len(list(store.incoming.iterdir())), 1)
            self.assertEqual(len(list(store.bundles.iterdir())), 1)
            self.assertEqual(len(list(store.datasets.iterdir())), 1)
            bundle = publisher.strict_load(
                store.bundles / first["bundle_id"] / "bundle.json"
            )
            terminal = publisher.strict_load(next(artifact.glob("*.json")))
            self.assertEqual(bundle["created_at"], terminal["generated_at"])

    def test_dataset_is_invariant_to_bundle_argument_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            store_root = root / "store"
            bundle_ids = []
            for run_id in (9, 11, 10):
                run = {**RUN, "run_id": str(run_id)}
                delivery = root / f"run-{run_id}"
                delivery.mkdir()
                matrix, artifact = _unsupported_delivery(delivery, run=run)
                result = publisher.ingest_command(
                    _args(store_root, matrix, artifact, run=run)
                )
                bundle_ids.append(result["bundle_id"])
            datasets = [
                publisher.build_dataset(
                    publisher.Store(store_root), order, promote=False,
                )
                for order in itertools.permutations(bundle_ids)
            ]
            self.assertTrue(all(dataset == datasets[0] for dataset in datasets[1:]))
            self.assertEqual(datasets[0]["generated_at"], "2026-07-04T00:00:00Z")
            selected = datasets[0]["coverage"][0]["selected_attempt_id"]
            selected_attempt = next(
                item for item in datasets[0]["attempts"]
                if item["attempt_id"] == selected
            )
            self.assertEqual(selected_attempt["run_id"], "11")

    def test_diagnostic_dataset_orders_reruns_by_run_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            store_root = root / "store"
            bundle_ids = []
            for run_attempt in (1, 2):
                run = {**RUN, "run_attempt": run_attempt}
                delivery = root / f"attempt-{run_attempt}"
                delivery.mkdir()
                matrix, artifact = _unsupported_delivery(delivery, run=run)
                result = publisher.ingest_command(
                    _args(store_root, matrix, artifact, run=run)
                )
                bundle_ids.append(result["bundle_id"])
            dataset = publisher.build_dataset(
                publisher.Store(store_root), bundle_ids, promote=False
            )
            selected_id = dataset["coverage"][0]["selected_attempt_id"]
            selected = next(
                item for item in dataset["attempts"]
                if item["attempt_id"] == selected_id
            )
            self.assertEqual(selected["run_attempt"], 2)

    def test_promotion_requires_every_runnable_case_to_succeed_in_every_bundle(self) -> None:
        cases = {
            "runnable": {"_disposition": "runnable"},
            "planned-unsupported": {"_disposition": "unsupported"},
        }
        bundles = []
        for _ in range(3):
            runnable = {
                "identity": {"case_id": "runnable"},
                "outcome": {"status": "success"},
            }
            unsupported = {
                "identity": {"case_id": "planned-unsupported"},
                "outcome": {"status": "unsupported"},
            }
            bundles.append({
                "selected": {"runnable": runnable, "planned-unsupported": unsupported},
                "documents": {"runnable": runnable, "planned-unsupported": unsupported},
            })
        publisher._require_runnable_promotion_success(bundles, cases)

        for status in ("failed", "invalid", "unsupported", "diagnostic"):
            with self.subTest(status=status):
                broken = copy.deepcopy(bundles)
                broken[1]["selected"]["runnable"]["outcome"]["status"] = status
                with self.assertRaisesRegex(
                    publisher.PublisherError, "every runnable matrix case"
                ):
                    publisher._require_runnable_promotion_success(broken, cases)

        broken = copy.deepcopy(bundles)
        broken[1]["documents"]["retry"] = {
            "identity": {"case_id": "runnable"},
            "outcome": {"status": "failed"},
        }
        with self.assertRaisesRegex(publisher.PublisherError, "rejects runnable cases"):
            publisher._require_runnable_promotion_success(broken, cases)

    def test_promoted_public_dataset_rejects_failed_retry_history(self) -> None:
        dataset = _promoted_dataset()
        successful = next(
            item for item in dataset["attempts"]
            if item["outcome"] == "success"
        )
        failed = copy.deepcopy(successful)
        old_attempt_id = successful["attempt_id"]
        successful["attempt_index"] = 2
        successful["attempt_id"] = identity.attempt_id(
            allocation=successful["allocation_id"], case=successful["case_id"], ordinal=2
        )
        failed.update({
            "attempt_id": old_attempt_id,
            "attempt_index": 1,
            "outcome": "failed",
            "failure_mode": "execution",
            "reason": "execution-failed",
            "series_id": None,
            "selected": False,
            "evidence": [],
        })
        dataset["attempts"].append(failed)
        dataset["attempts"].sort(key=lambda item: item["attempt_id"])
        coverage = next(
            item for item in dataset["coverage"]
            if item["case_id"] == failed["case_id"]
        )
        coverage["attempt_ids"] = [
            successful["attempt_id"] if value == old_attempt_id else value
            for value in coverage["attempt_ids"]
        ]
        coverage["attempt_ids"].append(failed["attempt_id"])
        coverage["attempt_ids"].sort()
        if coverage["selected_attempt_id"] == old_attempt_id:
            coverage["selected_attempt_id"] = successful["attempt_id"]

        fixture_catalog = publisher._case_disposition_catalog_sha256(dataset["coverage"])
        with mock.patch.object(
            publisher, "CANONICAL_FULL_V1_CASE_CATALOG_SHA256", fixture_catalog
        ), self.assertRaisesRegex(publisher.PublisherError, "rejects runnable cases"):
            publisher.validate_public_dataset(dataset)

    def test_unselected_success_does_not_reference_an_unpublished_series(self) -> None:
        raw, _ = _native_fixture()
        retained = publisher._public_attempt(raw, selected=False)
        selected = publisher._public_attempt(raw, selected=True)
        self.assertEqual(retained["outcome"], "success")
        self.assertIsNone(retained["series_id"])
        self.assertEqual(selected["series_id"], raw["identity"]["series_id"])

    def test_public_dataset_selects_latest_derived_retry(self) -> None:
        dataset = _dataset()
        first = dataset["attempts"][0]
        second = copy.deepcopy(first)
        second.update({
            "attempt_id": identity.attempt_id(
                allocation=first["allocation_id"], case=first["case_id"], ordinal=2
            ),
            "attempt_index": 2,
            "selected": False,
            "series_id": None,
            "evidence": [],
        })
        dataset["attempts"].append(second)
        dataset["attempts"].sort(key=lambda item: item["attempt_id"])
        dataset["coverage"][0]["attempt_ids"].append(second["attempt_id"])
        dataset["coverage"][0]["attempt_ids"].sort()
        with self.assertRaisesRegex(publisher.PublisherError, "select the latest retry"):
            publisher.validate_public_dataset(dataset)

        second["attempt_id"] = identity.digest("attempt", {"not": "derived"})
        dataset["attempts"].sort(key=lambda item: item["attempt_id"])
        dataset["coverage"][0]["attempt_ids"] = [
            item["attempt_id"] for item in dataset["attempts"]
        ]
        with self.assertRaisesRegex(publisher.PublisherError, "retry identity differs"):
            publisher.validate_public_dataset(dataset)

    def test_promotion_requires_an_eligible_cohort_for_every_comparison_kind(self) -> None:
        stable_fast, stable_fast_internal = _series(
            "stable-fast", "deepep", decision_grade=True
        )
        stable_slow, stable_slow_internal = _series(
            "stable-slow", "uccl", decision_grade=True
        )
        unstable_fast, unstable_fast_internal = _series(
            "unstable-fast", "deepep", decision_grade=True
        )
        unstable_slow, unstable_slow_internal = _series(
            "unstable-slow", "uccl", decision_grade=True
        )
        unstable_fast["phase"] = unstable_slow["phase"] = "prefill"
        unstable_fast["series_id"] = identity.series_id({"test": "unstable-fast"})
        unstable_slow["series_id"] = identity.series_id({"test": "unstable-slow"})
        for statistic in ("p50", "p99"):
            unstable_slow_internal["run_metrics"]["1"][8]["latency_us"][statistic] = (
                unstable_fast_internal["run_metrics"]["1"][8]["latency_us"][statistic]
                / 2
            )
            unstable_slow_internal["run_metrics"]["1"][8]["logical_payload_rate_gbps_at_latency_percentile"][statistic] = (
                unstable_fast_internal["run_metrics"]["1"][8]["logical_payload_rate_gbps_at_latency_percentile"][statistic]
                * 2
            )
        series = [stable_fast, stable_slow, unstable_fast, unstable_slow]
        internals = {
            stable_fast["series_id"]: stable_fast_internal,
            stable_slow["series_id"]: stable_slow_internal,
            unstable_fast["series_id"]: unstable_fast_internal,
            unstable_slow["series_id"]: unstable_slow_internal,
        }
        cohorts, _, _, _ = publisher.build_decisions(series, internals)
        eligible = [item for item in cohorts if item["eligibility"]["decision_grade"]]
        ineligible = [item for item in cohorts if not item["eligibility"]["decision_grade"]]
        self.assertEqual({item["kind"] for item in eligible}, {"library"})
        self.assertTrue(ineligible)
        anchor_series = [
            {
                "series_id": name,
                "workload": {"routing": routing, "eplb": eplb},
                "build": {"implementation_contract_sha256": "1" * 64},
            }
            for name, routing, eplb in (
                ("uniform", "uniform", False),
                ("zipf", "zipf", False),
                ("zipf-eplb", "zipf", True),
            )
        ]
        required = eligible + [
            {
                "kind": kind,
                "eligibility": {"decision_grade": True},
                **({"series_ids": [item["series_id"] for item in anchor_series]}
                   if kind == "routing" else {}),
            }
            for kind in publisher.REQUIRED_COHORT_KINDS
            if kind != "library"
        ]
        with mock.patch.object(
            publisher, "REQUIRED_PROMOTION_COHORT_COUNTS", {}
        ), mock.patch.object(
            publisher, "_expected_chip_cohort_count", return_value=1
        ):
            publisher._require_promotion_cohorts(
                required + ineligible, anchor_series
            )
            for kind in publisher.REQUIRED_COHORT_KINDS:
                with self.subTest(missing_kind=kind), self.assertRaisesRegex(
                    publisher.PublisherError, rf"cohort kinds:.*{kind}"
                ):
                    publisher._require_promotion_cohorts([
                        item for item in required + ineligible
                        if item["kind"] != kind or not item["eligibility"]["decision_grade"]
                    ], anchor_series)

    def test_promotion_requires_exact_counts_and_routing_anchors(self) -> None:
        dataset = _promoted_dataset()
        counts = _cohort_counts(dataset)
        with mock.patch.object(
            publisher, "REQUIRED_PROMOTION_COHORT_COUNTS", counts
        ):
            publisher._require_promotion_cohorts(
                dataset["cohorts"], dataset["series"]
            )
            routing = next(
                item for item in dataset["cohorts"] if item["kind"] == "routing"
            )
            eplb = next(
                item for item in dataset["series"]
                if item["series_id"] in routing["series_ids"]
                and item["workload"]["eplb"]
            )
            eplb["workload"]["eplb"] = False
            with self.assertRaisesRegex(publisher.PublisherError, "exact uniform"):
                publisher._require_promotion_cohorts(
                    dataset["cohorts"], dataset["series"]
                )

        dataset = _promoted_dataset()
        routing = next(item for item in dataset["cohorts"] if item["kind"] == "routing")
        zipf = next(
            item for item in dataset["series"]
            if item["series_id"] in routing["series_ids"]
            and item["workload"]["routing"] == "zipf"
            and not item["workload"]["eplb"]
        )
        zipf["build"]["implementation_contract_sha256"] = "f" * 64
        with mock.patch.object(
            publisher, "REQUIRED_PROMOTION_COHORT_COUNTS", counts
        ), self.assertRaisesRegex(publisher.PublisherError, "identical off-EPLB"):
            publisher._require_promotion_cohorts(dataset["cohorts"], dataset["series"])

        wrong_counts = {**counts, "library": counts["library"] + 1}
        with mock.patch.object(
            publisher, "REQUIRED_PROMOTION_COHORT_COUNTS", wrong_counts
        ), self.assertRaisesRegex(publisher.PublisherError, "exactly"):
            publisher._require_promotion_cohorts(
                dataset["cohorts"], dataset["series"]
            )

    def test_promotion_requires_every_derived_chip_cohort_to_be_stable(self) -> None:
        dataset = _promoted_dataset()
        chip = next(item for item in dataset["cohorts"] if item["kind"] == "chip")
        self.assertEqual(
            publisher._expected_chip_cohort_count(dataset["series"]),
            sum(item["kind"] == "chip" for item in dataset["cohorts"]),
        )
        with mock.patch.object(
            publisher, "REQUIRED_PROMOTION_COHORT_COUNTS", _cohort_counts(dataset)
        ):
            missing = [item for item in dataset["cohorts"] if item is not chip]
            with self.assertRaisesRegex(publisher.PublisherError, "derived chip cohorts"):
                publisher._require_promotion_cohorts(missing, dataset["series"])

            chip["eligibility"]["decision_grade"] = False
            with self.assertRaisesRegex(publisher.PublisherError, "derived chip cohorts"):
                publisher._require_promotion_cohorts(
                    dataset["cohorts"], dataset["series"]
                )

    def test_promotion_rejects_more_than_three_bundles(self) -> None:
        bundles = {
            str(run_id): {
                "id": str(run_id), "cases": [],
                "manifest": {
                    "matrix": {"sha256": publisher.CANONICAL_FULL_V1_MATRIX_SHA256},
                    "run": {"run_id": str(run_id), "run_attempt": 1},
                },
            }
            for run_id in range(1, 5)
        }
        with mock.patch.object(
            publisher, "load_bundle", side_effect=lambda _, bundle_id: bundles[bundle_id]
        ), self.assertRaisesRegex(publisher.PublisherError, "three independent"):
            publisher.build_dataset(object(), list(bundles), promote=True)

        dataset = _promoted_dataset()
        dataset["source_bundle_ids"].append("d" * 64)
        counts = _cohort_counts(dataset)
        with mock.patch.object(
            publisher,
            "CANONICAL_FULL_V1_CASE_CATALOG_SHA256",
            publisher._case_disposition_catalog_sha256(dataset["coverage"]),
        ), mock.patch.object(
            publisher, "REQUIRED_PROMOTION_COHORT_COUNTS", counts
        ), self.assertRaisesRegex(publisher.PublisherError, "complete coverage"):
            publisher.validate_public_dataset(dataset)

    def test_standalone_promotion_binds_matrix_and_requested_dispositions(self) -> None:
        dataset = _promoted_dataset()
        fixture_catalog = publisher._case_disposition_catalog_sha256(dataset["coverage"])
        with self.assertRaisesRegex(
            publisher.PublisherError, "canonical case/disposition catalog"
        ):
            publisher.validate_public_dataset(dataset)
        with mock.patch.object(
            publisher, "CANONICAL_FULL_V1_CASE_CATALOG_SHA256", fixture_catalog
        ), mock.patch.object(
            publisher,
            "REQUIRED_PROMOTION_COHORT_COUNTS",
            _cohort_counts(dataset),
        ):
            publisher.validate_public_dataset(dataset)

        diagnostic = copy.deepcopy(dataset)
        item = diagnostic["series"][0]
        item["status"] = "diagnostic"
        item["eligibility"].update({
            "decision_grade": False,
            "stable_p50": False,
            "p50_max_min_ratio": 1.20,
            "reasons": ["unstable-p50"],
        })
        with mock.patch.object(
            publisher, "CANONICAL_FULL_V1_CASE_CATALOG_SHA256", fixture_catalog
        ), mock.patch.object(
            publisher,
            "REQUIRED_PROMOTION_COHORT_COUNTS",
            _cohort_counts(dataset),
        ), self.assertRaisesRegex(
            publisher.PublisherError, "unstable or incomplete required series"
        ):
            publisher.validate_public_dataset(diagnostic)

        broken = copy.deepcopy(dataset)
        broken["promotion"]["matrix_id"] = "d" * 64
        with self.assertRaisesRegex(publisher.PublisherError, "canonical full-v1 matrix"):
            publisher.validate_public_dataset(broken)

        for original, replacement in (("runnable", "unsupported"),
                                      ("unsupported", "runnable")):
            with self.subTest(original=original):
                broken = copy.deepcopy(dataset)
                item = next(
                    coverage for coverage in broken["coverage"]
                    if coverage["disposition"] == original
                )
                item["disposition"] = replacement
                with mock.patch.object(
                    publisher,
                    "CANONICAL_FULL_V1_CASE_CATALOG_SHA256",
                    publisher._case_disposition_catalog_sha256(broken["coverage"]),
                ), self.assertRaisesRegex(
                    publisher.PublisherError, "requested dispositions"
                ):
                    publisher.validate_public_dataset(broken)

    def test_workflow_matrix_and_catalog_digests_do_not_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            matrix_path = Path(temporary) / "matrix_full.json"
            result = subprocess.run(
                [
                    sys.executable, str(ROOT / "sweep_matrix.py"),
                    "--suites", "all", "--max-cases", "128",
                    "--backends", "all", "--out", str(matrix_path),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                publisher.CANONICAL_FULL_V1_MATRIX_SHA256,
            )
            matrix = contracts.strict_load(matrix_path)
        coverage = [
            {
                "case_id": item["case"]["case_id"],
                "disposition": item["disposition"],
            }
            for item in matrix["requested_cases"]
        ]
        self.assertEqual(
            publisher._case_disposition_catalog_sha256(coverage),
            publisher.CANONICAL_FULL_V1_CASE_CATALOG_SHA256,
        )
        self.assertEqual(
            (
                len(matrix["include"]), len(coverage),
                sum(item["disposition"] == "runnable" for item in coverage),
                sum(item["disposition"] == "unsupported" for item in coverage),
                sum(
                    len(item["case"]["ladder"].split())
                    for item in matrix["requested_cases"]
                ),
                sum(
                    len(item["case"]["ladder"].split())
                    for item in matrix["requested_cases"]
                    if item["disposition"] == "runnable"
                ),
                sum(
                    len(item["case"]["ladder"].split())
                    for item in matrix["requested_cases"]
                    if item["disposition"] == "unsupported"
                ),
            ),
            (58, 608, 364, 244, 1600, 940, 660),
        )
        library: dict[tuple, set[str]] = {}
        system: dict[tuple, set[str]] = {}
        routing: dict[tuple, list[tuple[str, bool]]] = {}
        for requested in matrix["requested_cases"]:
            if requested["disposition"] != "runnable":
                continue
            case = requested["case"]
            shape = tuple(
                case[field]
                for field in (
                    "workload", "mode", "hidden", "topk", "experts", "ep", "phase"
                )
            )
            route = (case["routing"], case["eplb"])
            if case["backend"] != "nccl-ep":
                library.setdefault((requested["sku"], shape, route), set()).add(
                    case["backend"]
                )
            else:
                system.setdefault((shape, route), set()).add(requested["sku"])
            routing.setdefault(
                (requested["sku"], case["backend"], shape), []
            ).append(route)
        anchors = {("uniform", False), ("zipf", False), ("zipf", True)}
        self.assertEqual(
            {
                "library": sum(len(variants) >= 2 for variants in library.values()),
                "system": sum(len(variants) >= 2 for variants in system.values()),
                "routing": sum(
                    len(variants) == 3 and set(variants) == anchors
                    for variants in routing.values()
                ),
            },
            publisher.REQUIRED_PROMOTION_COHORT_COUNTS,
        )

    def test_build_promotion_requires_canonical_full_matrix(self) -> None:
        bundles = {
            str(run_id): {
                "id": str(run_id), "cases": [],
                "manifest": {
                    "matrix": {"sha256": "d" * 64},
                    "run": {"run_id": str(run_id), "run_attempt": 1},
                },
            }
            for run_id in range(1, 4)
        }
        with mock.patch.object(
            publisher, "load_bundle", side_effect=lambda _, bundle_id: bundles[bundle_id]
        ), self.assertRaisesRegex(publisher.PublisherError, "canonical full-v1 matrix"):
            publisher.build_dataset(object(), list(bundles), promote=True)

    def test_rejection_updates_latest_but_never_dev_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            store = publisher.Store(root / "store")
            sentinel = b"existing-promoted-pointer\n"
            (store.channels / "dev-latest.json").write_bytes(sentinel)
            (artifact / "unknown.json").write_text('{"format":"unknown"}')
            with self.assertRaises(publisher.PublisherError):
                publisher.ingest_command(_args(store.root, matrix, artifact))
            self.assertEqual((store.channels / "dev-latest.json").read_bytes(), sentinel)
            pointer = store.verify_channel("latest-attempt")
            dataset = publisher.strict_load(store.public / pointer["dataset"]["path"])
            self.assertEqual(dataset["promotion"]["status"], "quarantined")
            self.assertTrue(any(store.quarantine.iterdir()))

    def test_repeated_rejection_is_content_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            store = publisher.Store(root / "store")
            (artifact / "unknown.json").write_text('{"format":"unknown"}')
            with self.assertRaises(publisher.PublisherError):
                publisher.ingest_command(_args(store.root, matrix, artifact))
            pointer = (store.channels / "latest-attempt.json").read_bytes()
            counts = tuple(
                len(list(path.iterdir()))
                for path in (store.incoming, store.quarantine, store.datasets)
            )
            with self.assertRaises(publisher.PublisherError):
                publisher.ingest_command(_args(store.root, matrix, artifact))
            self.assertEqual((store.channels / "latest-attempt.json").read_bytes(), pointer)
            self.assertEqual(
                tuple(
                    len(list(path.iterdir()))
                    for path in (store.incoming, store.quarantine, store.datasets)
                ),
                counts,
            )

    def test_distinct_rejections_advance_latest_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            store = publisher.Store(root / "store")
            unknown = artifact / "unknown.json"
            unknown.write_text('{"format":"unknown-one"}')
            with self.assertRaises(publisher.PublisherError):
                publisher.ingest_command(_args(store.root, matrix, artifact))
            first = (store.channels / "latest-attempt.json").read_bytes()
            unknown.write_text('{"format":"unknown-two"}')
            with self.assertRaises(publisher.PublisherError):
                publisher.ingest_command(_args(store.root, matrix, artifact))
            second = (store.channels / "latest-attempt.json").read_bytes()
            self.assertNotEqual(second, first)
            self.assertEqual(len(list(store.datasets.iterdir())), 2)

    def test_zip_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.json", "{}")
            with self.assertRaisesRegex(publisher.PublisherError, "escapes"):
                publisher.extract_archive(archive, root / "out")

    def test_store_and_directory_archive_reject_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(publisher.PublisherError, "symlinked parent"):
                publisher.Store(alias / "store")
            self.assertFalse((real / "store").exists())
            artifact = root / f"cxunsupported-{RUN['run_id']}-{RUN['run_attempt']}"
            artifact.mkdir()
            target = root / "target.json"
            target.write_text("{}")
            (artifact / "linked.json").symlink_to(target)
            with self.assertRaisesRegex(publisher.PublisherError, "symlink"):
                publisher._archive_download_directory(artifact, root / "artifact.zip")

    def test_offline_caller_metadata_is_validated_before_store_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            store_root = root / "store"
            args = _args(store_root, matrix, artifact)
            args.run_id = "0"
            with self.assertRaisesRegex(publisher.PublisherError, "run-id"):
                publisher.ingest_command(args)
            self.assertFalse(store_root.exists())

            promote = types.SimpleNamespace(
                store_root=str(store_root), bundle=["not-a-digest"]
            )
            with self.assertRaisesRegex(publisher.PublisherError, "bundle IDs"):
                publisher.promote_command(promote)
            self.assertFalse(store_root.exists())
            with self.assertRaisesRegex(publisher.PublisherError, "absolute path"):
                publisher._store_from_args(types.SimpleNamespace(store_root="relative-store"))

    def test_store_rejects_group_or_world_writable_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "unsafe-store"
            root.mkdir()
            root.chmod(0o772)
            with self.assertRaisesRegex(publisher.PublisherError, "group/world writable"):
                publisher.Store(root)

    def test_retry_ordinals_must_be_contiguous_from_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root, (1, 3))
            with self.assertRaisesRegex(publisher.PublisherError, "contiguous ordinals"):
                publisher.ingest_command(_args(root / "store", matrix, artifact))

    def test_delivery_rejects_extra_archive_and_non_native_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            extra = root / f"cxshard-extra-{RUN['run_id']}-{RUN['run_attempt']}"
            extra.mkdir()
            (extra / "extra.json").write_text("{}")
            args = _args(root / "store-extra", matrix, artifact)
            args.artifact.append(str(extra))
            with self.assertRaisesRegex(publisher.PublisherError, "archive set"):
                publisher.ingest_command(args)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            (artifact / "notes.txt").write_text("not native evidence")
            with self.assertRaisesRegex(publisher.PublisherError, "unconsumed"):
                publisher.ingest_command(_args(root / "store-member", matrix, artifact))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            matrix, artifact = _unsupported_delivery(root)
            path = next(artifact.glob("*.json"))
            terminal = json.loads(path.read_text())
            terminal["outcome"]["reason"] = next(
                reason for reason in contracts.CAPABILITY_FAILURE_REASONS
                if reason != terminal["outcome"]["reason"]
            )
            path.write_text(json.dumps(terminal))
            with self.assertRaisesRegex(publisher.PublisherError, "reason differs"):
                publisher.ingest_command(_args(root / "store-reason", matrix, artifact))

    def test_rates_invert_latency_and_global_tokens_use_ep_size(self) -> None:
        dataset = _dataset()
        publisher.validate_public_dataset(dataset)
        rates = dataset["series"][0]["points"][0]["components"]["roundtrip"]["logical_payload_rate_gbps_at_latency_percentile"]
        self.assertGreater(rates["p50"], rates["p99"])
        broken = copy.deepcopy(dataset)
        broken["series"][0]["points"][0]["global_tokens"] = 128
        with self.assertRaisesRegex(publisher.PublisherError, "EP size"):
            publisher.validate_public_dataset(broken)
        broken = copy.deepcopy(dataset)
        broken["series"][0]["points"][0]["roundtrip_token_rate_at_latency_percentile"]["p99"] *= 2
        with self.assertRaisesRegex(publisher.PublisherError, "token throughput"):
            publisher.validate_public_dataset(broken)
        broken = copy.deepcopy(dataset)
        broken["attempts"][0]["evidence"][0]["point_id"] = identity.point_id(
            series=broken["series"][0]["series_id"], tokens_per_rank=16
        )
        with self.assertRaisesRegex(publisher.PublisherError, "point evidence"):
            publisher.validate_public_dataset(broken)
        broken = copy.deepcopy(dataset)
        broken["attempts"][0]["series_id"] = None
        with self.assertRaisesRegex(publisher.PublisherError, "present exactly for selected success"):
            publisher.validate_public_dataset(broken)
        broken = copy.deepcopy(dataset)
        component = broken["series"][0]["points"][0]["components"]["roundtrip"]
        component["logical_bytes"] = None
        component["logical_payload_rate_gbps_at_latency_percentile"] = None
        with self.assertRaisesRegex(publisher.PublisherError, "logical bandwidth is missing"):
            publisher.validate_public_dataset(broken)

        for mutate in (
            lambda item: item.update({"model": "different-model"}),
            lambda item: item["workload"].update({"hidden": 4096}),
            lambda item: item["workload"].update({"top_k": 4}),
            lambda item: item["workload"].update({"experts": 128}),
        ):
            broken = copy.deepcopy(dataset)
            mutate(broken["series"][0])
            with self.assertRaisesRegex(publisher.PublisherError, "frozen v1"):
                publisher.validate_public_dataset(broken)

        broken = copy.deepcopy(dataset)
        broken["series"][0]["eplb"]["mapping_sha256"] = "f" * 64
        with self.assertRaisesRegex(publisher.PublisherError, "claims a plan"):
            publisher.validate_public_dataset(broken)

        broken = copy.deepcopy(dataset)
        broken["series"][0]["backend"].update({
            "id": "nccl-ep", "label": publisher.BACKEND_LABELS["nccl-ep"],
            "role": "reference", "generation": "rccl",
        })
        broken["coverage"][0]["backend"] = "nccl-ep"
        with self.assertRaisesRegex(publisher.PublisherError, "configuration"):
            publisher.validate_public_dataset(broken)

    def test_public_coverage_binds_exact_topology_and_case_identity(self) -> None:
        dataset = _promoted_dataset()
        dataset["promotion"]["status"] = "diagnostic"
        self.assertEqual(
            {item["disposition"] for item in dataset["coverage"]},
            {"runnable", "unsupported"},
        )
        for item in dataset["coverage"]:
            self.assertEqual(
                tuple(item["topology"]), publisher.COVERAGE_TOPOLOGY_FIELDS
            )
        publisher.validate_public_dataset(dataset)

        broken = copy.deepcopy(dataset)
        unsupported = next(
            item for item in broken["coverage"]
            if item["disposition"] == "unsupported"
        )
        unsupported["topology"]["nodes"] = 2
        with self.assertRaisesRegex(publisher.PublisherError, "capability registry"):
            publisher.validate_public_dataset(broken)

        broken = copy.deepcopy(dataset)
        unsupported = next(
            item for item in broken["coverage"]
            if item["disposition"] == "unsupported"
        )
        unsupported["sku"] = "mi325x"
        topology = publisher.capability.topology_for("mi325x", 8)
        self.assertIsNotNone(topology)
        unsupported["topology"] = publisher._coverage_topology({
            "ep_size": 8, **topology,
        })
        with self.assertRaisesRegex(publisher.PublisherError, "case identity"):
            publisher.validate_public_dataset(broken)

    def test_cohort_contract_and_labels_name_mode_explicitly(self) -> None:
        dataset = _promoted_dataset()
        dataset["promotion"]["status"] = "diagnostic"
        publisher.validate_public_dataset(dataset)
        for cohort in dataset["cohorts"]:
            self.assertIn("mode", cohort["controlled_factors"])
            self.assertIn("/ normal /", cohort["label"])

        broken = copy.deepcopy(dataset)
        cohort = broken["cohorts"][0]
        cohort["controlled_factors"].remove("mode")
        cohort["cohort_id"] = publisher._derived_id("cxcohort-v1-", {
            "kind": cohort["kind"], "series_ids": cohort["series_ids"],
            "controlled_factors": cohort["controlled_factors"],
            "varying_factors": cohort["varying_factors"],
        })
        with self.assertRaisesRegex(publisher.PublisherError, "cohort factors"):
            publisher.validate_public_dataset(broken)

    def test_routing_and_eplb_facts_must_match_across_repeats(self) -> None:
        raw, _ = _native_fixture()
        descriptor = publisher._eplb_descriptor(raw)
        facts = publisher._routing_facts(raw["measurement"]["rows"][0])
        self.assertEqual(
            publisher._exact_repeat_value([descriptor, copy.deepcopy(descriptor)], "EPLB"),
            descriptor,
        )
        self.assertEqual(
            publisher._exact_repeat_value([facts, copy.deepcopy(facts)], "routing"),
            facts,
        )
        changed = copy.deepcopy(facts)
        changed["hotspot_ratio"] += 0.1
        with self.assertRaisesRegex(publisher.PublisherError, "routing differs"):
            publisher._exact_repeat_value([facts, changed], "routing")

        dataset = _promoted_dataset()
        dataset["promotion"]["status"] = "diagnostic"
        eplb = next(item for item in dataset["series"] if item["eplb"]["enabled"])
        eplb["points"][0]["routing"]["empty_expert_count"] = 280
        publisher.validate_public_dataset(dataset)
        eplb["points"][0]["routing"]["empty_expert_count"] = 288
        with self.assertRaisesRegex(publisher.PublisherError, "routing/load facts"):
            publisher.validate_public_dataset(dataset)

        for field, value in (
            ("mapping_sha256", "0" * 64),
            ("redundant_experts", 31),
            ("replicated_experts", 1),
            ("max_replicas", 2),
            ("replicated_experts", 257),
            ("max_replicas", 999),
            ("imbalance_after", 0.4),
            ("planner", "different-planner"),
            ("reference_tokens_per_rank", 1024),
        ):
            broken = _promoted_dataset()
            broken["promotion"]["status"] = "diagnostic"
            descriptor = next(
                item["eplb"] for item in broken["series"] if item["eplb"]["enabled"]
            )
            descriptor[field] = value
            with self.subTest(eplb_field=field), self.assertRaisesRegex(
                publisher.PublisherError, "EPLB descriptor"
            ):
                publisher.validate_public_dataset(broken)

    def test_publisher_owns_stable_rankings_and_recommendations(self) -> None:
        fast, fast_internal = _series("fast", "deepep", decision_grade=True)
        slow, slow_internal = _series("slow", "uccl", decision_grade=True)
        reference, reference_internal = _series("reference", "nccl-ep", decision_grade=True)
        reference_peer, reference_peer_internal = _series(
            "reference-peer", "nccl-ep", decision_grade=True
        )
        reference["backend"]["role"] = "reference"
        reference_peer["backend"]["role"] = "reference"
        reference_peer["system"].update({"sku": "h200-dgxc", "label": "NVIDIA H200"})
        cohorts, rankings, recommendations, _ = publisher.build_decisions(
            [fast, slow, reference, reference_peer], {
                fast["series_id"]: fast_internal,
                slow["series_id"]: slow_internal,
                reference["series_id"]: reference_internal,
                reference_peer["series_id"]: reference_peer_internal,
            }
        )
        library = next(item for item in cohorts if item["kind"] == "library")
        ranking = next(item for item in rankings if item["cohort_id"] == library["cohort_id"]
                       and item["metric"]["measure"] == "latency_us"
                       and item["metric"]["statistic"] == "p99")
        self.assertTrue(library["eligibility"]["decision_grade"])
        self.assertEqual(ranking["entries"][0]["series_id"], fast["series_id"])
        self.assertTrue(any(item["series_id"] == fast["series_id"] for item in recommendations))
        self.assertFalse(any(
            entry["series_id"] == reference["series_id"]
            for item in rankings if item["cohort_id"] == library["cohort_id"]
            for entry in item["entries"]
        ))
        self.assertTrue(any(
            item["kind"] == "system" and reference["series_id"] in item["series_ids"]
            for item in cohorts
        ))

    def test_routing_evidence_is_experimental_and_not_a_configuration_recommendation(self) -> None:
        dataset = _promoted_dataset()
        routing = next(item for item in dataset["cohorts"] if item["kind"] == "routing")
        members = [
            item for item in dataset["series"]
            if item["series_id"] in routing["series_ids"]
        ]
        self.assertEqual(
            {(item["workload"]["routing"], item["workload"]["eplb"]) for item in members},
            {("uniform", False), ("zipf", False), ("zipf", True)},
        )
        self.assertIn("implementation-static-build", routing["controlled_factors"])
        self.assertIn("resource", routing["controlled_factors"])
        self.assertEqual(
            routing["varying_factors"],
            ["workload.routing", "workload.eplb", "implementation-config"],
        )
        self.assertEqual(
            len({item["build"]["routing_control_sha256"] for item in members}),
            1,
        )
        self.assertGreater(
            len({item["build"]["implementation_contract_sha256"] for item in members}),
            1,
        )
        self.assertEqual(len({json.dumps(item["resource"], sort_keys=True) for item in members}), 1)
        self.assertEqual(routing["publication_tier"], "comparable-experimental")
        self.assertTrue(any(
            item["cohort_id"] == routing["cohort_id"] for item in dataset["rankings"]
        ))
        self.assertFalse(any(
            item["cohort_id"] == routing["cohort_id"] for item in dataset["recommendations"]
        ))
        self.assertTrue(all(
            item["publication_tier"] == "official"
            for item in dataset["recommendations"]
        ))
        self.assertFalse(any(
            dataset_cohort["publication_tier"] == "comparable-experimental"
            and item["cohort_id"] == dataset_cohort["cohort_id"]
            for item in dataset["recommendations"]
            for dataset_cohort in dataset["cohorts"]
        ))
        self.assertTrue(all(
            item["publication_tier"] == "comparable-experimental"
            for item in dataset["sensitivities"]
            if item["cohort_id"] == routing["cohort_id"]
        ))

    def test_routing_implementation_mismatch_blocks_all_decisions(self) -> None:
        dataset = _promoted_dataset()
        published = next(item for item in dataset["cohorts"] if item["kind"] == "routing")
        members = [
            item for item in dataset["series"]
            if item["series_id"] in published["series_ids"]
        ]
        zipf = next(
            item for item in members
            if item["workload"]["routing"] == "zipf" and not item["workload"]["eplb"]
        )
        zipf["build"]["implementation_contract_sha256"] = "f" * 64
        internals = {}
        for member in members:
            point = member["points"][0]
            roundtrip = point["components"]["roundtrip"]
            metrics = {
                "latency_us": {
                    name: roundtrip["latency_us"][name] for name in ("p50", "p99")
                },
                "logical_payload_rate_gbps_at_latency_percentile": {
                    name: roundtrip[
                        "logical_payload_rate_gbps_at_latency_percentile"
                    ][name]
                    for name in ("p50", "p99")
                },
            }
            internals[member["series_id"]] = {
                "run_metrics": {
                    str(run): {point["tokens_per_rank"]: metrics}
                    for run in range(3)
                }
            }
        cohorts, rankings, recommendations, sensitivities = publisher.build_decisions(
            members, internals
        )
        routing = next(item for item in cohorts if item["kind"] == "routing")
        self.assertFalse(routing["eligibility"]["decision_grade"])
        self.assertIn(
            "implementation-config-mismatch", routing["eligibility"]["reasons"]
        )
        self.assertEqual((rankings, recommendations, sensitivities), ([], [], []))

    def test_promoted_series_fields_are_bound_to_case_and_series_identities(self) -> None:
        dataset = _promoted_dataset()
        changed = copy.deepcopy(dataset)
        series = next(
            item for item in changed["series"]
            if item["system"]["sku"] == "h100-dgxc"
        )
        series["system"].update({
            "sku": "h200-dgxc", "label": "NVIDIA H200",
            "topology_class": "h200-nvlink-island",
        })
        for case_id in series["case_ids"]:
            coverage = next(
                item for item in changed["coverage"] if item["case_id"] == case_id
            )
            coverage["sku"] = "h200-dgxc"
            coverage["topology"] = publisher._coverage_topology(series["system"])
        with self.assertRaisesRegex(publisher.PublisherError, "configuration|case identity"):
            publisher.validate_public_dataset(changed)

        for field, value in (
            ("source_sha", "b" * 40),
            ("image_digest", "sha256:" + "4" * 64),
            ("squash_sha256", "5" * 64),
            ("runtime_fingerprint_sha256", "6" * 64),
            ("implementation_contract_sha256", "7" * 64),
            ("public_config_sha256", "9" * 64),
            ("routing_control_sha256", "8" * 64),
        ):
            changed = copy.deepcopy(dataset)
            changed["series"][0]["build"][field] = value
            with self.subTest(build_field=field), self.assertRaisesRegex(
                publisher.PublisherError, "commit"
            ):
                publisher.validate_public_dataset(changed)
        changed = copy.deepcopy(dataset)
        changed["series"][0]["workload"]["workload_id"] = identity.workload_id(
            {"changed": True}
        )
        with self.assertRaisesRegex(publisher.PublisherError, "committed factors"):
            publisher.validate_public_dataset(changed)

        for mutate, message in (
            (lambda item: item["backend"].update({
                "generation": "fabricated", "version": "fabricated-999",
            }), "configuration"),
            (lambda item: item["resource"].update({
                "profile": "profile-fabricated", "configured_units": 99,
            }), "configuration"),
            (lambda item: item["system"].update({"label": "Fabricated H100"}), "commit"),
        ):
            changed = copy.deepcopy(dataset)
            mutate(changed["series"][0])
            with self.assertRaisesRegex(publisher.PublisherError, message):
                publisher.validate_public_dataset(changed)

        diagnostic = _dataset()
        diagnostic["series"][0]["build"]["source_sha"] = "b" * 40
        with self.assertRaisesRegex(publisher.PublisherError, "committed factors"):
            publisher.validate_public_dataset(diagnostic)

    def test_all_decision_metrics_require_stable_repeat_ordering(self) -> None:
        fast, fast_internal = _series("ordering-fast", "deepep", decision_grade=True)
        slow, slow_internal = _series("ordering-slow", "uccl", decision_grade=True)
        internals = {
            fast["series_id"]: fast_internal,
            slow["series_id"]: slow_internal,
        }

        cohorts, rankings, recommendations, _ = publisher.build_decisions(
            [fast, slow], internals
        )
        library = next(item for item in cohorts if item["kind"] == "library")
        self.assertTrue(library["eligibility"]["decision_grade"])
        self.assertEqual(
            len([item for item in rankings if item["cohort_id"] == library["cohort_id"]]),
            4,
        )
        self.assertEqual(
            len([
                item for item in recommendations
                if item["cohort_id"] == library["cohort_id"]
            ]),
            4,
        )

        for statistic in ("p50", "p99"):
            slow_internal["run_metrics"]["1"][8]["logical_payload_rate_gbps_at_latency_percentile"][statistic] = (
                fast_internal["run_metrics"]["1"][8]["logical_payload_rate_gbps_at_latency_percentile"][statistic] * 2
            )
        cohorts, rankings, recommendations, _ = publisher.build_decisions(
            [fast, slow], internals
        )
        library = next(item for item in cohorts if item["kind"] == "library")
        self.assertFalse(library["eligibility"]["decision_grade"])
        self.assertIn("unstable-ordering", library["eligibility"]["reasons"])
        self.assertFalse(any(
            item["cohort_id"] == library["cohort_id"] for item in rankings
        ))
        self.assertFalse(any(
            item["cohort_id"] == library["cohort_id"] for item in recommendations
        ))

    def test_extra_eligibility_reason_blocks_decision_grade(self) -> None:
        allocations = [identity.allocation_id({"run": run}) for run in range(3)]
        eligibility = publisher._eligibility_record(
            allocations, complete=True, correct=True, measured=True,
            stable_ordering=True, p50_ratio=1.01, p99_ratio=1.02,
            extra_reasons=["incomplete-provenance"],
        )
        self.assertFalse(eligibility["decision_grade"])
        self.assertEqual(eligibility["reasons"], ["incomplete-provenance"])
        self.assertIs(publisher._eligibility(eligibility, "fixture"), eligibility)
        broken = {**eligibility, "decision_grade": True}
        with self.assertRaisesRegex(publisher.PublisherError, "promotion gates"):
            publisher._eligibility(broken, "fixture")

    def test_schema_is_strict_and_channel_target_must_be_complete(self) -> None:
        dataset = _dataset()
        dataset["unexpected"] = True
        with self.assertRaises(publisher.PublisherError):
            publisher.validate_public_dataset(dataset)
        with mock.patch.object(publisher, "MAX_PUBLIC_DATASET_BYTES", 1), self.assertRaisesRegex(
            publisher.PublisherError, "serving size limit"
        ):
            publisher.validate_public_dataset(_dataset())
        with tempfile.TemporaryDirectory() as temporary:
            store = publisher.Store(Path(temporary).resolve())
            dataset = _dataset()
            digest, size = store.install_dataset(dataset)
            store.update_channel("latest-attempt", digest, size, dataset["generated_at"])
            self.assertEqual(store.verify_channel("latest-attempt")["dataset"]["sha256"], digest)
            channel_path = store.channels / "latest-attempt.json"
            pointer = publisher.strict_load(channel_path)
            pointer["generated_at"] = "2099-01-01T00:00:00Z"
            channel_path.write_bytes(contracts.canonical_json_bytes(pointer))
            with self.assertRaisesRegex(publisher.PublisherError, "metadata differs"):
                store.verify_channel("latest-attempt")
            store.update_channel("latest-attempt", digest, size, dataset["generated_at"])
            with self.assertRaisesRegex(publisher.PublisherError, "metadata differs"):
                store.update_channel(
                    "latest-attempt", digest, size + 1, dataset["generated_at"]
                )
            with self.assertRaisesRegex(publisher.PublisherError, "metadata differs"):
                store.update_channel(
                    "latest-attempt", digest, size, "2026-07-05T00:00:00Z"
                )
            os.chmod(channel_path, 0o666)
            with self.assertRaisesRegex(publisher.PublisherError, "regular 644"):
                store.verify_channel("latest-attempt")
            os.chmod(channel_path, 0o644)
            dataset_dir = store.datasets / digest
            os.chmod(dataset_dir, 0o755)
            with self.assertRaisesRegex(publisher.PublisherError, "mode differs"):
                store.verify_channel("latest-attempt")
            os.chmod(dataset_dir, 0o555)
            os.chmod(dataset_dir / "dataset.json", 0o644)
            with self.assertRaisesRegex(publisher.PublisherError, "mode differs"):
                store.verify_channel("latest-attempt")
            os.chmod(dataset_dir / "dataset.json", 0o444)
            os.chmod(dataset_dir, 0o755)
            (dataset_dir / "COMPLETE").unlink()
            os.chmod(dataset_dir, 0o555)
            with self.assertRaisesRegex(publisher.PublisherError, "incomplete"):
                store.verify_channel("latest-attempt")

    def test_store_modes_do_not_depend_on_process_umask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.umask(0o077)
            try:
                store = publisher.Store(Path(temporary).resolve())
                dataset = _dataset()
                digest, size = store.install_dataset(dataset)
                store.update_channel(
                    "latest-attempt", digest, size, dataset["generated_at"]
                )
                with store.locked():
                    pass
            finally:
                os.umask(previous)
            self.assertEqual(
                store.root.stat().st_mode & 0o777,
                0o750,
            )
            self.assertEqual(
                (store.channels / "latest-attempt.json").stat().st_mode & 0o777,
                0o644,
            )
            self.assertEqual(
                (store.datasets / digest / "dataset.json").stat().st_mode & 0o777,
                0o444,
            )
            self.assertEqual(
                (store.locks / "publisher.lock").stat().st_mode & 0o777,
                0o600,
            )

    def test_verify_requires_bootstrap_but_dev_latest_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args = types.SimpleNamespace(
                store_root=str(root / "store"), channel=None, bundle=[]
            )
            with self.assertRaises(publisher.PublisherError):
                publisher.verify_command(args)
            store = publisher.Store(args.store_root)
            dataset = publisher._quarantine_dataset(
                "awaiting-v1-runs", "2026-07-04T00:00:00Z"
            )
            digest, size = store.install_dataset(dataset)
            store.update_channel(
                "latest-attempt", digest, size, "2026-07-04T00:00:00Z"
            )
            result = publisher.verify_command(args)
            self.assertEqual(set(result["channels"]), {"latest-attempt"})
            explicit = types.SimpleNamespace(
                store_root=args.store_root, channel=["dev-latest"], bundle=[]
            )
            with self.assertRaises(publisher.PublisherError):
                publisher.verify_command(explicit)
            dev_pointer = copy.deepcopy(store.verify_channel("latest-attempt"))
            dev_pointer["channel"] = "dev-latest"
            (store.channels / "dev-latest.json").write_bytes(
                contracts.canonical_json_bytes(dev_pointer)
            )
            with self.assertRaisesRegex(publisher.PublisherError, "non-promoted"):
                publisher.verify_command(args)


if __name__ == "__main__":
    unittest.main()
