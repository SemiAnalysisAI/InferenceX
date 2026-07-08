"""End-to-end raw-document contract test (no GPU, no torch).

`test_ep_backend.py` exercises the mocked benchmark up to the measurement rows, and
`test_schema_v1_contract.py` asserts facts about the schema files, but nothing drove a
complete raw attempt document plus its detached samples through
`contracts.validate_raw_document`. That left the emitter tail of `ep_harness.run_sweep`
(the doc assembly that only runs on a real allocation) unguarded: a field the emitter
writes but the frozen schema forbids, or vice versa, would only surface on a live sweep.

This builds a document whose field sets mirror `ep_harness.run_sweep`'s assembly exactly,
using the real `identity` module for every typed id and the real
`identity.profile_for_case` for the case profile, then validates it. The negative cases
pin the drift fields that were removed from the emitter so a re-introduction fails here
instead of on the cluster.
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import identity  # noqa: E402
import contracts  # noqa: E402

_HEX = "ab" * 32  # a syntactically valid lowercase sha256 digest (64 hex)
_SHA40 = "abcdef0123456789abcdef0123456789abcdef01"  # a git-style 40-hex sha
_WORKLOAD_ID = identity.workload_id({"name": "deepseek-v3-v1", "trace": _HEX})

# A representative EP8 H100 scheduled case, identical in shape to what sweep_matrix
# materializes and ep_harness rebuilds from args (no case_id key: it is computed from
# these factors).
SCHEDULED_CASE = {
    "backend": "deepep",
    "canonical": True,
    "ep": 8,
    "eplb": False,
    "experts": 256,
    "gpus_per_node": 8,
    "hidden": 7168,
    "ladder": "1 2 4 8 16 32 64 128",
    "mode": "normal",
    "nodes": 1,
    "phase": "decode",
    "routing": "uniform",
    "samples_per_point": 512,
    "scale_out_transport": None,
    "scale_up_domain": 8,
    "scale_up_transport": "nvlink",
    "scope": "scale-up",
    "suite": "ep-core-v1",
    "timing": "8:64:32",
    "topk": 8,
    "topology_class": "h100-nvlink-island",
    "transport": "nvlink",
    "warmup_semantics": "full-roundtrip-before-each-component-trial-point-v1",
    "workload": "deepseek-v3-v1",
}


def _percentiles():
    return {"p50": 10.0, "p90": 20.0, "p95": 25.0, "p99": 30.0}


def _component():
    return {
        "availability": "measured",
        "origin": "measured",
        "percentiles_us": _percentiles(),
        "sample_count": 512,
    }


def _samples_component():
    # 64 trials x 8 iterations of non-negative numbers, per samples-v1 trials def.
    return {
        "availability": "measured",
        "sample_count": 512,
        "trials": [[1.0] * 8 for _ in range(64)],
    }


def _byte_accounting():
    return {
        "accounting_contract": "activation-data-plus-scales-v1",
        "activation_data_bytes": 1024,
        "scale_bytes": 0,
        "total_logical_bytes": 1024,
    }


def _histogram():
    return {"bins": 4, "counts": [1, 2, 3, 4], "max": 30.0, "min": 10.0, "n": 10}


def _oracle():
    return {
        "atol": 0.02,
        "checks": {
            "combine_values": True,
            "counts": True,
            "metadata": True,
            "multiplicity": True,
            "payload": True,
            "source_set": True,
            "weights": True,
        },
        "combine_weight_semantics": "unweighted-rank-sum",
        "contract": "expert-specific-transform-v1",
        "dispatch_sha256": None,
        "max_absolute_error": 0.0,
        "max_elementwise_relative_error": 0.0,
        "max_relative_error": 0.0,
        "max_weight_error": None,
        "order_sha256": None,
        "ordering_contract": "roundtrip-first-then-components-v1",
        "passed": True,
        "receive_count": 8,
        "rtol": 0.05,
    }


def _routing():
    return {
        "empty_expert_count": 0,
        "empty_rank_count": 0,
        "expert_assignment_rank_cv": 0.0,
        "expert_assignments_per_rank": [32, 32, 32, 32, 32, 32, 32, 32],
        "expert_load_cv": 0.0,
        "expert_load_max": 8,
        "expert_load_mean": 8.0,
        "expert_load_min": 8,
        "fanout_histogram": [0, 8],
        "fanout_max": 1,
        "fanout_mean": 1.0,
        "fanout_min": 1,
        "hash": _HEX,
        "hotspot_ratio": 1.0,
        "locality": None,
        "payload_copies_per_rank": [8, 8, 8, 8, 8, 8, 8, 8],
        "payload_rank_cv": 0.0,
        "routed_copies": 64,
        "source_token_stats": None,
    }


def _correctness():
    return {
        "contract": "expert-specific-transform-v1",
        "max_relative_error": 0.0,
        "passed": True,
        "rank_evidence": [
            {
                "input_unchanged": True,
                "order_stable": True,
                "post_timing": _oracle(),
                "pre_timing": _oracle(),
                "rank": 0,
            }
        ],
        "scope": "dispatch-metadata-and-transformed-combine",
    }


def _build_documents():
    """Return (raw_document, samples_document) with fully reconciled identities."""
    profile = identity.profile_for_case({"mode": SCHEDULED_CASE["mode"]})
    case_factors = {"case": SCHEDULED_CASE, "profile": profile, "sku": "h100-dgxc"}
    case_id = identity.digest("case", case_factors)

    allocation_factors = {
        "artifact": "collectivex-results-h100-dgxc-deepep-n1",
        "execution_id": "exec-0001",
        "job": "sweep",
        "repo": "SemiAnalysisAI/InferenceX",
        "run_attempt": "1",
        "run_id": "100200300",
        "runner": "h100-dgxc",
        "source_sha": _SHA40,
    }
    allocation_id = identity.allocation_id(allocation_factors)
    attempt_ordinal = 1
    attempt_id = identity.attempt_id(
        allocation=allocation_id, case=case_id, ordinal=attempt_ordinal
    )

    series_factors = {
        "backend": "deepep",
        "case_id": case_id,
        "image_digest": "sha256:" + _HEX,
        "source_sha": _SHA40,
        "squash_sha256": _HEX,
        "workload_id": _WORKLOAD_ID,
    }
    series_id = identity.series_id(series_factors)

    rows = []
    sample_points = []
    for token_count in (64, 128):
        sample_sha256 = identity.digest("point", {"t": token_count})[-64:]
        point_id = identity.point_id(series=series_id, tokens_per_rank=token_count)
        evidence_id = identity.evidence_id(
            point=point_id,
            allocation=allocation_id,
            attempt=attempt_id,
            sample_sha256=sample_sha256,
        )
        rows.append({
            "anomalies": [],
            "byte_provenance": {c: _byte_accounting() for c in
                                ("combine", "dispatch", "roundtrip", "stage")},
            "components": {
                "combine": _component(),
                "dispatch": _component(),
                "isolated_sum": _component(),
                "roundtrip": _component(),
                "stage": _component(),
            },
            "correctness": _correctness(),
            "evidence_id": evidence_id,
            "global_tokens": token_count * 8,
            "point_id": point_id,
            "receive": {"max": 8, "mean": 8.0, "min": 8, "total": 64},
            "routing": _routing(),
            "sample_histograms": {
                "combine": None,
                "dispatch": None,
                "roundtrip": _histogram(),
                "stage": None,
            },
            "sample_sha256": sample_sha256,
            "token_rate_at_latency_percentile": _percentiles(),
            "tokens_per_rank": token_count,
        })
        sample_points.append({
            "components": {
                "combine": _samples_component(),
                "dispatch": _samples_component(),
                "roundtrip": _samples_component(),
                "stage": _samples_component(),
            },
            "evidence_id": evidence_id,
            "point_id": point_id,
            "sample_sha256": sample_sha256,
            "tokens_per_rank": token_count,
        })

    samples_document = {
        "allocation_id": allocation_id,
        "attempt_id": attempt_id,
        "case_id": case_id,
        "format": "collectivex.samples.v1",
        "points": sample_points,
        "sampling": {
            "iterations_per_trial": 8,
            "reduction": "cross-rank-max-per-iteration",
            "trials": 64,
        },
        "schema_version": 1,
        "series_id": series_id,
    }

    eplb_record = {
        "calibration_token_offset": None,
        "calibration_trace_sha256": None,
        "calibration_window": None,
        "calibration_workload_id": None,
        "enabled": False,
        "imbalance_after": None,
        "imbalance_before": None,
        "mapping_hash": None,
        "max_replicas": None,
        "num_logical_experts": 256,
        "num_physical_experts": 256,
        "num_redundant": 0,
        "planner": None,
        "reference_tokens_per_rank": None,
        "replicated_experts": 0,
    }

    document = {
        "format": "collectivex.ep.v1",
        "schema_version": 1,
        "record_type": "case-attempt",
        "generated_at": "2026-07-08T00:00:00Z",
        "identity": {
            "allocation_factors": allocation_factors,
            "allocation_id": allocation_id,
            "attempt_id": attempt_id,
            "attempt_ordinal": attempt_ordinal,
            "case_factors": case_factors,
            "case_id": case_id,
            "series_factors": series_factors,
            "series_id": series_id,
        },
        "case": {
            "attempt_ordinal": attempt_ordinal,
            "backend": "deepep",
            "eplb": eplb_record,
            "ep_size": 8,
            "mode": "normal",
            "phase": "decode",
            "resource_mode": "fixed-profile",
            "runner": "h100-dgxc",
            "shape": {
                "activation_profile": "canonical-counter-source-v4",
                "eplb": False,
                "experts": 256,
                "experts_per_rank": 32,
                "hidden": 7168,
                "kernel_gen": "image-pinned-v1",
                "num_logical_experts": 256,
                "routing": "uniform",
                "topk": 8,
            },
            "suite": "ep-core-v1",
            "workload_name": "deepseek-v3-v1",
        },
        "workload": {
            "activation_generator": "collectivex-activation-counter-v4",
            "activation_identity": _HEX,
            "activation_profile": "canonical-counter-source-v4",
            "cross_rank_consistent": True,
            "manifest_checksums": None,
            "members": None,
            "routing_generator": "collectivex-routing-counter-v3",
            "source": "seeded-runtime",
            "trace_hashes": [_HEX],
            "trace_signature": _HEX,
            "workload_id": _WORKLOAD_ID,
        },
        "measurement": {
            "component_order_contract": "qualification-hash-rotated-components-v1",
            "conditioning": {
                "contract": "fixed-phase-ramp-8-roundtrips-v1",
                "ladder": [1, 2, 4, 8, 16, 32, 64, 128],
                "roundtrips_per_shape": 8,
            },
            "contract": "layout-and-dispatch-v1",
            "rows": rows,
            "sampling": {
                "contract": "fixed-512-v1",
                "iterations_per_trial": 8,
                "percentile_method": "nearest-rank",
                "reduction": "cross-rank-max-per-iteration",
                "samples_per_component": 512,
                "trials": 64,
                "warmup_iterations": 32,
                "warmup_semantics": "full-roundtrip-before-each-component-trial-point-v1",
            },
            "source_allocation": "even",
        },
        "implementation": {
            "kernel_generation": "image-pinned-v1",
            "name": "deepep",
            "provenance": {},
            "resource_profile": {
                key: None for key in (
                    "achieved_fraction", "comm_units_kind", "configured_units",
                    "conformance_class", "device_units", "fixed_kernel",
                    "nonconforming", "pareto_eligible", "persistent_bytes",
                    "qps_per_rank", "requested_fraction", "resource_class",
                    "target_achieved_within_tol", "tolerance", "tuned_source",
                    "warps_combine", "warps_dispatch",
                )
            },
        },
        "topology": {
            "device_count": 8,
            "device_product": "NVIDIA H100 80GB HBM3",
            "gpus_per_node": 8,
            "nodes": 1,
            "placement": "packed",
            "realized_placement": {
                "gpus_per_node": 8,
                "nodes": 1,
                "ranks_per_node": 8,
                "unique_local_ranks": True,
                "valid": True,
            },
            "scale_up_domain": 8,
            "scale_up_transport": "nvlink",
            "scale_out_transport": None,
            "scope": "scale-up",
            "topology_class": "h100-nvlink-island",
            "transport": "nvlink",
            "world_size": 8,
        },
        "runtime_fingerprint": {
            "accelerator_runtime": {"kind": "cuda", "version": "12.4"},
            "collective_library": {"kind": "nccl", "version": "2.21.5"},
            "device": {
                "arch": "sm90",
                "compute_units": 132,
                "memory_bytes": 85899345920,
                "product": "NVIDIA H100 80GB HBM3",
                "warp_size": 32,
            },
            "driver_version": "550.54.15",
            "framework": {"kind": "torch", "version": "2.5.1"},
            "machine": "x86_64",
            "python_version": "3.11.9",
            "vendor": "nvidia",
        },
        "provenance": {
            "command": "run_ep.py ...",
            "distributed_launcher": None,
            "git_run": {
                "artifact": "collectivex-results-h100-dgxc-deepep-n1",
                "job": "sweep",
                "ref": "collectivex",
                "repo": "SemiAnalysisAI/InferenceX",
                "run_attempt": "1",
                "run_id": "100200300",
                "source_sha": _SHA40,
            },
            "image": {
                "arch": "x86_64",
                "digest": "sha256:" + _HEX,
                "digest_verified": True,
                "reference": "example/image:tag",
                "squash_sha256": _HEX,
            },
            "redaction": "sanitized-v1",
        },
        "sample_artifact": {
            "bytes": 1,
            "format": "collectivex.samples.v1",
            "path": "case.samples.json",
            "sha256": _HEX,
        },
        "outcome": {
            "reasons": [],
            "status": "success",
            "validity": {
                "anomaly_free": True,
                "execution_status": "complete",
                "measurement_conformance": "conformant",
                "provenance_complete": True,
                "resource_conformance": "conformant",
                "sampling_conformance": "conformant",
                "semantic_correctness": "pass",
                "workload_identity": "consistent-across-ranks",
                "workload_source": "seeded-runtime",
            },
        },
    }
    return document, samples_document


class RawDocumentRoundTripTest(unittest.TestCase):
    def test_representative_raw_document_validates(self):
        document, samples = _build_documents()
        # Must not raise: schema shape + all cross-file identity checks pass.
        result = contracts.validate_raw_document(document, samples)
        self.assertEqual(result["identity"]["case_id"], document["identity"]["case_id"])

    def test_profile_is_bf16_fixed_fact(self):
        profile = identity.profile_for_case({"mode": "normal"})
        self.assertEqual(profile["combine_dtype"], "bf16")
        self.assertEqual(profile["combine_quant_mode"], "none")

    def test_removed_drift_fields_are_rejected(self):
        # Each of these keys was emitted by an earlier build but is forbidden by the
        # frozen schema. Re-introducing any of them must fail closed here.
        base_doc, base_samples = _build_documents()
        cases = [
            ("identity", "allocation_factors", "qualification_index"),
            ("identity", "series_factors", "runtime_fingerprint_sha256"),
            ("measurement", None, "execution_order_sha256"),
            ("measurement", None, "qualification_index"),
            ("provenance", None, "allocation_stratum_sha256"),
            ("provenance", "git_run", "qualification_index"),
            ("case", "shape", "precision_profile"),
            ("case", "shape", "dispatch_precision"),
        ]
        for top, sub, key in cases:
            with self.subTest(field=f"{top}.{sub or ''}.{key}"):
                doc = copy.deepcopy(base_doc)
                target = doc[top] if sub is None else doc[top][sub]
                target[key] = 1
                with self.assertRaises(contracts.ContractError):
                    contracts.validate_raw_document(doc, copy.deepcopy(base_samples))

    def test_samples_qualification_index_is_rejected(self):
        doc, samples = _build_documents()
        samples["qualification_index"] = 1
        with self.assertRaises(contracts.ContractError):
            contracts.validate_raw_document(doc, samples)


if __name__ == "__main__":
    unittest.main()
