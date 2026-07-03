#!/usr/bin/env python3
"""Focused tests for the CollectiveX fixed EP sampling contract."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import ep_harness  # noqa: E402
import aggregate_results as ar  # noqa: E402
import capability  # noqa: E402
import generate_matrix  # noqa: E402
import make_bundle  # noqa: E402
import summarize  # noqa: E402
import sweep_matrix  # noqa: E402
import validate_results as vr  # noqa: E402


def _hist(n: int) -> dict:
    return {"n": n, "min": 1.0, "max": 1.0, "bins": 40, "counts": [n]}


def _doc(iters: int = 8, trials: int = 64, warmup: int = 32, samples: int = 512) -> dict:
    validity = {
        "execution_status": "complete",
        "semantic_correctness": "pass",
        "workload_identity": "consistent-across-ranks",
        "workload_source": "seeded-runtime",
        "measurement_conformance": "conformant",
        "sampling_conformance": "conformant",
        "resource_conformance": "backend-default",
        "provenance_complete": False,
        "anomaly_free": True,
    }
    pcts = {"p50": 1.0, "p90": 1.0, "p95": 1.0, "p99": 1.0}
    return {
        "schema_version": 5,
        "family": "moe",
        "runner": "test-runner",
        "backend": "deepep",
        "mode": "normal",
        "phase": "decode",
        "ep_size": 8,
        "publication_status": "comparable-experimental",
        "measurement_contract": "layout-and-dispatch-v1",
        "shape": {
            "hidden": 7168,
            "topk": 8,
            "experts": 256,
            "experts_per_rank": 32,
            "dispatch_dtype": "bf16",
            "routing": "uniform",
        },
        "validity": validity,
        "workload": {
            "source": "seeded-runtime",
            "workload_id": None,
            "trace_signature": "abc",
            "cross_rank_consistent": True,
        },
        "reproduction": {
            "command": "python3 tests/run_ep.py",
            "seed": 67,
            "measurement_contract": "layout-and-dispatch-v1",
            "sampling_contract": "fixed-512-v1",
            "samples_per_point": samples,
            "iters": iters,
            "trials": trials,
            "warmup": warmup,
            "warmup_semantics": "full-roundtrip-per-trial-point-v1",
        },
        "placement": {
            "kind": "packed", "nodes": 1, "gpus_per_node": 8,
            "scale_up_domain": 8, "ranks": 8,
        },
        "backend_provenance": {},
        "comparison_key": "fixture-comparison-key",
        "anomalies": [],
        "anomaly_summary": {"waived": False},
        "rows": [{
            "tokens_per_rank": 8,
            "global_tokens": 64,
            "samples_pooled": samples,
            "trials": trials,
            "dispatch": dict(pcts),
            "combine": dict(pcts),
            "roundtrip": dict(pcts),
            "isolated_sum": {},
            "byte_contracts": {
                "token_rank_payload_copies": 64,
                "token_expert_payload_copies": 512,
                "dispatch_bytes": 1,
                "combine_bytes": 1,
            },
            "correct": True,
            "raw_samples": {
                "dispatch": _hist(samples),
                "combine": _hist(samples),
                "roundtrip": _hist(samples),
            },
        }],
    }


def _failed(case: dict, generated_at="2026-07-03T00:00:00Z", **fields) -> dict:
    return {
        "schema_version": 5, "family": "moe", "record_type": "failed-case",
        "runner": "h100-dgxc_01", "topology_class": "h100-nvlink-island",
        "backend": case["backend"], "phase": case["phase"],
        "publication_status": "failed", "generated_at": generated_at, "rows": [],
        "failure": {"failure_mode": "timeout", "return_code": 124, "case": case}, **fields,
    }


class SamplingContractTest(unittest.TestCase):
    def test_constants_and_default_profile_match_validator(self) -> None:
        self.assertEqual(ep_harness.SCHEMA_VERSION, 5)
        self.assertEqual(ep_harness.SAMPLING_CONTRACT, vr.SAMPLING_CONTRACT)
        self.assertEqual(ep_harness.TIMED_SAMPLES_PER_POINT, vr.TIMED_SAMPLES_PER_POINT)
        self.assertEqual(ep_harness.TIMED_ITERS_PER_TRIAL, vr.TIMED_ITERS_PER_TRIAL)
        self.assertEqual(ep_harness.TRIALS_PER_POINT, vr.TRIALS_PER_POINT)
        self.assertEqual(ep_harness.WARMUP_ITERS_PER_TRIAL, vr.WARMUP_ITERS_PER_TRIAL)
        self.assertEqual(ep_harness.WARMUP_SEMANTICS, vr.WARMUP_SEMANTICS)
        self.assertIsNone(ep_harness.sampling_contract_error(8, 64, 32))

        parser = argparse.ArgumentParser()
        ep_harness.add_common_args(parser)
        args = parser.parse_args([
            "--runner", "test", "--topology-class", "test-topology", "--out", "result.json",
        ])
        self.assertEqual((args.iters, args.trials, args.warmup), (8, 64, 32))

        schemas = vr.load_schema_registry()
        self.assertEqual(sorted(schemas), [3, 4, 5])
        self.assertIs(schemas[3], schemas[4])
        self.assertEqual(schemas[5]["properties"]["schema_version"]["const"], 5)
        reproduction = schemas[5]["properties"]["reproduction"]["properties"]
        self.assertEqual((reproduction["iters"]["const"], reproduction["trials"]["const"],
                          reproduction["warmup"]["const"]), (8, 64, 32))
        self.assertEqual(reproduction["warmup_semantics"]["const"],
                         "full-roundtrip-per-trial-point-v1")

    def test_non_exact_profiles_are_rejected_even_when_the_product_is_512(self) -> None:
        self.assertIn("got 200:3:32", ep_harness.sampling_contract_error(200, 3, 32))
        self.assertIn("got 8:1:4", ep_harness.sampling_contract_error(8, 1, 4))
        self.assertIn("got 128:4:32", ep_harness.sampling_contract_error(128, 4, 32))
        self.assertIn("got 8:64:4", ep_harness.sampling_contract_error(8, 64, 4))
        self.assertIn("got 0:64:32", ep_harness.sampling_contract_error(0, 64, 32))

    def test_valid_comparison_grade_fixture_passes(self) -> None:
        doc = _doc()
        errors, warnings, status = vr.validate_doc(doc, vr.load_schema_registry(), "fixture.json")
        self.assertEqual(status, "comparable-experimental")
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_tampered_sample_counts_cannot_remain_comparison_grade(self) -> None:
        for mutate in (
            lambda d: d["reproduction"].update(iters=200, trials=3, samples_per_point=600),
            lambda d: d["reproduction"].update(iters=128, trials=4),
            lambda d: d["reproduction"].update(warmup=4),
            lambda d: d["reproduction"].update(warmup_semantics="operation-specific-v0"),
            lambda d: d["rows"][0].update(samples_pooled=600),
            lambda d: d["rows"][0]["raw_samples"]["roundtrip"].update(n=8, counts=[8]),
            lambda d: d["rows"][0]["raw_samples"]["dispatch"].update(counts=[511]),
        ):
            with self.subTest(mutate=mutate):
                doc = copy.deepcopy(_doc())
                mutate(doc)
                errors, _warnings, _status = vr.validate_doc(doc, None, "tampered.json")
                self.assertTrue(any("sampling" in error for error in errors), errors)

    def test_all_sweep_cases_use_the_exact_profile(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "matrix.json")
            proc = subprocess.run(
                [sys.executable, os.path.join(ROOT, "sweep_matrix.py"), "--suites", "all",
                 "--backends", "all", "--out", out],
                cwd=ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            with open(out) as fh:
                matrix = json.load(fh)
        cases = [case for shard in matrix["include"] for case in shard["cases"]]
        self.assertTrue(cases)
        self.assertEqual(len(matrix["include"]), 39)
        self.assertEqual(len(cases), 232)
        points = sum(len(case["ladder"].split()) if case["ladder"] else
                     (8 if case["phase"] == "decode" else 6) for case in cases)
        self.assertEqual(points, 618)
        self.assertEqual({case["timing"] for case in cases}, {"8:64:32"})
        self.assertEqual({case["samples_per_point"] for case in cases}, {512})
        self.assertEqual({case["warmup_semantics"] for case in cases},
                         {"full-roundtrip-per-trial-point-v1"})
        self.assertEqual({shard["sku"] for shard in matrix["include"]},
                         {"b200-dgxc", "b300", "gb200", "gb300", "h100-dgxc", "h200-dgxc",
                          "mi325x", "mi355x"})
        for shard in matrix["include"]:
            platform = capability.PLATFORMS[shard["sku"]]
            self.assertEqual(shard["launcher"], platform["launcher"])
            self.assertEqual(shard["gpus_per_node"], platform["gpus_per_node"])
            self.assertEqual(shard["scale_up_domain"], platform["scale_up_domain"])
            self.assertTrue(all(case["gpus_per_node"] == platform["gpus_per_node"]
                                and case["scale_up_domain"] == platform["scale_up_domain"]
                                for case in shard["cases"]))
            self.assertTrue(os.path.isfile(os.path.join(
                ROOT, "launchers", f"launch_{shard['launcher']}.sh"
            )))
        self.assertEqual({case["suite"] for case in cases}, {"ep-core-v1", "ep-routing-v1"})
        self.assertEqual({case["mode"] for case in cases}, {"normal"})
        self.assertEqual({case["dtype"] for case in cases}, {"bf16"})
        self.assertEqual({case["contract"] for case in cases}, {"layout-and-dispatch-v1"})
        self.assertEqual({case["workload"] for case in cases}, {"deepseek-v3-v1"})
        case_ids = [case["case_id"] for case in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertTrue(all(case_id.startswith("cxv1-") for case_id in case_ids))
        self.assertTrue(all(case["canonical"] for case in cases))
        self.assertTrue(all(not case["eplb"] or case["routing"] == "zipf" for case in cases))

    def test_matrix_uses_public_gha_platform_registry(self) -> None:
        original_load = generate_matrix._load

        def public_load(name: str):
            self.assertNotIn(name, {"platforms.yaml", "backends.yaml"})
            return original_load(name)

        with mock.patch.object(generate_matrix, "_load", side_effect=public_load):
            generated = generate_matrix.generate("ep-core-v1")
        self.assertTrue(generated["cases"])
        suite_platforms = set(
            generate_matrix._load("suites.yaml")["suites"]["ep-core-v1"]["platforms"]
        )
        self.assertLessEqual(suite_platforms, set(capability.PLATFORMS))
        self.assertEqual(
            {case["platform"] for case in generated["cases"]},
            {"h100-dgxc", "h200-dgxc", "b200-dgxc", "b300", "gb200", "gb300", "mi325x", "mi355x"},
        )
        self.assertEqual(
            set(capability.PLATFORMS),
            {"h100-dgxc", "h200-dgxc", "b200-dgxc", "b300", "gb200", "gb300",
             "mi325x", "mi355x"},
        )
        self.assertFalse(capability.resolve("b300", "deepep", mode="ll")[0])
        self.assertFalse(capability.resolve("h200", "deepep")[0])

    def test_backend_ladder_limits_apply_after_backend_expansion(self) -> None:
        self.assertEqual(
            sweep_matrix._resolved_ladder(
                "128 256 512", "prefill", "mori", "uniform", "mi355x"),
            "128 256 512",
        )
        self.assertIsNone(sweep_matrix._resolved_ladder(
            "512 2048", "prefill", "mori", "zipf", "mi355x"))
        self.assertEqual(
            sweep_matrix._resolved_ladder(
                "512 2048", "prefill", "mori", "zipf", "mi325x"),
            "512",
        )
        self.assertEqual(
            sweep_matrix._resolved_ladder(
                "512 2048", "prefill", "nccl-ep", "zipf", "mi355x"),
            "512 2048",
        )

    def test_backend_filter_does_not_add_the_amd_native_backend(self) -> None:
        def selected(option: str, backend: str) -> tuple[set[str], set[str]]:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "matrix.json")
                proc = subprocess.run(
                    [sys.executable, os.path.join(ROOT, "sweep_matrix.py"), "--suites", "all",
                     option, backend, "--out", out],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
                with open(out) as fh:
                    shards = json.load(fh)["include"]
            return ({shard["backend"] for shard in shards}, {shard["sku"] for shard in shards})

        self.assertEqual(selected("--backend", "deepep")[0], {"deepep"})
        self.assertEqual(selected("--backend", "mori"), ({"mori"}, {"mi325x", "mi355x"}))
        backends, skus = selected("--backend", "nccl-ep")
        self.assertEqual(backends, {"nccl-ep"})
        self.assertEqual(skus, set(capability.PLATFORMS))

    def test_official_workloads_require_a_pinned_source(self) -> None:
        suite = {"workloads": ["deepseek-v3-v1"], "required_publication": "official"}
        workloads = {"model_derived": {"deepseek-v3-v1": {"verified_against": "pinned"}}}
        generate_matrix.validate_workloads("core", suite, workloads)
        workloads["model_derived"]["deepseek-v3-v1"].pop("verified_against")
        with self.assertRaises(SystemExit):
            generate_matrix.validate_workloads("core", suite, workloads)

    def test_gradual_conditioning_does_not_expand_scored_ladder(self) -> None:
        scored = [512]
        self.assertEqual(ep_harness.conditioning_ladder(scored, True),
                         [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.assertEqual(scored, [512])

    def test_bundle_coverage_requires_one_result_at_the_required_tier(self) -> None:
        case_id = "cxv1-0123456789abcdefabcd"
        matrix = {"include": [{"cases": [{
            "case_id": case_id, "required_publication": "official",
        }]}]}
        doc = {"family": "moe", "case_id": case_id, "publication_status": "official",
               "required_publication": "official"}
        self.assertEqual(
            make_bundle.validate_expected_coverage([doc], matrix),
            {"expected": 1, "observed": 1, "complete": True},
        )
        with self.assertRaises(SystemExit):
            make_bundle.validate_expected_coverage(
                [{**doc, "publication_status": "comparable-experimental"}], matrix)
        with self.assertRaises(SystemExit):
            make_bundle.validate_expected_coverage([], matrix)
        with self.assertRaises(SystemExit):
            make_bundle.validate_expected_coverage([doc, doc], matrix)

    def test_ep_result_producer_never_inlines_environment_documents(self) -> None:
        path = os.path.join(ROOT, "tests", "ep_harness.py")
        with open(path) as fh:
            tree = ast.parse(fh.read(), path)
        self.assertFalse(any(
            isinstance(node, ast.Constant) and node.value == "environment"
            for node in ast.walk(tree)
        ))

    def test_environment_capture_calls_are_redacted(self) -> None:
        callsites = ("runtime/run_in_container.sh",)
        for relative in callsites:
            with self.subTest(callsite=relative):
                with open(os.path.join(ROOT, relative)) as fh:
                    calls = [line for line in fh if "env_capture.py" in line]
                self.assertTrue(calls)
                self.assertTrue(all("--redact" in line for line in calls))

    def test_flashinfer_retries_preserve_attempt_evidence(self) -> None:
        with open(os.path.join(ROOT, "runtime", "run_in_container.sh")) as fh:
            runtime = fh.read()
        self.assertIn('export CX_ATTEMPT_ID="$a"', runtime)
        self.assertNotIn('rm -f results/failed_', runtime)
        for launcher in ("launch_gb200-nv.sh", "launch_gb300-nv.sh"):
            with open(os.path.join(ROOT, "launchers", launcher)) as fh:
                rack = fh.read()
            self.assertIn('CX_FLASHINFER_RETRIES:-3', rack)
            self.assertIn('export CX_ATTEMPT_ID="$attempt"', rack)
        with open(os.path.join(ROOT, "runtime", "common.sh")) as fh:
            self.assertIn('"attempt_id": env("CX_ATTEMPT_ID", "1")', fh.read())

    def test_rack_build_only_uses_shared_backend_preparation(self) -> None:
        with open(os.path.join(ROOT, "runtime", "run_in_container.sh")) as fh:
            runtime = fh.read()
        self.assertIn("cx_prepare_backend()", runtime)
        self.assertIn('cx_prepare_backend "${CX_BENCH:-}"', runtime)
        self.assertIn("cx_persist_backend_env", runtime)

    def test_uccl_build_is_idempotent_within_a_shard(self) -> None:
        with open(os.path.join(ROOT, "runtime", "run_in_container.sh")) as fh:
            runtime = fh.read()
        self.assertIn("[ -f /tmp/.cx_built_uccl ]", runtime)
        self.assertIn(": > /tmp/.cx_built_uccl", runtime)
        self.assertIn("DEEPEP_COMMIT", runtime)
        self.assertIn("FLASHINFER_COMMIT", runtime)
        self.assertIn("CX_FLASHINFER_STACK", runtime)
        self.assertIn('python3 -c "from deep_ep import Buffer"', runtime)
        self.assertIn('[ "${CX_FLASHINFER_UPGRADE:-}" = "1" ]', runtime)
        for backend in ("deepep", "deepep-hybrid", "flashinfer"):
            self.assertIn(f"cx_prepare_backend {backend}", runtime)
        for launcher in ("launch_gb200-nv.sh", "launch_gb300-nv.sh"):
            with self.subTest(launcher=launcher):
                with open(os.path.join(ROOT, "launchers", launcher)) as fh:
                    source = fh.read()
                self.assertIn("CX_BUILD_ONLY=1", source)
                self.assertIn('cx_die "EP backend preparation failed"', source)
                self.assertIn("/tmp/.cx_backend_env", source)
                self.assertNotIn("/tmp/.cx_hybrid_env", source)

    def test_rack_launchers_pass_public_topology_and_manual_gb300_defaults_one_node(self) -> None:
        for launcher, gpn in (("launch_gb200-nv.sh", "GPUS_PER_NODE"),
                              ("launch_gb300-nv.sh", "GPN")):
            with self.subTest(launcher=launcher):
                with open(os.path.join(ROOT, "launchers", launcher)) as fh:
                    source = fh.read()
                self.assertIn(f'--gpus-per-node "${gpn}"', source)
                self.assertIn('--scale-up-domain "$SCALE_UP_DOMAIN"', source)
        with open(os.path.join(ROOT, "launchers", "launch_gb300-nv.sh")) as fh:
            gb300 = fh.read()
        self.assertIn('NODES="${CX_NODES:-1}"', gb300)
        self.assertNotIn('NODES="${CX_NODES:-2}"', gb300)

    def test_flashinfer_rack_mapping_never_falls_back_to_world_as_node_size(self) -> None:
        with open(os.path.join(ROOT, "tests", "ep_flashinfer.py")) as fh:
            source = fh.read()
        tree = ast.parse(source)
        mapping = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "_build_mapping")
        self.assertEqual([arg.arg for arg in mapping.args.args],
                         ["world_size", "rank", "gpus_per_node"])
        self.assertNotIn("gpus_per_node=world_size", source)
        self.assertIn("if gpus_per_node == world_size", source)

    def test_sm_budget_setters_fail_instead_of_recording_an_unapplied_request(self) -> None:
        for adapter, library in (("ep_deepep.py", "DeepEP"), ("ep_uccl.py", "UCCL")):
            with self.subTest(adapter=adapter):
                with open(os.path.join(ROOT, "tests", adapter)) as fh:
                    source = fh.read()
                self.assertIn(f'raise RuntimeError(f"{library} did not apply requested num_sms=', source)
                self.assertIn('"requested_num_sms": num_sms', source)
                self.assertIn('"num_sms": applied_num_sms', source)

    def test_nccl_version_normalizes_integer_and_tuple_and_labels_rccl(self) -> None:
        path = os.path.join(ROOT, "tests", "ep_nccl.py")
        with open(path) as fh:
            source = fh.read()
        tree = ast.parse(source, path)
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_format_collective_version")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), namespace)
        self.assertEqual(namespace["_format_collective_version"](21805), "2.18.5")
        self.assertEqual(namespace["_format_collective_version"](2809), "2.8.9")
        self.assertEqual(namespace["_format_collective_version"]((2, 21, 5)), "2.21.5")
        self.assertIn('"rccl" if torch.version.hip else "nccl"', source)

    def test_result_doc_probe_distinguishes_terminal_invalid_results(self) -> None:
        common = os.path.join(ROOT, "runtime", "common.sh")
        env = {**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"}
        with tempfile.TemporaryDirectory() as tmp:
            valid = os.path.join(tmp, "invalid-result.json")
            incomplete = os.path.join(tmp, "incomplete.json")
            malformed = os.path.join(tmp, "malformed.json")
            with open(valid, "w") as fh:
                json.dump({"schema_version": 5, "family": "moe", "status": "invalid"}, fh)
            with open(incomplete, "w") as fh:
                json.dump({"schema_version": 5, "family": "moe"}, fh)
            with open(malformed, "w") as fh:
                fh.write("{")
            command = 'source "$1"; cx_has_result_doc "$2"'
            self.assertEqual(
                subprocess.run(["bash", "-c", command, "_", common, valid], env=env).returncode,
                0,
            )
            for path in (incomplete, malformed):
                self.assertNotEqual(
                    subprocess.run(["bash", "-c", command, "_", common, path], env=env).returncode,
                    0,
                )

    def test_nonzero_command_demotes_an_emitted_result(self) -> None:
        common = os.path.join(ROOT, "runtime", "common.sh")
        env = {**os.environ, "COLLECTIVEX_OPERATOR_CONFIG": "/dev/null"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "result.json")
            with open(path, "w") as fh:
                json.dump(_doc(), fh)
            subprocess.run(
                ["bash", "-c", 'source "$1"; cx_demote_result_doc "$2" 17', "_", common, path],
                check=True,
                env=env,
            )
            with open(path) as fh:
                result = json.load(fh)
        self.assertEqual(result["publication_status"], "failed")
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["validity"]["execution_status"], "failed")
        self.assertEqual(result["post_emit_failure"]["return_code"], 17)

    def test_failed_commands_cannot_leave_accepted_results(self) -> None:
        with open(os.path.join(ROOT, "runtime", "run_in_container.sh")) as fh:
            runtime = fh.read()
        self.assertIn('cx_has_result_doc "$out"', runtime)
        self.assertIn('cx_demote_result_doc "$out"', runtime)
        for launcher in ("launch_gb200-nv.sh", "launch_gb300-nv.sh"):
            with open(os.path.join(ROOT, "launchers", launcher)) as fh:
                rack = fh.read()
            self.assertIn('cx_has_result_doc "$expected_out"', rack)
            self.assertIn('cx_demote_result_doc "$expected_out"', rack)
            self.assertIn('failed_cases=$((failed_cases + 1))', rack)

    def test_non_rack_launchers_reject_multi_node_runs(self) -> None:
        launchers = (
            "launch_h100-dgxc-slurm.sh", "launch_h200.sh", "launch_b200-dgxc.sh",
            "launch_b300.sh", "launch_mi355x-amds.sh",
        )
        for launcher in launchers:
            with self.subTest(launcher=launcher):
                with open(os.path.join(ROOT, "launchers", launcher)) as fh:
                    self.assertIn('cx_require_single_node "$RUNNER_NAME"', fh.read())

    def test_image_digest_matches_the_selected_image(self) -> None:
        common = os.path.join(ROOT, "runtime", "common.sh")
        script = f'''
          export HOME="$(mktemp -d)"
          source {common!r}
          test -n "$(cx_default_image_digest "$CX_IMAGE_MULTIARCH")"
          test -z "$(cx_default_image_digest "$CX_IMAGE_AMD_MORI")"
        '''
        proc = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)

    def test_official_provenance_requires_every_declared_run_field(self) -> None:
        provenance = {"version": "1.0", "commit": "abc123"}
        run = {key: "value" for key in ep_harness.REQUIRED_GIT_RUN_FIELDS}
        args = argparse.Namespace(image_digest="sha256:test", git_run=run)
        self.assertTrue(ep_harness._provenance_complete(provenance, args))
        for field in ep_harness.REQUIRED_GIT_RUN_FIELDS:
            with self.subTest(field=field):
                incomplete = argparse.Namespace(
                    image_digest="sha256:test", git_run={**run, field: None}
                )
                self.assertFalse(ep_harness._provenance_complete(provenance, incomplete))

    def test_official_provenance_requires_resolved_backend_build_identity(self) -> None:
        run = {key: "value" for key in ep_harness.REQUIRED_GIT_RUN_FIELDS}
        args = argparse.Namespace(backend="flashinfer", image_digest="sha256:test", git_run=run)
        complete = {
            "flashinfer_version": "0.6.14", "flashinfer_commit": "pkg-0.6.14",
            "flashinfer_stack": "flashinfer-python=0.6.14 torch=2.9.0",
        }
        self.assertTrue(ep_harness._provenance_complete(complete, args))
        for field, value in (("flashinfer_commit", "pkg-unknown"),
                             ("flashinfer_stack", None),
                             ("flashinfer_stack", "capture-failed")):
            with self.subTest(field=field, value=value):
                self.assertFalse(ep_harness._provenance_complete(
                    {**complete, field: value}, args))

        doc = _doc()
        doc["validity"].update(provenance_complete=True, workload_source="canonical-serialized")
        doc["publication_status"] = "official"
        doc["workload"].update(source="canonical-serialized", workload_id="workload-1")
        doc["backend_provenance"] = {"deepep_version": "1.2.1", "deepep_commit": "pkg-unknown"}
        errors, _warnings, _status = vr.validate_doc(doc, None, "bad-provenance.json")
        self.assertTrue(any("unresolved backend identity" in error for error in errors), errors)

    def test_validator_rejects_platform_topology_mismatch(self) -> None:
        doc = _doc()
        doc["runner"] = "gb200-8x"
        doc["placement"].update(nodes=2, gpus_per_node=4, scale_up_domain=72)
        errors, _warnings, _status = vr.validate_doc(doc, None, "good-topology.json")
        self.assertEqual(errors, [])
        doc["placement"]["scale_up_domain"] = 8
        errors, _warnings, _status = vr.validate_doc(doc, None, "bad-topology.json")
        self.assertTrue(any("expected 72 for gb200" in error for error in errors), errors)

    def test_aggregate_fails_closed_on_malformed_or_non_object_documents(self) -> None:
        fixtures = (
            ("broken.json", "{"),
            ("broken.ndjson", '{"family":"moe"}\nnot-json\n'),
            ("scalar.json", '"not-an-object"'),
        )
        for name, contents in fixtures:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, name), "w") as fh:
                    fh.write(contents)
                with self.assertRaises(SystemExit):
                    ar.aggregate(tmp)

    def test_bundle_recursively_rejects_sensitive_fields_and_value_shapes(self) -> None:
        make_bundle.assert_publication_safe([{
            "family": "moe",
            "runner": "test-runner",
            "provenance": {"source_sha": "abc123"},
        }])
        unsafe = (
            {"nested": {"environment": {}}},
            {"nested": {"hostname": "private-host"}},
            {"nested": {"detail": "/home/private/result.json"}},
            {"nested": {"detail": "192.0.2.1"}},
            {"nested": {"detail": "2001:db8::1"}},
            {"nested": {"detail": "ssh://user@private-host"}},
        )
        for document in unsafe:
            with self.subTest(document=list(document)):
                with self.assertRaises(SystemExit):
                    make_bundle.assert_publication_safe([document])

    def test_bundle_rejects_non_ep_families(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unsupported family"):
            make_bundle.validate([{
                "family": "kv-cache",
                "publication_status": "official",
                "rows": [],
            }], None)

    def test_summary_keeps_only_ep_docs_and_reports_failed_attempts(self) -> None:
        valid = _doc()
        valid["status"] = "valid"
        failure = _failed({"backend": "deepep", "phase": "decode"}, attempt_id="2")
        failure["status"] = "failed"
        with tempfile.TemporaryDirectory() as tmp:
            for name, document in (
                ("valid.json", valid),
                ("failed.json", failure),
                ("foreign.json", {"family": "kv-cache", "status": "valid"}),
            ):
                with open(os.path.join(tmp, name), "w") as fh:
                    json.dump(document, fh)
            docs = summarize.load_results(tmp, None, None)

        self.assertEqual(len(docs), 2)
        self.assertEqual({doc["family"] for doc in docs}, {"moe"})
        rendered = summarize.render_markdown(docs)
        self.assertIn("Failed attempts", rendered)
        self.assertIn("attempt", rendered)
        self.assertNotIn("kv-cache", rendered)

    def test_bundle_rejects_cross_chip_canonical_workload_drift(self) -> None:
        def canonical(runner: str, routing_hash: str) -> dict:
            doc = _doc()
            doc.update(
                runner=runner,
                case_id=f"case-{runner}",
                suite="ep-core-v1",
                workload_name="deepseek-v3-v1",
                required_publication="comparable-experimental",
                phase="decode",
                ep_size=8,
                eplb={"enabled": False},
            )
            doc["shape"]["activation_profile"] = "normal"
            doc["workload"].update(
                source="canonical-serialized",
                activation_identity="activation-a",
            )
            doc["rows"][0]["routing_hash"] = routing_hash
            return doc

        docs = [canonical("h100-dgxc", "route-a"), canonical("b300", "route-b")]
        self.assertEqual(len(vr.cross_document_workload_issues(docs)), 1)
        with self.assertRaisesRegex(SystemExit, "cross-document workload identity"):
            make_bundle.validate(docs, None)

    def test_bundle_coverage_rejects_case_id_with_wrong_semantics_or_rows(self) -> None:
        case_id = "cxv1-0123456789abcdefabcd"
        case = {
            "case_id": case_id, "suite": "ep-core-v1", "workload": "deepseek-v3-v1",
            "required_publication": "comparable-experimental", "backend": "deepep",
            "mode": "normal", "dtype": "bf16",
            "contract": "layout-and-dispatch-v1", "routing": "uniform", "phase": "decode",
            "ep": 8, "eplb": False, "combine_quant_mode": "none",
            "resource_mode": "tuned", "activation_profile": "normal",
            "placement": "packed", "routing_step": "0", "uneven_tokens": "none",
            "hidden": "", "topk": "", "experts": "", "samples_per_point": 512,
            "warmup_semantics": "full-roundtrip-per-trial-point-v1", "ladder": "8",
            "timing": "8:64:32", "canonical": False, "nodes": "1",
            "gpus_per_node": 8, "scale_up_domain": 8,
        }
        matrix = {"include": [{"sku": "h100-dgxc", "gpus_per_node": 8,
                                "scale_up_domain": 8, "cases": [case]}]}
        doc = _doc()
        doc.update(case_id=case_id, suite=case["suite"], workload_name=case["workload"],
                   required_publication=case["required_publication"], resource_mode="tuned",
                   runner="h100-dgxc-slurm_19")
        self.assertEqual(
            make_bundle.validate_expected_coverage([doc], matrix),
            {"expected": 1, "observed": 1, "complete": True},
        )

        mutations = (
            lambda value: value.update(suite="wrong-suite"),
            lambda value: value.update(phase="prefill"),
            lambda value: value["shape"].update(routing="zipf"),
            lambda value: value["rows"][0].update(tokens_per_rank=16),
            lambda value: value.update(runner="b200-dgxc-slurm_19"),
            lambda value: value["placement"].update(gpus_per_node=4),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                mismatched = copy.deepcopy(doc)
                mutate(mismatched)
                with self.assertRaisesRegex(SystemExit, "identity_mismatch"):
                    make_bundle.validate_expected_coverage([mismatched], matrix)

    def test_bundle_coverage_resolves_blank_ladder_to_v1_phase_default(self) -> None:
        case_id = "cxv1-0123456789abcdefabcd"
        case = {"case_id": case_id, "required_publication": "diagnostic",
                "phase": "decode", "ladder": ""}
        doc = {"family": "moe", "case_id": case_id,
               "required_publication": "diagnostic", "publication_status": "diagnostic",
               "phase": "decode", "rows": [
            {"tokens_per_rank": token}
            for token in (1, 2, 4, 8, 16, 32, 64, 128)
        ]}
        matrix = {"include": [{"cases": [case]}]}
        self.assertEqual(
            make_bundle.validate_expected_coverage([doc], matrix),
            {"expected": 1, "observed": 1, "complete": True},
        )
        doc["rows"].pop()
        with self.assertRaisesRegex(SystemExit, "identity_mismatch"):
            make_bundle.validate_expected_coverage([doc], matrix)

    def test_aggregate_preserves_distinct_failed_cases(self) -> None:
        case = {
            "suite": "ep-routing-v1", "workload": "deepseek-v3-v1",
            "backend": "deepep", "phase": "decode", "ep": 8, "mode": "normal",
            "dispatch_dtype": "bf16", "contract": "layout-and-dispatch-v1",
            "routing": "zipf", "eplb": False, "combine_quant_mode": "none",
            "resource_mode": "tuned", "tokens_ladder": "128",
        }
        docs = [
            _failed(case),
            _failed({**case, "eplb": True}, "2026-07-03T00:00:01Z"),
            _failed(case, "2026-07-03T00:00:02Z"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, doc in enumerate(docs):
                with open(os.path.join(tmp, f"{index}.json"), "w") as fh:
                    json.dump(doc, fh)
            got = ar.aggregate(tmp)
        self.assertEqual(len(got), 2)
        by_eplb = {doc["failure"]["case"]["eplb"]: doc for doc in got}
        self.assertEqual(by_eplb[False]["generated_at"], "2026-07-03T00:00:02Z")

    def test_aggregate_projects_one_newest_usable_outcome_per_case(self) -> None:
        older = _doc()
        older.update(case_id="case-a", generated_at="2026-07-03T00:00:01Z")
        newer = copy.deepcopy(older)
        newer["generated_at"] = "2026-07-03T00:00:02Z"
        failed = _failed({"backend": "deepep", "phase": "decode"},
                         "2026-07-03T00:00:03Z", case_id="case-a")
        with tempfile.TemporaryDirectory() as tmp:
            for index, doc in enumerate((older, newer, failed)):
                with open(os.path.join(tmp, f"{index}.json"), "w") as fh:
                    json.dump(doc, fh)
            got = ar.aggregate(tmp)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["generated_at"], newer["generated_at"])

    def test_aggregate_failed_identity_covers_scheduled_axes(self) -> None:
        case = {
            "suite": "ep-routing-v1", "workload": "deepseek-v3-v1",
            "backend": "deepep", "phase": "decode", "ep": 8, "mode": "normal",
            "dispatch_dtype": "bf16", "contract": "layout-and-dispatch-v1",
            "routing": "zipf", "eplb": False, "combine_quant_mode": "none",
            "resource_mode": "tuned", "tokens_ladder": "128",
        }
        replacements = {
            "suite": "ep-core-v1", "workload": "other", "backend": "uccl",
            "phase": "prefill", "ep": 4, "mode": "ll", "dispatch_dtype": "fp8",
            "contract": "runtime-visible-v1", "routing": "uniform", "eplb": True,
            "combine_quant_mode": "fp8", "resource_mode": "normalized",
            "tokens_ladder": "512 2048",
        }
        baseline = ar._key(_failed(case))
        for field, value in replacements.items():
            with self.subTest(field=field):
                self.assertNotEqual(baseline, ar._key(_failed({**case, field: value})))

        self.assertEqual(ar._key(_failed(case, case_id="case-a")),
                         ar._key(_failed({**case, "routing": "uniform"}, case_id="case-a")))
        self.assertNotEqual(ar._key(_failed(case, case_id="case-a")),
                            ar._key(_failed(case, case_id="case-b")))

    def test_sampling_nonconformance_is_diagnostic(self) -> None:
        validity = _doc()["validity"]
        validity["sampling_conformance"] = "nonconformant"
        self.assertEqual(vr.derive_publication_status(validity), "diagnostic")
        self.assertEqual(ep_harness._derive_publication_status(validity), "diagnostic")

    def test_historical_v4_keeps_variable_sample_semantics(self) -> None:
        doc = _doc(iters=200, trials=3, samples=600)
        doc["schema_version"] = 4
        doc["validity"].pop("sampling_conformance")
        doc["reproduction"].pop("sampling_contract")
        doc["reproduction"].pop("samples_per_point")
        errors, warnings, status = vr.validate_doc(doc, None, "historical-v4.json")
        self.assertEqual(status, "comparable-experimental")
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

        doc["schema_version"] = 3
        registry = vr.load_schema_registry()
        selected, schema_errors = vr._schema_for_doc(doc, registry)
        self.assertIs(selected, registry[4])
        self.assertEqual(schema_errors, [])
        errors, warnings, status = vr.validate_doc(doc, None, "historical-v3.json")
        self.assertEqual(status, "comparable-experimental")
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_v5_failed_case_is_schema_selected_but_sampling_exempt(self) -> None:
        doc = {
            "schema_version": 5,
            "family": "moe",
            "record_type": "failed-case",
            "runner": "test",
            "backend": "deepep",
            "publication_status": "failed",
            "rows": [],
            "failure": {"failure_mode": "timeout", "return_code": 124, "case": {}},
        }
        errors, warnings, status = vr.validate_doc(doc, vr.load_schema_registry(), "failed-v5.json")
        self.assertEqual((errors, warnings, status), ([], [], "failed"))

        doc["schema_version"] = 6
        errors, _warnings, _status = vr.validate_doc(doc, vr.load_schema_registry(), "failed-v6.json")
        self.assertTrue(any("unsupported schema_version" in error for error in errors), errors)

    def test_scheduled_failed_case_requires_attributable_identity(self) -> None:
        case_id = "cxv1-0123456789abcdefabcd"
        case = {
            "case_id": case_id, "suite": "ep-core-v1", "workload": "deepseek-v3-v1",
            "required_publication": "official", "backend": "deepep", "phase": "decode",
            "ep": 8, "dispatch_dtype": "bf16", "mode": "normal",
            "contract": "layout-and-dispatch-v1", "routing": "uniform", "eplb": False,
            "combine_quant_mode": "none", "resource_mode": "tuned", "tokens_ladder": "",
            "gpus_per_node": 8, "scale_up_domain": 8,
            "sampling_contract": "fixed-512-v1", "samples_per_point": 512,
            "iters": 8, "trials": 64, "warmup": 32,
            "warmup_semantics": "full-roundtrip-per-trial-point-v1",
        }
        doc = _failed(case, case_id=case_id, suite="ep-core-v1",
                      workload_name="deepseek-v3-v1", required_publication="official",
                      mode="normal", ep_size=8,
                      measurement_contract="layout-and-dispatch-v1")
        errors, _warnings, status = vr.validate_doc(
            doc, vr.load_schema_registry(), "scheduled-failure.json")
        self.assertEqual((errors, status), ([], "failed"))
        del case["routing"]
        errors, _warnings, _status = vr.validate_doc(
            doc, vr.load_schema_registry(), "missing-routing.json")
        self.assertTrue(any("failure.case.routing" in error for error in errors), errors)

    def test_v5_missing_publication_status_is_not_legacy(self) -> None:
        doc = _doc()
        doc.pop("publication_status")
        errors, _warnings, status = vr.validate_doc(doc, vr.load_schema_registry(), "malformed-v5.json")
        self.assertNotEqual(status, "legacy-experimental")
        self.assertTrue(errors)

if __name__ == "__main__":
    unittest.main()
