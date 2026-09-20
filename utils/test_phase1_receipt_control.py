"""Behavioral checks for the independently reviewed qualification and publication controls."""

import hashlib
import io
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from infx.results.publication_receipt import (
    ExpectedContract,
    Issuer,
    PublicationRecord,
    canonical_bytes,
    seal_receipt,
)
from infx.workflows.phase1_publication import Approval, expected_contract
from infx.workflows.phase1_record import validate_record


def archive_bytes(name, content):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.approval = {
            "schema_version": 1,
            "repository": "SemiAnalysisAI/InferenceX",
            "source_run_id": "100",
            "source_attempt": 1,
            "source_head_sha": "a" * 40,
            "points": [
                {
                    "point_id": f"{index + 1:064x}",
                    "execution_id": f"{index + 10:064x}",
                    "bundle_digest": f"{index + 20:064x}",
                    "native_manifest_sha256": f"{index + 30:064x}",
                    "kind": kind,
                    "concurrency": concurrency,
                    "dataset_revision": "c" * 40 if kind == "throughput" else None,
                }
                for index, (kind, concurrency) in enumerate(
                    [("throughput", c) for c in [1, 2, 4, 8, 16, 20, 24, 28]]
                    + [("eval", 28)]
                )
            ],
        }
        self.inventory = []
        for index, point in enumerate(self.approval["points"]):
            point_id = point["point_id"]
            self.inventory.append(
                {
                    "id": 1001 + index * 10,
                    "name": f"native-execution-{point_id}",
                    "expired": False,
                }
            )
            if point["kind"] == "throughput":
                self.inventory.extend(
                    [
                        {
                            "id": 1000 + index * 10,
                            "name": f"bmk_agentic_{point_id}",
                            "expired": False,
                        },
                        {
                            "id": 1002 + index * 10,
                            "name": f"agentic_{point_id}",
                            "expired": False,
                        },
                    ]
                )
            else:
                self.inventory.append(
                    {
                        "id": 1080,
                        "name": f"eval_{point_id}_lm-eval__1",
                        "expired": False,
                    }
                )
        self.run = {
            "id": 100,
            "head_sha": "a" * 40,
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "success",
            "path": ".github/workflows/run-sweep.yml",
        }
        self.patchers = [
            patch("infx.workflows.phase1_publication.api", side_effect=self.api),
            patch(
                "infx.workflows.phase1_publication.subprocess.run",
                side_effect=self.download,
            ),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def api(self, endpoint, **kwargs):
        if endpoint.endswith("/actions/runs/100"):
            return self.run
        if endpoint.endswith("/actions/runs/100/artifacts?per_page=100"):
            return [{"artifacts": self.inventory}]
        raise AssertionError(endpoint)

    def test_qualification_is_rejected_even_with_all_approved_normal_artifacts(self):
        self.inventory.append(
            {"id": 9999, "name": "native-qualification-run", "expired": True}
        )
        with self.assertRaisesRegex(ValueError, "nonpublishing qualification"):
            expected_contract(Approval.model_validate(self.approval))

    def download(self, argv, *, stdout, check):
        self.assertEqual(
            argv[-1], "repos/SemiAnalysisAI/InferenceX/actions/artifacts/1080/zip"
        )
        with zipfile.ZipFile(stdout, "w") as archive:
            for name in [
                "results_fixed_conc28.json",
                "samples_gsm8k_fixed_conc28.jsonl",
                "meta_env.json",
            ]:
                archive.writestr(name, "{}")
        return subprocess.CompletedProcess(argv, 0)

    def test_complete_pilot_binds_each_artifact_and_preserves_corpus_and_precision(
        self,
    ):
        contract = expected_contract(Approval.model_validate(self.approval))
        self.assertEqual(len(contract.points), 9)
        self.assertEqual(contract.points[0].artifact_ids, [1001, 1000, 1002])
        self.assertEqual(contract.points[0].config["precision"], "fp4")
        self.assertEqual(contract.points[0].dataset["hf_revision"], "c" * 40)
        self.assertEqual(contract.points[0].topology.serving_gpus, 8)
        self.assertNotEqual(
            contract.points[0].bundle_digest, contract.points[1].bundle_digest
        )
        evaluation = contract.points[-1]
        self.assertEqual(evaluation.artifact_ids, [1081, 1080])
        self.assertEqual(evaluation.normalized_format, "lm-eval")
        self.assertEqual(evaluation.normalized_path, "results_fixed_conc28.json")
        self.assertEqual(evaluation.samples_path, "samples_gsm8k_fixed_conc28.jsonl")
        self.assertEqual(evaluation.sample_count, 1319)
        self.assertEqual(evaluation.filters, ["strict-match", "flexible-extract"])

    def test_partial_pilot_wrong_source_and_ambiguous_artifacts_fail(self):
        with self.assertRaisesRegex(ValueError, "exactly eight throughput"):
            Approval.model_validate(
                self.approval | {"points": self.approval["points"][:-1]}
            )
        self.run["id"] = 101
        with self.assertRaisesRegex(ValueError, "completed successful sweep"):
            expected_contract(Approval.model_validate(self.approval))
        self.run["id"] = 100
        self.inventory.append(self.inventory[0] | {"id": 9999})
        with self.assertRaisesRegex(ValueError, "expected one unexpired artifact"):
            expected_contract(Approval.model_validate(self.approval))


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        with zipfile.ZipFile(root / "101.zip", "w") as archive:
            archive.writestr(
                "result.json",
                json.dumps(
                    {
                        "conc": 1,
                        "tp": 8,
                        "ep": 1,
                        "num_gpus": 8,
                        "disagg": False,
                        "is_multinode": False,
                        "infmax_model_prefix": "dsr1",
                        "output_tput_tps": 100,
                    }
                ),
            )
            archive.writestr(
                "execution.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "point_id": "c" * 64,
                        "execution_id": "owned:55",
                        "bundle_digest": "b" * 64,
                        "source": {
                            "repository": "org/repo",
                            "run_id": 100,
                            "attempt": 1,
                            "head_sha": "a" * 40,
                        },
                        "mode": "throughput",
                        "native_receipt": {
                            "job_id": "55",
                            "manifest_sha256": "f" * 64,
                            "state": "COMPLETED",
                        },
                        "client_exit_code": 0,
                    }
                ),
            )
        expected = ExpectedContract.model_validate(
            {
                "repository": "org/repo",
                "source_run_id": "100",
                "source_attempt": 1,
                "source_head_sha": "a" * 40,
                "bundle_digest": "b" * 64,
                "contracts": {
                    "raw": "aiperf-1.4",
                    "normalized": "agentx-v1",
                    "publication": 1,
                },
                "points": [
                    {
                        "point_id": "c" * 64,
                        "bundle_digest": "b" * 64,
                        "execution_id": "owned:55",
                        "source_run_id": "100",
                        "source_attempt": 1,
                        "kind": "throughput",
                        "concurrency": 1,
                        "topology": {
                            "kind": "aggregate",
                            "nodes": 1,
                            "serving_gpus": 8,
                            "tp": 8,
                            "ep": 1,
                        },
                        "artifact_ids": [101],
                        "execution_artifact_id": 101,
                        "execution_path": "execution.json",
                        "native_manifest_sha256": "f" * 64,
                        "normalized_artifact_id": 101,
                        "normalized_path": "result.json",
                        "required_metrics": ["output_tput_tps"],
                        "config": {"model": "dsr1"},
                    }
                ],
            }
        )
        receipt = seal_receipt(
            expected,
            Issuer(
                repository="org/repo",
                run_id="200",
                job="seal",
                workflow_sha="d" * 40,
                collector_sha="d" * 40,
            ),
            [
                {
                    "id": 101,
                    "name": "native-execution-point",
                    "expired": False,
                    "workflow_run": {"id": 100},
                    "digest": "sha256:"
                    + hashlib.sha256((root / "101.zip").read_bytes()).hexdigest(),
                }
            ],
            root,
        )
        self.receipt = receipt.model_dump()
        self.archives = {
            301: archive_bytes("receipt.json", json.dumps(self.receipt)),
            501: archive_bytes("changelog-metadata.json", "{}"),
        }
        self.metadata = {
            key: {
                "id": key,
                "name": name,
                "expired": False,
                "workflow_run": {"id": owner},
                "digest": "sha256:" + hashlib.sha256(self.archives[key]).hexdigest(),
            }
            for key, name, owner in [
                (301, "measurement-receipt", 200),
                (501, "changelog-metadata", 150),
            ]
        }
        self.runs = {
            "200": {
                "id": 200,
                "head_sha": "d" * 40,
                "status": "completed",
                "conclusion": "success",
                "head_branch": "main",
                "event": "workflow_dispatch",
                "path": ".github/workflows/phase1-receipt.yml",
            },
            "100/attempts/1": {
                "id": 100,
                "run_attempt": 1,
                "head_sha": "a" * 40,
                "status": "completed",
                "conclusion": "success",
            },
            "150": {
                "id": 150,
                "head_sha": "e" * 40,
                "status": "completed",
                "conclusion": "success",
                "head_branch": "main",
                "event": "push",
                "path": ".github/workflows/run-sweep.yml",
            },
        }
        self.record = PublicationRecord.model_validate(
            {
                "kind": "publication-record",
                "version": 1,
                "receipt_id": receipt.receipt_id,
                "receipt_artifact_id": 301,
                "receipt_artifact_sha256": self.metadata[301]["digest"][7:],
                "source_run_id": "100",
                "merge_run_id": "150",
                "merge_sha": "e" * 40,
                "changelog_artifact_id": 501,
                "changelog_artifact_sha256": self.metadata[501]["digest"][7:],
                "ingest_sha": "f" * 40,
                "app_sha": "f" * 40,
            }
        )
        for patcher in [
            patch("infx.workflows.phase1_record.api", side_effect=self.api),
            patch(
                "infx.workflows.phase1_record.subprocess.run", side_effect=self.download
            ),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def api(self, endpoint, **kwargs):
        if endpoint.endswith("/runs/100/artifacts?per_page=100"):
            return [{"artifacts": getattr(self, "source_artifacts", [])}]
        prefix = "repos/org/repo/actions/"
        if endpoint.startswith(prefix + "artifacts/"):
            return self.metadata[int(endpoint.rsplit("/", 1)[-1])]
        if endpoint.startswith(prefix + "runs/"):
            return self.runs[endpoint.removeprefix(prefix + "runs/")]
        raise AssertionError(endpoint)

    def test_later_publication_cannot_accept_a_nonpublishing_source(self):
        self.source_artifacts = [
            {"id": 9999, "name": "native-qualification-run", "expired": True}
        ]
        with self.assertRaisesRegex(ValueError, "nonpublishing qualification"):
            self.validate()

    def download(self, argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout=self.archives[int(argv[-1].split("/")[-2])]
        )

    def validate(self):
        return validate_record(self.record, "org/repo", {"d" * 40}, "f" * 40)

    def replace_receipt_archive(self, text):
        self.archives[301] = archive_bytes("receipt.json", text)
        digest = hashlib.sha256(self.archives[301]).hexdigest()
        self.metadata[301]["digest"] = "sha256:" + digest
        self.record = self.record.model_copy(update={"receipt_artifact_sha256": digest})

    def test_publication_preserves_original_measurement_and_pins_later_merge(self):
        record = self.validate()
        self.assertEqual(record.source_run_id, "100")
        self.assertEqual(record.merge_run_id, "150")
        self.assertEqual(record.receipt_artifact_id, 301)
        self.assertEqual(record.changelog_artifact_id, 501)

    def test_wrong_api_owner_source_attempt_and_reader_pin_fail(self):
        for mapping, key, invalid, message in [
            (self.metadata[301], "id", 999, "missing, expired or misnamed"),
            (
                self.runs["200"],
                "event",
                "pull_request",
                "approved completed trusted workflow",
            ),
            (
                self.runs["100/attempts/1"],
                "head_sha",
                "b" * 40,
                "completed original attempt",
            ),
            (self.runs["150"], "id", 151, "completed successful main sweep"),
            (self.metadata[501]["workflow_run"], "id", 151, "different merge run"),
        ]:
            with self.subTest(key=key, invalid=invalid):
                original, mapping[key] = mapping[key], invalid
                with self.assertRaisesRegex(ValueError, message):
                    self.validate()
                mapping[key] = original
        self.record = self.record.model_copy(update={"app_sha": "9" * 40})
        with self.assertRaisesRegex(ValueError, "deployed reader pin"):
            self.validate()

    def test_receipt_repository_and_duplicate_json_keys_cannot_change_authority(self):
        original = json.dumps(self.receipt)
        self.replace_receipt_archive('{"receipt_id":"' + "0" * 64 + '",' + original[1:])
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.validate()
        self.receipt["issuer"]["repository"] = "another/repository"
        payload = {
            key: value for key, value in self.receipt.items() if key != "receipt_id"
        }
        self.receipt["receipt_id"] = hashlib.sha256(
            canonical_bytes(payload)
        ).hexdigest()
        self.record = self.record.model_copy(
            update={"receipt_id": self.receipt["receipt_id"]}
        )
        self.replace_receipt_archive(json.dumps(self.receipt))
        with self.assertRaisesRegex(ValueError, "replace source measurement identity"):
            self.validate()
