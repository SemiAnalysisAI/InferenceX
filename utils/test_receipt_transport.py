"""Read-only transport resolution against controlled GitHub API/archive responses."""

import hashlib
import io
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from infx.results.publication_receipt import ExpectedContract, Issuer, seal_receipt
from infx.workflows.receipt_transport import PendingReceiptError, resolve_transport


def archive_bytes(name, payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, json.dumps(payload))
    return buffer.getvalue()


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        execution = {
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
            archive.writestr("execution.json", json.dumps(execution))
        self.source_archive_sha = hashlib.sha256(
            (root / "101.zip").read_bytes()
        ).hexdigest()
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
        source_meta = {
            "id": 101,
            "name": "native-execution-point",
            "expired": False,
            "workflow_run": {"id": 100},
            "digest": "sha256:" + self.source_archive_sha,
        }
        receipt = seal_receipt(
            expected,
            Issuer(
                repository="org/repo",
                run_id="200",
                job="seal",
                workflow_sha="d" * 40,
                collector_sha="d" * 40,
            ),
            [source_meta],
            root,
        )
        self.receipt_id = receipt.receipt_id
        self.archives = {301: archive_bytes("receipt.json", receipt.model_dump())}
        receipt_zip_sha = hashlib.sha256(self.archives[301]).hexdigest()
        publication = {
            "kind": "publication-record",
            "version": 1,
            "receipt_id": receipt.receipt_id,
            "receipt_artifact_id": 301,
            "receipt_artifact_sha256": receipt_zip_sha,
            "source_run_id": "100",
            "merge_run_id": "150",
            "merge_sha": "f" * 40,
            "changelog_artifact_id": 501,
            "changelog_artifact_sha256": "0" * 64,
            "ingest_sha": "1" * 40,
            "app_sha": "2" * 40,
        }
        self.archives[401] = archive_bytes("publication.json", publication)
        self.metadata = {
            101: source_meta,
            301: {
                "id": 301,
                "name": "measurement-receipt",
                "workflow_run": {"id": 200},
                "digest": "sha256:" + receipt_zip_sha,
                "expired": False,
            },
            401: {
                "id": 401,
                "name": "publication-record",
                "workflow_run": {"id": 300},
                "digest": "sha256:" + hashlib.sha256(self.archives[401]).hexdigest(),
                "expired": False,
            },
            501: {
                "id": 501,
                "name": "changelog-metadata",
                "workflow_run": {"id": 150},
                "digest": "sha256:" + "0" * 64,
                "expired": False,
            },
        }
        self.runs = [
            {
                "id": run,
                "head_sha": sha * 40,
                "status": "completed",
                "conclusion": "success",
                "path": ".github/workflows/phase1-receipt.yml",
                "head_branch": "main",
                "event": "workflow_dispatch",
            }
            for run, sha in [(200, "d"), (300, "e")]
        ]
        self.native = True
        self.patchers = [
            patch("infx.workflows.receipt_transport.github.api", side_effect=self.api),
            patch(
                "infx.workflows.receipt_transport.github.paginate",
                side_effect=self.pages,
            ),
            patch(
                "infx.workflows.receipt_transport.subprocess.run",
                side_effect=self.download,
            ),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def api(self, repo, endpoint, *args, **kwargs):
        if endpoint.startswith("/actions/artifacts/"):
            return self.metadata[int(endpoint.rsplit("/", 1)[-1])]
        if endpoint == "/actions/runs/100/attempts/1":
            return {
                "id": 100,
                "run_attempt": 1,
                "head_sha": "a" * 40,
                "status": "completed",
                "conclusion": "success",
            }
        if endpoint == "/actions/runs/150":
            return {
                "id": 150,
                "head_sha": "f" * 40,
                "status": "completed",
                "conclusion": "success",
                "head_branch": "main",
                "event": "push",
            }
        raise AssertionError(endpoint)

    def pages(self, repo, endpoint, *args, **kwargs):
        if endpoint == "/actions/runs/100/artifacts":
            return [self.metadata[101]] if self.native else []
        if endpoint.endswith("/phase1-receipt.yml/runs"):
            return self.runs
        if endpoint == "/actions/runs/200/artifacts":
            return [self.metadata[301]]
        if endpoint == "/actions/runs/300/artifacts":
            return [self.metadata[401]]
        raise AssertionError(endpoint)

    def download(self, args, *, stdout, check):
        artifact = int(args[-1].split("/")[-2])
        stdout.write(self.archives[artifact])
        return subprocess.CompletedProcess(args, 0)

    def test_preserves_source_receipt_and_separate_later_publication(self):
        payload = resolve_transport(
            "org/repo", "100", "150", {"d" * 40, "e" * 40}, publication_required=True
        )
        self.assertEqual(payload["receipt-artifact-id"], "301")
        self.assertEqual(payload["receipt-issuer-run-id"], "200")
        self.assertEqual(payload["publication-artifact-id"], "401")
        self.assertEqual(payload["publication-issuer-run-id"], "300")
        self.assertEqual(payload["receipt-required"], "true")

    def test_missing_required_receipt_never_enters_legacy(self):
        self.runs = []
        with self.assertRaises(PendingReceiptError):
            resolve_transport("org/repo", "100", "100", {"d" * 40})
        self.native = False
        self.assertEqual(
            resolve_transport("org/repo", "100", "100", set()),
            {"receipt-required": "false"},
        )

    def test_wrong_issuer_digest_and_missing_accepted_artifact_fail(self):
        with self.assertRaises(PendingReceiptError):
            resolve_transport("org/repo", "100", "100", {"9" * 40})
        original = self.metadata[301]["digest"]
        self.metadata[301]["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "API digest"):
            resolve_transport("org/repo", "100", "100", {"d" * 40})
        self.metadata[301]["digest"] = original
        self.metadata[101]["expired"] = True
        with self.assertRaisesRegex(ValueError, "Accepted artifact set"):
            resolve_transport("org/repo", "100", "100", {"d" * 40})
