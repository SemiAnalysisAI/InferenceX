"""Behavioral checks for independently issued immutable artifact snapshots."""

import hashlib
import json
import stat
import struct
import tempfile
import tracemalloc
import unittest
import zipfile
from pathlib import Path

from infx.results.publication_receipt import (
    ExpectedContract,
    Issuer,
    inspect_archive,
    seal_receipt,
    verify_receipt,
)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        with zipfile.ZipFile(self.root / "101.zip", "w") as archive:
            archive.writestr(
                "result.json",
                '{"conc":1,"tp":8,"ep":1,"num_gpus":8,"disagg":false,"is_multinode":false,"infmax_model_prefix":"dsr1","output_tput_tps":100}',
            )
            archive.writestr(
                "execution.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "point_id": "c" * 64,
                        "execution_id": "intent:cluster:55:0",
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
        self.expected = ExpectedContract.model_validate(
            {
                "repository": "org/repo",
                "source_run_id": "100",
                "source_attempt": 2,
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
                        "execution_id": "intent:cluster:55:0",
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
                        "normalized_artifact_id": 101,
                        "normalized_path": "result.json",
                        "required_metrics": ["output_tput_tps"],
                        "config": {"model": "dsr1"},
                        "execution_artifact_id": 101,
                        "execution_path": "execution.json",
                        "native_manifest_sha256": "f" * 64,
                    }
                ],
            }
        )
        self.issuer = Issuer(
            repository="org/repo",
            run_id="200",
            job="validate",
            workflow_sha="d" * 40,
            collector_sha="e" * 40,
        )
        digest = hashlib.sha256((self.root / "101.zip").read_bytes()).hexdigest()
        self.inventory = [
            {
                "id": 101,
                "name": "bmk_result",
                "expired": False,
                "workflow_run": {"id": 100},
                "digest": f"sha256:{digest}",
            }
        ]

    def test_preserves_original_execution_and_exact_uploaded_member(self):
        receipt = seal_receipt(self.expected, self.issuer, self.inventory, self.root)
        self.assertEqual(receipt.points[0].source_attempt, 1)
        self.assertEqual(receipt.source_attempt, 2)
        self.assertEqual(receipt.artifacts[0].members[1].path, "result.json")
        self.assertEqual(
            verify_receipt(json.loads(receipt.model_dump_json())).receipt_id,
            receipt.receipt_id,
        )
        changed = receipt.model_dump()
        changed["points"][0]["source_attempt"] = 2
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            verify_receipt(changed)

    def test_missing_wrong_run_expired_and_changed_bytes_fail(self):
        bad_inventory = [
            [],
            [self.inventory[0] | {"workflow_run": {"id": 99}}],
            [self.inventory[0] | {"expired": True}],
            [self.inventory[0] | {"digest": "sha256:" + "0" * 64}],
        ]
        for rows in bad_inventory:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                seal_receipt(self.expected, self.issuer, rows, self.root)

    def test_rejects_archive_traversal_duplicate_and_link_members(self):
        for names in [["../escape"], ["same", "same"], ["/absolute"], ["a\\b"]]:
            with self.subTest(names=names):
                archive_path = self.root / "bad.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    for name in names:
                        archive.writestr(name, "data")
                with self.assertRaises(ValueError):
                    inspect_archive(archive_path)
        with zipfile.ZipFile(self.root / "link.zip", "w") as archive:
            item = zipfile.ZipInfo("link")
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(item, "target")
        with self.assertRaisesRegex(ValueError, "link"):
            inspect_archive(self.root / "link.zip")

    def test_hashes_large_compressed_member_with_bounded_memory(self):
        archive_path = self.root / "large.zip"
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            with archive.open("trace.jsonl", "w") as trace:
                for _ in range(64):
                    trace.write(b"x" * 1024**2)
            archive.writestr("empty", b"")
        tracemalloc.start()
        try:
            members = inspect_archive(archive_path)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(
            [member.model_dump() for member in members],
            [
                {
                    "path": "empty",
                    "size": 0,
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                },
                {
                    "path": "trace.jsonl",
                    "size": 64 * 1024**2,
                    "sha256": "e20a69eca39368572e90b9135738a613838f954987a0b44b6220889c171cbb76",
                },
            ],
        )
        self.assertLess(peak, 8 * 1024**2)

    def test_streamed_member_rejects_crc_corruption_at_end(self):
        archive_path = self.root / "bad-crc.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("trace.jsonl", b"x" * (1024**2 + 3))
            item = archive.getinfo("trace.jsonl")
            payload_offset = item.header_offset + 30 + len(item.filename.encode())
        with archive_path.open("r+b") as source:
            source.seek(payload_offset + item.file_size - 1)
            source.write(b"y")
        with self.assertRaisesRegex(zipfile.BadZipFile, "CRC"):
            inspect_archive(archive_path)

    def test_streamed_member_rejects_incomplete_declared_size(self):
        archive_path = self.root / "bad-size.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("trace.jsonl", b"abc")
        data = bytearray(archive_path.read_bytes())
        central_header = data.index(b"PK\x01\x02")
        # ZIP central-directory uncompressed size; payload and its CRC remain intact.
        struct.pack_into("<I", data, central_header + 24, 4)
        archive_path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "declared size"):
            inspect_archive(archive_path)

    def test_semantic_failure_is_not_repaired_by_a_fresh_api_digest(self):
        with zipfile.ZipFile(self.root / "101.zip") as archive:
            execution = archive.read("execution.json")
        for change in (
            {"output_tput_tps": -1},
            {"infmax_model_prefix": "wrong-model"},
            {"conc": 28},
            {"num_gpus": 4},
        ):
            with self.subTest(change=change):
                row = {
                    "conc": 1,
                    "tp": 8,
                    "ep": 1,
                    "num_gpus": 8,
                    "disagg": False,
                    "is_multinode": False,
                    "infmax_model_prefix": "dsr1",
                    "output_tput_tps": 100,
                } | change
                with zipfile.ZipFile(self.root / "101.zip", "w") as archive:
                    archive.writestr("result.json", json.dumps(row))
                    archive.writestr("execution.json", execution)
                inventory = [
                    self.inventory[0]
                    | {
                        "digest": "sha256:"
                        + hashlib.sha256(
                            (self.root / "101.zip").read_bytes()
                        ).hexdigest()
                    }
                ]
                with self.assertRaises(ValueError):
                    seal_receipt(self.expected, self.issuer, inventory, self.root)

    def test_raw_eval_gate_matches_summary_to_both_filter_sets(self):
        from infx.results.publication_receipt import Point, validate_point_content

        point = Point.model_validate(
            self.expected.points[0].model_dump()
            | {
                "kind": "eval",
                "concurrency": 28,
                "normalized_format": "lm-eval",
                "metadata_path": "meta_env.json",
                "task": "gsm8k",
                "sample_count": 2,
                "filters": ["strict-match", "flexible-extract"],
                "samples_artifact_id": 101,
                "samples_path": "samples.jsonl",
                "required_metrics": ["em_strict", "n_eff"],
            }
        )
        meta = {
            "conc": 28,
            "tp": 8,
            "ep": 1,
            "num_gpus": 8,
            "disagg": False,
            "is_multinode": False,
            "infmax_model_prefix": "dsr1",
            "prefill_num_workers": 0,
            "decode_num_workers": 0,
        }
        samples = [
            {
                "doc_id": doc,
                "task_name": "gsm8k",
                "filter": name,
                "exact_match": int(doc == 0 or name == "flexible-extract"),
            }
            for doc in range(2)
            for name in point.filters
        ]
        for summary, succeeds in [(0.5, True), (1.0, False)]:
            with zipfile.ZipFile(
                self.root / "101.zip", "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.writestr(
                    "result.json",
                    json.dumps(
                        {
                            "results": {"gsm8k": {"exact_match,strict-match": summary}},
                            "configs": {
                                "gsm8k": {
                                    "metric_list": [{"metric": "exact_match"}],
                                    "filter_list": [
                                        {"name": "strict-match"},
                                        {"name": "flexible-extract"},
                                    ],
                                }
                            },
                            "n-samples": {"gsm8k": {"effective": 2}},
                        }
                    ),
                )
                archive.writestr("meta_env.json", json.dumps(meta))
                with archive.open("samples.jsonl", "w") as output:
                    if succeeds:
                        blank_lines = (b" " * 4095 + b"\n") * 256
                        for _ in range(32):
                            output.write(blank_lines)
                    output.write(
                        "\r\n".join(json.dumps(sample) for sample in samples).encode()
                    )
            if succeeds:
                tracemalloc.start()
                try:
                    validate_point_content(point, self.root)
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 8 * 1024**2)
            else:
                with self.assertRaisesRegex(ValueError, "strict summary"):
                    validate_point_content(point, self.root)

    def test_execution_evidence_rejects_ambiguous_json_and_boolean_status(self):
        with zipfile.ZipFile(self.root / "101.zip") as archive:
            result = archive.read("result.json")
            execution = json.loads(archive.read("execution.json"))
        candidates = [
            json.dumps(execution | {"schema_version": True}),
            json.dumps(execution | {"client_exit_code": False}),
            '{"client_exit_code":1,' + json.dumps(execution)[1:],
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                with zipfile.ZipFile(self.root / "101.zip", "w") as archive:
                    archive.writestr("result.json", result)
                    archive.writestr("execution.json", candidate)
                metadata = self.inventory[0] | {
                    "digest": "sha256:"
                    + hashlib.sha256((self.root / "101.zip").read_bytes()).hexdigest()
                }
                with self.assertRaises(ValueError):
                    seal_receipt(self.expected, self.issuer, [metadata], self.root)
