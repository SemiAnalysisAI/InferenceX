#!/usr/bin/env python3
"""CPU-only tests for qualification-specific shard execution planning."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

import identity  # noqa: E402
import sweep_matrix  # noqa: E402
import capability  # noqa: E402


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()


class QualificationPlanningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(
                backend="deepep", only_sku="h100-dgxc", max_cases=128
            )
        )
        cls.shard = next(item for item in cls.matrix["include"] if item["n"] >= 3)

    def test_matrix_semantics_do_not_depend_on_qualification_index(self) -> None:
        expected = _canonical(self.matrix)
        for qualification_index in (1, 2, 3):
            with mock.patch.dict(
                os.environ,
                {"CX_QUALIFICATION_INDEX": str(qualification_index)},
            ):
                observed = sweep_matrix.validate_matrix_document(
                    sweep_matrix.resolve_matrix(
                        backend="deepep", only_sku="h100-dgxc", max_cases=128
                    )
                )
            self.assertEqual(_canonical(observed), expected)

    def test_extract_shard_has_deterministic_distinct_exact_plans(self) -> None:
        matrix_bytes = _canonical(self.matrix) + b"\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_bytes(matrix_bytes)
            original_digest = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
            controls = []
            for qualification_index in (1, 2, 3):
                first = sweep_matrix.extract_shard(
                    matrix_path,
                    self.shard["id"],
                    root / f"q{qualification_index}-first.json",
                    sku=self.shard["sku"],
                    backend=self.shard["backend"],
                    nodes=self.shard["nodes"],
                    qualification_index=qualification_index,
                )
                repeated = sweep_matrix.extract_shard(
                    matrix_path,
                    self.shard["id"],
                    root / f"q{qualification_index}-repeated.json",
                    sku=self.shard["sku"],
                    backend=self.shard["backend"],
                    nodes=self.shard["nodes"],
                    qualification_index=qualification_index,
                )
                self.assertEqual(first, repeated)
                self.assertEqual(first["qualification_index"], qualification_index)
                self.assertEqual(
                    {case["case_id"] for case in first["cases"]},
                    set(self.shard["case_ids"]),
                )
                plan = [
                    [
                        case["case_id"],
                        case.get(
                            "precision_profile",
                            identity.V1_CONTROL_PRECISION_PROFILE,
                        ),
                    ]
                    for case in first["cases"]
                ]
                self.assertEqual(
                    first["execution_plan_sha256"],
                    hashlib.sha256(
                        json.dumps(plan, separators=(",", ":")).encode()
                    ).hexdigest(),
                )
                sweep_matrix.validate_shard_control(
                    first,
                    sku=self.shard["sku"],
                    backend=self.shard["backend"],
                    nodes=self.shard["nodes"],
                    qualification_index=qualification_index,
                )
                controls.append(first)
            self.assertEqual(
                len({control["execution_plan_sha256"] for control in controls}), 3
            )
            self.assertEqual(
                len({tuple(case["case_id"] for case in control["cases"])
                     for control in controls}),
                3,
            )
            self.assertEqual(
                hashlib.sha256(matrix_path.read_bytes()).hexdigest(), original_digest
            )

    def test_precision_profiles_and_cases_rotate_across_repeats(self) -> None:
        profiles = list(identity.V1_PRECISION_PROFILES)[:3]
        cases = [
            {
                "case_id": identity.digest("case", {"fixture": index}),
                "precision_profile": profiles[index % len(profiles)],
            }
            for index in range(9)
        ]
        plans = [
            sweep_matrix.qualification_execution_order(
                "qualification-fixture", cases, qualification_index
            )
            for qualification_index in (1, 2, 3)
        ]
        expected_ids = {case["case_id"] for case in cases}
        self.assertTrue(all(
            {case["case_id"] for case in plan} == expected_ids for plan in plans
        ))
        self.assertEqual(
            len({tuple(case["case_id"] for case in plan) for plan in plans}), 3
        )
        self.assertEqual(
            len({tuple(case["precision_profile"] for case in plan) for plan in plans}),
            3,
        )

    def test_matrix_execution_plan_digest_is_repeat_specific_and_stable(self) -> None:
        digests = [
            sweep_matrix.qualification_execution_plan_sha256(self.matrix, index)
            for index in (1, 2, 3)
        ]
        self.assertEqual(len(set(digests)), 3)
        self.assertTrue(all(len(digest) == 64 for digest in digests))
        self.assertEqual(
            digests,
            [
                sweep_matrix.qualification_execution_plan_sha256(
                    copy.deepcopy(self.matrix), index
                )
                for index in (1, 2, 3)
            ],
        )
        self.assertTrue(all(shard["execution_weight"] > 0 for shard in self.matrix["include"]))
        tampered = copy.deepcopy(self.matrix)
        tampered["include"][0]["execution_weight"] += 1
        with self.assertRaisesRegex(
            sweep_matrix.MatrixError, "execution_weight differs from its cases"
        ):
            sweep_matrix.qualification_execution_plan_sha256(tampered, 1)

    def test_full_v1_execution_plans_match_the_frozen_digests(self) -> None:
        matrix = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(suites="all", backends="all", max_cases=128)
        )
        if capability.provisional_precision_targets():
            self.assertNotEqual(
                [
                    sweep_matrix.qualification_execution_plan_sha256(matrix, index)
                    for index in (1, 2, 3)
                ],
                [
                    sweep_matrix.CANONICAL_V1_EXECUTION_PLAN_SHA256[index]
                    for index in (1, 2, 3)
                ],
            )
            return
        self.assertEqual(
            [
                sweep_matrix.qualification_execution_plan_sha256(matrix, index)
                for index in (1, 2, 3)
            ],
            [
                sweep_matrix.CANONICAL_V1_EXECUTION_PLAN_SHA256[index]
                for index in (1, 2, 3)
            ],
        )

    def test_full_v1_matrix_emits_every_unsupported_terminal_outcome(self) -> None:
        matrix = sweep_matrix.validate_matrix_document(
            sweep_matrix.resolve_matrix(suites="all", backends="all", max_cases=128)
        )
        expected = [
            item for item in matrix["requested_cases"]
            if item["disposition"] == "unsupported"
        ]
        environment = {
            "COLLECTIVEX_ARTIFACT_NAME": "cxunsupported-123-1",
            "COLLECTIVEX_EXECUTION_ID": "123_1_unsupported",
            "COLLECTIVEX_SOURCE_SHA": "a" * 40,
            "CX_QUALIFICATION_INDEX": "1",
            "GITHUB_JOB": "setup",
            "GITHUB_REF_NAME": "collectivex",
            "GITHUB_REPOSITORY": "SemiAnalysisAI/InferenceX",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "123",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_bytes(_canonical(matrix))
            with mock.patch.dict(os.environ, environment, clear=False):
                written = sweep_matrix.emit_unsupported(matrix_path, root / "unsupported")
            documents = [json.loads(path.read_text()) for path in written]

        self.assertEqual(len(documents), len(expected))
        self.assertEqual(len(documents), 361)
        self.assertEqual(
            {document["outcome"]["reason"] for document in documents},
            {item["reason"] for item in expected},
        )
        self.assertEqual(
            {document["identity"]["case_id"] for document in documents},
            {item["case"]["case_id"] for item in expected},
        )

    def test_frontend_catalog_covers_every_requested_case_and_point(self) -> None:
        catalog = sweep_matrix.frontend_catalog(self.matrix)
        self.assertEqual(catalog["format"], "collectivex.frontend-catalog.v1")
        self.assertEqual(catalog["case_count"], len(self.matrix["requested_cases"]))
        self.assertEqual(
            catalog["point_count"],
            sum(
                len(item["case"]["ladder"].split())
                for item in self.matrix["requested_cases"]
            ),
        )
        self.assertEqual(
            {item["case_id"] for item in catalog["cases"]},
            {item["case"]["case_id"] for item in self.matrix["requested_cases"]},
        )
        self.assertEqual(
            set(catalog["precision_profiles"]),
            {item["precision_profile"] for item in catalog["cases"]},
        )
        self.assertTrue(all(item["required"] for item in catalog["cases"]))
        self.assertTrue(all(item["backend_generation"] is None for item in catalog["cases"]))
        self.assertEqual(
            {item["resource"]["mode"] for item in catalog["cases"]},
            {"fixed-profile"},
        )
        for item in catalog["cases"]:
            profile = identity.precision_profile(item["precision_profile"])
            projected = catalog["precision_profiles"][item["precision_profile"]]
            self.assertEqual(projected["dispatch"], profile["dispatch"])
            self.assertEqual(projected["combine"], profile["combine"])
        self.assertLess(len(_canonical(catalog)) + 1, 1024 * 1024)

    def test_invalid_qualification_controls_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix_path = root / "matrix.json"
            matrix_path.write_bytes(_canonical(self.matrix) + b"\n")
            arguments = {
                "sku": self.shard["sku"],
                "backend": self.shard["backend"],
                "nodes": self.shard["nodes"],
            }
            for invalid in (0, 4, True):
                with self.subTest(qualification_index=invalid), self.assertRaisesRegex(
                    sweep_matrix.MatrixError, "integer in 1..3"
                ):
                    sweep_matrix.extract_shard(
                        matrix_path,
                        self.shard["id"],
                        root / "invalid.json",
                        qualification_index=invalid,
                        **arguments,
                    )
            with mock.patch.dict(os.environ, {"CX_QUALIFICATION_INDEX": "invalid"}):
                with self.assertRaisesRegex(
                    sweep_matrix.MatrixError, "integer in 1..3"
                ):
                    sweep_matrix.extract_shard(
                        matrix_path,
                        self.shard["id"],
                        root / "invalid-env.json",
                        **arguments,
                    )

            with mock.patch.dict(os.environ, {}, clear=True):
                control = sweep_matrix.extract_shard(
                    matrix_path,
                    self.shard["id"],
                    root / "default.json",
                    **arguments,
                )
            self.assertEqual(control["qualification_index"], 1)

            invalid_control = copy.deepcopy(control)
            invalid_control["qualification_index"] = 4
            with self.assertRaisesRegex(
                sweep_matrix.MatrixError, "integer in 1..3"
            ):
                sweep_matrix.validate_shard_control(invalid_control, **arguments)
            tampered = copy.deepcopy(control)
            tampered["execution_plan_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                sweep_matrix.MatrixError, "differs from its ordered cases"
            ):
                sweep_matrix.validate_shard_control(tampered, **arguments)


if __name__ == "__main__":
    unittest.main()
