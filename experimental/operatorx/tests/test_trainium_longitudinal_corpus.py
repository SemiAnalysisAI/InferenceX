from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

BUILDER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_trainium_longitudinal_corpus.py"
)
BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_trainium_longitudinal_corpus", BUILDER
)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER_MODULE = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER_MODULE)

SWEEP = BUILDER.parent / "trainium_gemm_sweep.py"
SWEEP_SPEC = importlib.util.spec_from_file_location("trainium_gemm_sweep", SWEEP)
assert SWEEP_SPEC is not None and SWEEP_SPEC.loader is not None
SWEEP_MODULE = importlib.util.module_from_spec(SWEEP_SPEC)
SWEEP_SPEC.loader.exec_module(SWEEP_MODULE)


def test_checked_longitudinal_corpus_is_reproducible_and_broad() -> None:
    operatorx = BUILDER.parents[1]
    source = json.loads((operatorx / "testlists" / "gemm.json").read_text())
    checked = json.loads(
        (
            operatorx
            / "testlists"
            / "trainium_gemm_longitudinal_validation.json"
        ).read_text()
    )
    reproduced = BUILDER_MODULE.build(source)
    roles = Counter(row["corpus_role"] for row in checked)
    executables = {SWEEP_MODULE.executed_shape(row) for row in checked}
    calibration = {
        SWEEP_MODULE.executed_shape(row)
        for row in json.loads(
            (operatorx / "testlists" / "trainium_gemm_exact_holdout.json").read_text()
        )
    }

    assert reproduced == checked
    assert len(checked) == 103
    assert len(executables) == 38
    assert roles == {
        "operatorx-derived-request": 75,
        "exact-extrapolation": 24,
        "calibration-anchor": 4,
    }
    assert len(executables - calibration) == 29
    assert max(row["args"]["n"] for row in checked) == 12288
    assert max(row["args"]["k"] for row in checked) == 8192


def test_checked_longitudinal_snapshots_pass_contract() -> None:
    operatorx = BUILDER.parents[1]
    corpus = json.loads(
        (
            operatorx
            / "testlists"
            / "trainium_gemm_longitudinal_validation.json"
        ).read_text()
    )
    for suffix in ("a", "b"):
        dataset = json.loads(
            (
                operatorx
                / "data"
                / f"trn3_lnc2_gemm_longitudinal_20260801_snapshot_{suffix}.json"
            ).read_text()
        )
        rows = dataset["rows"]

        assert dataset["measurement_run"]["label"] == (
            f"20260801-snapshot-{suffix}"
        )
        assert dataset["measurement_run"]["started_at_utc"].endswith("+00:00")
        assert dataset["measurement_run"]["completed_at_utc"].endswith("+00:00")
        assert [row["name"] for row in rows] == [row["name"] for row in corpus]
        assert len(rows) == 103
        assert len({SWEEP_MODULE.executed_shape(row) for row in corpus}) == 38
        assert all(row["device_execution_us"]["count"] == 21 for row in rows)
        assert all(len(row["samples"]) == 21 for row in rows)
        assert all(row["correctness"]["validated"] for row in rows)
        assert all(row["placement"]["compiler_info_lnc"] == 2 for row in rows)
        assert dataset["methodology"]["full_artifacts_retained"] is False
