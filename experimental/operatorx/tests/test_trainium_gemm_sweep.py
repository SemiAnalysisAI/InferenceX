from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trainium_gemm_sweep.py"
SPEC = importlib.util.spec_from_file_location("trainium_gemm_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_percentile_interpolates() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert MODULE.percentile(values, 0.0) == 1.0
    assert MODULE.percentile(values, 0.5) == 3.0
    assert MODULE.percentile(values, 0.9) == pytest.approx(4.6)
    assert MODULE.percentile(values, 1.0) == 5.0


def test_executed_shape_applies_nki_padding() -> None:
    row = {"args": {"m": 16, "n": 28672, "k": 8192}}
    assert MODULE.executed_shape(row) == (2048, 28672, 8192)


def test_load_rows_rejects_non_bf16(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "bad",
                    "type": "gemm",
                    "args": {
                        "m": 1,
                        "n": 1,
                        "k": 1,
                        "dtype_a": "fp16",
                        "dtype_b": "fp16",
                        "dtype_out": "fp16",
                    },
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="requires BF16"):
        MODULE.load_rows(path)


def test_validate_summary_requires_executed_flops() -> None:
    valid = {
        "hardware_flops": 2 * 2048 * 2048 * 1024,
        "total_time": 0.001,
        "matmul_instruction_count": 1,
        "hbm_read_bytes": 1,
        "hbm_write_bytes": 1,
    }
    MODULE.validate_summary(valid, m=2048, n=2048, k=1024)
    valid["hardware_flops"] -= 1
    MODULE.validate_summary(valid, m=2048, n=2048, k=1024)
    valid["hardware_flops"] = 2 * 2048 * 2048 * 1024 - 2 * 2048 * 1024
    MODULE.validate_summary(valid, m=2048, n=2048, k=1024)
    valid["hardware_flops"] -= 1
    with pytest.raises(RuntimeError, match="hardware FLOPs"):
        MODULE.validate_summary(valid, m=2048, n=2048, k=1024)
    valid["hardware_flops"] = 1
    with pytest.raises(RuntimeError, match="hardware FLOPs"):
        MODULE.validate_summary(valid, m=2048, n=2048, k=1024)


def test_build_rows_records_padding_cost() -> None:
    test_row = {
        "name": "skinny",
        "type": "gemm",
        "args": {
            "m": 16,
            "n": 4096,
            "k": 4096,
            "dtype_a": "bf16",
            "dtype_b": "bf16",
            "dtype_out": "bf16",
        },
    }
    padded = MODULE.executed_shape(test_row)
    measurement = {"device_execution_us": {"p50": 10.0}}
    row = MODULE.build_rows([test_row], {padded: measurement})[0]
    assert row["executed_shape"] == {"m": 2048, "n": 4096, "k": 4096}
    assert row["padding_flops_ratio"] == 128.0
    assert row["device_execution_us"]["p50"] == 10.0


def test_checked_in_trainium_dataset_passes_contract() -> None:
    operatorx = SCRIPT.parents[1]
    dataset = json.loads(
        (operatorx / "data" / "trn3_lnc2_gemm_sweep_20260731.json").read_text()
    )
    corpus = json.loads(
        (operatorx / "testlists" / "tpu_gemm_sweep.json").read_text()
    )
    rows = dataset["rows"]

    assert [row["name"] for row in rows] == [row["name"] for row in corpus]
    assert len(rows) == 20
    assert dataset["methodology"]["full_artifacts_retained"] is False
    assert all(row["device_execution_us"]["count"] == 21 for row in rows)
    assert all(len(row["samples"]) == 21 for row in rows)
    assert all(row["correctness"]["validated"] for row in rows)
    assert all(row["placement"]["compiler_info_lnc"] == 2 for row in rows)
    assert all(row["compiled_program"]["lnc"] == 2 for row in rows)


def test_tile_boundary_corpus_has_expected_plateaus_and_exact_controls() -> None:
    operatorx = SCRIPT.parents[1]
    corpus = MODULE.load_rows(
        operatorx / "testlists" / "trainium_gemm_tile_boundaries.json"
    )
    executed = {MODULE.executed_shape(row) for row in corpus}
    exact = [
        row
        for row in corpus
        if MODULE.executed_shape(row)
        == tuple(int(row["args"][dimension]) for dimension in ("m", "n", "k"))
    ]

    assert len(corpus) == 20
    assert len(executed) == 11
    assert len(exact) == 8
    assert MODULE.executed_shape(corpus[1]) == (2048, 2048, 1024)
    assert MODULE.executed_shape(corpus[2]) == (4096, 2048, 1024)


def test_checked_in_tile_boundary_dataset_passes_contract() -> None:
    operatorx = SCRIPT.parents[1]
    dataset = json.loads(
        (
            operatorx
            / "data"
            / "trn3_lnc2_gemm_tile_boundaries_20260731.json"
        ).read_text()
    )
    corpus = json.loads(
        (
            operatorx
            / "testlists"
            / "trainium_gemm_tile_boundaries.json"
        ).read_text()
    )
    rows = dataset["rows"]

    assert [row["name"] for row in rows] == [row["name"] for row in corpus]
    assert len(rows) == 20
    assert len(
        {
            tuple(row["executed_shape"][dimension] for dimension in ("m", "n", "k"))
            for row in rows
        }
    ) == 11
    assert sum(row["padding_flops_ratio"] == 1.0 for row in rows) == 8
    assert dataset["methodology"]["full_artifacts_retained"] is False
    assert all(row["device_execution_us"]["count"] == 21 for row in rows)
    assert all(len(row["samples"]) == 21 for row in rows)
    assert all(row["correctness"]["validated"] for row in rows)
    assert all(row["placement"]["compiler_info_lnc"] == 2 for row in rows)


def test_exact_holdout_corpus_is_balanced_and_holds_out_mn_pairs() -> None:
    operatorx = SCRIPT.parents[1]
    corpus = MODULE.load_rows(
        operatorx / "testlists" / "trainium_gemm_exact_holdout.json"
    )
    split_counts = {
        split: sum(row["evaluation_split"] == split for row in corpus)
        for split in ("train", "holdout")
    }
    k_counts = {
        k: sum(int(row["args"]["k"]) == k for row in corpus)
        for k in (1024, 2048, 3072, 4096)
    }
    train_pairs = {
        (int(row["args"]["m"]), int(row["args"]["n"]))
        for row in corpus
        if row["evaluation_split"] == "train"
    }
    holdout_pairs = {
        (int(row["args"]["m"]), int(row["args"]["n"]))
        for row in corpus
        if row["evaluation_split"] == "holdout"
    }

    assert len(corpus) == 24
    assert len({MODULE.executed_shape(row) for row in corpus}) == 24
    assert all(
        MODULE.executed_shape(row)
        == tuple(int(row["args"][dimension]) for dimension in ("m", "n", "k"))
        for row in corpus
    )
    assert split_counts == {"train": 16, "holdout": 8}
    assert k_counts == {1024: 6, 2048: 6, 3072: 6, 4096: 6}
    assert train_pairs.isdisjoint(holdout_pairs)
    assert len(holdout_pairs) == 4


def test_checked_in_exact_holdout_dataset_passes_contract() -> None:
    operatorx = SCRIPT.parents[1]
    dataset = json.loads(
        (
            operatorx
            / "data"
            / "trn3_lnc2_gemm_exact_holdout_20260731.json"
        ).read_text()
    )
    corpus = json.loads(
        (operatorx / "testlists" / "trainium_gemm_exact_holdout.json").read_text()
    )
    rows = dataset["rows"]

    assert [row["name"] for row in rows] == [row["name"] for row in corpus]
    assert len(rows) == 24
    assert {row["evaluation_split"] for row in rows} == {"train", "holdout"}
    assert sum(row["evaluation_split"] == "train" for row in rows) == 16
    assert sum(row["evaluation_split"] == "holdout" for row in rows) == 8
    assert all(row["shape"] == row["executed_shape"] for row in rows)
    assert dataset["methodology"]["full_artifacts_retained"] is False
    assert all(row["device_execution_us"]["count"] == 21 for row in rows)
    assert all(len(row["samples"]) == 21 for row in rows)
    assert all(row["correctness"]["validated"] for row in rows)
    assert all(row["placement"]["compiler_info_lnc"] == 2 for row in rows)
    assert all(row["compiled_program"]["lnc"] == 2 for row in rows)
