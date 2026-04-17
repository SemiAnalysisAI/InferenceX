"""Unit tests for the ISB1 mechanism_eval schema helpers."""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

SCRIPTS_DIR = UTILS_DIR.parent / "datasets" / "isb1" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

mechanism_eval = importlib.import_module("mechanism_eval")
isb1_results_db = importlib.import_module("isb1_results_db")


def test_build_mechanism_fields_defaults_to_baseline(monkeypatch):
    for _, env_var, _ in mechanism_eval.MECHANISM_FIELDS:
        monkeypatch.delenv(env_var, raising=False)

    fields = mechanism_eval.build_mechanism_fields()

    assert fields["mechanism"] == "baseline"
    assert fields["mechanism_variant"] is None
    for row_key in (
        "compression_method",
        "compression_scope",
        "compression_ratio",
        "compression_overhead_ms",
        "decompression_overhead_ms",
        "quality_eval_id",
        "quality_eval_status",
        "quality_delta_summary",
        "draft_model_id",
        "speculative_acceptance_rate",
        "speculative_wasted_tokens",
        "mechanism_notes",
    ):
        assert fields[row_key] is None, f"expected {row_key} to default to None"


def test_build_mechanism_fields_coerces_numeric_env(monkeypatch):
    monkeypatch.setenv("MECHANISM", "kv_quantization")
    monkeypatch.setenv("MECHANISM_VARIANT", "fp8_e4m3")
    monkeypatch.setenv("COMPRESSION_METHOD", "fp8_e4m3")
    monkeypatch.setenv("COMPRESSION_SCOPE", "kv_cache")
    monkeypatch.setenv("COMPRESSION_RATIO", "0.5")
    monkeypatch.setenv("COMPRESSION_OVERHEAD_MS", "12.5")
    monkeypatch.setenv("DECOMPRESSION_OVERHEAD_MS", "4.25")
    monkeypatch.setenv("SPECULATIVE_WASTED_TOKENS", "128")
    monkeypatch.setenv("SPECULATIVE_ACCEPTANCE_RATE", "0.82")
    monkeypatch.setenv("QUALITY_EVAL_ID", "ruler_v1")
    monkeypatch.setenv("QUALITY_EVAL_STATUS", "completed")

    fields = mechanism_eval.build_mechanism_fields()

    assert fields["mechanism"] == "kv_quantization"
    assert fields["mechanism_variant"] == "fp8_e4m3"
    assert fields["compression_ratio"] == pytest.approx(0.5)
    assert fields["compression_overhead_ms"] == pytest.approx(12.5)
    assert fields["decompression_overhead_ms"] == pytest.approx(4.25)
    assert fields["speculative_wasted_tokens"] == 128
    assert fields["speculative_acceptance_rate"] == pytest.approx(0.82)
    assert fields["quality_eval_id"] == "ruler_v1"
    assert fields["quality_eval_status"] == "completed"


def test_build_mechanism_fields_rejects_bad_numeric(monkeypatch):
    monkeypatch.setenv("COMPRESSION_RATIO", "not-a-number")
    monkeypatch.setenv("SPECULATIVE_WASTED_TOKENS", "oops")

    fields = mechanism_eval.build_mechanism_fields()

    assert fields["compression_ratio"] is None
    assert fields["speculative_wasted_tokens"] is None


def test_validate_mechanism_fields_registered_pair():
    fields = {
        "mechanism": "kv_quantization",
        "mechanism_variant": "fp8_e4m3",
        "quality_eval_id": "ruler_v1",
        "quality_eval_status": "completed",
    }
    record = mechanism_eval.validate_mechanism_fields(fields)

    assert record["mechanism_eval_registered"] is True
    assert record["quality_eval_registered"] is True
    assert record["quality_eval_status_known"] is True
    assert record["issues"] == []


def test_validate_mechanism_fields_unregistered_variant_flags_issue():
    fields = {
        "mechanism": "kv_quantization",
        "mechanism_variant": "made_up_variant",
        "quality_eval_id": "ruler_v1",
        "quality_eval_status": "completed",
    }
    record = mechanism_eval.validate_mechanism_fields(fields)

    assert record["mechanism_eval_registered"] is False
    assert any(
        "not registered in mechanism_variant_registry.json" in issue
        for issue in record["issues"]
    )


def test_validate_mechanism_fields_unregistered_quality_eval_id():
    fields = {
        "mechanism": "kv_quantization",
        "mechanism_variant": "fp8_e4m3",
        "quality_eval_id": "nonexistent_eval",
        "quality_eval_status": "completed",
    }
    record = mechanism_eval.validate_mechanism_fields(fields)

    assert record["quality_eval_registered"] is False
    assert any(
        "not registered in quality_eval_registry.json" in issue
        for issue in record["issues"]
    )


def test_validate_mechanism_fields_unknown_status():
    fields = {
        "mechanism": "kv_quantization",
        "mechanism_variant": "fp8_e4m3",
        "quality_eval_id": "ruler_v1",
        "quality_eval_status": "maybe",
    }
    record = mechanism_eval.validate_mechanism_fields(fields)

    assert record["quality_eval_status_known"] is False
    assert any("outside the accepted set" in issue for issue in record["issues"])


def test_validate_mechanism_fields_speculative_requires_draft_model():
    fields = {
        "mechanism": "speculative_decoding",
        "mechanism_variant": "eagle3",
        "draft_model_id": None,
    }
    record = mechanism_eval.validate_mechanism_fields(fields)

    assert any("requires draft_model_id" in issue for issue in record["issues"])


def test_validate_mechanism_fields_baseline_passes_without_quality():
    fields = {"mechanism": "baseline", "mechanism_variant": "none"}
    record = mechanism_eval.validate_mechanism_fields(fields)

    assert record["mechanism_eval_registered"] is True
    assert record["quality_eval_registered"] is None
    assert record["issues"] == []


def test_row_requires_completed_quality_eval_matrix():
    assert mechanism_eval.row_requires_completed_quality_eval(
        "kv_quantization", "supported"
    ) is True
    assert mechanism_eval.row_requires_completed_quality_eval(
        "kv_compression", "supported"
    ) is True
    assert mechanism_eval.row_requires_completed_quality_eval(
        "compressed_attention", "supported"
    ) is True
    # Non-supported tier never requires completed eval.
    assert mechanism_eval.row_requires_completed_quality_eval(
        "kv_quantization", "reviewed_preview"
    ) is False
    # Baseline never requires.
    assert mechanism_eval.row_requires_completed_quality_eval(
        "baseline", "supported"
    ) is False
    # Speculative decoding is governed by a separate predicate, not this hard rule.
    assert mechanism_eval.row_requires_completed_quality_eval(
        "speculative_decoding", "supported"
    ) is False


def test_registry_files_are_valid_json_and_match_expectations():
    mechanism_registry = mechanism_eval.load_mechanism_registry()
    quality_registry = mechanism_eval.load_quality_registry()

    # Every compression mechanism in the module matches a variant in the registry.
    variants_by_mechanism: dict[str, set[str]] = {}
    for entry in mechanism_registry["variants"]:
        variants_by_mechanism.setdefault(entry["mechanism"], set()).add(
            entry["mechanism_variant"]
        )

    for compression_mechanism in mechanism_eval.COMPRESSION_MECHANISMS:
        assert compression_mechanism in variants_by_mechanism, (
            f"compression mechanism {compression_mechanism} is missing from the registry"
        )

    quality_ids = mechanism_eval.registered_quality_ids(quality_registry)
    assert {"ruler_v1", "longbench_v2", "humaneval", "math_500"}.issubset(quality_ids)


def test_isb1_results_db_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "idempotent.db"
    conn = isb1_results_db.connect_db(db_path)
    # Second ensure_db call must not raise; the migration uses IF NOT EXISTS
    # logic in the form of try/except OperationalError for ALTER TABLE.
    isb1_results_db.ensure_db(conn)

    cursor = conn.execute(f"PRAGMA table_info({isb1_results_db.TABLE_NAME})")
    columns = {row[1] for row in cursor.fetchall()}

    expected = {
        "mechanism",
        "mechanism_variant",
        "compression_method",
        "compression_scope",
        "compression_ratio",
        "compression_overhead_ms",
        "decompression_overhead_ms",
        "quality_eval_id",
        "quality_eval_status",
        "quality_delta_summary",
        "draft_model_id",
        "speculative_acceptance_rate",
        "speculative_wasted_tokens",
        "mechanism_notes",
        "mechanism_eval_registered",
        "quality_eval_registered",
    }
    missing = expected - columns
    assert not missing, f"expected mechanism columns missing after migration: {missing}"

    conn.close()


def test_isb1_results_db_migration_upgrades_legacy_schema(tmp_path):
    """A pre-mechanism_eval database should gain the new columns on re-open."""
    db_path = tmp_path / "legacy.db"

    # Construct a legacy-shaped database by running only a minimal CREATE TABLE
    # with the pre-mechanism_eval column set, then re-open via ensure_db.
    legacy_conn = sqlite3.connect(db_path)
    legacy_conn.execute(
        f"""
        CREATE TABLE {isb1_results_db.TABLE_NAME} (
            id INTEGER PRIMARY KEY,
            run_id TEXT,
            timestamp TEXT,
            gpu_type TEXT,
            model TEXT,
            engine TEXT,
            context_band TEXT,
            max_model_len INTEGER,
            tp INTEGER,
            raw_result_json TEXT,
            status TEXT,
            error_message TEXT
        )
        """
    )
    legacy_conn.commit()
    legacy_conn.close()

    conn = isb1_results_db.connect_db(db_path)
    cursor = conn.execute(f"PRAGMA table_info({isb1_results_db.TABLE_NAME})")
    columns = {row[1] for row in cursor.fetchall()}

    assert "mechanism" in columns
    assert "quality_eval_id" in columns
    assert "speculative_acceptance_rate" in columns

    conn.close()
