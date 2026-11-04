"""The PR summary compares each fixed-sequence result with main's latest row of its config."""

import json
import re
import sqlite3
import sys

import pytest

import infx.results.compare_results as compare

SCHEMA = """
CREATE TABLE configs (
    id INTEGER PRIMARY KEY, hardware TEXT, framework TEXT, model TEXT, precision TEXT,
    spec_method TEXT, disagg BOOLEAN, is_multinode BOOLEAN, prefill_tp INTEGER,
    prefill_ep INTEGER, prefill_dp_attention BOOLEAN, decode_tp INTEGER, decode_ep INTEGER,
    decode_dp_attention BOOLEAN
);
CREATE TABLE workflow_runs (id INTEGER PRIMARY KEY, head_branch TEXT);
CREATE TABLE benchmark_results (
    config_id INTEGER, workflow_run_id INTEGER, metrics TEXT, isl INTEGER, osl INTEGER,
    conc INTEGER, date TEXT, error TEXT
);
INSERT INTO workflow_runs VALUES (1, 'main');
"""


class Database:
    """An in-memory SQL database behind psycopg2's pyformat parameters."""

    def __init__(self, configs: list[tuple[str, str, str, float]], framework: str = "sglang"):
        self.connection = sqlite3.connect(":memory:")
        self.connection.executescript(SCHEMA)
        for index, (model, hardware, date, tput) in enumerate(configs, start=1):
            self.connection.execute(
                "INSERT INTO configs VALUES (?, ?, ?, ?, 'fp4', 'none', 0, 0, 8, 1, 0, 8, 1, 0)",
                (index, hardware, framework, model),
            )
            self.connection.execute(
                "INSERT INTO benchmark_results VALUES (?, 1, ?, 8192, 1024, 4, ?, NULL)",
                (index, json.dumps({"tput_per_gpu": tput}), date),
            )

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query: str, params: dict) -> None:
        query = re.sub(r"%\((\w+)\)s", r":\1", query).replace("%%", "%")
        self.rows = self.connection.execute(query, params).fetchall()

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def close(self) -> None:
        self.connection.close()


def summary(tmp_path, monkeypatch, capsys, database: Database, **result) -> str:
    row = {
        "hw": "b200", "infmax_model_prefix": "dsv4", "model": "deepseek-ai/DeepSeek-V4-Pro",
        "framework": "sglang", "precision": "fp4", "spec_decoding": "none", "disagg": False,
        "is_multinode": False, "tp": 8, "ep": 1, "dp_attention": False, "isl": 8192,
        "osl": 1024, "conc": 4, "tput_per_gpu": 110.0, **result,
    }  # fmt: skip
    (tmp_path / "agg.json").write_text(json.dumps([row]))
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(compare.psycopg2, "connect", lambda url: database)
    monkeypatch.setattr(sys, "argv", ["compare_results", str(tmp_path)])
    compare.main()
    return capsys.readouterr().out


def test_a_model_prefix_is_not_compared_with_a_longer_prefix_it_begins(
    tmp_path, monkeypatch, capsys
):
    database = Database(
        [("dsv4", "b200", "2026-10-01", 100.0), ("dsv41flash", "b200", "2026-10-20", 400.0)]
    )

    out = summary(tmp_path, monkeypatch, capsys, database)

    assert r"\text{+10.00}$ (+10.0%)" in out


@pytest.mark.parametrize(
    ("runner", "hardware"), [("cluster:b200-nscale", "b200"), ("gb300-nv", "gb300")]
)
def test_a_runner_label_finds_the_baseline_of_its_gpu(
    tmp_path, monkeypatch, capsys, runner, hardware
):
    database = Database([("dsv4", hardware, "2026-10-01", 100.0)])

    out = summary(tmp_path, monkeypatch, capsys, database, hw=runner)

    assert r"\text{+10.00}$ (+10.0%)" in out


def test_sglang_disagg_finds_the_baseline_ingestion_stored_as_mori_sglang(
    tmp_path, monkeypatch, capsys
):
    database = Database([("dsr1", "mi355x", "2026-10-01", 100.0)], framework="mori-sglang")

    out = summary(
        tmp_path, monkeypatch, capsys, database,
        hw="cluster:mi355x-amds", infmax_model_prefix="dsr1", framework="sglang-disagg",
    )  # fmt: skip

    assert r"\text{+10.00}$ (+10.0%)" in out
