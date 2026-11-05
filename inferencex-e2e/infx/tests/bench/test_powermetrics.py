"""``powermetrics`` with a stub sudo on PATH."""

from __future__ import annotations

import plistlib

import pytest

from infx.bench import powermetrics

SAMPLE = plistlib.dumps({
    "processor": {
        "combined_power": 120500.0, "cpu_power": 20000.0, "gpu_power": 100000.0, "ane_power": 500.0,
    },
})  # fmt: skip


@pytest.fixture
def sudo(tmp_path, monkeypatch):
    """Put a stub sudo first on PATH; it runs the script written to the returned path."""
    script = tmp_path / "sudo"
    monkeypatch.setenv("PATH", str(tmp_path), prepend=":")
    return script


def write_stub(script, body: bytes, rc: int = 0) -> None:
    (script.parent / "output").write_bytes(body)
    script.write_text(f"#!/bin/sh\ncat {script.parent}/output\nexit {rc}\n")
    script.chmod(0o755)


def test_each_finished_plist_is_one_row_in_watts(sudo, tmp_path):
    write_stub(sudo, b"\n" + SAMPLE + b"\0\n" + SAMPLE + b"\0\n" + SAMPLE[:40])
    output = tmp_path / "samples.csv"

    assert powermetrics.main(["--output", str(output)]) == 0

    header, *rows = output.read_text().splitlines()
    assert header == "timestamp_unix,combined_w,cpu_w,gpu_w,ane_w"
    assert [row.split(",")[1:] for row in rows] == [["120.500", "20.000", "100.000", "0.500"]] * 2
    first, second = (float(row.split(",")[0]) for row in rows)
    assert first < second


def test_a_refused_sudo_leaves_only_the_header_and_its_status(sudo, tmp_path):
    write_stub(sudo, b"", rc=1)
    output = tmp_path / "samples.csv"

    assert powermetrics.main(["--output", str(output)]) == 1

    assert output.read_text() == "timestamp_unix,combined_w,cpu_w,gpu_w,ane_w\n"
