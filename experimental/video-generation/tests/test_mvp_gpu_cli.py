"""CLI routing tests only: mocked receipts are not GPU benchmark evidence."""

import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.cli import _load_mvp_object, main


def test_gpu_preview_is_default_and_never_executes(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    calls = []
    monkeypatch.setitem(sys.modules, "evaluator.mvp_gpu_job", SimpleNamespace(
        preview_gpu_job=lambda value: {"test_fixture": True, "preview": value},
        run_gpu_job=lambda *args: calls.append(args),
    ))
    assert main(["gpu-job", str(spec)]) == 0
    assert json.loads(capsys.readouterr().out)["preview"] == {}
    assert calls == []
    assert list(tmp_path.iterdir()) == [spec]


@pytest.mark.parametrize("status,regression,accepted,expected", [
    ("complete", "pass", True, 0),
    ("complete", "inconclusive", False, 2),
    ("complete", "pass", False, 2),
    ("complete", "fail", False, 1),
    ("failed", "inconclusive", False, 1),
    ("aborted", "inconclusive", False, 1),
])
def test_gpu_cli_never_confuses_completion_with_acceptance(tmp_path, monkeypatch, status, regression, accepted, expected):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    monkeypatch.setitem(sys.modules, "evaluator.mvp_gpu_job", SimpleNamespace(
        preview_gpu_job=lambda value: value,
        run_gpu_job=lambda *args: {
            "test_fixture": True, "status": status,
            "regression_status": regression, "ci_accepted": accepted,
            "measurement_status": "complete", "cleanup_status": "clean",
        },
    ))
    assert main(["gpu-job", str(spec), "--execute", "--output", str(tmp_path / "out")]) == expected


@pytest.mark.parametrize("changes,expected", [
    ({"status": "failed"}, 1), ({"measurement_status": "incomplete"}, 2),
    ({"cleanup_status": "failed"}, 1), ({"regression_status": "inconclusive"}, 2),
])
def test_contradictory_gpu_acceptance_flag_cannot_return_success(tmp_path, monkeypatch, changes, expected):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    result = {"status": "complete", "measurement_status": "complete", "cleanup_status": "clean",
              "regression_status": "pass", "ci_accepted": True, **changes}
    monkeypatch.setitem(sys.modules, "evaluator.mvp_gpu_job", SimpleNamespace(
        preview_gpu_job=lambda value: value, run_gpu_job=lambda *args: result,
    ))
    assert main(["gpu-job", str(spec), "--execute", "--output", str(tmp_path / "out")]) == expected


def test_gpu_execute_requires_output_before_supervision(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    calls = []
    monkeypatch.setitem(sys.modules, "evaluator.mvp_gpu_job", SimpleNamespace(
        preview_gpu_job=lambda value: value,
        run_gpu_job=lambda *args: calls.append(args),
    ))
    assert main(["gpu-job", str(spec), "--execute"]) == 2
    assert "requires --output" in capsys.readouterr().err
    assert calls == []


def test_gpu_job_configuration_refuses_fifo_without_blocking(tmp_path, capsys):
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    assert main(["gpu-job", str(fifo)]) == 2
    assert "regular file" in capsys.readouterr().err


def test_mvp_configuration_refuses_device_and_symlink(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        _load_mvp_object(Path("/dev/zero"))
    original = tmp_path / "original.json"
    original.write_text("{}")
    linked = tmp_path / "linked.json"
    linked.symlink_to(original)
    with pytest.raises(OSError):
        _load_mvp_object(linked)


def test_configuration_read_remains_bounded_if_file_grows_after_stat(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    with path.open("wb") as stream:
        stream.truncate(4 * 1024 * 1024 + 10)
    monkeypatch.setattr(os, "fstat", lambda _: SimpleNamespace(st_mode=stat.S_IFREG, st_size=0))
    with pytest.raises(ValueError, match="exceeds 4 MiB while being read"):
        _load_mvp_object(path)


def test_rendering_failure_evidence_is_not_a_gpu_acceptance(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "evaluator.mvp_gpu_report", SimpleNamespace(
        write_gpu_report=lambda *args: {"test_fixture": True, "ci_accepted": False, "status": "incomplete"},
    ))
    assert main(["gpu-report", str(tmp_path), "--output", str(tmp_path / "report")]) == 0
    assert json.loads(capsys.readouterr().out)["ci_accepted"] is False
