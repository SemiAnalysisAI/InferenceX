"""CLI routing tests only: mocked receipts are not GPU benchmark evidence."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluator.cli import _load_mvp_object, main


def test_gpu_preview_is_default_and_never_executes(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    calls = []
    monkeypatch.setitem(sys.modules, "evaluator.gpu_job", SimpleNamespace(
        preview_gpu_job=lambda value: {"test_fixture": True, "preview": value},
        run_gpu_job=lambda *args: calls.append(args),
    ))
    assert main(["gpu-job", str(spec)]) == 0
    assert json.loads(capsys.readouterr().out)["preview"] == {}
    assert calls == []
    assert list(tmp_path.iterdir()) == [spec]


@pytest.mark.parametrize("changes,expected", [
    ({}, 0),
    ({"ci_accepted": False}, 2),
    ({"regression_status": "fail"}, 1),
])
def test_gpu_cli_never_confuses_completion_with_acceptance(tmp_path, monkeypatch, changes, expected):
    path = tmp_path / "spec.json"
    path.write_text("{}")
    result = {"status": "complete", "measurement_status": "complete", "cleanup_status": "clean",
              "regression_status": "pass", "ci_accepted": True, **changes}
    monkeypatch.setitem(sys.modules, "evaluator.gpu_job", SimpleNamespace(
        preview_gpu_job=lambda value: value, run_gpu_job=lambda *args: result,
    ))
    assert main(["gpu-job", str(path), "--execute", "--output", str(tmp_path / "out")]) == expected


def test_gpu_execute_requires_output_before_supervision(tmp_path, monkeypatch, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    calls = []
    monkeypatch.setitem(sys.modules, "evaluator.gpu_job", SimpleNamespace(
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
