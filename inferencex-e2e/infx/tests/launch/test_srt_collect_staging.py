"""Exit staging of a single-node point: power-package copies are best-effort unless power is required."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from infx.launch.drivers.srt import collect
from infx.launch.drivers.srt.config import EXPORTER_PROVENANCE


class _Backend:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.cancelled: list[object] = []

    def cancel(self, job: object) -> None:
        self.cancelled.append(job)

    def fetch_outputs(self, job: object, fetched: Path) -> Path:
        return self.output


def _point(tmp_path: Path, *, require_power: bool) -> tuple[SimpleNamespace, SimpleNamespace, Path]:
    output = tmp_path / "outputs" / "42"
    logs = output / "logs"
    (logs / "power").mkdir(parents=True)
    (logs / "power" / "samples.csv").write_text("retained native samples\n")
    (logs / "point.json").write_text('{"completed": 2}\n')
    (logs / "power_validation_point.json").write_text('{"power_valid": true}\n')
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in (EXPORTER_PROVENANCE, "power-producer-sha.txt"):
        (workspace / name).write_text(f"{name}\n")
    request = SimpleNamespace(eval_only=False, require_power=require_power, result_filename="point")
    run = SimpleNamespace(backend=_Backend(output), request=request, workspace=workspace)
    submitted = SimpleNamespace(recover=lambda backend: object())
    return run, submitted, workspace


@pytest.mark.parametrize("require_power", [False, True])
def test_power_package_copy_failure_fails_the_point_only_when_power_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], require_power: bool
) -> None:
    run, submitted, workspace = _point(tmp_path, require_power=require_power)

    def exploding_copytree(*args: object, **kwargs: object) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(shutil, "copytree", exploding_copytree)

    rc = collect.finish_single_node(run, submitted, tmp_path / "fetched")

    assert rc == int(require_power)
    assert (workspace / "point.json").read_text() == '{"completed": 2}\n'
    assert (workspace / "power_validation_point.json").is_file()
    level = "ERROR" if require_power else "WARNING"
    assert f"{level}: failed to stage the native power package" in capsys.readouterr().err


def test_power_package_is_staged_beside_the_result(tmp_path: Path) -> None:
    run, submitted, workspace = _point(tmp_path, require_power=True)

    assert collect.finish_single_node(run, submitted, tmp_path / "fetched") == 0
    assert (workspace / "LOGS" / "power" / "samples.csv").read_text() == "retained native samples\n"
    assert (workspace / "LOGS" / "power" / "power-producer-sha.txt").is_file()
    assert (workspace / "point.json").is_file()
    assert (workspace / "power_validation_point.json").is_file()
