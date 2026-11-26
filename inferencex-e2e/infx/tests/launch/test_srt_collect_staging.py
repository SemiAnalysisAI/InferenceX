"""Exit staging of a single-node point: power-package copies are best-effort unless power is required."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from infx.launch.drivers.srt import collect
from infx.launch.drivers.srt.config import EXPORTER_PROVENANCE
from infx.launch.lifecycle import Lifecycle
from infx.matrix.validation import VIDEO_SPEC_DIR
from infx.srt_slurm.video import load_spec
from infx.tests.srt_recipes import VIDEO_SET, VIDEO_SPEC, write_video_specs


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
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in (EXPORTER_PROVENANCE, "power-producer-sha.txt"):
        (workspace / name).write_text(f"{name}\n")
    request = SimpleNamespace(eval_only=False, al_gate=False, require_power=require_power, result_filename="point")
    run = SimpleNamespace(backend=_Backend(output), request=request, workspace=workspace, life=Lifecycle())
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
    level = "ERROR" if require_power else "WARNING"
    assert f"{level}: failed to stage the native power package" in capsys.readouterr().err
    error = run.life.event.finish(rc).error
    if require_power:
        assert (error.stage, error.type) == ("collect", "ArtifactError")
    else:
        assert error is None


def test_power_package_is_staged_beside_the_result(tmp_path: Path) -> None:
    run, submitted, workspace = _point(tmp_path, require_power=True)

    assert collect.finish_single_node(run, submitted, tmp_path / "fetched") == 0
    assert (workspace / "LOGS" / "power" / "samples.csv").read_text() == "retained native samples\n"
    assert (workspace / "LOGS" / "power" / "power-producer-sha.txt").is_file()
    assert (workspace / "point.json").is_file()


def _video_point(tmp_path: Path) -> tuple[SimpleNamespace, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_video_specs(tmp_path)
    bound = tmp_path / "bound.yaml"
    bound.write_text(yaml.safe_dump({"benchmark": {"env": {
        "VIDEO_SET": VIDEO_SET, "VIDEO_RUNTIME": "sglang", "VIDEO_RUNTIME_REVISION": "b" * 40,
        "VIDEO_PLAN_SHA256": load_spec(VIDEO_SPEC, root=workspace, set_id=VIDEO_SET)["plan_sha256"],
    }}}))  # fmt: skip
    run = SimpleNamespace(workspace=workspace, request=SimpleNamespace(env={"VIDEO_SPEC": VIDEO_SPEC}))
    return run, bound


def test_video_identity_carries_the_size_computed_from_the_run_tree_and_returns_the_plan(tmp_path: Path) -> None:
    run, bound = _video_point(tmp_path)

    plan = collect.stage_video_identity(run, bound)

    identity = json.loads((run.workspace / "video-identity.json").read_text())
    assert plan == identity["workload_plan"] and plan["plan_id"] == "fixture-plan"
    # The fixture spec: 3 latent frames of one patch plus 2 audio tokens, in one pass.
    size = identity["workload_size"]
    assert (size["latent_tokens"], size["dit_forward_passes"], size["dit_flops_per_clip"]) == (5, 1, 110)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda spec: spec.pop("workload_size"), "declares no workload_size"),
        # The bound recipe keeps the plan hash of the spec as the binder read it.
        (
            lambda spec: spec["generation"].update(num_inference_steps=2),
            "bound video identity disagrees with the selected workload",
        ),
    ],
)
def test_video_identity_refuses_a_spec_without_a_size_or_edited_after_binding(
    tmp_path: Path, edit: Callable[[dict[str, Any]], Any], message: str
) -> None:
    run, bound = _video_point(tmp_path)
    path = tmp_path / VIDEO_SPEC_DIR / VIDEO_SPEC
    spec = yaml.safe_load(path.read_text())
    edit(spec)
    path.write_text(yaml.safe_dump(spec))

    with pytest.raises(ValueError, match=message):
        collect.stage_video_identity(run, bound)
