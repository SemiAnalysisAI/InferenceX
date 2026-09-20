"""Provisioning works from observed shared assets, without inventing site identities."""

import json
from pathlib import Path

import pytest

from infx.srt_slurm.provision import ProvisionConfig, inspect_assets, main


def asset_config(tmp_path: Path) -> ProvisionConfig:
    hub = tmp_path / "hub"
    model = hub / "models--example--model/snapshots" / ("a" * 40)
    traces = hub / "datasets--example--traces/snapshots" / ("b" * 40)
    model.mkdir(parents=True)
    traces.mkdir(parents=True)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shard.safetensors"}})
    )
    (model / "shard.safetensors").write_bytes(b"weights")
    (tmp_path / "image.sqsh").write_bytes(b"squash")
    return ProvisionConfig(
        schema_version=1,
        shared_root=str(tmp_path / "prepared"),
        hub_cache=str(hub),
        image_path=str(tmp_path / "image.sqsh"),
        image_reference="example/image:version",
        model_repository="example/model",
        model_revision="a" * 40,
        dataset_repository="example/traces",
        dataset_revision="b" * 40,
    )


def test_inspection_observes_missing_payload_and_creates_no_shared_root(tmp_path):
    config = asset_config(tmp_path)
    report = inspect_assets(config)
    assert report["assets_present"] is True
    assert report["qualification_complete"] is False
    assert report["paths"]["image"]["size"] == 6
    model = Path(report["paths"]["model_snapshot"]["path"])
    (model / "shard.safetensors").unlink()
    report = inspect_assets(config)
    assert report["assets_present"] is False
    assert report["missing_model_shards"] == ["shard.safetensors"]
    assert not Path(config.shared_root).exists()


def test_inspection_rejects_shard_path_escape(tmp_path):
    config = asset_config(tmp_path)
    model = Path(config.hub_cache) / "models--example--model/snapshots" / ("a" * 40)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../escape"}})
    )
    with pytest.raises(ValueError, match="unsafe shard path"):
        inspect_assets(config)


def test_inspection_cli_requires_worker_and_controller_slurm_tools(
    tmp_path, monkeypatch
):
    config = asset_config(tmp_path)
    path = tmp_path / "site.json"
    path.write_text(config.model_dump_json())
    output = tmp_path / "report"
    monkeypatch.setattr(
        "infx.srt_slurm.provision.shutil.which",
        lambda name: None if name in {"srun", "scontrol"} else "/usr/bin/" + name,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "provision",
            "--config",
            str(path),
            "--output",
            str(output),
            "--operation",
            "inspect",
        ],
    )
    assert main() == 1
    report = json.loads((output / "inventory.json").read_text())
    assert report["assets_present"] is True
    assert report["missing_slurm_tools"] == ["srun", "scontrol"]
    assert not Path(config.shared_root).exists()
