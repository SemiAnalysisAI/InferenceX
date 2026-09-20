"""Inspect and provision explicit shared H100 assets on a CI login runner."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from infx.benchmarks.common import read_json, write_json


class ProvisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    shared_root: str
    hub_cache: str
    image_path: str
    image_reference: str
    model_repository: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_repository: str
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")

    @field_validator("shared_root", "hub_cache", "image_path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path.resolve().is_relative_to("/workspace"):
            raise ValueError("provisioning requires explicit shared paths outside /workspace")
        return value

    @field_validator("model_repository", "dataset_repository")
    @classmethod
    def repository_name(cls, value: str) -> str:
        if len(value.split("/")) != 2 or any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("expected owner/name Hugging Face repository")
        return value


def snapshot(config: ProvisionConfig, *, dataset: bool) -> Path:
    repository = config.dataset_repository if dataset else config.model_repository
    revision = config.dataset_revision if dataset else config.model_revision
    prefix = "datasets--" if dataset else "models--"
    return (
        Path(config.hub_cache) / (prefix + repository.replace("/", "--")) / "snapshots" / revision
    )


def inspect_assets(config: ProvisionConfig) -> dict[str, Any]:
    """Report actual existing paths without altering shared caches or taking an allocation."""
    paths = {
        "shared_parent": Path(config.shared_root).parent,
        "hub_cache": Path(config.hub_cache),
        "image": Path(config.image_path),
        "model_snapshot": snapshot(config, dataset=False),
        "dataset_snapshot": snapshot(config, dataset=True),
    }
    entries = {}
    for name, path in paths.items():
        entries[name] = {
            "path": str(path),
            "canonical_path": str(path.resolve()),
            "exists": path.exists(),
            "is_directory": path.is_dir(),
            "size": path.stat().st_size if path.is_file() else None,
        }
    model = paths["model_snapshot"]
    indexes = sorted(model.glob("*.safetensors.index.json")) + sorted(
        model.glob("*.bin.index.json")
    )
    missing_shards = []
    for index in indexes:
        for name in set(read_json(index).get("weight_map", {}).values()):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("model index contains an unsafe shard path")
            if not (model / relative).is_file():
                missing_shards.append(name)
    ready = (
        all(value["exists"] for value in entries.values()) and bool(indexes) and not missing_shards
    )
    slurm_tools = {
        name: shutil.which(name)
        for name in ("sbatch", "squeue", "sacct", "scancel", "srun", "scontrol")
    }
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "assets_present": ready,
        "paths": entries,
        "model_indexes": [path.name for path in indexes],
        "missing_model_shards": sorted(missing_shards),
        "slurm_tools": slurm_tools,
        "missing_slurm_tools": [name for name, path in slurm_tools.items() if path is None],
        "qualification_complete": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation", choices=("inspect", "provision"), required=True)
    args = parser.parse_args()
    config = ProvisionConfig.model_validate(read_json(args.config))
    report = inspect_assets(config)
    report["head_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    write_json(args.output / "inventory.json", report)
    print(f"Prepared asset inventory written; assets_present={report['assets_present']}")
    if not report["assets_present"] or report["missing_slurm_tools"]:
        return 1
    if args.operation == "provision":
        from infx.srt_slurm.provision_runtime import provision

        namespace = f"{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}"
        provision(config, Path.cwd(), args.output.resolve(), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
