"""Freeze already materialized clients, model assets and evaluation resources before allocation."""

from __future__ import annotations

import argparse
import os
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from .common import read_json, sha256_file, verify_snapshot_assets, write_json
from .identity import (
    AGENTX_REVISION,
    LM_EVAL_REVISION,
    capture_identity,
    require_source_revision,
)
from .spec import (
    PositiveInt,
    PreparedFile,
    RuntimeSpec,
    StrictModel,
    secret_environment_key,
    validate_environment,
)


class ClientSite(StrictModel):
    python: str
    distributions: Annotated[list[str], Field(min_length=1)]
    env: dict[str, str]
    env_unset: list[str]
    asset_roots: Annotated[list[str], Field(min_length=1)]
    asset_files: list[str]
    model_path: str
    timeout_seconds: PositiveInt
    terminate_grace_seconds: PositiveInt

    @field_validator("python", "model_path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        return PreparedFile.absolute_path(value)

    @field_validator("asset_roots", "asset_files")
    @classmethod
    def absolute_paths(cls, values: list[str]) -> list[str]:
        return [PreparedFile.absolute_path(value) for value in values]

    @model_validator(mode="after")
    def environment_contract(self) -> Self:
        validate_environment(self.env, self.env_unset)
        return self


def bind_file(path: Path) -> PreparedFile:
    """Hash the actual bytes, rejecting replacement/truncation while reading."""
    before = path.stat()
    if not path.is_file():
        raise ValueError(f"required prepared asset is not a file: {path}")
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"asset changed during preparation: {path}")
    return PreparedFile(path=str(path.absolute()), sha256=digest)


def collect_assets(site: ClientSite) -> list[PreparedFile]:
    paths = {Path(path) for path in site.asset_files}
    for directory in site.asset_roots:
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"prepared asset root is unavailable: {root}")
        children = list(root.rglob("*"))
        if any(child.is_symlink() and not child.exists() for child in children):
            raise ValueError(f"prepared asset root contains dangling blob links: {root}")
        payloads = [child for child in children if child.is_file()]
        if not payloads:
            raise ValueError(f"prepared asset root is empty: {root}")
        paths.update(payloads)
    assets = [bind_file(path) for path in sorted(paths)]
    bound = {Path(asset.path).resolve() for asset in assets}
    model = Path(site.model_path)
    config = model / "config.json"
    indexes = list(model.glob("*.safetensors.index.json")) + list(model.glob("*.bin.index.json"))
    if config.resolve() not in bound or not indexes:
        raise ValueError("prepared model requires bound config.json and a weight shard index")
    for index in indexes:
        if index.resolve() not in bound:
            raise ValueError(f"model shard index is not bound: {index}")
        mapping = read_json(index).get("weight_map", {})
        if not mapping:
            raise ValueError(f"model shard index contains no weights: {index}")
        for name in set(mapping.values()):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"model shard escapes model directory: {name}")
            if (model / relative).resolve() not in bound or (model / relative).stat().st_size == 0:
                raise ValueError(f"model shard is missing or unbound: {name}")
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "tokenizer.tiktoken")
    if not any((model / name).resolve() in bound for name in tokenizer_files):
        raise ValueError("prepared model tokenizer payload is missing or unbound")
    return assets


def prepare(site: ClientSite, kind: Literal["agentx", "eval"], output: Path) -> dict[str, Any]:
    """Prepare a new, exclusive directory; never mutate a previously prepared environment."""
    output = output.absolute()
    assets = collect_assets(site)
    distribution, revision, loader = (
        ("aiperf", AGENTX_REVISION, "semianalysis_cc_traces_weka_062126")
        if kind == "agentx"
        else ("lm-eval", LM_EVAL_REVISION, None)
    )
    if distribution not in site.distributions:
        raise ValueError(f"site must bind the {distribution} installed distribution")
    env = dict(os.environ)
    for key in (*site.env_unset, "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    for key in list(env):
        if key.startswith("AIPERF_") or secret_environment_key(key):
            env.pop(key)
    env.update(site.env)
    identity = capture_identity(site.python, site.distributions, dataset_loader=loader, env=env)
    require_source_revision(identity, distribution, revision)
    resources: dict[str, Any] = {"schema_version": 1, "kind": kind, "model_path": site.model_path}
    if kind == "agentx":
        dataset = Path(site.env["HF_HUB_CACHE"]) / "datasets--semianalysisai--cc-traces-weka-062126"
        reference = dataset / "refs/main"
        resources["dataset_revision"] = reference.read_text().strip()
        if (
            identity.get("dataset_resolution", {}).get("metadata", {}).get("hf_dataset_name")
            != "semianalysisai/cc-traces-weka-062126"
        ):
            raise ValueError("installed plugin resolves a different dataset")
    output.mkdir(parents=True, exist_ok=False)
    try:
        identity_path = output / "identity.json"
        write_json(identity_path, identity)
        runtime = RuntimeSpec(
            python=site.python,
            identity=bind_file(identity_path),
            distributions=site.distributions,
            env=site.env,
            env_unset=site.env_unset,
            assets=assets,
            timeout_seconds=site.timeout_seconds,
            terminate_grace_seconds=site.terminate_grace_seconds,
        )
        if kind == "agentx":
            # Reuse the same actual client cache boundary before allocating a server.
            from .agentx import verify_prepared_corpus

            verify_prepared_corpus(runtime, resources["dataset_revision"])
        else:
            resources["dataset_revision"] = verify_snapshot_assets(
                runtime, "openai/gsm8k", expected_revision=None, only_snapshot=False
            )
            for key, source in {
                "task": files("infx.evals").joinpath("gsm8k.yaml"),
                "document_identities": files("infx.benchmarks").joinpath(
                    "resources/gsm8k-test-doc-hashes.json"
                ),
            }.items():
                target = output / source.name
                target.write_bytes(source.read_bytes())
                resources[key] = bind_file(target).model_dump(mode="json")
        write_json(output / "runtime.json", runtime.model_dump(mode="json"))
        write_json(output / "prepared-resources.json", resources)
        for path in output.iterdir():
            path.chmod(0o444)
    except BaseException:
        shutil.rmtree(output)
        raise
    return resources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--kind", choices=("agentx", "eval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(ClientSite.model_validate(read_json(args.site)), args.kind, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
