"""Versioned execution references and content identities for native SRT jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExecutionReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    runtime: Literal["srt-slurm"]
    contract_version: Literal[1] = Field(alias="contract-version")
    recipe: str
    profile: str
    runtime_lock: str = Field(alias="runtime-lock")
    client_policy: str = Field(alias="client-policy")
    input_digests: dict[str, str] | None = Field(default=None, alias="input-digests")

    @field_validator("recipe", "profile", "runtime_lock", "client_policy")
    @classmethod
    def repository_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("execution references must be contained repository paths")
        return value


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject accidental duplicate mappings before native schema validation."""


def _unique_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r} at {key_node.start_mark}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(), Loader=UniqueKeyLoader)  # noqa: S506
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {path}")
    return value


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def referenced_inputs(reference: ExecutionReference, root: Path) -> dict[str, str]:
    """Bind referenced bytes from the caller's selected checkout, never a live fallback."""
    inputs = {}
    resolved_root = root.resolve()
    names = [reference.recipe, reference.profile, reference.runtime_lock, reference.client_policy]
    policy_path = (root / reference.client_policy).resolve(strict=True)
    if not policy_path.is_relative_to(resolved_root):
        raise ValueError(f"Execution input escapes checkout: {reference.client_policy}")
    policy = load_mapping(policy_path)
    golden = policy.get("golden_curve")
    if not isinstance(golden, str):
        raise ValueError("client policy must name its committed golden curve")
    names.append(ExecutionReference.repository_path(golden))
    for name in names:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"Execution input escapes checkout: {name}")
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return inputs


def resolve_reference(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    reference = ExecutionReference.model_validate(raw)
    actual = referenced_inputs(reference, root)
    if reference.input_digests is not None and reference.input_digests != actual:
        raise ValueError("Execution inputs changed after matrix generation")
    return reference.model_copy(update={"input_digests": actual}).model_dump(by_alias=True)
