"""Bind master-owned workload values to a fully selected native SRT recipe."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value.strip():
        raise ValueError(f"Missing runtime input: {name}")
    return value


def _positive_integer(value: str, name: str) -> int:
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer: {value!r}")
    return int(value)


def workload_concurrencies(environment: Mapping[str, str]) -> list[int]:
    """Read the selected matrix concurrency, rejecting conflicting inputs."""
    if environment.get("CONC_LIST", "").strip():
        values = [
            _positive_integer(word, "CONC_LIST")
            for word in _required(environment, "CONC_LIST").split()
        ]
        if len(values) != len(set(values)):
            raise ValueError("CONC_LIST must contain unique positive integers")
        if environment.get("CONC", "").strip():
            point = _positive_integer(_required(environment, "CONC"), "CONC")
            if values != [point]:
                raise ValueError("CONC must match the single CONC_LIST value")
        return values
    return [_positive_integer(_required(environment, "CONC"), "CONC")]


def _replace_model_references(recipe: dict[str, Any], previous: set[str], model: str) -> None:
    """Update duplicated model IDs, preserving distinct aliases and draft models."""
    argument_keys = {"served-model-name", "served_model_name", "tokenizer-path", "tokenizer"}
    environment_keys = {"MODEL", "SERVED_MODEL_NAME", "DYN_TRTLLM_SERVED_MODEL_NAME", "TOKENIZER"}
    engine = recipe.get("engine")
    if isinstance(engine, dict) and engine.get("served_model_name") in previous:
        engine["served_model_name"] = model
    sections = [recipe["benchmark"], recipe.get("frontend", {}), *recipe.get("roles", {}).values()]
    for section in sections:
        for mapping_name, keys in (("args", argument_keys), ("env", environment_keys)):
            mapping = section.get(mapping_name, {})
            for key in keys:
                value = mapping.get(key)
                if isinstance(value, str) and value in previous:
                    mapping[key] = model


def bind_workload(recipe: dict[str, Any], environment: Mapping[str, str]) -> dict[str, Any]:
    """Return a job-local recipe populated from the selected master matrix point.

    Call this after native variant expansion: binding a zip override before expansion
    would destroy the relationship between its concurrency and server tuning.
    Role/frontend images, topology, precision, and engine tuning stay recipe-owned.
    """
    image = _required(environment, "IMAGE")
    model = _required(environment, "MODEL")
    agentic = _required(environment, "IS_AGENTIC")
    if agentic not in {"0", "1"}:
        raise ValueError("IS_AGENTIC must be 0 or 1")
    concurrencies = workload_concurrencies(environment)
    lengths = (
        {name: _positive_integer(_required(environment, name), name) for name in ("ISL", "OSL")}
        if agentic == "0"
        else {}
    )
    bound = deepcopy(recipe)
    model_config = bound.setdefault("model", {})
    benchmark = bound.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("Recipe must contain a benchmark mapping")
    workload = benchmark.setdefault("env", {})
    previous: set[str] = set()
    old_path = model_config.get("path")
    if isinstance(old_path, str) and (old_path.startswith("hf:") or "/" in old_path):
        previous.add(old_path.removeprefix("hf:"))
    identity = bound.get("identity", {})
    old_identity_model = identity.get("model", {}).get("repo")
    if isinstance(old_identity_model, str) and old_identity_model:
        previous.add(old_identity_model)
    old_client_model = workload.get("MODEL")
    if isinstance(old_client_model, str) and old_client_model:
        previous.add(old_client_model)
    _replace_model_references(bound, previous, model)
    model_config.update({"path": f"hf:{model}", "container": image})
    if "identity" in bound:
        identity.setdefault("model", {})["repo"] = model
        identity.setdefault("container", {})["image"] = image
    custom = benchmark.get("type") == "custom"
    if custom or "MODEL" in workload:
        workload["MODEL"] = model
    if "IMAGE" in workload:
        workload["IMAGE"] = image
    for name, value in lengths.items():
        if custom or name in workload:
            workload[name] = str(value)
        if not custom or name.lower() in benchmark:
            benchmark[name.lower()] = value
    multinode = environment.get("IS_MULTINODE") == "true"
    if not custom or multinode or "concurrencies" in benchmark:
        benchmark["concurrencies"] = concurrencies
    if custom:
        if multinode and agentic == "0":
            workload["CONC_LIST"] = " ".join(map(str, concurrencies))
            if "CONC" in workload:
                if len(concurrencies) == 1:
                    workload["CONC"] = str(concurrencies[0])
                else:
                    del workload["CONC"]
        else:
            if len(concurrencies) != 1:
                raise ValueError("Custom client CONC requires exactly one concurrency")
            workload["CONC"] = str(concurrencies[0])
            if "CONC_LIST" in workload:
                workload["CONC_LIST"] = str(concurrencies[0])
    return bound
