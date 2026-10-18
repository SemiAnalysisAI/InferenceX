"""Bind a services-only video-generation recipe to one master-config point.

A video point serves one frozen H3 workload (``experimental/video-generation/specs``)
from a recipe-managed diffusion server and runs the H3 client as srt-slurm's custom
benchmark, every ``CONC_LIST`` concurrency against the same server. The launcher owns
the model, image, GPU count, native telemetry block and every ``H3_*`` path; the
fragment owns only the service and client shape.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from infx.config import repository_root
from infx.srt_slurm.workload import (
    BOUND_RECIPE,
    NATIVE_TELEMETRY_BLOCK,
    _block,
    _lookup,
    add_fabric_argument,
    merge_blocks,
    parse_concurrencies,
    resolve_fabric,
)

VIDEO_PACKAGE = "experimental/video-generation"
SPEC_DIR = "specs"
PACKAGE_MOUNT = "/h3-package"
RESULT_DIR = "/logs/h3"
CLIENT_SCRIPT = "benchmarks/h3/run_client.sh"
SERVER_SCRIPT = "benchmarks/h3/run_server.sh"
READINESS_PATH = "/health"
_POWER_FLAGS = {"1", "true", "TRUE", "yes", "YES"}
VIDEO_ENV = (
    "MODEL",
    "CONC_LIST",
    "RESULT_FILENAME",
    "RESULT_DIR",
    "REQUIRE_POWER",
    "H3_SPEC_PATH",
    "H3_PACKAGE_ROOT",
    "H3_MODEL_PATH",
    "H3_RESULT_DIR",
    "H3_SET",
    "H3_PLAN_SHA256",
    "H3_RUNTIME",
    "H3_RUNTIME_REVISION",
)
_BOUND_KEYS = (
    ("model",),
    ("slurm",),
    ("telemetry",),
    ("resources", "gpus_per_node"),
    ("benchmark", "concurrencies"),
    *(("benchmark", "env", name) for name in VIDEO_ENV),
)
_SERVICE_ENV = (
    "H3_SPEC_PATH", "H3_PACKAGE_ROOT", "H3_MODEL_PATH", "H3_RESULT_DIR",
    "H3_SET", "H3_PLAN_SHA256", "H3_RUNTIME", "H3_RUNTIME_REVISION", "GPU_COUNT",
)


def video_package(root: Path | None = None) -> Path:
    """The H3 harness beside the end-to-end project (the repository's ``experimental`` tree)."""
    project = root or repository_root()
    nested = project / VIDEO_PACKAGE
    return nested if nested.is_dir() else project.parent / VIDEO_PACKAGE


def check_video_fragment(raw: Mapping[str, Any], source: Path) -> None:
    """Reject a fragment that sets a launcher-owned key or carries engine roles."""
    found = [
        f"{'.'.join(key)} (= {value!r})"
        for key in _BOUND_KEYS
        for present, value in (_lookup(raw, key),)
        if present
    ]
    if "base" in raw or any(key.startswith(("override_", "zip_override_")) for key in raw):
        raise ValueError(f"{source}: video fragments do not take variant bundles")
    if raw.get("roles"):
        raise ValueError(f"{source}: video fragments own no engine roles; the server is a service")
    if found:
        raise ValueError(
            f"{source}: remove {', '.join(found)} from the fragment; the launcher sets them"
        )


def compose_video_recipe(path: Path, *, root: Path) -> dict[str, Any]:
    """The fragment at ``path`` over the native telemetry block."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a mapping")
    check_video_fragment(raw, path)
    return merge_blocks(_block(root / NATIVE_TELEMETRY_BLOCK), raw)


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value.strip():
        raise ValueError(f"Missing video input: {name}")
    return value


def _positive(value: str, name: str) -> int:
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer: {value!r}")
    return int(value)


def spec_model_path(hf_hub_cache: str, model: str, revision: str) -> str:
    """Where the HF hub cache the job mounts holds ``model`` at ``revision``."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", model):
        raise ValueError(f"MODEL must be an org/name Hugging Face repo id: {model!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"spec model.revision must be a 40-hex commit: {revision!r}")
    cache = PurePosixPath(hf_hub_cache)
    if not cache.is_absolute():
        raise ValueError(f"HF_HUB_CACHE must be absolute: {hf_hub_cache!r}")
    return str(cache / f"models--{model.replace('/', '--')}" / "snapshots" / revision)


def load_spec(name: str, *, root: Path, set_id: str | None = None) -> dict[str, Any]:
    """The selected video workload under the harness's ``specs`` directory."""
    if name != PurePosixPath(name).name or not name.endswith((".json", ".yaml", ".yml")):
        raise ValueError(
            f"H3_SPEC must name a JSON or YAML file under {VIDEO_PACKAGE}/{SPEC_DIR}: {name!r}"
        )
    path = video_package(root) / SPEC_DIR / name
    if not path.is_file():
        raise ValueError(f"H3 spec not found: {path}")
    if path.suffix in {".yaml", ".yml"}:
        if not set_id or not re.fullmatch(r"[A-Za-z0-9_-]+", set_id):
            raise ValueError("H3_SET must select a safe nonempty YAML workload set")
        helper = path.with_name("h3_spec.py")
        module_spec = importlib.util.spec_from_file_location("video_workload_spec", helper)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load workload helper: {helper}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        source = module.load_spec(path)
        if set_id not in source["sets"]:
            raise ValueError(f"H3_SET {set_id!r} is not defined in {name}")
        plan = module.plan_for(source, set_id)
        return {
            "plan": plan,
            "model": {"revision": source["model"]["revision"]["value"]},
            "plan_sha256": module.plan_sha256(plan),
        }
    if set_id:
        raise ValueError("H3_SET is only valid with a YAML workload")
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: spec must be an object")
    for key in (("plan", "model_id"), ("model", "revision"), ("model", "path"), ("port",)):
        present, value = _lookup(spec, key)
        if not present or value in (None, ""):
            raise ValueError(f"{path}: spec lacks {'.'.join(key)}")
    return spec


def _pool_service(recipe: Mapping[str, Any]) -> dict[str, Any]:
    services = recipe.get("services")
    owners = [s for s in services or [] if isinstance(s, Mapping) and s.get("nodes")]
    if len(owners) != 1:
        raise ValueError("video recipes declare exactly one service that owns nodes (the server)")
    [service] = owners
    if service.get("type") != "generic" or not service.get("critical", False):
        raise ValueError("the video server service must be a critical generic service")
    command = service.get("command")
    if not (isinstance(command, list) and command and str(command[-1]).endswith(SERVER_SCRIPT)):
        raise ValueError(f"the video server service must run {SERVER_SCRIPT}")
    return service


def bind_video(
    recipe: Mapping[str, Any], environment: Mapping[str, str], *, root: Path
) -> dict[str, Any]:
    """Write the point's model, image, GPU count, concurrencies and H3 paths into ``recipe``."""
    image, model = _required(environment, "IMAGE"), _required(environment, "MODEL")
    gpus = _positive(_required(environment, "GPU_COUNT"), "GPU_COUNT")
    concurrencies = parse_concurrencies(_required(environment, "CONC_LIST"))
    spec_name = _required(environment, "H3_SPEC")
    set_id = environment.get("H3_SET") or None
    spec = load_spec(spec_name, root=root, set_id=set_id)
    if spec["plan"]["model_id"] != model:
        raise ValueError(f"H3 spec serves {spec['plan']['model_id']!r}, the point {model!r}")
    model_path = spec_model_path(
        _required(environment, "HF_HUB_CACHE"), model, spec["model"]["revision"]
    )
    if "path" in spec["model"] and spec["model"]["path"] != model_path:
        raise ValueError(
            f"H3 spec model.path {spec['model']['path']!r} is not the mounted hub cache "
            f"snapshot {model_path!r}"
        )
    if recipe.get("frontend", {}).get("type") != "none":
        raise ValueError("video recipes are services-only: frontend.type must be none")
    bound = {key: deepcopy(value) for key, value in recipe.items() if key in ("schema", "name")}
    bound["model"] = {
        "path": f"hf:{model}",
        "container": image,
        "precision": _required(environment, "PRECISION"),
    }
    bound.update((key, deepcopy(value)) for key, value in recipe.items() if key not in bound)
    bound["resources"] = {**bound.get("resources", {}), "gpus_per_node": gpus}
    service = _pool_service(bound)
    benchmark = bound.setdefault("benchmark", {})
    if benchmark.get("type") != "custom" or not str(benchmark.get("command", "")).endswith(
        CLIENT_SCRIPT
    ):
        raise ValueError(f"video recipes run {CLIENT_SCRIPT} as benchmark.type custom")
    workload = benchmark.setdefault("env", {})
    port = str(spec.get("port", (service.get("readiness") or {}).get("http", {}).get("port", "")))
    readiness = (service.get("readiness") or {}).get("http") or {}
    if (
        str(workload.get("H3_SERVER_PORT")) != port
        or str((service.get("env") or {}).get("H3_SERVER_PORT")) != port
        or str(readiness.get("port")) != port
        or readiness.get("path") != READINESS_PATH
    ):
        raise ValueError(
            f"the server service must probe {READINESS_PATH} on the spec port {port} and the "
            "client must address that port (H3_SERVER_PORT)"
        )
    if workload.get("H3_SERVER_SERVICE") != service["name"]:
        raise ValueError("benchmark.env.H3_SERVER_SERVICE must name the server service")
    values = {
        "MODEL": model,
        "CONC_LIST": " ".join(map(str, concurrencies)),
        "RESULT_FILENAME": _required(environment, "RESULT_FILENAME"),
        "RESULT_DIR": "/logs",
        "H3_SPEC_PATH": f"{PACKAGE_MOUNT}/{SPEC_DIR}/{spec_name}",
        "H3_PACKAGE_ROOT": PACKAGE_MOUNT,
        "H3_MODEL_PATH": model_path,
        "H3_RESULT_DIR": RESULT_DIR,
    }
    if set_id:
        runtime = _required(environment, "FRAMEWORK")
        if runtime not in {"sglang", "vllm-omni"}:
            raise ValueError(f"unsupported YAML video runtime: {runtime!r}")
        revision = str((service.get("env") or {}).get("VIDEO_RUNTIME_REVISION", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("video service requires a pinned VIDEO_RUNTIME_REVISION")
        values.update({
            "H3_SET": set_id,
            "H3_PLAN_SHA256": spec["plan_sha256"],
            "H3_RUNTIME": runtime,
            "H3_RUNTIME_REVISION": revision,
            "GPU_COUNT": str(gpus),
        })
    workload.update(values)
    service["env"] = {
        **(service.get("env") or {}),
        **{k: values[k] for k in _SERVICE_ENV if k in values},
    }
    if (bound.get("telemetry") or {}).get("enabled") is True:
        benchmark["concurrencies"] = list(concurrencies)
        required = environment.get("REQUIRE_POWER", "0") in _POWER_FLAGS
        bound["telemetry"]["required"] = required
        workload["REQUIRE_POWER"] = "1" if required else "0"
    return bound


def runtime_arguments(environment: Mapping[str, str]) -> list[str]:
    """Bind only runtime-owned srun values after the recipe is bound."""
    gpus = _positive(_required(environment, "GPU_COUNT"), "GPU_COUNT")
    overrides = ["--set", f"srun_options.gpus-per-node={json.dumps(str(gpus))}"]
    if environment.get("SRT_SRUN_OPTIONS"):
        options = json.loads(environment["SRT_SRUN_OPTIONS"])
        if not isinstance(options, dict) or any(
            not re.fullmatch(r"[a-z][a-z0-9-]*", key) or not isinstance(value, str)
            for key, value in options.items()
        ):
            raise ValueError("SRT_SRUN_OPTIONS must map option names to string values")
        for key, value in options.items():
            overrides += ["--set", f"srun_options.{key}={json.dumps(value)}"]
    return overrides


def prepare(recipe: Path, environment: Mapping[str, str], *, root: Path) -> dict[str, Any]:
    """Compose and bind ``recipe`` for ``environment``: the one recipe the launcher submits."""
    return bind_video(compose_video_recipe(recipe, root=root), environment, root=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("recipe", type=Path)
    command.add_argument("output", type=Path)
    add_fabric_argument(command)
    parsed = parser.parse_args()
    try:
        root = repository_root()
        recipe = resolve_fabric(prepare(parsed.recipe, os.environ, root=root), parsed.fabric)
        config = parsed.output.with_name(BOUND_RECIPE)
        config.write_text(yaml.safe_dump(recipe, sort_keys=False))
        arguments = runtime_arguments(os.environ)
        parsed.output.write_bytes(
            "\0".join([str(parsed.recipe), str(config), *arguments, ""]).encode()
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
