from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from infx.bench.env import BenchError
from infx.bench.video import parse_args
from infx.config import repository_root
from infx.srt_slurm.workload import (
    BOUND_RECIPE,
    NATIVE_TELEMETRY_BLOCK,
    POWER_FLAGS,
    _block,
    _lookup,
    _positive,
    _required,
    add_fabric_argument,
    merge_blocks,
    parse_concurrencies,
    resolve_fabric,
)

VIDEO_PACKAGE = "experimental/video-generation"
SPEC_DIR = "specs"
PACKAGE_MOUNT = "/h3-package"
RESULT_DIR = "/logs/h3"
READINESS_PATH = "/health"
VIDEO_ENV = (
    "MODEL",
    "CONC_LIST",
    "RESULT_FILENAME",
    "RESULT_DIR",
    "REQUIRE_POWER",
    "H3_SPEC_PATH",
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
    "H3_SPEC_PATH",
    "H3_MODEL_PATH",
    "H3_RESULT_DIR",
    "H3_SET",
    "H3_PLAN_SHA256",
    "H3_RUNTIME",
    "H3_RUNTIME_REVISION",
    "GPU_COUNT",
)
_RECIPE_ENV_KEYS = "H3_RECIPE_ENV_KEYS"
_RESERVED_ENV = (
    set(VIDEO_ENV)
    | set(_SERVICE_ENV)
    | {
        _RECIPE_ENV_KEYS,
        "CUDA_VISIBLE_DEVICES",
        "VGBENCH_GPU_UUIDS",
        "SRT_MEASUREMENT_WINDOW_DIR",
    }
)
_CREDENTIAL_ENV = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|AUTH|BEARER|"
    r"API_KEY|ACCESS_KEY|PRIVATE_KEY|SSH_KEY)(?:_|$)",
    re.IGNORECASE,
)


def _declared_server_env_keys(recipe: Mapping[str, Any], server: Mapping[str, Any]) -> list[str]:
    """Validate recipe env scopes and name only values declared for the server."""
    scopes = [
        ("environment", recipe.get("environment")),
        ("benchmark.env", (recipe.get("benchmark") or {}).get("env")),
    ]
    scopes.extend(
        (f"services[{index}].env", service.get("env"))
        for index, service in enumerate(recipe.get("services") or [])
    )
    for scope, values in scopes:
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"{scope} must map environment names to values")
        for name in values:
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"{scope} has an invalid environment name: {name!r}")
            if name in _RESERVED_ENV or name.startswith("SRT_SERVICE_"):
                raise ValueError(f"{scope}.{name} is owned by the launcher or runtime")
            if _CREDENTIAL_ENV.search(name):
                raise ValueError(f"{scope}.{name} could expose a credential in the launch receipt")
    return sorted(set(recipe.get("environment") or {}) | set(server.get("env") or {}))


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


def load_spec(name: str, *, root: Path, set_id: str) -> dict[str, Any]:
    """The selected set of the YAML workload ``name`` under the harness's ``specs`` directory."""
    if name != PurePosixPath(name).name or not name.endswith((".yaml", ".yml")):
        raise ValueError(
            f"H3_SPEC must name a YAML file under {VIDEO_PACKAGE}/{SPEC_DIR}: {name!r}"
        )
    path = video_package(root) / SPEC_DIR / name
    if not path.is_file():
        raise ValueError(f"H3 spec not found: {path}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", set_id):
        raise ValueError(f"H3_SET must name a workload set of {name}: {set_id!r}")
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


def _pool_service(recipe: Mapping[str, Any]) -> dict[str, Any]:
    services = recipe.get("services")
    owners = [s for s in services or [] if isinstance(s, Mapping) and s.get("nodes")]
    if len(owners) != 1:
        raise ValueError("video recipes declare exactly one service that owns nodes (the server)")
    [service] = owners
    if service.get("type") != "generic" or not service.get("critical", False):
        raise ValueError("the video server service must be a critical generic service")
    command = service.get("command")
    if not (isinstance(command, list) and command[1:] == ["-m", "infx.bench", "video", "serve"]):
        raise ValueError("the video server service must run python -m infx.bench video serve")
    return service


def _argument(command: list[str], flag: str) -> str:
    values = []
    for index, token in enumerate(command):
        if token == flag and index + 1 < len(command):
            values.append(command[index + 1])
        elif token.startswith(f"{flag}="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1 or not values[0] or values[0].startswith("--"):
        raise ValueError(f"video engine command requires exactly one {flag} value")
    return values[0]


def _bind_command(
    service: dict[str, Any],
    *,
    runtime: str,
    model: str,
    model_path: str,
    revision: str,
    port: str,
    gpus: int,
) -> None:
    values = {
        "model_path": model_path,
        "model_id": model,
        "model_revision": revision,
        "gpus_per_node": str(gpus),
    }

    def render(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("video service args and env values must be strings")
        for name, bound in values.items():
            value = value.replace("{" + name + "}", bound)
        return value

    args = service.get("args")
    if not isinstance(args, list):
        raise ValueError("video service args must declare the engine command")
    service["args"] = [render(arg) for arg in args]
    service["env"] = {key: render(value) for key, value in service.get("env", {}).items()}
    try:
        parsed = parse_args(["serve", *service["args"]])
    except (SystemExit, ValueError, BenchError) as error:
        raise ValueError("invalid video serve arguments") from error
    command = parsed.command
    expected = {"--port": port, "--num-gpus": str(gpus)}
    if runtime == "sglang":
        expected.update({"--model-path": model_path, "--model-id": model, "--revision": revision})
    elif runtime == "vllm-omni":
        if command[:3] != ["vllm", "serve", model_path]:
            raise ValueError("vLLM video command must serve the bound model snapshot")
        expected["--served-model-name"] = model
    else:
        raise ValueError(f"unsupported video runtime: {runtime!r}")
    for flag, value in expected.items():
        if _argument(command, flag) != value:
            raise ValueError(f"video engine {flag} must match the bound value {value!r}")
    _argument(command, "--host")


def bind_video(
    recipe: Mapping[str, Any], environment: Mapping[str, str], *, root: Path
) -> dict[str, Any]:
    """Write the point's model, image, GPU count, concurrencies and H3 paths into ``recipe``."""
    image, model = _required(environment, "IMAGE"), _required(environment, "MODEL")
    gpus = _positive(_required(environment, "GPU_COUNT"), "GPU_COUNT")
    concurrencies = parse_concurrencies(_required(environment, "CONC_LIST"))
    spec_name, set_id = _required(environment, "H3_SPEC"), _required(environment, "H3_SET")
    spec = load_spec(spec_name, root=root, set_id=set_id)
    if spec["plan"]["model_id"] != model:
        raise ValueError(f"H3 spec serves {spec['plan']['model_id']!r}, the point {model!r}")
    model_path = spec_model_path(
        _required(environment, "HF_HUB_CACHE"), model, spec["model"]["revision"]
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
    recipe_env_keys = _declared_server_env_keys(bound, service)
    benchmark = bound.setdefault("benchmark", {})
    client_command = shlex.split(str(benchmark.get("command", "")))
    if benchmark.get("type") != "custom" or client_command[1:5] != [
        "-m",
        "infx.bench",
        "video",
        "client",
    ]:
        raise ValueError(
            "video recipes run python -m infx.bench video client as benchmark.type custom"
        )
    try:
        parse_args(client_command[4:])
    except (SystemExit, ValueError, BenchError) as error:
        raise ValueError("invalid video client arguments") from error
    workload = benchmark.setdefault("env", {})
    readiness = (service.get("readiness") or {}).get("http") or {}
    port = str(readiness.get("port", ""))
    if (
        str(workload.get("H3_SERVER_PORT")) != port
        or str((service.get("env") or {}).get("H3_SERVER_PORT")) != port
        or str(readiness.get("port")) != port
        or readiness.get("path") != READINESS_PATH
    ):
        raise ValueError(
            f"the server service must probe {READINESS_PATH} on its port and the client "
            "must address that port (H3_SERVER_PORT)"
        )
    if workload.get("H3_SERVER_SERVICE") != service["name"]:
        raise ValueError("benchmark.env.H3_SERVER_SERVICE must name the server service")
    runtime = _required(environment, "FRAMEWORK")
    _bind_command(
        service,
        runtime=runtime,
        model=model,
        model_path=model_path,
        revision=spec["model"]["revision"],
        port=port,
        gpus=gpus,
    )
    revision = str((service.get("env") or {}).get("VIDEO_RUNTIME_REVISION", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("video service requires a pinned VIDEO_RUNTIME_REVISION")
    values = {
        "MODEL": model,
        "CONC_LIST": " ".join(map(str, concurrencies)),
        "RESULT_FILENAME": _required(environment, "RESULT_FILENAME"),
        "RESULT_DIR": "/logs",
        "H3_SPEC_PATH": f"{PACKAGE_MOUNT}/{SPEC_DIR}/{spec_name}",
        "H3_MODEL_PATH": model_path,
        "H3_RESULT_DIR": RESULT_DIR,
        "H3_SET": set_id,
        "H3_PLAN_SHA256": spec["plan_sha256"],
        "H3_RUNTIME": runtime,
        "H3_RUNTIME_REVISION": revision,
        "GPU_COUNT": str(gpus),
    }
    workload.update(values)
    service["env"] = {
        **(service.get("env") or {}),
        **{k: values[k] for k in _SERVICE_ENV if k in values},
        _RECIPE_ENV_KEYS: json.dumps(recipe_env_keys),
    }
    if (bound.get("telemetry") or {}).get("enabled") is True:
        benchmark["concurrencies"] = list(concurrencies)
        required = environment.get("REQUIRE_POWER", "0") in POWER_FLAGS
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
