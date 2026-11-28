from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from infx.bench.env import BenchError
from infx.bench.video import CREDENTIAL_ENV, parse_args
from infx.config import repository_root
from infx.srt_slurm.single_node import srun_overrides
from infx.srt_slurm.workload import (
    BOUND_RECIPE,
    NATIVE_TELEMETRY_BLOCK,
    POWER_FLAGS,
    VIDEO_BLOCK,
    _block,
    _lookup,
    _positive,
    _required,
    add_fabric_argument,
    merge_blocks,
    parse_concurrencies,
    resolve_fabric,
)

VIDEO_PACKAGE = "video-generation"
SPEC_DIR = "specs"
PACKAGE_MOUNT = "/video-package"
RESULT_DIR = "/logs/video"
VIDEO_ENV = (
    "MODEL",
    "CONC_LIST",
    "RESULT_FILENAME",
    "RESULT_DIR",
    "REQUIRE_POWER",
    "VIDEO_SPEC_PATH",
    "VIDEO_MODEL_PATH",
    "VIDEO_RESULT_DIR",
    "VIDEO_SET",
    "VIDEO_PLAN_SHA256",
    "VIDEO_CONTAINER_IMAGE",
    "VIDEO_RUNTIME",
    "VIDEO_RUNTIME_REVISION",
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
    "VIDEO_SPEC_PATH",
    "VIDEO_MODEL_PATH",
    "VIDEO_RESULT_DIR",
    "VIDEO_SET",
    "VIDEO_PLAN_SHA256",
    "VIDEO_CONTAINER_IMAGE",
    "VIDEO_RUNTIME",
    "GPU_COUNT",
)
_RECIPE_ENV_KEYS = "VIDEO_RECIPE_ENV_KEYS"
# The server recipe pins VIDEO_RUNTIME_REVISION itself; the binder copies it to the client.
_RESERVED_ENV = (
    set(VIDEO_ENV)
    | set(_SERVICE_ENV)
    | {
        _RECIPE_ENV_KEYS,
        "CUDA_VISIBLE_DEVICES",
        "VGBENCH_GPU_UUIDS",
        "SRT_MEASUREMENT_WINDOW_DIR",
    }
) - {"VIDEO_RUNTIME_REVISION"}


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
            if CREDENTIAL_ENV.search(name):
                raise ValueError(f"{scope}.{name} could expose a credential in the launch receipt")
    return sorted(set(recipe.get("environment") or {}) | set(server.get("env") or {}))


def video_package(root: Path) -> Path:
    """The video harness beside the end-to-end project ``root`` in the repository."""
    return root.parent / VIDEO_PACKAGE


def _restated(shared: Mapping[str, Any], fragment: Mapping[str, Any], depth: int) -> list[str]:
    """The keys of ``fragment``, down to ``depth`` nested mappings, that ``shared`` sets."""
    found = []
    for key, value in fragment.items():
        if key == "name" or key not in shared:
            continue
        if depth and isinstance(shared[key], Mapping) and isinstance(value, Mapping):
            found.extend(f"{key}.{inner}" for inner in _restated(shared[key], value, depth - 1))
        else:
            found.append(key)
    return found


def check_video_fragment(
    raw: Mapping[str, Any],
    source: Path,
    shared: Mapping[str, Any],
    services: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject a fragment that sets a launcher-owned or shared key, or carries engine roles.

    ``shared`` is the shared block and ``services`` its services by name. The fragment may
    add to a shared block, and each of its services to the one of its name, but neither may
    set a key the shared one already sets.
    """
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
    blocks = {key: value for key, value in shared.items() if key != "services"}
    scopes = [("the fragment", blocks, raw, 2)]
    if isinstance(raw.get("services"), list):
        scopes += [
            (f"service {service['name']!r}", services.get(service["name"], {}), service, 1)
            for service in raw["services"]
        ]
    for where, fixed, declared, depth in scopes:
        restated = _restated(fixed, declared, depth)
        if restated:
            raise ValueError(
                f"{source}: remove {', '.join(restated)} from {where}; "
                f"{VIDEO_BLOCK} sets them for every recipe"
            )


def compose_video_recipe(path: Path, *, root: Path) -> dict[str, Any]:
    """The fragment at ``path`` over the shared video and native telemetry blocks.

    Each fragment service also merges over the shared service of the same name. The fragment
    may extend what the shared blocks set but not restate it.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a mapping")
    shared = merge_blocks(_block(root / VIDEO_BLOCK), _block(root / NATIVE_TELEMETRY_BLOCK))
    # infx.matrix.changed binds a base revision's recipes with this code; its block may hold none.
    by_name = {service["name"]: service for service in shared.get("services", [])}
    check_video_fragment(raw, path, shared, by_name)
    recipe = merge_blocks(shared, raw)
    if isinstance(raw.get("services"), list):
        recipe["services"] = [
            merge_blocks(by_name.get(service["name"], {}), service) for service in raw["services"]
        ]
    return recipe


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
            f"VIDEO_SPEC must name a YAML file under {VIDEO_PACKAGE}/{SPEC_DIR}: {name!r}"
        )
    path = video_package(root) / SPEC_DIR / name
    if not path.is_file():
        raise ValueError(f"video spec not found: {path}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", set_id):
        raise ValueError(f"VIDEO_SET must name a workload set of {name}: {set_id!r}")
    helper = path.with_name("video_spec.py")
    module_spec = importlib.util.spec_from_file_location("video_workload_spec", helper)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot load workload helper: {helper}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    source = module.load_spec(path)
    if set_id not in source["sets"]:
        raise ValueError(f"VIDEO_SET {set_id!r} is not defined in {name}")
    plan = module.plan_for(source, set_id)
    return {"plan": plan, "plan_sha256": module.plan_sha256(plan)}


def _pool_service(recipe: Mapping[str, Any]) -> dict[str, Any]:
    services = recipe.get("services")
    owners = [s for s in services or [] if isinstance(s, Mapping) and s.get("nodes")]
    if len(owners) != 1:
        raise ValueError("video recipes declare exactly one service that owns nodes (the server)")
    return owners[0]


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
) -> list[str]:
    """Render the engine command, check the values the binder owns and return the engine's own
    arguments."""
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
        # The harness entry hands everything after it to SGLang's CLI, so the registry sees all.
        if command[1:4] != ["-m", "evaluator.srt_selected", "serve"]:
            raise ValueError("SGLang video command must run python -m evaluator.srt_selected serve")
        expected.update({"--model-path": model_path, "--model-id": model, "--revision": revision})
        arguments = command[4:]
    elif runtime == "vllm-omni":
        if command[:3] != ["vllm", "serve", model_path]:
            raise ValueError("vLLM video command must serve the bound model snapshot")
        expected["--served-model-name"] = model
        arguments = command[3:]
    else:
        raise ValueError(f"unsupported video runtime: {runtime!r}")
    for flag, value in expected.items():
        if _argument(command, flag) != value:
            raise ValueError(f"video engine {flag} must match the bound value {value!r}")
    _argument(command, "--host")
    return arguments


@dataclass(frozen=True)
class _Reviewed:
    """One engine's reviewed lossless settings.

    A row may give its engine only these flags, spelled in full, environment variables and
    preambles; anything else is refused, so a new engine option enters VideoGenX v0.1 only by
    review here. A value set limits what a setting may be given; None allows any value.
    """

    engine: str
    switches: frozenset[str]
    flags: Mapping[str, frozenset[str] | None]
    env: Mapping[str, frozenset[str] | None]
    preambles: frozenset[str]
    required: frozenset[str]


_REVIEWED = {
    # SGLang's log attestation runs after the sweep and sees only the settings it names; this
    # refuses any other setting before a GPU runs it.
    "sglang": _Reviewed(
        engine="SGLang",
        switches=frozenset({"--disable-conditioning-cache"}),
        flags={
            "--model-type": None,
            "--model-path": None,
            "--model-id": None,
            "--revision": None,
            "--num-gpus": None,
            "--host": None,
            "--port": None,
            "--tp-size": None,
            "--ulysses-degree": None,
            "--cfg-parallel-size": None,
            "--encoder-parallel": None,
            "--warmup-resolutions": None,
            "--warmup-num-frames": None,
            # auto and memory pick offload and sharding from free GPU memory.
            "--performance-mode": frozenset({"speed"}),
            "--enable-torch-compile": None,
            "--dit-cpu-offload": None,
            # The plan pins it; srt_selected and the attestation compare.
            "--flow-shift": None,
            # Copies SGLANG.dense_attention in attestation.py, which infx has no import path to.
            "--attention-backend": frozenset(
                {"fa", "fa2", "torch_sdpa", "torch_cudnn_sdpa", "dynamic_cudnn_sdpa", "aiter"}
            ),
        },
        env={
            "HF_HUB_OFFLINE": None,
            "TRANSFORMERS_OFFLINE": None,
            "PYTHONPATH": None,
            "PYTHONUNBUFFERED": None,
            "ROCM_QUICK_REDUCE_QUANTIZATION": frozenset({"NONE"}),
            "SGLANG_USE_AITER": None,
            "VIDEO_RUNTIME_REVISION": None,
            "VIDEO_SERVER_PORT": None,
        },
        preambles=frozenset(),
        required=frozenset({"--performance-mode"}),
    ),
    # vLLM-Omni has no log attestation, so this is its only check. --cache-config is judged by
    # its JSON.
    "vllm-omni": _Reviewed(
        engine="vLLM-Omni",
        switches=frozenset({"--omni", "--trust-remote-code", "--vae-use-tiling"}),
        flags={
            "--served-model-name": None,
            "--host": None,
            "--port": None,
            "--num-gpus": None,
            "--init-timeout": None,
            "--stage-init-timeout": None,
            "--task-type": None,
            "--tensor-parallel-size": None,
            "--cfg-parallel-size": None,
            "--usp": None,
            "--ring": None,
            "--text-encoder-tp-size": None,
            "--vae-parallel-mode": None,
            "--vae-patch-parallel-size": None,
            "--diffusion-attention-backend": frozenset(
                {
                    "FLASH_ATTN",
                    "TORCH_SDPA",
                    "CUDNN_ATTN",
                    "FLASHINFER_ATTN",
                    "FLASH_ATTN_HUB",
                    "FLASH_ATTN_3_HUB",
                    "TRTLLM_ATTN",
                }
            ),
            "--cache-config": None,
        },
        env={
            "HF_HUB_OFFLINE": None,
            "TRANSFORMERS_OFFLINE": None,
            "OMNI_DIFFUSION_PROMPT_EMBED_CACHE": None,
            "PYTHONPATH": None,
            "PYTHONUNBUFFERED": None,
            "VIDEO_PREFETCH_MODEL": None,
            "VIDEO_RUNTIME_REVISION": None,
            "VIDEO_SERVER_PORT": None,
            "VLLM_OMNI_SERVER_STORAGE__PATH": None,
            "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": None,
            "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": frozenset({"NONE"}),
            "VLLM_WORKER_MULTIPROC_METHOD": None,
        },
        # ROCm vLLM starts only with HIP_VISIBLE_DEVICES set, numbered inside the GPU mask.
        preambles=frozenset({"export HIP_VISIBLE_DEVICES=$(seq -s, 0 $(( {gpus_per_node} - 1 )))"}),
        required=frozenset(),
    ),
}


@dataclass(frozen=True)
class _ReviewedModel:
    """What one model adds to the engine registries, so a new model is one row.

    ``flags`` maps each runtime to the flags only this model's rows may pass on it, with value
    sets as in ``_Reviewed``. ``vllm_omni_default_on`` names the --cache-config keys vLLM-Omni
    turns on by default for the model; its rows must pass each as false.
    """

    flags: Mapping[str, Mapping[str, frozenset[str] | None]]
    vllm_omni_default_on: tuple[str, ...]


_REVIEWED_MODELS = {
    "MiniMaxAI/MiniMax-H3": _ReviewedModel(
        # The pipeline the spec's workload names.
        flags={"sglang": {"--model-variant": frozenset({"fl2va"})}, "vllm-omni": {}},
        vllm_omni_default_on=("minimax_h3_adaln_cache",),
    ),
}
# A model without a row adds nothing to its engine's registry.
_NO_ADDITIONS = _ReviewedModel(flags={name: {} for name in _REVIEWED}, vllm_omni_default_on=())

# The flags each precision a row declares must pass, per engine, so the label names what ran: an
# fp8 row has to quantize and a bf16 row may not. vLLM-Omni's --quantization fp8 converts the BF16
# checkpoint's DiT linears to FP8 when it loads them.
_PRECISION_FLAGS: dict[str, dict[str, dict[str, str]]] = {
    "sglang": {"bf16": {}},
    "vllm-omni": {"bf16": {}, "fp8": {"--quantization": "fp8"}},
}


def _refuse_value(
    engine: str, kind: str, name: str, value: str, allowed: frozenset[str] | None
) -> None:
    if allowed is not None and value not in allowed:
        raise ValueError(f"{engine} {kind} {name}={value!r} is not a reviewed value")


def _refuse_unreviewed(
    reviewed: _Reviewed, arguments: list[str], env: Mapping[str, Any], preamble: Any
) -> dict[str, str]:
    """Refuse a row that gives its engine a setting outside ``reviewed``, or one flag or switch
    twice; return each flag's value."""
    engine = reviewed.engine
    if preamble and preamble not in reviewed.preambles:
        raise ValueError(f"{engine} preamble {preamble!r} is not a reviewed preamble")
    for name, value in env.items():
        if name not in reviewed.env:
            raise ValueError(
                f"{engine} environment variable {name} is not in the reviewed registry"
            )
        _refuse_value(engine, "environment variable", name, str(value), reviewed.env[name])
    values: dict[str, str] = {}
    given: set[str] = set()
    tokens = iter(arguments)
    for flag in tokens:
        if flag not in reviewed.switches:
            if flag not in reviewed.flags:
                raise ValueError(
                    f"{engine} argument {flag!r} is not a reviewed flag spelled in full"
                )
            # A flag with an optional value would otherwise hide the switch that follows it.
            value = next(tokens, "")
            if not value or value.startswith("--"):
                raise ValueError(f"{engine} flag {flag} needs a value, not {value!r}")
            _refuse_value(engine, "flag", flag, value, reviewed.flags[flag])
            values[flag] = value
        # The engine would keep the last value and ignore the earlier one.
        if flag in given:
            raise ValueError(f"{engine} argument {flag} is given twice")
        given.add(flag)
    missing = sorted(reviewed.required - values.keys())
    if missing:
        raise ValueError(f"{engine} rows must set {', '.join(missing)}")
    return values


def _refuse_cached_vllm_omni(
    model: str, default_on: tuple[str, ...], cache_config: str | None
) -> None:
    config: dict[str, Any] = {}
    if cache_config is not None:
        try:
            config = json.loads(cache_config)
            lossless = all(enabled is False for enabled in config.values())
        except (ValueError, AttributeError):
            lossless = False
        if not lossless:
            raise ValueError(f"vLLM-Omni flag --cache-config={cache_config!r} is not lossless")
    for key in default_on:
        if key not in config:
            raise ValueError(
                f"vLLM-Omni turns {key} on by default for {model}: the recipe must pass "
                "it as false in --cache-config"
            )


def bind_video(
    recipe: Mapping[str, Any], environment: Mapping[str, str], *, root: Path
) -> dict[str, Any]:
    """Write the point's model, image, GPU count, concurrencies and video paths into ``recipe``."""
    image, model = _required(environment, "IMAGE"), _required(environment, "MODEL")
    gpus = _positive(_required(environment, "GPU_COUNT"), "GPU_COUNT")
    concurrencies = parse_concurrencies(_required(environment, "CONC_LIST"))
    spec_name, set_id = _required(environment, "VIDEO_SPEC"), _required(environment, "VIDEO_SET")
    spec = load_spec(spec_name, root=root, set_id=set_id)
    if spec["plan"]["model_id"] != model:
        raise ValueError(f"video spec serves {spec['plan']['model_id']!r}, the point {model!r}")
    model_path = spec_model_path(
        _required(environment, "HF_HUB_CACHE"), model, spec["plan"]["model_revision"]
    )
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
    workload = benchmark.setdefault("env", {})
    if workload.get("VIDEO_SERVER_SERVICE") != service["name"]:
        raise ValueError("benchmark.env.VIDEO_SERVER_SERVICE must name the server service")
    runtime = _required(environment, "FRAMEWORK")
    arguments = _bind_command(
        service,
        runtime=runtime,
        model=model,
        model_path=model_path,
        revision=spec["plan"]["model_revision"],
        port=workload["VIDEO_SERVER_PORT"],
        gpus=gpus,
    )
    if runtime == "sglang":
        if service["args"].count("--disable-conditioning-cache") != 1:
            raise ValueError("fairness workload requires --disable-conditioning-cache")
    elif {**(bound.get("environment") or {}), **service["env"]}.get(
        "OMNI_DIFFUSION_PROMPT_EMBED_CACHE"
    ) != "0":
        raise ValueError("fairness workload requires OMNI_DIFFUSION_PROMPT_EMBED_CACHE=0")
    revision = str((service.get("env") or {}).get("VIDEO_RUNTIME_REVISION", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("video service requires a pinned VIDEO_RUNTIME_REVISION")
    reviewed = _REVIEWED[runtime]
    additions = _REVIEWED_MODELS.get(model, _NO_ADDITIONS)
    precision = bound["model"]["precision"]
    precision_flags = _PRECISION_FLAGS[runtime].get(precision)
    if precision_flags is None:
        raise ValueError(f"{reviewed.engine} video rows have no reviewed {precision} precision")
    # Only the row's precision grants a precision flag, even if a registry above lists it.
    owned = {flag for spec in _PRECISION_FLAGS[runtime].values() for flag in spec}
    allowed = {**reviewed.flags, **additions.flags[runtime]}
    flags = _refuse_unreviewed(
        replace(
            reviewed,
            flags={
                **{flag: values for flag, values in allowed.items() if flag not in owned},
                **{flag: frozenset({value}) for flag, value in precision_flags.items()},
            },
            required=reviewed.required | frozenset(precision_flags),
        ),
        arguments,
        {**(bound.get("environment") or {}), **service["env"]},
        service.get("preamble"),
    )
    if runtime == "vllm-omni":
        _refuse_cached_vllm_omni(model, additions.vllm_omni_default_on, flags.get("--cache-config"))
    values = {
        "MODEL": model,
        "CONC_LIST": " ".join(map(str, concurrencies)),
        "RESULT_FILENAME": _required(environment, "RESULT_FILENAME"),
        "RESULT_DIR": "/logs",
        "VIDEO_SPEC_PATH": f"{PACKAGE_MOUNT}/{SPEC_DIR}/{spec_name}",
        "VIDEO_MODEL_PATH": model_path,
        "VIDEO_RESULT_DIR": RESULT_DIR,
        "VIDEO_SET": set_id,
        "VIDEO_PLAN_SHA256": spec["plan_sha256"],
        "VIDEO_RUNTIME": runtime,
        "VIDEO_CONTAINER_IMAGE": image,
        "VIDEO_RUNTIME_REVISION": revision,
        "GPU_COUNT": str(gpus),
    }
    workload.update(values)
    service["env"] = {
        **(service.get("env") or {}),
        **{k: values[k] for k in _SERVICE_ENV},
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
    return [
        "--set",
        f"srun_options.gpus-per-node={json.dumps(str(gpus))}",
        *srun_overrides(environment),
    ]


def prepare(recipe: Path, environment: Mapping[str, str], *, root: Path) -> dict[str, Any]:
    """Compose and bind ``recipe`` for ``environment``: the one recipe the launcher submits."""
    return bind_video(compose_video_recipe(recipe, root=root), environment, root=root)


def main() -> None:
    parser = argparse.ArgumentParser()
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
