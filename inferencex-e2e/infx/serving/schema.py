"""Serving YAML: the one file a team member writes to benchmark a serving config.

Engine command (serve:), optional router and storage sidecars, replicas, and the scenario
with its variant (overrides / workload / slo). Every error is one line
"<field>: <problem and fix>" so a person or an agent can act on it.
Spec: docs/superpowers/specs/2026-10-25-vng-serving-refactor-design.md
"""

from __future__ import annotations

import itertools
import re
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from infx.clusters import RunnerInventory, load_inventory
from infx.serving import scenarios
from infx.serving.scenarios import Point, Rendered, ScenarioError

NAME_PATTERN = r"^[a-z0-9][a-z0-9.-]*$"
LOCAL_PATH_MARKERS = ("/models/", "/mnt/", "/data/")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# image is spliced into srun arguments on the GPU node; model is passed to the client argv.
IMAGE_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._/:@-]*|/[A-Za-z0-9._/-]+\.sqsh"
)  # a registry ref, or a squash file on the node
MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+")
VNG_CLUSTER_PREFIXES = ("hcm-", "han-")  # VNG sites; cluster ids are <site>-<node>
# Variables job.sh / bench.sh / sidecar.sh / client.py set themselves.
RESERVED_ENV = (
    "PORT", "MODEL", "CONC", "RESULT_DIR", "HF_HUB_CACHE", "FRAMEWORK",
    "SERVE_CMD", "SERVE_ENV", "SERVED_MODEL_NAME", "SERVING_YAML", "SERVING_SCENARIO_YAML",
    "SERVING_SCENARIO_LABEL", "SERVING_DATASET", "SERVING_DATASET_KIND", "SERVING_SEARCH",
    "SERVING_ENGINE_IMAGE", "SERVING_DCGM_IMAGE", "SERVING_HF_CACHE_HOST", "IX_CUSTOM_DATASET_FILE",
    "AIPERF_SERVER_METRICS_URLS", "AIPERF_GPU_TELEMETRY_URL",
    "DYN_DISCOVERY_BACKEND", "DYN_FILE_KV", "DYN_SYSTEM_ENABLED", "DYN_SYSTEM_PORT",
    "ROUTER_IMAGE", "ROUTER_CMD", "ROUTER_ENV", "ROUTER_PROFILE", "ROUTER_METRICS_PORT",
    "ENGINE_PORT", "ENGINE_PORTS", "REPLICAS",
    "STORAGE_IMAGE", "STORAGE_CMD", "STORAGE_ENV", "STORAGE_MASTER_PORT", "STORAGE_META_PORT",
)  # fmt: skip
# {{name}} placeholders sidecar.sh / bench.sh expand at run time; {{ x }} (spaces) is inline Jinja.
ROUTER_PLACEHOLDERS = (
    "router_port",
    "router_metrics_port",
    "engine_url",
    "engine_port",
    "engine_urls",
    "storage_master_port",
    "storage_meta_port",
)
ENV_PLACEHOLDERS = ("storage_master_port", "storage_meta_port")
STORAGE_PLACEHOLDERS = ("storage_master_port", "storage_meta_port")
PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")
TOKENIZER_ENCODINGS = (
    "builtin",
    "cl100k_base",
    "o200k_base",
    "o200k_harmony",
)  # tiktoken, no HF access
SECRET_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
# Engine fields a remote endpoint has no use for.
REMOTE_EXCLUDED = (
    "image",
    "serve",
    "gpus",
    "model",
    "replicas",
    "precision",
    "env",
    "router",
    "storage",
    "arms",
)


class ServingError(ValueError):
    """Validation failure; each line names a field and the fix."""


def _coerce_env(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if isinstance(item, bool):
            raise ValueError(f'{key}: quote boolean values, e.g. "1" or "true"')
        out[key] = str(item) if isinstance(item, (int, float)) else item
    return out


def _listed(value: Any) -> Any:
    return value if isinstance(value, list) else [value]


class RouterSpec(BaseModel):
    """A router sidecar in front of the engine(s)."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    serve: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    metrics_port: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


class StorageSpec(BaseModel):
    """A non-GPU storage sidecar started before the engine (e.g. Mooncake master)."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    serve: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


class ArmSpec(BaseModel):
    """One engine variant of a serving YAML: ``serve:`` plus ``append``, ``env:`` overlaid by ``env``."""

    model_config = ConfigDict(extra="forbid")

    append: str = ""
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


class EndpointSpec(BaseModel):
    """An endpoint we do not launch: an OpenAI-compatible URL, its served model and its key."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    url: str
    model: str = Field(min_length=1)
    api_key_secret: str | None = Field(default=None, alias="api-key-secret")
    tokenizer: str = "builtin"
    tls_verify: bool = Field(default=True, alias="tls-verify")

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        if not re.fullmatch(r"https?://[^\s'\"]+", value):
            raise ValueError("must start with http:// or https:// and hold no whitespace or quotes")
        return (
            value.rstrip("/").removesuffix("/v1").rstrip("/")
        )  # the scenario appends /v1/chat/completions

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        if re.search(r"[\s'\"]", value):
            raise ValueError("no whitespace or quotes; use the served model name")
        return value

    @field_validator("api_key_secret")
    @classmethod
    def _secret(cls, value: str | None) -> str | None:
        if value is not None and not SECRET_NAME.fullmatch(value):
            raise ValueError("a GitHub secret name: capitals, digits and _ (e.g. OPENAI_API_KEY)")
        return value

    @field_validator("tokenizer")
    @classmethod
    def _tokenizer(cls, value: str) -> str:
        if value not in TOKENIZER_ENCODINGS and not MODEL_PATTERN.fullmatch(value):
            raise ValueError(f"{', '.join(TOKENIZER_ENCODINGS)}, or an HF repo id <org>/<name>")
        return value


class Workload(BaseModel):
    """The synthetic grid: one job per (isl, osl, cache) point."""

    model_config = ConfigDict(extra="forbid")

    isl: list[str] = Field(min_length=1)
    osl: list[str] = Field(min_length=1)
    cache: list[int] = Field(min_length=1)
    search: bool = False

    @field_validator("isl", "osl", mode="before")
    @classmethod
    def _labels(cls, value: Any) -> Any:
        return [str(item) for item in _listed(value)]

    @field_validator("cache", mode="before")
    @classmethod
    def _cache(cls, value: Any) -> Any:
        return _listed(value)

    def points(self) -> list[Point]:
        return [Point(i, o, c) for i, o, c in itertools.product(self.isl, self.osl, self.cache)]


class ServingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    runner: str = Field(min_length=1)
    image: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    gpus: int | None = Field(default=None, gt=0)
    replicas: int = Field(default=1, ge=1)
    precision: str = Field(default="bf16", pattern=r"^[a-z0-9]+$")
    env: dict[str, str] = Field(default_factory=dict)
    serve: str | None = Field(default=None, min_length=1)
    endpoint: EndpointSpec | None = None
    router: RouterSpec | None = None
    storage: StorageSpec | None = None
    arms: dict[str, ArmSpec] | None = Field(default=None, min_length=1)
    scenario: Literal[
        "agentx",
        "tau2",
        "livekit",
        "shared-prefix-e2",
        "hidden-prefix-e3",
        "synthetic",
        "greenspec",
        "speedbench",
    ]
    overrides: dict[str, Any] = Field(default_factory=dict)
    workload: Workload | None = None
    slo: str | None = None
    conc: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)

    @field_validator("conc", mode="before")
    @classmethod
    def _scalar_conc(cls, value: Any) -> Any:
        return [value] if isinstance(value, int) and not isinstance(value, bool) else value

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)

    @field_validator("arms")
    @classmethod
    def _arm_names(cls, value: dict[str, ArmSpec] | None) -> dict[str, ArmSpec] | None:
        for name in value or {}:
            if not re.fullmatch(NAME_PATTERN, name):
                raise ValueError(f"{name!r}: arm names use lowercase letters, digits, . and -")
        return value


def _format_validation_error(error: ValidationError) -> str:
    models = {
        "router": RouterSpec,
        "storage": StorageSpec,
        "workload": Workload,
        "endpoint": EndpointSpec,
    }
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "serving"
        message = item["msg"].removeprefix("Value error, ")
        if item["type"] == "extra_forbidden":
            model = (
                models.get(str(item["loc"][0]), ServingSpec)
                if len(item["loc"]) > 1
                else ServingSpec
            )
            allowed = (field.alias or name for name, field in model.model_fields.items())
            message = "unknown key (allowed: " + ", ".join(allowed) + ")"
        elif item["loc"] == ("scenario",):
            message = f"must be one of {', '.join(scenarios.FAMILIES)} (variants go in overrides:)"
        lines.append(f"{location}: {message}")
    return "\n".join(lines)


def is_local_squash(image: str) -> bool:
    """A squash file already on one node; pyxis starts it as is, so there is nothing to stage."""
    return image.startswith("/") and image.endswith(".sqsh")


def runner_gpus(runner: str, inventory: RunnerInventory) -> int:
    """GPUs per node behind ``runner`` (a cluster label, a pool label or a slot name)."""
    vng = sorted(cid for cid in inventory.clusters if cid.startswith(VNG_CLUSTER_PREFIXES))
    hint = f"runner: {runner} is not a VNG serving runner; use cluster:<id> with <id> in {', '.join(vng)}, or a pool: h200-hcm, rtx6000pro-hcm, h200-han"
    names = inventory.labels.get(runner, [runner])
    gpus = set()
    for name in names:
        try:
            cluster = inventory.cluster_for(name)
        except ValueError:
            raise ServingError(hint) from None
        if not cluster.id.startswith(VNG_CLUSTER_PREFIXES):
            raise ServingError(hint)
        gpus.add(cluster.gpus_per_node)
    return min(gpus)


def arm_specs(spec: ServingSpec) -> list[tuple[str | None, ServingSpec]]:
    """The engine configurations of a serving YAML: one per arm (``serve:`` + the arm's append), or just the YAML."""
    if spec.arms is None:
        return [(None, spec)]
    out = []
    for name, arm in spec.arms.items():
        serve = f"{spec.serve.rstrip()}\n{arm.append}" if spec.serve and arm.append else spec.serve
        out.append(
            (
                name,
                spec.model_copy(
                    update={"serve": serve, "env": {**spec.env, **arm.env}, "arms": None}
                ),
            )
        )
    return out


def variants(spec: ServingSpec) -> list[Rendered]:
    """Every scenario variant the YAML asks for (one per workload point, or exactly one)."""
    try:
        if spec.workload is None:
            return [scenarios.render(spec.scenario, spec.overrides, slo=spec.slo)]
        return [
            scenarios.render(spec.scenario, spec.overrides, point, spec.workload.search, spec.slo)
            for point in spec.workload.points()
        ]
    except ScenarioError as error:
        raise ServingError(str(error)) from None


def _command_problems(spec: ServingSpec) -> list[str]:
    problems = []
    commands = [("serve", spec.serve)]
    commands += [("router.serve", spec.router.serve)] if spec.router else []
    commands += [("storage.serve", spec.storage.serve)] if spec.storage else []
    for field, command in commands:
        marker = next((m for m in LOCAL_PATH_MARKERS if m in command), None)
        if marker:
            problems.append(
                f"{field}: local path containing {marker!r}; replace it with the HF repo id from model"
            )
        if "$" in command:
            problems.append(
                f"{field}: shell variables ($...) are not expanded; write literal values"
            )
        if "--numa-node" in command:
            problems.append(f"{field}: drop --numa-node (benchmark convention)")
    envs = [("env", spec.env)]
    envs += [("router.env", spec.router.env)] if spec.router else []
    envs += [("storage.env", spec.storage.env)] if spec.storage else []
    for prefix, env in envs:
        for key, value in env.items():
            if not ENV_NAME.fullmatch(key):
                problems.append(f"{prefix}.{key}: not a valid environment variable name")
            elif key in RESERVED_ENV:
                problems.append(
                    f"{prefix}.{key}: reserved, the benchmark sets it; remove it from env"
                )
            if "\n" in value:
                problems.append(f"{prefix}.{key}: value must be a single line")
    return problems


def _placeholder_problems(spec: ServingSpec) -> list[str]:
    problems = []
    blocks = []
    if spec.router:
        if not IMAGE_PATTERN.fullmatch(spec.router.image):
            problems.append("router.image: only letters, digits and ._/:@- are allowed")
        blocks.append(("router", spec.router, ROUTER_PLACEHOLDERS))
        router_texts = [spec.router.serve, *spec.router.env.values()]
        if spec.router.metrics_port is not None and any(
            "{{router_metrics_port}}" in t for t in router_texts
        ):
            problems.append(
                "router.metrics_port: drop it; {{router_metrics_port}} in router.serve picks the port at run time"
            )
    if spec.storage:
        if not IMAGE_PATTERN.fullmatch(spec.storage.image):
            problems.append("storage.image: only letters, digits and ._/:@- are allowed")
        blocks.append(("storage", spec.storage, STORAGE_PLACEHOLDERS))
    for name, block, allowed in blocks:
        texts = [(f"{name}.serve", block.serve)] + [
            (f"{name}.env.{k}", v) for k, v in block.env.items()
        ]
        for field, body in texts:
            problems.extend(
                f"{field}: unknown placeholder {{{{{found}}}}}; allowed: {', '.join(allowed)}"
                for found in PLACEHOLDER.findall(body)
                if found not in allowed
            )
    allowed_env = ENV_PLACEHOLDERS if spec.storage else ()
    for key, value in spec.env.items():
        for found in PLACEHOLDER.findall(value):
            if found not in allowed_env:
                hint = (
                    f"allowed: {', '.join(ENV_PLACEHOLDERS)}"
                    if spec.storage
                    else "no placeholders are allowed in env: without a storage block"
                )
                problems.append(f"env.{key}: unknown placeholder {{{{{found}}}}}; {hint}")
    return problems


def _local_squash_problems(spec: ServingSpec, inventory: RunnerInventory) -> list[str]:
    images = [("image", spec.image)]
    images += [("router.image", spec.router.image)] if spec.router else []
    images += [("storage.image", spec.storage.image)] if spec.storage else []
    local = [field for field, image in images if is_local_squash(image)]
    nodes = sorted(
        {
            inventory.cluster_for(name).id
            for name in inventory.labels.get(spec.runner, [spec.runner])
        }
    )
    if local and len(nodes) > 1:
        return [
            (
                f"{local[0]}: a .sqsh file sits on one node, but runner {spec.runner} spans {', '.join(nodes)}; "
                "use runner: cluster:<id> of the node that holds the file"
            )
        ]
    return []


def _engine_problems(spec: ServingSpec) -> list[str]:
    """Checks of the engine fields (image, model, commands, replicas) a launched config needs."""
    problems = []
    if not IMAGE_PATTERN.fullmatch(spec.image):
        problems.append(
            "image: only letters, digits and ._/:@- are allowed (e.g. vllm/vllm-openai:v0.29.0)"
        )
    if spec.model.startswith("/"):
        problems.append("model: use the HF repo id (e.g. Qwen/Qwen3.8-27B), not a local path")
    elif not MODEL_PATTERN.fullmatch(spec.model):
        problems.append("model: must be an HF repo id <org>/<name> using letters, digits and ._-")
    problems += _command_problems(spec)
    problems += _placeholder_problems(spec)
    if spec.gpus % spec.replicas:
        problems.append(
            f"replicas: gpus {spec.gpus} is not a multiple of replicas {spec.replicas}; "
            "each replica gets gpus/replicas GPUs"
        )
    if spec.replicas > 1:
        if spec.router is None:
            problems.append(
                "replicas: more than 1 replica needs a router: block whose serve lists {{engine_urls}}"
            )
        elif "dynamo.frontend" in spec.router.serve:
            problems.append(
                "replicas: the Dynamo frontend discovers its engines itself; use replicas: 1"
            )
    return problems


def load_serving(text: str, inventory: RunnerInventory | None = None) -> ServingSpec:
    """Validate a serving YAML; raise ``ServingError`` listing every problem."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ServingError(f"serving: not valid YAML ({error})".replace("\n", " ")) from error
    if not isinstance(raw, dict):
        raise ServingError("serving: the top level must be a mapping of keys")
    if "duration" in raw:
        raise ServingError(
            "duration: set it with overrides: {duration: 180} (180 s is the smoke run)"
        )
    try:
        spec = ServingSpec.model_validate(raw)
    except ValidationError as error:
        raise ServingError(_format_validation_error(error)) from error

    problems: list[str] = []
    if spec.endpoint is not None:
        problems += [
            f"{field}: not used with endpoint: (remote mode); remove it"
            for field in REMOTE_EXCLUDED
            if field in spec.model_fields_set
        ]
    else:
        missing = [
            f"{field}: required (or use endpoint: for a remote endpoint)"
            for field in ("image", "model", "gpus", "serve")
            if getattr(spec, field) is None
        ]
        if missing:
            raise ServingError("\n".join(missing))
        seen: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}
        for name, derived in arm_specs(spec):
            problems += [
                f"arms.{name}.{line}" if name else line for line in _engine_problems(derived)
            ]
            key = (" ".join(derived.serve.split()), tuple(sorted(derived.env.items())))
            if name and key in seen:
                problems.append(f"arms.{name}: the same engine command and env as arms.{seen[key]}")
            seen[key] = name or ""
    if len(set(spec.conc)) != len(spec.conc):
        problems.append("conc: duplicate values")
    try:
        rendered = variants(spec)
    except ServingError as error:
        problems.append(str(error))
        rendered = []
    if any(variant.search for variant in rendered) and spec.conc != [1]:
        problems.append(
            f"conc: workload.search finds the concurrency itself; set conc: 1 (got {spec.conc})"
        )
    elif len(rendered) > 1 and len(spec.conc) > 1:
        problems.append(
            f"conc: a workload grid runs one job per point; set exactly one conc (got {len(spec.conc)})"
        )
    inventory = inventory or load_inventory()
    try:
        per_node = runner_gpus(spec.runner, inventory)  # also checks runner is a VNG runner
        if spec.endpoint is None:
            if spec.gpus > per_node:
                problems.append(
                    f"gpus: {spec.gpus} > {per_node} GPUs per node on runner {spec.runner}; "
                    "pick a bigger node"
                )
            problems += _local_squash_problems(spec, inventory)
    except ServingError as error:
        problems.append(str(error))
    if problems:
        raise ServingError("\n".join(problems))
    return spec
