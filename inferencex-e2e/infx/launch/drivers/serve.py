"""The serve driver: an AgentX point whose server the launcher starts itself, on any backend.

One container runs ``python3 -m infx.bench serve``, built from the point's bound srt-slurm recipe.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml

from infx.launch import policy
from infx.launch.artifacts import (
    ArtifactError,
    bundle_server_logs,
    copy_al_gate_result,
    copy_eval_artifacts,
    copy_to_workspace,
)
from infx.launch.backends.base import Container, Mount
from infx.launch.context import Launch, LaunchError
from infx.launch.drivers.script import CONTAINER_MODELS
from infx.launch.drivers.srt.collect import SINGLE_NODE_LOGS
from infx.launch.drivers.srt.models import checkpoint
from infx.launch.request import RequestError, SingleNodeRequest
from infx.results.power import powermetrics
from infx.results.power.multinode import SAMPLES_FILENAME, WINDOWS_DIRNAME
from infx.srt_slurm import al_gate
from infx.srt_slurm.single_node import select_recipe
from infx.srt_slurm.synthetic_acceptance import acceptance_length
from infx.srt_slurm.variants import expand_variants
from infx.srt_slurm.workload import resolve_fabric

if TYPE_CHECKING:
    from infx.clusters import Cluster
    from infx.launch.backends.base import Job

HOST = "127.0.0.1"
PORT = 8888
BENCH = ("python3", "-m", "infx.bench")
WORKDIR = PurePosixPath("/infmax-workspace")
HF_HUB_CACHE_VOLUME = "hf-hub-cache"
POWER_DIR = "power"
# Where the workflow's power audit uploads a job's samples from.
STAGED_POWER_DIR = Path("LOGS") / POWER_DIR
OUTPUTS = ("results", "agentic", "eval")
RECIPE_FIELDS: dict[str, frozenset[str] | None] = {
    "schema": None,
    "name": None,
    "resources": frozenset({"gpu_type", "gpus_per_node"}),
    "model": frozenset({"path", "container", "precision"}),
    "engine": frozenset({"type"}),
    "frontend": frozenset({"type", "enable_multiple_frontends"}),
    "roles": frozenset({"agg"}),
    "benchmark": frozenset({"type", "command", "env"}),
    "health_check": frozenset({"max_attempts", "interval_seconds"}),
    "environment": None,
}
ROLE_KEYS = frozenset({"nodes", "workers", "gpus", "args", "env"})
LAUNCHER_ARGS = frozenset({
    "host", "port", "hf-repo", "model-url", "docker-repo", "models-dir", "models-preset",
    "spec-synth-len", "spec-synth-rates",
})  # fmt: skip


def _cli_flags(args: Mapping[str, Any]) -> list[str]:
    """``args`` as ``--name value`` pairs, sorted by name.

    True gives a bare ``--name``; False and None give nothing.
    """
    flags: list[str] = []
    for name, value in sorted(args.items()):
        if value is True:
            flags.append(f"--{name}")
        elif value is not False and value is not None:
            flags += [f"--{name}", str(value)]
    return flags


def _kebab(name: str) -> str:
    """A recipe arg with hyphens for underscores: ``max_model_len`` is ``max-model-len``."""
    return name.replace("_", "-")


def _llamacpp_server(
    model_root: str, served_name: str, port: int, args: Mapping[str, Any]
) -> list[str]:
    """The ``llama-server`` command, bound to localhost, for the recipe's GGUF file.

    The recipe's ``model`` and any ``model-draft`` args are paths relative to ``model_root``, the
    staged checkpoint.

    ``lazy-mode`` and ``flash-attn`` must be strings because YAML reads an unquoted on or off as a
    boolean.
    """
    args = dict(args)
    gguf = args.pop("model", None)
    if not gguf:
        raise LaunchError("llama.cpp recipes name their GGUF file as roles.agg.args.model")
    if draft := args.get("model-draft"):
        args["model-draft"] = str(PurePosixPath(model_root) / draft)
    for name in ("lazy-mode", "flash-attn"):
        if (value := args.get(name)) not in ("on", "off", "auto"):
            raise LaunchError(
                f"llama.cpp recipes set {name} to the string on, off or auto, not {value!r}"
            )
    return [
        "llama-server", "--model", str(PurePosixPath(model_root) / gguf), "--host", HOST,
        "--port", str(port), *_cli_flags({"alias": served_name, **args}),
    ]  # fmt: skip


SERVERS: dict[str, Callable[[str, str, int, Mapping[str, Any]], list[str]]] = {
    "llamacpp": _llamacpp_server,
}


def _bound_recipe(request: SingleNodeRequest) -> tuple[str, dict[str, Any]]:
    """The point's variant of its recipe, bound as srt-slurm binds it, without its power telemetry.

    Raises ``LaunchError`` for anything in the recipe the serve path would otherwise ignore.
    """
    config = f"{request.workspace}/{request.srt_recipe}"
    try:
        selected, recipe = select_recipe(
            config, request.env, root=request.workspace, expand=expand_variants, native_power=False
        )
        recipe = resolve_fabric(recipe, {})
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise LaunchError(f"{request.srt_recipe}: {error}") from error
    if unsupported := _unsupported_fields(recipe):
        raise LaunchError(f"{selected}: the serve path does not run {', '.join(unsupported)}")
    frontend = recipe.get("frontend") or {}
    engine = recipe["engine"]["type"] if isinstance(recipe["engine"], dict) else recipe["engine"]
    if (kind := frontend.get("type", engine)) != engine:
        raise LaunchError(f"{selected}: the serve path runs no {kind} frontend")
    return selected, recipe


def _unsupported_fields(recipe: Mapping[str, Any]) -> list[str]:
    """The fields of the bound ``recipe`` the serve path would ignore, as sorted dotted paths.

    For each field, ``RECIPE_FIELDS`` lists the keys the serve path reads from its mapping, or
    None for a scalar or a mapping of variables. The engine's own server is the only frontend, so
    ``enable_multiple_frontends`` must stay off.
    """
    found: list[str] = []
    for field, value in recipe.items():
        if field not in RECIPE_FIELDS:
            found.append(str(field))
        elif (keys := RECIPE_FIELDS[field]) is not None and isinstance(value, Mapping):
            found += [f"{field}.{key}" for key in value if key not in keys]
    found += [f"roles.agg.{key}" for key in recipe["roles"]["agg"] if key not in ROLE_KEYS]
    if (recipe.get("frontend") or {}).get("enable_multiple_frontends"):
        found.append("frontend.enable_multiple_frontends")
    return sorted(found)


def _merged_env(*blocks: Mapping[str, Any] | None) -> dict[str, str]:
    """Recipe env mappings merged in order, later ones winning; a null value sets nothing."""
    return {
        str(name): str(value)
        for block in blocks
        for name, value in (block or {}).items()
        if value is not None
    }


def _server_argv(recipe: Mapping[str, Any], model: str, request: SingleNodeRequest) -> list[str]:
    """The framework server's argv for the bound recipe's ``agg`` role.

    Refuses ``LAUNCHER_ARGS``: they set the address the launcher probes, would serve a different
    model from the one the launcher names, or would replace the golden acceptance length.
    """
    server = SERVERS.get(request.framework)
    if server is None:
        raise LaunchError(f"the serve path runs no {request.framework} server")
    args = {_kebab(str(key)): value for key, value in recipe["roles"]["agg"]["args"].items()}
    if owned := sorted(args.keys() & LAUNCHER_ARGS):
        raise LaunchError(f"{request.srt_recipe}: the launcher sets {', '.join(owned)}")
    try:
        acceptance = acceptance_length(recipe, request.framework, request.env)
    except (KeyError, ValueError) as error:
        raise LaunchError(f"{request.srt_recipe}: {error}") from error
    if acceptance is not None:
        args["spec-synth-len"] = f"{acceptance:g}"
    return server(model, request.model, PORT, args)


def _clients(request: SingleNodeRequest, root: PurePosixPath, url: str) -> list[list[str]]:
    """The client argvs to run against the server at ``url``.

    The AgentX client runs unless the point is eval-only; the eval runs if the point has one.
    """
    argvs = [] if request.eval_only else [[*BENCH, "agentic"]]
    if request.run_eval or request.eval_only:
        argvs.append([
            *BENCH, "eval", "--endpoint", url, "--concurrency", str(request.conc),
            "--stage-to", str(root / "eval"),
        ])  # fmt: skip
    return argvs


def _al_gate(request: SingleNodeRequest, recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """The acceptance gate an agentic speculative eval runs first, as the srt driver plans it.

    A ``skipped`` record is written where the workflow uploads it, and gives None like a point
    with no gate at all.
    """
    if not request.eval_only or request.spec_decoding == "none":
        return None
    if al_gate.prompts_staged(request.env, request.workspace):
        gate = al_gate.plan(recipe, request.framework, request.env)
    else:
        gate = al_gate.skipped_without_prompts(request.env)
    if gate is None or gate.get("status") != "skipped":
        return gate
    al_gate.write_result(request.workspace, request.result_filename, gate)
    return None


def _readiness_budget_s(recipe: Mapping[str, Any]) -> int:
    """The seconds the server has to become ready; the defaults are srtctl's ``health_check``."""
    health = recipe.get("health_check") or {}
    return health.get("max_attempts", 180) * health.get("interval_seconds", 10)


def _server_log(root: PurePosixPath) -> PurePosixPath:
    """The job's server log, in RESULT_DIR because the AgentX aggregate reads its KV pool there."""
    return root / "agentic" / "server.log"


def _samples_power(cluster: Cluster, request: SingleNodeRequest) -> bool:
    """Whether the job samples power, which needs a Mac's powermetrics and a benchmark window."""
    return cluster.os == "darwin" and not request.eval_only


def _container(
    launch: Launch, request: SingleNodeRequest, recipe: Mapping[str, Any], root: PurePosixPath
) -> Container:
    """The one container that serves the point and runs its clients in the same environment.

    A MODEL_PATH in the point's additional settings is refused: srt-slurm would serve it, but the
    serve path would ignore it and serve MODEL or its staged checkpoint.
    """
    cluster = launch.cluster
    time_limit = policy.salloc_time_limit(cluster.id, request)
    if time_limit is None:
        raise RequestError.missing("SALLOC_TIME_LIMIT")
    role = recipe["roles"]["agg"]
    point_settings = policy.point_settings(request)
    recipe_env = _merged_env(
        recipe.get("environment"), role.get("env"), recipe["benchmark"].get("env")
    )
    if "MODEL_PATH" in point_settings:
        raise LaunchError("the serve path serves MODEL or its staged checkpoint, not MODEL_PATH")
    mounts = [Mount(HF_HUB_CACHE_VOLUME, PurePosixPath(request.hf_hub_cache), create=True)]
    staged_env: dict[str, str] = {}
    model = request.model
    if (staged := checkpoint(cluster, request)) is not None:
        model_path = CONTAINER_MODELS / staged.dir
        mounts.append(Mount(staged.volume, CONTAINER_MODELS))
        model = str(model_path)
        staged_env["MODEL_PATH"] = model
    elif request.framework == "llamacpp":
        raise LaunchError(
            f"llama.cpp serves only a staged checkpoint; {cluster.id!r} stages no {request.model!r}"
        )
    url = f"http://{HOST}:{PORT}"
    client_env = {
        "RESULT_DIR": str(root / "agentic"),
        "AGENTIC_OUTPUT_DIR": str(root / "results"),
        "AIPERF_SERVER_URL": url,
        "AIPERF_SERVER_METRICS_URLS": f"{url}/metrics",
    }
    command = [
        *BENCH, "serve",
        "--ready-url", f"{url}/health",
        "--ready-timeout", str(_readiness_budget_s(recipe)),
        "--log", str(_server_log(root)),
    ]  # fmt: skip
    if _samples_power(cluster, request):
        samples = root / POWER_DIR / SAMPLES_FILENAME
        sampler = ["python3", "-m", "infx.bench.powermetrics", "--output", str(samples)]
        command += ["--sampler", json.dumps(sampler)]
        # The client writes the benchmark window here, as for srt-slurm's power telemetry.
        client_env["ENABLE_AGENTX_POWER"] = "1"
        client_env["SRT_MEASUREMENT_WINDOW_DIR"] = str(WORKDIR / root / POWER_DIR / WINDOWS_DIRNAME)
    if (gate := _al_gate(request, recipe)) is not None:
        timeout = request.env.get("AL_GATE_TIMEOUT_S") or al_gate.TIMEOUT_S
        client_env |= al_gate.client_env(gate, request.env, str(root / "agentic"))
        client_env["AL_GATE_TIMEOUT_S"] = timeout
        command += ["--gate", json.dumps([*BENCH, "al-gate"]), "--gate-timeout", timeout]
    for client in _clients(request, root, url):
        command += ["--client", json.dumps(client)]
    command += ["--", *_server_argv(recipe, model, request)]
    return Container(
        image=launch.backend.prepare_image(request.image),
        command=command,
        gpus=request.gpu_count,
        time_limit_min=time_limit,
        workspace=request.workspace,
        workdir=WORKDIR,
        env={
            **{name: value for name, value in cluster.env.items() if name not in point_settings},
            **recipe_env,
            # Import infx.bench from the checkout only, as srt_agentic.sh does.
            "PYTHONPATH": str(WORKDIR),
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            **staged_env,
            **client_env,
        },
        mounts=mounts,
        outputs=tuple(root / directory for directory in (*OUTPUTS, POWER_DIR)),
        exclude=(".git",),
    )


@dataclass
class _Staging:
    """Copies the job's server log, result, artifacts, eval and power into the workspace, once.

    ``run`` calls it once the job has ended. It is also a lifecycle cleanup, registered after the
    job's cancel so that it runs first, for a signal or an error that ends the launch sooner.
    ``done`` stops the second call from staging again.
    """

    launch: Launch
    job: Job
    request: SingleNodeRequest
    root: PurePosixPath
    done: bool = False

    def __call__(self) -> int:
        """Stage what the job left; 1 if an artifact it left could not be copied."""
        if self.done:
            return 0
        self.done = True
        request, workspace = self.request, self.request.workspace
        fetched = self.launch.backend.fetch_outputs(self.job, workspace)
        outputs = fetched / self.root
        try:
            _copy_if_present(fetched / _server_log(self.root), workspace)
            if not request.eval_only:
                result = outputs / "results" / f"{request.result_filename}.json"
                _copy_if_present(result, workspace)
            # The workflow uploads results/ as a benchmark point's raw artifact, and an eval-only
            # job has no point for it to match.
            if not request.eval_only and (outputs / "agentic").is_dir():
                shutil.copytree(
                    outputs / "agentic", workspace / "results",
                    ignore=shutil.ignore_patterns("server.log"), dirs_exist_ok=True,
                )  # fmt: skip
            if (outputs / POWER_DIR).is_dir():
                shutil.copytree(
                    outputs / POWER_DIR, workspace / STAGED_POWER_DIR, dirs_exist_ok=True
                )
            if request.run_eval or request.eval_only:
                bundle_server_logs(outputs / "agentic" / "al_gate", workspace / SINGLE_NODE_LOGS)
                copy_eval_artifacts(outputs / "eval", workspace)
                copy_al_gate_result(outputs / "agentic", workspace, request.result_filename)
        except ArtifactError as error:
            self.launch.life.event.error(error)
            return 1
        except OSError as error:
            self.launch.life.event.fail("ArtifactError", f"failed to stage artifacts: {error}")
            return 1
        return 0


def _copy_if_present(source: Path, workspace: Path) -> None:
    """Copy ``source`` into ``workspace`` under its own name if the job left it."""
    if source.is_file():
        copy_to_workspace(source, workspace / source.name)


def _validate_power(launch: Launch, request: SingleNodeRequest) -> int:
    """Join the job's power samples to its AgentX window; 1 if ``REQUIRE_POWER`` went unmet."""
    workspace = request.workspace
    sidecar = f"power_validation_{request.result_filename}.json"
    try:
        valid = powermetrics.validate(
            samples_path=workspace / STAGED_POWER_DIR / SAMPLES_FILENAME,
            bench_result=workspace / "results" / f"agentic_power_concurrency_{request.conc}.json",
            agg_result=workspace / f"{request.result_filename}.json",
            validation_result=workspace / sidecar,
            audit_source=sidecar,
        )
    except (OSError, ValueError) as error:
        print(f"WARNING: AgentX power validation failed: {error}", file=sys.stderr)
        valid = False
    if request.require_power and not valid:
        launch.life.event.fail("PowerValidationFailed", "AgentX power validation failed")
        return 1
    return 0


def run(launch: Launch) -> int:
    """Serve the point in one container, stage its artifacts, and return its exit code."""
    event, backend = launch.life.event, launch.backend
    with event.stage("prepare"):
        request = SingleNodeRequest.from_env(launch.request.env)
        if not request.is_agentic:
            raise LaunchError("the serve path runs only AgentX points")
        if request.require_power and not _samples_power(launch.cluster, request):
            raise LaunchError("this point measures no power, which REQUIRE_POWER demands")
        selected, recipe = _bound_recipe(request)
        event.set(recipe=selected.removeprefix(f"{request.workspace}/"))
        serve_dir = Path(tempfile.mkdtemp(prefix="serve.", dir=request.workspace))
        for directory in OUTPUTS:
            (serve_dir / directory).mkdir()
        if _samples_power(launch.cluster, request):
            (serve_dir / POWER_DIR / WINDOWS_DIRNAME).mkdir(parents=True)
        root = PurePosixPath(serve_dir.relative_to(request.workspace).as_posix())
        container = _container(launch, request, recipe, root)
    job = backend.run_container(container)
    staging = _Staging(launch, job, request, root)
    launch.life.callback(staging)
    backend.stream_logs(job)
    status = backend.state(job)
    with event.stage("collect"):
        rc = staging()
    if not status.succeeded:
        return status.exit_code or 1
    if rc or request.eval_only:
        return rc
    result = request.workspace / f"{request.result_filename}.json"
    if not result.is_file() or result.stat().st_size == 0:
        event.fail("ResultMissing", f"benchmark result {result} is missing or empty")
        return 1
    return _validate_power(launch, request) if _samples_power(launch.cluster, request) else 0
