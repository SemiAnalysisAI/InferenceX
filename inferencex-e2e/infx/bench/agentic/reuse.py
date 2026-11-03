"""Opt-in reuse of one ready server across the points of a same-launch-config group.

A point is admitted only after the previous one drained cleanly, every expected engine is
idle with continuous lifetime counters, and one non-preempting local prefix-cache reset
succeeded.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from infx.bench import env as inputs, proc
from infx.bench.agentic import drain, reuse_vllm
from infx.bench.agentic.venv import REQUIREMENTS, Runtime

if TYPE_CHECKING:
    from infx.bench.agentic.run import Plan

    Execute = Callable[..., int]

ENABLE = "AGENTX_SERVER_REUSE"
REQUIRED = (
    "CONC_LIST",
    "RESULT_DIR",
    "RESULT_FILENAME",
    "MODEL",
    "KV_OFFLOADING",
    "AGENTX_REUSE_ENGINES",
    "AGENTX_REUSE_LAUNCH_IDENTITIES",
    "AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS",
    "AGENTX_REUSE_CLIENT_EXIT_TIMEOUT_SECONDS",
)
SCHEMA = "infx-agentx-server-reuse/v1"
POINT_KEYS = frozenset({"image", "model", "server"})
RECEIPT = "server_reuse.json"
POLL_S = 1.0
STABLE_POLLS = 2
# The reset clears only the local GPU prefix cache, so any server key that can configure a
# KV connector, transfer or offload (vLLM flags, LMCache/Mooncake env) is out of scope.
_EXTERNAL_KV = re.compile(r"kv[-_]?(?:transfer|connector)|lmcache|mooncake|offload", re.IGNORECASE)


class Control(Protocol):
    """One server's control API, which must own every engine the group expects."""

    base: str

    def require_reset_route(self) -> None:
        """Raise ``BenchError`` unless the reset is reachable, without resetting anything."""

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Per-engine samples from one scrape; ``BenchError`` if an engine changed."""

    def idle(self, snapshot: dict[str, dict[str, float]]) -> bool:
        """Whether ``snapshot`` shows no request in flight on any engine."""

    def reset(self) -> str:
        """Reset the local prefix cache once without preempting; return the response."""


@dataclass(frozen=True)
class Backend:
    """What reuse needs from one engine family, selected by the bound recipe's engine type."""

    check_server: Callable[[str, dict[str, Any]], None]
    """Raise ``InputError`` unless a point's bound server recipe is in scope."""
    routers: frozenset[str]
    """Frontend types that only route traffic, so control must address the backend directly."""
    control: Callable[[str, Sequence[str]], Control]


BACKENDS = {
    "vllm": Backend(reuse_vllm.check_server, reuse_vllm.ROUTERS, reuse_vllm.Control),
}


class Interrupted(inputs.BenchError):
    """A signal arrived, so no further point may be admitted."""

    def __init__(self, signum: int) -> None:
        """Remember ``signum``."""
        super().__init__(f"signal {signum} received")
        self.signum = signum


def enabled(env: Mapping[str, str]) -> bool:
    """Whether the caller opted into server reuse; unset keeps the single-point default."""
    return ENABLE in env and inputs.flag(ENABLE, env)


@dataclass(frozen=True)
class Group:
    """Validated points that share one server deployment, in run order."""

    concurrencies: tuple[int, ...]
    points: tuple[dict[str, str], ...]
    """Each point's environment: one ``CONC``, its own ``RESULT_DIR`` and ``RESULT_FILENAME``."""
    engines: tuple[str, ...]
    backend: Backend
    control_url: str | None
    launch_identity_sha256: str
    drain_timeout: int
    client_exit_timeout: int
    receipt: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str], requirements: Path = REQUIREMENTS) -> Group:
        """Validate the whole group before any point runs."""
        values = inputs.require(*REQUIRED, env=env)
        _require_supported(env, requirements)
        concurrencies = _concurrencies(values["CONC_LIST"])
        conc = env.get("CONC")
        if conc and conc != str(concurrencies[0]):
            raise inputs.InputError(
                f"CONC={conc!r} must be the first CONC_LIST point ({concurrencies[0]}), "
                "the one measured on the fresh server"
            )
        engines = tuple(values["AGENTX_REUSE_ENGINES"].split())
        if not engines or len(set(engines)) != len(engines):
            raise inputs.InputError("AGENTX_REUSE_ENGINES must list unique engine indices")
        for engine in engines:
            if not re.fullmatch(r"0|[1-9][0-9]*", engine):
                raise inputs.InputError(f"AGENTX_REUSE_ENGINES has a non-index entry {engine!r}")
        path = Path(values["AGENTX_REUSE_LAUNCH_IDENTITIES"])
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise inputs.InputError(f"cannot read launch identities {path}: {error}") from None
        try:
            identity = check_identities(document, concurrencies, env["MODEL"], env.get("IMAGE"))
        except inputs.InputError as error:
            raise inputs.InputError(f"{path}: {error}") from None
        launch = document["points"][str(concurrencies[0])]["server"]
        backend = _backend(str(concurrencies[0]), launch)
        control_url = _control_url(env, launch["frontend"]["type"], backend)
        point_env = {key: value for key, value in env.items() if key != "CONC_LIST"}
        if control_url is not None:
            metrics_url = f"{control_url}/metrics"
            configured_metrics = inputs.optional("AIPERF_SERVER_METRICS_URLS", env)
            if configured_metrics is not None and configured_metrics != metrics_url:
                raise inputs.InputError(
                    "server reuse requires AIPERF_SERVER_METRICS_URLS to be the control "
                    f"API's single metrics endpoint {metrics_url!r}"
                )
            point_env["AIPERF_SERVER_METRICS_URLS"] = metrics_url
        base = Path(values["RESULT_DIR"]).absolute()
        name = values["RESULT_FILENAME"]
        points = tuple(
            {
                **point_env,
                "CONC": str(c),
                "RESULT_DIR": str(base / f"conc_{c}"),
                "RESULT_FILENAME": f"{name}_conc{c}",
            }
            for c in concurrencies
        )
        paths = (base / RECEIPT, *(Path(point["RESULT_DIR"]) for point in points))
        stale = [str(path) for path in paths if path.exists()]
        if stale:
            raise inputs.InputError(
                "server reuse needs fresh result paths so no earlier log or metric is mixed "
                f"into a point; already present: {', '.join(stale)}"
            )
        return cls(
            concurrencies=concurrencies,
            points=points,
            engines=engines,
            backend=backend,
            control_url=control_url,
            launch_identity_sha256=identity,
            drain_timeout=inputs.positive_int("AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS", env),
            client_exit_timeout=inputs.positive_int(
                "AGENTX_REUSE_CLIENT_EXIT_TIMEOUT_SECONDS", env
            ),
            receipt=base / RECEIPT,
        )


def _control_url(env: Mapping[str, str], frontend: str, backend: Backend) -> str | None:
    """Resolve the one aggregated engine API independently of the replay router."""
    endpoints = inputs.optional("SRT_AGG_ENDPOINTS", env)
    result = None
    if endpoints is not None:
        if "," in endpoints or any(char.isspace() for char in endpoints):
            raise inputs.InputError("server reuse requires exactly one SRT_AGG_ENDPOINTS endpoint")
        result = _api_root(f"http://{endpoints}")
    if frontend in backend.routers and result is None:
        raise inputs.InputError(
            f"{frontend} reuse needs one SRT_AGG_ENDPOINTS backend; "
            "reset must not be sent through the router"
        )
    return result


def _api_root(value: str) -> str:
    """Validate an HTTP API root without credentials, query, fragment or subpath."""
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise inputs.InputError(f"invalid reuse control URL: {error}") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or (port is not None and port <= 0)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in value)
    ):
        raise inputs.InputError(f"reuse control URL must be an HTTP API root, got {value!r}")
    return value.rstrip("/")


def _require_supported(env: Mapping[str, str], requirements: Path) -> None:
    """Raise ``InputError`` for a client release or topology whose invariants are unproven."""
    pinned = None
    for line in requirements.read_text().splitlines():
        name, exact, version = line.partition("#")[0].strip().partition("==")
        if exact and name == drain.CLIENT:
            pinned = version
    drain.require_validated(pinned, str(requirements))
    unsupported = {
        "IS_MULTINODE": inputs.flag("IS_MULTINODE", env),
        "EVAL_ONLY": inputs.flag("EVAL_ONLY", env),
        "ENABLE_AGENTX_POWER": env.get("ENABLE_AGENTX_POWER") not in {None, "", "0", "false"},
        "KV_OFFLOADING": env["KV_OFFLOADING"] != "none",
        "KV_OFFLOAD_BACKEND": bool(env.get("KV_OFFLOAD_BACKEND")),
    }
    named = [name for name, present in unsupported.items() if present]
    if named:
        raise inputs.InputError(
            "server reuse supports only a single-node server with a local KV cache, "
            f"no eval-only run and no AgentX power windows; unsupported: {', '.join(named)}"
        )


def _concurrencies(value: str) -> tuple[int, ...]:
    """At least two unique positive concurrencies, in run order."""
    points = tuple(inputs.parse_positive_int("CONC_LIST", word) for word in value.split())
    if len(points) < 2 or len(set(points)) != len(points):
        raise inputs.InputError(
            f"server reuse needs at least two unique CONC_LIST points, got {value!r}"
        )
    return points


def check_identities(
    document: object, concurrencies: Sequence[int], model: str, image: str | None
) -> str:
    """The sha256 of the one server identity every point declares, else ``InputError``.

    ``document`` is ``{"schema": SCHEMA, "points": {"<conc>": {"image", "model", "server"}}}``,
    where ``server`` is the point's bound single-node recipe without its client-side
    ``benchmark`` block and variant ``name``. Every point must declare the identical
    server; ``model`` and ``image`` must match it and this job; and no server key may
    configure an external, connector-managed or offloaded KV cache.
    """
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise inputs.InputError(f"launch identities must declare schema {SCHEMA!r}")
    if set(document) != {"schema", "points"} or not isinstance(document["points"], dict):
        raise inputs.InputError("launch identities must hold exactly 'schema' and 'points'")
    declared = document["points"]
    expected = [str(c) for c in concurrencies]
    if set(declared) != set(expected):
        raise inputs.InputError(
            f"launch identities must cover exactly the CONC_LIST points {expected}; "
            f"got {sorted(declared, key=lambda key: (len(key), key))}"
        )
    for conc in expected:
        _check_point(conc, declared[conc], model, image)
    first = declared[expected[0]]
    for conc in expected[1:]:
        if declared[conc] != first:
            differ = ", ".join(_differences(first, declared[conc]))
            raise inputs.InputError(
                f"points {expected[0]} and {conc} declare different server launches ({differ}); "
                "they cannot share one server"
            )
    canonical = json.dumps(first, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _check_point(conc: str, point: object, model: str, image: str | None) -> None:
    """Raise ``InputError`` unless ``point`` is a well-formed, in-scope server identity."""
    if not isinstance(point, dict) or set(point) != POINT_KEYS:
        raise inputs.InputError(f"point {conc} must hold exactly {sorted(POINT_KEYS)}")
    if point["model"] != model:
        raise inputs.InputError(f"point {conc} model {point['model']!r} is not MODEL={model!r}")
    if (
        not isinstance(point["image"], str)
        or not point["image"]
        or image not in {None, point["image"]}
    ):
        raise inputs.InputError(f"point {conc} image {point['image']!r} does not match IMAGE")
    launch = point["server"]
    if not isinstance(launch, dict) or "benchmark" in launch or "name" in launch:
        raise inputs.InputError(
            f"point {conc} server must be the bound recipe without 'benchmark' and 'name'"
        )
    recipe_model = launch.get("model")
    if (
        not isinstance(recipe_model, dict)
        or recipe_model.get("container") != point["image"]
        or recipe_model.get("path") not in {model, f"hf:{model}"}
    ):
        raise inputs.InputError(
            f"point {conc} server model block does not serve {model} from image"
        )
    backend = _backend(conc, launch)
    engine = launch["engine"]
    if isinstance(engine, dict) and engine.get("connector") is not None:
        raise inputs.InputError(
            f"point {conc} server engine uses connector {engine['connector']!r}; the reset "
            "clears only the local prefix cache"
        )
    backend.check_server(conc, launch)
    roles = launch.get("roles")
    agg = roles.get("agg") if isinstance(roles, dict) else None
    if set(roles or ()) != {"agg"} or not isinstance(agg, dict) or agg.get("workers") != 1:
        raise inputs.InputError(f"point {conc} server must have exactly one 'agg' role, one worker")
    if not isinstance(agg.get("args"), dict) or not agg["args"]:
        raise inputs.InputError(f"point {conc} server role 'agg' has no engine args")
    external = sorted(set(_keys_matching(launch, _EXTERNAL_KV)))
    if external:
        raise inputs.InputError(
            f"point {conc} server configures an external or offloaded KV cache "
            f"({', '.join(external)}); the reset clears only the local prefix cache"
        )


def _backend(conc: str, launch: dict[str, Any]) -> Backend:
    """The ``BACKENDS`` entry for ``launch``'s engine type, else ``InputError``."""
    engine = launch.get("engine")
    name = engine.get("type") if isinstance(engine, dict) else engine
    if not isinstance(name, str) or name not in BACKENDS:
        raise inputs.InputError(
            f"point {conc} server engine must be {' or '.join(BACKENDS)}, got {engine!r}"
        )
    return BACKENDS[name]


def _keys_matching(value: object, pattern: re.Pattern[str], path: str = "") -> list[str]:
    """Dotted paths of every mapping key below ``value`` that ``pattern`` finds."""
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            where = f"{path}.{key}" if path else str(key)
            if pattern.search(str(key)):
                found.append(where)
            found += _keys_matching(child, pattern, where)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found += _keys_matching(child, pattern, f"{path}[{index}]")
    return found


def _differences(left: object, right: object, path: str = "") -> list[str]:
    """Dotted paths where ``left`` and ``right`` differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            difference
            for key in sorted({*left, *right}, key=str)
            for difference in _differences(
                left.get(key), right.get(key), f"{path}.{key}" if path else str(key)
            )
        ]
    return [] if left == right else [path or "<root>"]


def _require_idle(
    control: Control, timeout: float, interrupted: Callable[[], int | None]
) -> dict[str, dict[str, float]]:
    """Wait up to ``timeout`` seconds for consecutive scrapes with every engine idle.

    Raises ``Interrupted`` as soon as ``interrupted()`` reports a signal.
    """
    deadline = time.monotonic() + timeout
    idle = 0
    while True:
        if signum := interrupted():
            raise Interrupted(signum)
        snapshot = control.snapshot()
        idle = idle + 1 if control.idle(snapshot) else 0
        if idle >= STABLE_POLLS:
            return snapshot
        if time.monotonic() >= deadline:
            raise inputs.BenchError(f"engines did not stay idle within {timeout:g}s: {snapshot}")
        time.sleep(POLL_S)


def run(group: Group, plans: Sequence[Plan], runtime: Runtime, execute: Execute) -> int:
    """Run ``plans`` in order on one server; stop at the first point or boundary that fails.

    A SIGINT, SIGTERM or SIGHUP between points stops admission and returns 128 + N; one
    during a point reaches only that point's own client session.
    """
    control = group.backend.control(
        group.control_url or _api_root(plans[0].replay.url), group.engines
    )
    receipt: dict[str, Any] = {
        "status": "running",
        "control_url": control.base,
        "engines": list(group.engines),
        "concurrencies": list(group.concurrencies),
        "launch_identity_sha256": group.launch_identity_sha256,
        "points": [],
    }

    def save() -> None:
        group.receipt.parent.mkdir(parents=True, exist_ok=True)
        partial = group.receipt.with_suffix(".tmp")
        partial.write_text(json.dumps(receipt, indent=2) + "\n")
        partial.replace(group.receipt)

    def stop(status: str, error: str, *, announce: bool = True) -> None:
        receipt.update(status=status, error=error)
        save()
        if announce:
            print(f"ERROR: {error}", flush=True)

    def after() -> str:
        done = [entry["concurrency"] for entry in receipt["points"] if "drain_barrier" in entry]
        return f"after c{done[-1]}" if done else "before the first point"

    save()
    with proc.DeferSignals() as signals:
        try:
            control.require_reset_route()
            receipt["engines_before_first_point"] = _require_idle(
                control, group.drain_timeout, lambda: signals.received
            )
            for index, (plan, point) in enumerate(zip(plans, group.points, strict=True)):
                if signals.received:
                    raise Interrupted(signals.received)
                entry: dict[str, Any] = {"concurrency": plan.replay.concurrency}
                receipt["points"].append(entry)
                save()
                rc = execute(plan, runtime, point, client_exit_timeout=group.client_exit_timeout)
                entry["exit_code"] = rc
                if rc:
                    stop(
                        "terminated" if rc > 128 else "failed",
                        f"point c{plan.replay.concurrency} exited {rc}; later points were not run",
                    )
                    return rc
                entry["drain_barrier"] = drain.check(plan.replay.artifact_dir / drain.LOG)
                save()
                if index == len(plans) - 1:
                    break
                entry["engines_drained"] = _require_idle(
                    control, group.drain_timeout, lambda: signals.received
                )
                entry["reset_response"] = control.reset()
                entry["engines_after_reset"] = _require_idle(
                    control, group.drain_timeout, lambda: signals.received
                )
                print(f"Reset the prefix cache after c{plan.replay.concurrency}", flush=True)
            if signals.received:
                raise Interrupted(signals.received)
        except Interrupted as error:
            stop("terminated", f"signal {error.signum} received {after()}; no later point admitted")
            return 128 + error.signum
        except inputs.BenchError as error:
            stop("failed", str(error), announce=False)
            raise
    receipt["status"] = "passed"
    save()
    return 0
