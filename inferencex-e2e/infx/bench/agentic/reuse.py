"""Opt-in reuse of one ready vLLM server across the points of a same-launch-config group.

A point is admitted only after the previous one drained cleanly, every expected engine is
idle with continuous lifetime counters, and one non-preempting local prefix-cache reset
succeeded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from infx.bench import env as inputs, proc, server
from infx.bench.agentic import drain
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
RUNNING, WAITING = "vllm:num_requests_running", "vllm:num_requests_waiting"
QUERIES, HITS = "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"
GAUGES, COUNTERS = (RUNNING, WAITING), (QUERIES, HITS)
RESET_PATH = "/reset_prefix_cache?reset_running_requests=false&reset_external=false"
DEV_PROBE_PATH = "/is_sleeping"
RESET_TIMEOUT_S = 120
POLL_S = 1.0
STABLE_POLLS = 2
# The reset clears only the local GPU prefix cache, so any server key that can configure a
# KV connector, transfer or offload (vLLM flags, LMCache/Mooncake env) is out of scope.
_EXTERNAL_KV = re.compile(r"kv[-_]?(?:transfer|connector)|lmcache|mooncake|offload", re.IGNORECASE)
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>\S+)(?:\s+-?\d+)?$"
)
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


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
        frontend = document["points"][str(concurrencies[0])]["server"]["frontend"]["type"]
        control_url = _control_url(env, frontend)
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
            control_url=control_url,
            launch_identity_sha256=identity,
            drain_timeout=inputs.positive_int("AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS", env),
            client_exit_timeout=inputs.positive_int(
                "AGENTX_REUSE_CLIENT_EXIT_TIMEOUT_SECONDS", env
            ),
            receipt=base / RECEIPT,
        )


def _control_url(env: Mapping[str, str], frontend: str) -> str | None:
    """Resolve the one aggregated engine API independently of the replay router."""
    endpoints = inputs.optional("SRT_AGG_ENDPOINTS", env)
    result = None
    if endpoints is not None:
        if "," in endpoints or any(char.isspace() for char in endpoints):
            raise inputs.InputError("server reuse requires exactly one SRT_AGG_ENDPOINTS endpoint")
        result = _api_root(f"http://{endpoints}")
    if frontend == "vllm-router" and result is None:
        raise inputs.InputError(
            "vllm-router reuse needs one SRT_AGG_ENDPOINTS backend; "
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
    pins = requirements.read_text().split()
    if drain.HARNESS not in pins:
        raise inputs.InputError(
            f"server reuse validates the drain barrier only for {drain.HARNESS}, "
            f"but {requirements} pins {' '.join(pins)}"
        )
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
            "server reuse supports only a single-node vLLM backend with a local KV cache, "
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
    where ``server`` is the point's bound single-node vLLM recipe without its client-side
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
    engine = launch.get("engine")
    if (engine.get("type") if isinstance(engine, dict) else engine) != "vllm":
        raise inputs.InputError(f"point {conc} server engine must be vllm, got {engine!r}")
    if isinstance(engine, dict) and engine.get("connector") is not None:
        raise inputs.InputError(
            f"point {conc} server engine uses connector {engine['connector']!r}; the reset "
            "clears only the local prefix cache"
        )
    frontend = launch.get("frontend")
    if (
        not isinstance(frontend, dict)
        or frontend.get("type") not in {"vllm", "vllm-router"}
        or frontend.get("enable_multiple_frontends") is not False
    ):
        raise inputs.InputError(
            f"point {conc} server frontend must be vllm or vllm-router with "
            f"enable_multiple_frontends false, got {frontend!r}"
        )
    if frontend["type"] == "vllm-router":
        router_args = frontend.get("args", {})
        if (
            not isinstance(router_args, dict)
            or router_args.get("policy") != "consistent_hash"
            or router_args.get("disable-retries") is not True
        ):
            raise inputs.InputError(
                f"point {conc} vllm-router reuse requires consistent_hash and disable-retries true"
            )
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


class Control:
    """The one serving API that owns every internal DP engine and fans a reset out to them."""

    def __init__(self, base_url: str, engines: Sequence[str]) -> None:
        """Control ``base_url``, which must report exactly ``engines``."""
        self.base = base_url.rstrip("/")
        self.engines = sorted(engines)
        self._counters: dict[str, dict[str, float]] = {}

    def require_dev_routes(self) -> None:
        """Raise ``BenchError`` unless the read-only dev route mounted with the reset answers.

        vLLM's Python and Rust API servers mount ``/is_sleeping`` and ``/reset_prefix_cache``
        under the same ``VLLM_SERVER_DEV_MODE`` switch, so this proves the route exists
        without resetting anything; it cannot prove the later reset succeeds.
        """
        status, body = self._request("GET", DEV_PROBE_PATH, server.REQUEST_TIMEOUT_S)
        if status == 404:
            raise inputs.BenchError(
                f"{self.base}{DEV_PROBE_PATH} is not mounted, so neither is the prefix-cache "
                "reset; launch the server with VLLM_SERVER_DEV_MODE=1"
            )
        try:
            answer = json.loads(body) if status == 200 else None
        except ValueError:
            answer = None
        if not isinstance(answer, dict) or answer.get("is_sleeping") is not False:
            raise inputs.BenchError(
                f"{self.base}{DEV_PROBE_PATH} answered HTTP {status} {body[:200]!r}; expected "
                'an awake server: {"is_sleeping": false}'
            )

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Each engine's request gauges and lifetime prefix-cache counters from one scrape.

        Raises ``BenchError`` when an engine is missing, extra, duplicated or incomplete, or
        when a lifetime counter fell since the previous scrape, as it does when an engine
        restarts under the same index.
        """
        status, body = self._request("GET", "/metrics", server.REQUEST_TIMEOUT_S)
        if status != 200:
            raise inputs.BenchError(f"{self.base}/metrics answered HTTP {status}")
        found: dict[str, dict[str, float]] = {}
        for line in body.splitlines():
            match = _SAMPLE.match(line.strip())
            if not match or match["name"] not in {*GAUGES, *COUNTERS}:
                continue
            engine = dict(_LABEL.findall(match["labels"] or "")).get("engine")
            if engine is None:
                continue
            try:
                value = float(match["value"])
            except ValueError:
                value = math.nan
            samples = found.setdefault(engine, {})
            if not math.isfinite(value) or match["name"] in samples:
                raise inputs.BenchError(
                    f"{match['name']} for engine {engine} is non-finite or duplicated"
                )
            samples[match["name"]] = value
        if sorted(found) != self.engines:
            raise inputs.BenchError(
                f"server reports engines {sorted(found)}, expected {self.engines}; an engine "
                "is missing or changed"
            )
        for engine, samples in found.items():
            missing = [name for name in (*GAUGES, *COUNTERS) if name not in samples]
            if missing:
                raise inputs.BenchError(f"engine {engine} lacks {', '.join(missing)}")
            for name in COUNTERS:
                before = self._counters.get(engine, {}).get(name, 0.0)
                if samples[name] < before:
                    raise inputs.BenchError(
                        f"engine {engine} {name} fell from {before:g} to {samples[name]:g}; "
                        "the engine restarted, so the boundary is not continuous"
                    )
        self._counters = {
            engine: {name: samples[name] for name in COUNTERS} for engine, samples in found.items()
        }
        return found

    def require_idle(
        self, timeout: float, interrupted: Callable[[], int | None]
    ) -> dict[str, dict[str, float]]:
        """Wait up to ``timeout`` seconds for consecutive scrapes with every engine idle.

        Raises ``Interrupted`` as soon as ``interrupted()`` reports a signal.
        """
        deadline = time.monotonic() + timeout
        idle = 0
        while True:
            if signum := interrupted():
                raise Interrupted(signum)
            snapshot = self.snapshot()
            busy = any(samples[name] for samples in snapshot.values() for name in GAUGES)
            idle = 0 if busy else idle + 1
            if idle >= STABLE_POLLS:
                return snapshot
            if time.monotonic() >= deadline:
                raise inputs.BenchError(
                    f"engines did not stay idle within {timeout:g}s: {snapshot}"
                )
            time.sleep(POLL_S)

    def reset(self) -> str:
        """Reset the local prefix cache once without preempting; return the response body."""
        status, body = self._request("POST", RESET_PATH, RESET_TIMEOUT_S)
        if status != 200:
            raise inputs.BenchError(f"prefix-cache reset answered HTTP {status}: {body[:200]!r}")
        try:
            outcome = json.loads(body)
        except ValueError:
            outcome = None
        if not isinstance(outcome, dict) or outcome.get("success") is not True:
            raise inputs.BenchError(
                f"prefix-cache reset did not return success: true: {body[:200]!r}"
            )
        return body[:200]

    def _request(self, method: str, path: str, timeout: float) -> tuple[int, str]:
        """Return ``(status, body)``; a missing HTTP response raises ``BenchError``."""
        data = b"{}" if method == "POST" else None
        request = urllib.request.Request(  # noqa: S310
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8", "replace")
        except OSError as error:
            raise inputs.BenchError(f"{method} {self.base}{path} failed: {error}") from None


def run(group: Group, plans: Sequence[Plan], runtime: Runtime, execute: Execute) -> int:
    """Run ``plans`` in order on one server; stop at the first point or boundary that fails.

    A SIGINT, SIGTERM or SIGHUP between points stops admission and returns 128 + N; one
    during a point reaches only that point's own client session.
    """
    control = Control(group.control_url or _api_root(plans[0].replay.url), group.engines)
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
            control.require_dev_routes()
            receipt["engines_before_first_point"] = control.require_idle(
                group.drain_timeout, lambda: signals.received
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
                entry["engines_drained"] = control.require_idle(
                    group.drain_timeout, lambda: signals.received
                )
                entry["reset_response"] = control.reset()
                entry["engines_after_reset"] = control.require_idle(
                    group.drain_timeout, lambda: signals.received
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
