"""The vLLM server-reuse backend: recipe rules, metrics, the dev-route probe and the reset."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from infx.bench import env as inputs, server

ROUTERS = frozenset({"vllm-router"})
FRONTENDS = frozenset({"vllm", *ROUTERS})
RUNNING, WAITING = "vllm:num_requests_running", "vllm:num_requests_waiting"
QUERIES, HITS = "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"
GAUGES, COUNTERS = (RUNNING, WAITING), (QUERIES, HITS)
RESET_PATH = "/reset_prefix_cache?reset_running_requests=false&reset_external=false"
DEV_PROBE_PATH = "/is_sleeping"
RESET_TIMEOUT_S = 120
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>\S+)(?:\s+-?\d+)?$"
)
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


def check_server(conc: str, launch: dict[str, Any]) -> None:
    """Raise ``InputError`` unless ``launch`` uses one qualified vLLM frontend or router."""
    frontend = launch.get("frontend")
    if (
        not isinstance(frontend, dict)
        or frontend.get("type") not in FRONTENDS
        or frontend.get("enable_multiple_frontends") is not False
    ):
        raise inputs.InputError(
            f"point {conc} server frontend must be vllm or vllm-router with "
            f"enable_multiple_frontends false, got {frontend!r}"
        )
    if frontend["type"] in ROUTERS:
        router_args = frontend.get("args", {})
        if (
            not isinstance(router_args, dict)
            or router_args.get("policy") != "consistent_hash"
            or router_args.get("disable-retries") is not True
        ):
            raise inputs.InputError(
                f"point {conc} vllm-router reuse requires consistent_hash and disable-retries true"
            )


class Control:
    """The one serving API that owns every internal DP engine and fans a reset out to them."""

    def __init__(self, base_url: str, engines: Sequence[str]) -> None:
        """Control ``base_url``, which must report exactly ``engines``."""
        self.base = base_url.rstrip("/")
        self.engines = sorted(engines)
        self._counters: dict[str, dict[str, float]] = {}

    def require_reset_route(self) -> None:
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

    def idle(self, snapshot: dict[str, dict[str, float]]) -> bool:
        """Whether ``snapshot`` shows no running or waiting request on any engine."""
        return not any(samples[name] for samples in snapshot.values() for name in GAUGES)

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
