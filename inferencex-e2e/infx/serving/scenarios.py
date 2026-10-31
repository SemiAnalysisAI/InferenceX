"""Serving scenarios: six base AIPerf configs plus the variant a serving YAML asks for.

render() returns the final AIPerf config and the legacy label its results are filed under
(agentx-cc-062126-smoke, sweep-isl8k-osl1k-cache50, ...), so old and new runs line up.
Every error is one line "<field>: <problem>".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infx.serving import greenspec, speedbench
from infx.serving.datasets import LockEntry, load_lock
from infx.serving.slo import SLOError, parse_slo

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "benchmarks/single_node/serving/scenarios"
FAMILIES = (
    "agentx",
    "tau2",
    "livekit",
    "shared-prefix-e2",
    "hidden-prefix-e3",
    "synthetic",
    "greenspec",
    "speedbench",
)
OVERRIDE_KEYS = (
    "duration",
    "dataset",
    "max_context_length",
    "generation",
    "trace_idle_gap_cap",
    "warmup_requests_per_lane",
)
SYNTHETIC_OVERRIDES = ("duration",)
GENERATION_KEYS = ("top_p", "temperature")
ISL = {"8k": 8192, "32k": 32768, "64k": 65536, "128k": 131072}
OSL = {"512": 512, "1k": 1024}
CACHE = (0, 50, 90)
DEFAULT_SLO = "time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50"
PUBLIC_DATASET = "semianalysis_cc_traces_weka_062126"
LABELS = {
    "agentx": "agentx-cc-062126",
    "tau2": "custom-agentic-tau2",
    "shared-prefix-e2": "agentx-shared-prefix-e2",
    "hidden-prefix-e3": "agentx-hidden-prefix-e3",
}
LIVEKIT_LABEL = "custom-agentic-livekit111868-"


class ScenarioError(ValueError):
    """A scenario variant problem, as one line."""


@dataclass(frozen=True)
class Point:
    """One synthetic grid point."""

    isl: str
    osl: str
    cache: int

    @property
    def label(self) -> str:
        return f"isl{self.isl}-osl{self.osl}-cache{self.cache}"


@dataclass(frozen=True)
class Rendered:
    """The AIPerf config one benchmark job replays, and what it is filed under."""

    doc: dict[str, Any]
    label: str
    kind: str  # public | file | synthetic
    dataset: str  # public loader alias, lock key, or "synthetic"
    search: bool
    duration: float

    def yaml(self) -> str:
        return yaml.safe_dump(self.doc, sort_keys=False)


def load_base(family: str) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ScenarioError(f"scenario: {family!r} is not one of {', '.join(FAMILIES)}")
    return yaml.safe_load((SCENARIOS_DIR / f"{family}.yaml").read_text())


def _profiling(doc: dict[str, Any]) -> dict[str, Any]:
    [phase] = [p for p in doc["benchmark"]["phases"] if p.get("kind") == "profiling"]
    return phase


def _positive(field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ScenarioError(f"{field}: must be a positive number, got {value!r}")
    return float(value)


def _apply(
    doc: dict[str, Any], family: str, overrides: dict[str, Any], lock: dict[str, LockEntry]
) -> list[str]:
    """Apply ``overrides`` to ``doc`` in place; return the label suffixes, in a fixed order."""
    if unknown := sorted(set(overrides) - set(OVERRIDE_KEYS)):
        raise ScenarioError(
            f"overrides: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(OVERRIDE_KEYS)}"
        )
    dataset = doc["benchmark"]["datasets"][0]
    suffixes: list[str] = []
    if "dataset" in overrides:
        value = overrides["dataset"]
        if dataset["type"] == "public":
            if not (isinstance(value, str) and value.startswith(PUBLIC_DATASET)):
                raise ScenarioError(
                    f"overrides.dataset: {family} takes a public {PUBLIC_DATASET}* dataset, got {value!r}"
                )
            dataset["dataset"] = value
            if value != PUBLIC_DATASET:
                suffixes.append("-" + value.removeprefix(PUBLIC_DATASET).lstrip("_"))
        else:
            entry = lock.get(value) if isinstance(value, str) else None
            if entry is None:
                raise ScenarioError(
                    f"overrides.dataset: {value!r} is not in the dataset lock; available: {', '.join(lock)}"
                )
            if entry.entries is None:
                raise ScenarioError(f"overrides.dataset: lock entry {value} has no entries count")
            dataset["name"], dataset["entries"] = value, entry.entries
    if "max_context_length" in overrides:
        value = overrides["max_context_length"]
        if value is None:
            dataset.pop("maxContextLength", None)
            suffixes.append("-1m")
        else:
            dataset["maxContextLength"] = int(_positive("overrides.max_context_length", value))
            suffixes.append(f"-ctx{dataset['maxContextLength']}")
    if "generation" in overrides:
        value = overrides["generation"]
        if not isinstance(value, dict) or not value or set(value) - set(GENERATION_KEYS):
            raise ScenarioError(f"overrides.generation: a mapping of {', '.join(GENERATION_KEYS)}")
        extra = doc["benchmark"]["endpoint"].setdefault("extra", {})
        extra.update({key: float(item) for key, item in value.items()})
        suffixes.append("-topp95" if value == {"top_p": 0.95, "temperature": 1.0} else "-gen")
    if "trace_idle_gap_cap" in overrides:
        value = overrides["trace_idle_gap_cap"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ScenarioError(f"overrides.trace_idle_gap_cap: seconds >= 0, got {value!r}")
        dataset["traceIdleGapCapSeconds"] = float(value)
        suffixes.append("-nothink" if value == 0 else f"-idle{value:g}")
    if "warmup_requests_per_lane" in overrides:
        value = overrides["warmup_requests_per_lane"]
        phase = _profiling(doc)
        # 0 removes the key: aiperf's warmup budget is None (only the mandatory
        # snapshot primers run), for endpoints whose token quota the zero-idle
        # cache-pressure burst would exceed.
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ScenarioError(
                f"overrides.warmup_requests_per_lane: 0 to disable or a positive int, got {value!r}"
            )
        if value == 0:
            phase.pop("warmupRequestsPerLane", None)
            suffixes.append("-nowarmup")
        else:
            phase["warmupRequestsPerLane"] = value
            suffixes.append(f"-warm{value}")
    if "duration" in overrides:
        value = _positive("overrides.duration", overrides["duration"])
        _profiling(doc)["duration"] = value
        suffixes.append("-smoke" if value == 180 else f"-d{int(value)}")
    return suffixes


def _synthetic(point: Point, search: bool, slo: str | None) -> dict[str, Any]:
    if point.isl not in ISL:
        raise ScenarioError(f"workload.isl: {point.isl!r} is not one of {', '.join(ISL)}")
    if point.osl not in OSL:
        raise ScenarioError(f"workload.osl: {point.osl!r} is not one of {', '.join(OSL)}")
    if point.cache not in CACHE:
        raise ScenarioError(
            f"workload.cache: {point.cache!r} is not one of {', '.join(map(str, CACHE))}"
        )
    if slo and not search:
        raise ScenarioError("slo: only applies with workload.search: true")
    isl = ISL[point.isl]
    prefix = round(isl * point.cache / 100)  # added on top of isl, so isl - prefix keeps the total
    doc = load_base("synthetic")
    dataset = doc["benchmark"]["datasets"][0]
    dataset["isl"], dataset["osl"] = isl - prefix, OSL[point.osl]
    dataset.pop("prefix_prompts", None)
    if prefix:
        dataset["prefix_prompts"] = {"shared_system_length": prefix}
    if search:
        try:
            filters = parse_slo(slo or DEFAULT_SLO)
        except SLOError as error:
            raise ScenarioError(str(error)) from None
        doc["sweep"] = {
            "type": "adaptive_search",
            "planner": "smooth_isotonic",
            "objectives": [
                {"metric": "output_token_throughput", "stat": "avg", "direction": "maximize"}
            ],
            "search_space": [
                {"path": "phases.profiling.concurrency", "lo": 1, "hi": 512, "kind": "int"}
            ],
            "sla_filters": [
                {"metric_tag": s.metric, "stat": s.stat, "op": s.op, "threshold": s.threshold}
                for s in filters
            ],
            "max_iterations": 16,
            "sla_precision": "normal",  # 1000 requests/probe cap
        }
    return doc


def _greenspec(
    overrides: dict[str, Any],
    point: Point | None,
    search: bool,
    slo: str | None,
    lock: dict[str, LockEntry],
) -> Rendered:
    """The greenspec scenario with its selection; the run length is the selection, so there is no duration."""
    if point is not None or search or slo:
        raise ScenarioError("workload/slo: only for scenario synthetic, not greenspec")
    try:
        chosen = greenspec.normalize(overrides)
    except greenspec.GreenSpecError as error:
        raise ScenarioError(str(error)) from None
    doc = load_base("greenspec")
    name = doc["benchmark"]["datasets"][0]["name"]
    if name not in lock:
        raise ScenarioError(f"scenario: dataset {name} of greenspec is not in the dataset lock")
    if chosen.requests:
        _profiling(doc)["requests"] = chosen.requests
    return Rendered(doc, greenspec.label(chosen), "file", name, False, 0.0)


def _speedbench(
    overrides: dict[str, Any],
    point: Point | None,
    search: bool,
    slo: str | None,
    lock: dict[str, LockEntry],
) -> Rendered:
    """The speedbench scenario with its selection; the run length is the selection, so there is no duration."""
    if point is not None or search or slo:
        raise ScenarioError("workload/slo: only for scenario synthetic, not speedbench")
    try:
        chosen = speedbench.normalize(overrides)
    except speedbench.SpeedBenchError as error:
        raise ScenarioError(str(error)) from None
    if chosen.dataset not in lock:
        raise ScenarioError(
            f"scenario: dataset {chosen.dataset} of speedbench is not in the dataset lock"
        )
    doc = load_base("speedbench")
    doc["benchmark"]["datasets"][0]["name"] = chosen.dataset
    if chosen.requests:
        _profiling(doc)["requests"] = chosen.requests
    return Rendered(doc, speedbench.label(chosen), "file", chosen.dataset, False, 0.0)


def render(
    family: str,
    overrides: dict[str, Any] | None = None,
    point: Point | None = None,
    search: bool = False,
    slo: str | None = None,
    lock: dict[str, LockEntry] | None = None,
) -> Rendered:
    """The final AIPerf config of ``family`` with its variant applied."""
    overrides = dict(overrides or {})
    lock = load_lock() if lock is None else lock
    if family == "greenspec":
        return _greenspec(overrides, point, search, slo, lock)
    if family == "speedbench":
        return _speedbench(overrides, point, search, slo, lock)
    if family == "synthetic":
        if point is None:
            raise ScenarioError("workload: synthetic needs isl, osl and cache")
        if extra := sorted(set(overrides) - set(SYNTHETIC_OVERRIDES)):
            raise ScenarioError(
                f"overrides: synthetic only takes {', '.join(SYNTHETIC_OVERRIDES)}, not {', '.join(extra)}"
            )
        doc = _synthetic(point, search, slo)
        suffixes = _apply(doc, family, overrides, lock)
        label = f"{'sweep' if search else 'fixed'}-{point.label}{''.join(suffixes)}"
        return Rendered(doc, label, "synthetic", "synthetic", search, _profiling(doc)["duration"])
    if point is not None or search or slo:
        raise ScenarioError(f"workload/slo: only for scenario synthetic, not {family}")
    doc = load_base(family)
    suffixes = _apply(doc, family, overrides, lock)
    dataset = doc["benchmark"]["datasets"][0]
    if dataset["type"] == "public":
        kind, name = "public", dataset["dataset"]
    else:
        kind, name = "file", dataset["name"]
        if name not in lock:
            raise ScenarioError(f"scenario: dataset {name} of {family} is not in the dataset lock")
    base = LABELS.get(family) or LIVEKIT_LABEL + name.removeprefix("livekit-111868-")
    return Rendered(doc, base + "".join(suffixes), kind, name, False, _profiling(doc)["duration"])
