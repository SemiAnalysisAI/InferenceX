"""Keep the GB200 Kimi-K3 AgentX telemetry blocks on the fleet-standard cadence.

The srt-slurm power collector runs on a fixed cadence (``next_cycle +=
interval``) and every scrape is bounded by ``request_timeout_seconds``. On the
GB200 fleet the head node's own dcgm-exporter shares the node with the Dynamo
frontend, NATS/etcd and the AIPerf client and shows a 1.0-1.6 s response tail
(run 35492708340: max scrape duration 1.2-1.6 s on the passing 1000 ms / 2 s
lanes, hundreds of timed-out scrapes on the 500 ms / 1 s lanes). A request
timeout at or below that tail turns ordinary slow responses into repeated
misses and ``sample_gap_exceeded``, which sets ``publication_valid=False`` after
the benchmark has already spent its GPU hour.

These tests pin the invariants that keep the four lanes publishable:

* the telemetry block matches the sibling GB200 Kimi-K3 recipes that already
  pass the power gate, so the four lanes cannot silently diverge again;
* ``request_timeout_seconds`` stays above the observed exporter latency tail;
* ``collect_interval_ms`` stays inside the validator's coverage limit;
* ``collector_join_timeout_seconds`` still covers two worst-case collector
  cycles (mirrors srt-slurm's ``_validate_collector_budget``).
"""

from pathlib import Path

import pytest
import yaml

from infx.results.power.multinode import MAX_SAMPLE_GAP_SECONDS


ROOT = Path(__file__).resolve().parents[1]
LANE_DIR = ROOT / "benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/gb200-fp4/agentx"

# The four lanes #3287 adds measured power to.
LANES = (
    "agg-dep16-vllm-simple-offload.yaml",
    "agg-dep16.yaml",
    "agg-tep16-balanced.yaml",
    "agg-tp16-latency.yaml",
)

# Sibling lanes that already collect required power and pass the gate.
REFERENCE_LANE = "agg-tp8pp2-mooncake-c16.yaml"

# Largest head-node exporter response observed on a passing GB200 Kimi-K3 lane
# (run 35492708340, max_scrape_duration_seconds 1.62). The request timeout has
# to sit above this tail or the collector manufactures its own gaps.
OBSERVED_EXPORTER_TAIL_SECONDS = 1.62

# Mirrors srt-slurm's COLLECT_CYCLE_TIMEOUT_GRACE_SECONDS.
_COLLECT_CYCLE_GRACE_SECONDS = 1.0

_COMPARED_KEYS = (
    "collect_interval_ms",
    "request_timeout_seconds",
    "required",
    "startup_timeout_seconds",
    "collector_join_timeout_seconds",
)


def _telemetry(name):
    recipe = yaml.safe_load((LANE_DIR / name).read_text())
    telemetry = recipe.get("telemetry")
    assert telemetry is not None, f"{name} declares no telemetry block"
    return telemetry


def test_every_lane_recipe_is_present():
    """A renamed recipe would make the checks below vacuous."""
    missing = [name for name in (*LANES, REFERENCE_LANE) if not (LANE_DIR / name).exists()]
    assert not missing, f"recipes moved or renamed: {missing}"


@pytest.mark.parametrize("name", LANES)
def test_lane_matches_the_passing_sibling_cadence(name):
    reference = _telemetry(REFERENCE_LANE)
    telemetry = _telemetry(name)
    diverged = {
        key: (telemetry.get(key), reference.get(key))
        for key in _COMPARED_KEYS
        if telemetry.get(key) != reference.get(key)
    }
    assert not diverged, f"{name} diverges from {REFERENCE_LANE} on {diverged}; keep the fleet-standard cadence"


@pytest.mark.parametrize("name", LANES)
def test_request_timeout_clears_the_observed_exporter_tail(name):
    timeout = _telemetry(name)["request_timeout_seconds"]
    assert timeout > OBSERVED_EXPORTER_TAIL_SECONDS, (
        f"{name}: request_timeout_seconds={timeout} is inside the {OBSERVED_EXPORTER_TAIL_SECONDS}s "
        "head-node exporter tail; slow responses would become timed-out scrapes"
    )


@pytest.mark.parametrize("name", LANES)
def test_sample_interval_is_inside_the_coverage_limit(name):
    interval_seconds = _telemetry(name)["collect_interval_ms"] / 1000
    assert interval_seconds < MAX_SAMPLE_GAP_SECONDS, (
        f"{name}: collect_interval_ms alone would exceed the {MAX_SAMPLE_GAP_SECONDS}s gap limit"
    )


@pytest.mark.parametrize("name", LANES)
def test_join_timeout_still_covers_two_collector_cycles(name):
    """Mirrors srt-slurm's schema check so a recipe cannot be gate-valid but schema-invalid."""
    telemetry = _telemetry(name)
    worst_case_join = 2 * (2 * telemetry["request_timeout_seconds"] + _COLLECT_CYCLE_GRACE_SECONDS)
    assert telemetry["collector_join_timeout_seconds"] > worst_case_join, (
        f"{name}: collector_join_timeout_seconds must exceed {worst_case_join}s"
    )
