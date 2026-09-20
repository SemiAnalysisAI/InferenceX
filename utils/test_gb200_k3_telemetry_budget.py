"""Keep the GB200 Kimi-K3 AgentX lanes inside the power validator's gap budget.

The tachometer scraper sleeps ``collect_interval_ms`` *after* each scrape
returns rather than on a fixed schedule, so one scrape that runs into
``request_timeout_seconds`` leaves a hole of ``timeout + interval`` in the
sample stream. The power validator rejects any window whose largest hole
exceeds ``MAX_SAMPLE_GAP_SECONDS``, and that rejection is what sets
``publication_valid=False`` -> ``package_recompute_invalid`` -> job failure.

At the 1000 ms / 2 s these recipes were first written with, one timed-out
scrape produces exactly 3.0 s against a 3.0 s budget. Every one of these lanes
that finished its benchmark was then thrown away at the telemetry gate, after
one to four hours of GB200 time. These are 16-way expert-parallel lanes that
saturate the node, so a timed-out scrape is not a rare event for them.
"""

from pathlib import Path

import pytest
import yaml

from infx.results.power.multinode import MAX_SAMPLE_GAP_SECONDS


ROOT = Path(__file__).resolve().parents[1]
LANE_DIR = ROOT / "benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/gb200-fp4/agentx"

# The four lanes this PR adds measured power to.
LANES = (
    "agg-dep16-vllm-simple-offload.yaml",
    "agg-dep16.yaml",
    "agg-tep16-balanced.yaml",
    "agg-tp16-latency.yaml",
)

# A single timed-out scrape must not come close to the validator's limit.
REQUIRED_HEADROOM_SECONDS = 1.0

# Mirrors srt-slurm's _validate_collector_budget so a recipe cannot become
# schema-invalid while satisfying the gap budget.
_COLLECT_CYCLE_GRACE_SECONDS = 1.0


def _telemetry(name):
    recipe = yaml.safe_load((LANE_DIR / name).read_text())
    telemetry = recipe.get("telemetry")
    assert telemetry is not None, f"{name} declares no telemetry block"
    return telemetry


def test_every_lane_recipe_is_present():
    """A renamed recipe would make the budget checks below vacuous."""
    missing = [name for name in LANES if not (LANE_DIR / name).exists()]
    assert not missing, f"recipes moved or renamed: {missing}"


@pytest.mark.parametrize("name", LANES)
def test_one_timed_out_scrape_stays_inside_the_sample_gap_budget(name):
    telemetry = _telemetry(name)
    interval_seconds = telemetry["collect_interval_ms"] / 1000
    worst_case_gap = telemetry["request_timeout_seconds"] + interval_seconds
    headroom = MAX_SAMPLE_GAP_SECONDS - worst_case_gap
    assert headroom >= REQUIRED_HEADROOM_SECONDS, (
        f"{name}: one timed-out scrape leaves a {worst_case_gap}s hole against the "
        f"validator's {MAX_SAMPLE_GAP_SECONDS}s limit ({headroom}s headroom); "
        f"lower request_timeout_seconds or collect_interval_ms"
    )


@pytest.mark.parametrize("name", LANES)
def test_join_timeout_still_covers_two_collector_cycles(name):
    """Lowering the scrape timeout must not invalidate srt-slurm's join budget."""
    telemetry = _telemetry(name)
    worst_case_join = 2 * (2 * telemetry["request_timeout_seconds"] + _COLLECT_CYCLE_GRACE_SECONDS)
    assert telemetry["collector_join_timeout_seconds"] > worst_case_join, (
        f"{name}: collector_join_timeout_seconds must exceed {worst_case_join}s"
    )
