from decimal import Decimal
from pathlib import Path

import pytest

from ci_priority import PriorityContext, annotate_jobs, calculate_priority, load_policy


POLICY = load_policy(Path(__file__).parents[1] / "configs" / "ci-priority.yaml")


def test_combined_high_value_signals_outrank_baseline_job():
    baseline = {
        "runner": "h100",
        "framework": "trt",
        "model-prefix": "other",
        "precision": "fp8",
        "spec-decoding": "none",
    }
    high_value = {
        **baseline,
        "runner": "b200-multinode",
        "framework": "sglang",
        "model-prefix": "dsv4",
        "precision": "fp4",
        "spec-decoding": "mtp",
        "scenario-type": "agentic-coding",
        "prefill": {"hardware": "b200"},
        "decode": {"hardware": "b200"},
    }

    assert calculate_priority(high_value, POLICY) == Decimal("0.000")
    assert calculate_priority(baseline, POLICY) == Decimal("5.000")


def test_main_branch_jobs_receive_an_automatic_boost():
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(event_name="push"),
    ) == Decimal("3.000")


def test_patchwork_label_forces_bottom_priority_without_waiver():
    entry = {"runner": "b200", "framework": "sglang", "precision": "fp4"}

    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(labels=frozenset({"ci-patchwork"})),
    ) == Decimal("999.000")
    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(labels=frozenset({"ci-patchwork", "ci-patchwork-waived"})),
    ) < Decimal("999.000")


def test_maintainer_override_wins_and_conflicts_are_rejected():
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(labels=frozenset({"ci-priority:p0.7"})),
    ) == Decimal("0.700")
    with pytest.raises(ValueError, match="Multiple ci-priority override"):
        calculate_priority(
            entry,
            POLICY,
            PriorityContext(labels=frozenset({"ci-priority:p0.7", "ci-priority:p2.5"})),
        )


def test_annotation_only_touches_runnable_matrix_entries():
    payload = {
        "single_node": {
            "1k1k": [
                {
                    "runner": "b200",
                    "framework": "sglang",
                    "model-prefix": "qwen3.5",
                    "precision": "fp4",
                    "spec-decoding": "mtp",
                }
            ]
        },
        "changelog_metadata": {"runner": "not-a-job"},
    }

    annotated = annotate_jobs(payload, POLICY)

    assert annotated["single_node"]["1k1k"][0]["priority"] == "2.250"
    assert len(annotated["single_node"]["1k1k"][0]["queue-token"]) == 20
    assert "priority" not in annotated["changelog_metadata"]
    assert "priority" not in payload["single_node"]["1k1k"][0]
    assert "queue-token" not in payload["single_node"]["1k1k"][0]
