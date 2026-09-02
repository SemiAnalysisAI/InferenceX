import json
import shlex
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from ci_priority import (
    PriorityContext,
    annotate_jobs,
    calculate_priority,
    load_policy,
    queue_token,
    supported_criteria,
)


@pytest.fixture
def policy():
    """Controlled weights test scoring independently of the live scheduling policy."""
    return {
        "version": 1,
        "base-score": 10,
        "adjustments": {
            "event": {"push": 7},
            "additional-node": -0.125,
            "multi-node": 1,
            "agentic": 2,
            "eval-only": 0.375,
            "precision": {"fp4": 3},
            "spec-decoding": {"mtp": 4},
            "framework-prefix": {"vllm": 5, "sglang": 6},
            "model-prefix": {"dsv4": 8, "dsr1": 9, "qwen3.5": 11},
        },
        "labels": {
            "patchwork": {
                "names": ["ci-patchwork"],
                "waived-by": ["ci-patchwork-waived"],
                "score": -5,
            },
            "checklist-complete": {
                "names": ["ci-checklist-complete"],
                "adjustment": 1.5,
            },
            "skip-queue": {"name": "skip_queue"},
        },
    }


def test_combined_signals_add_their_configured_weights(policy):
    baseline = {
        "runner": "h100",
        "framework": "trt",
        "model-prefix": "other",
        "precision": "fp8",
        "spec-decoding": "none",
    }
    high_value = {
        **baseline,
        "runner": "cluster:b200-nscale",
        "framework": "sglang",
        "model-prefix": "dsv4",
        "precision": "fp4",
        "spec-decoding": "mtp",
        "scenario-type": "agentic-coding",
        "prefill": {"hardware": "b200"},
        "decode": {"hardware": "b200"},
    }

    assert calculate_priority(high_value, policy) == Decimal("34.000")
    assert calculate_priority(baseline, policy) == Decimal("10.000")


def test_event_adjustment_is_applied(policy):
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(event_name="push"),
    ) == Decimal("17.000")


def test_node_adjustment_applies_only_to_additional_nodes(policy):
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority({**entry, "node-count": 1}, policy) == Decimal("10.000")
    assert calculate_priority({**entry, "node-count": 2}, policy) == Decimal("9.875")
    assert calculate_priority({**entry, "node-count": 3}, policy) == Decimal("9.750")


def test_node_count_tiebreaker_survives_classifier_projection(policy):
    entry = {"runner": "h100", "framework": "trt", "node-count": 3}

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(criteria=frozenset()),
    ) == Decimal("9.750")


@pytest.mark.parametrize("node_count", [0, -1, 1.5, True, None, "2"])
def test_node_count_must_be_a_positive_integer(node_count, policy):
    with pytest.raises(ValueError, match="positive integer"):
        calculate_priority(
            {"runner": "h100", "framework": "trt", "node-count": node_count},
            policy,
        )


def test_patchwork_score_uses_half_up_rounding(policy):
    policy["labels"]["patchwork"]["score"] = 0.7225
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(labels=frozenset({"ci-patchwork"})),
    ) == Decimal("0.723")


def test_skip_queue_request_keeps_numeric_priority(policy):
    entry = {"runner": "h100", "framework": "sglang", "precision": "fp4"}

    annotated = annotate_jobs(
        [entry],
        policy,
        PriorityContext(
            labels=frozenset({"skip_queue"}),
            pr_number=2124,
        ),
    )

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(labels=frozenset({"skip_queue"})),
    ) == Decimal("19.000")
    assert annotated[0]["priority"] == "19.000"
    assert annotated[0]["skip-queue-pr"] == 2124


def test_patchwork_override_precedes_other_adjustments_unless_waived(policy):
    entry = {"runner": "b200", "framework": "sglang", "precision": "fp4"}

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(labels=frozenset({"ci-patchwork"})),
    ) == Decimal("-5.000")
    assert calculate_priority(
        entry,
        policy,
        PriorityContext(labels=frozenset({"ci-patchwork", "ci-patchwork-waived"})),
    ) == Decimal("19.000")
    assert calculate_priority(
        entry,
        policy,
        PriorityContext(criteria=frozenset({"patchwork"})),
    ) == Decimal("-5.000")
    assert calculate_priority(
        entry,
        policy,
        PriorityContext(
            labels=frozenset({"ci-patchwork-waived"}),
            criteria=frozenset({"patchwork"}),
        ),
    ) == Decimal("10.000")


def test_priority_criteria_require_matching_job_fields(policy):
    criteria = frozenset({"multi-node", "agentic", "fp4", "mtp", "vllm", "dsr1"})
    equivalent_entry = {
        "prefill": {},
        "scenario-type": "agentic-coding",
        "precision": "fp4",
        "spec-decoding": "mtp",
        "framework": "vllm",
        "model-prefix": "dsr1",
    }
    entry = dict(equivalent_entry)

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(criteria=criteria),
    ) == Decimal("34.000")
    unrelated_entry = {"runner": "h100", "framework": "trt"}
    assert calculate_priority(
        unrelated_entry,
        policy,
        PriorityContext(criteria=criteria),
    ) == Decimal("10.000")


def test_checklist_label_applies_alongside_classifier_criteria(policy):
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(
            labels=frozenset({"ci-checklist-complete"}),
            criteria=frozenset(),
        ),
    ) == Decimal("11.500")


def test_priority_criteria_reject_unknown_values_and_allow_mixed_jobs(policy):
    entry = {"runner": "h100", "framework": "vllm"}

    with pytest.raises(ValueError, match="Unknown CI priority criteria"):
        calculate_priority(
            entry,
            policy,
            PriorityContext(criteria=frozenset({"unknown"})),
        )
    assert calculate_priority(
        entry,
        policy,
        PriorityContext(criteria=frozenset({"vllm", "sglang"})),
    ) == Decimal("15.000")


def test_priority_labels_do_not_override_automatic_score(policy):
    entry = {"runner": "h100", "framework": "trt"}
    labels = frozenset(
        {"ci-priority:p0", "ci-priority:p4.5", "ci-priority:p1000000"}
    )

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(labels=labels),
    ) == Decimal("10.000")


def test_annotation_only_touches_runnable_matrix_entries(policy):
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

    annotated = annotate_jobs(payload, policy)

    assert annotated["single_node"]["1k1k"][0]["priority"] == "34.000"
    assert annotated["single_node"]["1k1k"][0]["queue-token"]
    assert "priority" not in annotated["changelog_metadata"]
    assert "priority" not in payload["single_node"]["1k1k"][0]
    assert "queue-token" not in payload["single_node"]["1k1k"][0]


def test_classifier_schema_matches_the_policy_vocabulary():
    policy = load_policy(Path(__file__).parents[1] / "configs" / "ci-priority.yaml")
    workflow = yaml.safe_load(
        (
            Path(__file__).parents[1] / ".github" / "workflows" / "run-sweep.yml"
        ).read_text()
    )
    classifier = next(
        step
        for step in workflow["jobs"]["setup"]["steps"]
        if step.get("id") == "classify"
    )
    arguments = shlex.split(classifier["with"]["claude_args"])
    schema = json.loads(arguments[arguments.index("--json-schema") + 1])
    schema_criteria = schema["properties"]["criteria"]["items"]["enum"]

    assert set(schema_criteria) == set(supported_criteria(policy))


def test_queue_tokens_change_between_run_attempts():
    entry = {"runner": "b200", "framework": "sglang"}

    assert queue_token(entry, "123:1", ("0",)) != queue_token(
        entry,
        "123:2",
        ("0",),
    )


def test_queue_tokens_are_stable_for_reordered_keys_but_distinct_for_duplicate_jobs():
    entry = {"runner": "example", "framework": "vllm"}
    reordered = {"framework": "vllm", "runner": "example"}

    assert queue_token(entry, "run", ("0",)) == queue_token(reordered, "run", ("0",))
    assert queue_token(entry, "run", ("0",)) != queue_token(entry, "run", ("1",))


@pytest.mark.parametrize("framework,expected", [("vllm", "15"), ("vllm-disagg", "15"), ("vllmish", "10")])
def test_framework_prefix_matching_requires_a_separator(policy, framework, expected):
    assert calculate_priority({"framework": framework}, policy) == Decimal(expected)
