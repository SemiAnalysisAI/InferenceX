import json
import shlex
from copy import deepcopy
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

    assert calculate_priority(high_value, POLICY) == Decimal("6.000")
    assert calculate_priority(baseline, POLICY) == Decimal("1.000")


def test_main_branch_jobs_receive_an_automatic_boost():
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(event_name="push"),
    ) == Decimal("3.000")


def test_patchwork_score_uses_half_up_rounding():
    policy = deepcopy(POLICY)
    policy["labels"]["patchwork"]["score"] = 0.7225
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority(
        entry,
        policy,
        PriorityContext(labels=frozenset({"ci-patchwork"})),
    ) == Decimal("0.723")


def test_skip_queue_request_keeps_numeric_priority():
    entry = {"runner": "h100", "framework": "sglang", "precision": "fp4"}

    annotated = annotate_jobs(
        [entry],
        POLICY,
        PriorityContext(
            labels=frozenset({"skip_queue"}),
            pr_number=2124,
        ),
    )

    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(labels=frozenset({"skip_queue"})),
    ) == Decimal("2.250")
    assert annotated[0]["priority"] == "2.250"
    assert annotated[0]["skip-queue-pr"] == 2124


def test_patchwork_label_forces_bottom_priority_without_waiver():
    entry = {"runner": "b200", "framework": "sglang", "precision": "fp4"}

    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(labels=frozenset({"ci-patchwork"})),
    ) == Decimal("0.000")
    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(labels=frozenset({"ci-patchwork", "ci-patchwork-waived"})),
    ) > Decimal("0.000")
    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(criteria=frozenset({"patchwork"})),
    ) == Decimal("0.000")
    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(
            labels=frozenset({"ci-patchwork-waived"}),
            criteria=frozenset({"patchwork"}),
        ),
    ) > Decimal("0.000")


def test_fable_criteria_drive_all_configured_adjustments():
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
        POLICY,
        PriorityContext(criteria=criteria),
    ) == calculate_priority(equivalent_entry, POLICY)
    unrelated_entry = {"runner": "h100", "framework": "trt"}
    assert calculate_priority(
        unrelated_entry,
        POLICY,
        PriorityContext(criteria=criteria),
    ) == Decimal("1.000")


def test_checklist_label_applies_alongside_classifier_criteria():
    entry = {"runner": "h100", "framework": "trt"}

    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(
            labels=frozenset({"ci-checklist-complete"}),
            criteria=frozenset(),
        ),
    ) == Decimal("1.250")


def test_fable_criteria_reject_unknown_values_and_allow_mixed_jobs():
    entry = {"runner": "h100", "framework": "vllm"}

    with pytest.raises(ValueError, match="Unknown CI priority criteria"):
        calculate_priority(
            entry,
            POLICY,
            PriorityContext(criteria=frozenset({"unknown"})),
        )
    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(criteria=frozenset({"vllm", "sglang"})),
    ) == Decimal("1.500")


def test_priority_labels_do_not_override_automatic_score():
    entry = {"runner": "h100", "framework": "trt"}
    labels = frozenset(
        {"ci-priority:p0", "ci-priority:p4.5", "ci-priority:p1000000"}
    )

    assert calculate_priority(
        entry,
        POLICY,
        PriorityContext(labels=labels),
    ) == Decimal("1.000")


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

    assert annotated["single_node"]["1k1k"][0]["priority"] == "3.750"
    assert len(annotated["single_node"]["1k1k"][0]["queue-token"]) == 32
    assert "priority" not in annotated["changelog_metadata"]
    assert "priority" not in payload["single_node"]["1k1k"][0]
    assert "queue-token" not in payload["single_node"]["1k1k"][0]


def test_classifier_schema_matches_the_policy_vocabulary():
    workflow = yaml.safe_load(
        (
            Path(__file__).parents[1] / ".github" / "workflows" / "run-sweep.yml"
        ).read_text()
    )
    classifier = next(
        step
        for step in workflow["jobs"]["setup"]["steps"]
        if step.get("id") == "fable"
    )
    arguments = shlex.split(classifier["with"]["claude_args"])
    schema = json.loads(arguments[arguments.index("--json-schema") + 1])
    schema_criteria = schema["properties"]["criteria"]["items"]["enum"]

    assert set(schema_criteria) == set(supported_criteria(POLICY))


def test_queue_tokens_change_between_run_attempts():
    entry = {"runner": "b200", "framework": "sglang"}

    assert queue_token(entry, "123:1", ("0",)) != queue_token(
        entry,
        "123:2",
        ("0",),
    )
