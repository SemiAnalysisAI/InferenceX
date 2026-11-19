import io
import json
import sys

import pytest
import yaml

from infx.config import repository_root
from infx.workflows import scheduler_facts
from infx.workflows.scheduler_facts import (
    SchedulingContext,
    annotate_jobs,
    merge_facts,
    queue_token,
    run_facts,
    supported_criteria,
)


@pytest.fixture
def root(tmp_path):
    """A repository whose master configs define a small, controlled criteria catalog."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/amd-master.yaml").write_text(yaml.safe_dump({
        "dsv4-fp4-mi355x-sglang": {
            "model-prefix": "dsv4", "precision": "fp4", "framework": "sglang",
            "scenarios": {"fixed-seq-len": [{"search-space": [{"spec-decoding": "mtp"}]}]},
        },
    }))
    (tmp_path / "configs/nvidia-master.yaml").write_text(yaml.safe_dump({
        "glm5.3-fp8-b200-vllm": {"model-prefix": "glm5.3", "precision": "fp8", "framework": "vllm"},
        "glm5.3flash-fp8-b200-sglang-disagg": {
            "model-prefix": "glm5.3flash", "precision": "fp8", "framework": "sglang-disagg",
        },
    }))
    return tmp_path


def facts_for(entry, criteria=None, event=""):
    """Return the facts recorded for a single job."""
    _, jobs = annotate_jobs(
        [{"runner": "b200", **entry}],
        SchedulingContext(event_name=event, criteria=criteria),
    )
    (facts,) = jobs.values()
    return facts


FULL_ENTRY = {
    "framework": "sglang-disagg",
    "model-prefix": "dsv4",
    "precision": "fp4",
    "spec-decoding": "mtp",
    "scenario-type": "agentic-coding",
    "eval-only": True,
    "prefill": {"num-worker": 1},
    "node-count": 3,
}
FULL_FACTS = {
    "framework": "sglang-disagg",
    "model-prefix": "dsv4",
    "precision": "fp4",
    "spec-decoding": "mtp",
    "agentic": True,
    "eval-only": True,
    "multi-node": True,
    "node-count": 3,
}
BLANK_FACTS = {
    "framework": "",
    "model-prefix": "",
    "precision": "",
    "spec-decoding": "",
    "agentic": False,
    "eval-only": False,
    "multi-node": False,
}


def test_queue_tokens_match_the_previous_implementation():
    """Tokens pinned from the scorer this module replaced, so runner labels keep their identity."""
    payload = {
        "single_node": {"1k1k": [{
            "runner": "b200", "framework": "sglang", "model-prefix": "qwen3.5",
            "precision": "fp4", "spec-decoding": "mtp",
        }]},
        "multi": [{
            "runner": "gb200", "framework": "dynamo-vllm",
            "prefill": {"num-worker": 1}, "node-count": 3,
        }],
    }

    annotated, jobs = annotate_jobs(payload, SchedulingContext(queue_namespace="12345:2"))

    assert annotated["single_node"]["1k1k"][0]["queue-token"] == "e9cfb12b5015b44816211f732e7f1148"
    assert annotated["multi"][0]["queue-token"] == "0ef47611d612d3e57b4dba04dff4ef4c"
    assert set(jobs) == {"e9cfb12b5015b44816211f732e7f1148", "0ef47611d612d3e57b4dba04dff4ef4c"}


def test_annotation_only_touches_runnable_matrix_entries():
    payload = {
        "single_node": {"1k1k": [{"runner": "b200", "framework": "sglang"}]},
        "changelog_metadata": {"runner": "not-a-job"},
    }

    annotated, jobs = annotate_jobs(payload)

    token = annotated["single_node"]["1k1k"][0]["queue-token"]
    assert list(jobs) == [token]
    assert "queue-token" not in annotated["changelog_metadata"]
    assert "queue-token" not in payload["single_node"]["1k1k"][0]
    assert "priority" not in annotated["single_node"]["1k1k"][0]


def test_unscoped_runs_report_every_fact_unprojected():
    assert facts_for(FULL_ENTRY) == FULL_FACTS
    assert facts_for({"framework": "trt"}) == {**BLANK_FACTS, "framework": "trt", "node-count": 1}


def test_criteria_credit_only_the_facts_they_name():
    criteria = frozenset({"multi-node", "agentic", "eval-only", "fp4", "mtp", "sglang", "dsv4"})

    assert facts_for(FULL_ENTRY, criteria) == FULL_FACTS
    assert facts_for(FULL_ENTRY, frozenset({"fp4"})) == {
        **BLANK_FACTS, "precision": "fp4", "node-count": 3,
    }
    assert facts_for(FULL_ENTRY, frozenset()) == {**BLANK_FACTS, "node-count": 3}


def test_criteria_do_not_credit_facts_the_job_lacks():
    criteria = frozenset({"multi-node", "agentic", "eval-only", "fp4", "mtp", "sglang", "dsv4"})
    entry = {"framework": "vllm", "model-prefix": "glm5.3", "precision": "fp8", "node-count": 2}

    assert facts_for(entry, criteria) == {**BLANK_FACTS, "node-count": 2}


@pytest.mark.parametrize(("framework", "credited"), [
    ("sglang", True), ("sglang-disagg", True), ("sglangx", False), ("vllm", False),
])
def test_framework_criteria_match_hyphenated_prefixes(framework, credited):
    facts = facts_for({"framework": framework}, frozenset({"sglang"}))

    assert facts["framework"] == (framework if credited else "")


@pytest.mark.parametrize(("model_prefix", "credited"), [
    ("glm5.3", True), ("glm5.3-mini", True), ("glm5.3flash", False),
])
def test_model_criteria_match_hyphenated_prefixes(model_prefix, credited):
    facts = facts_for({"framework": "vllm", "model-prefix": model_prefix}, frozenset({"glm5.3"}))

    assert facts["model-prefix"] == (model_prefix if credited else "")


@pytest.mark.parametrize("field", ["precision", "spec-decoding"])
def test_precision_and_spec_decoding_criteria_match_exactly(field):
    assert facts_for({"framework": "vllm", field: "fp4-mx"}, frozenset({"fp4"}))[field] == ""
    assert facts_for({"framework": "vllm", field: "fp4"}, frozenset({"fp4"}))[field] == "fp4"


def test_flags_are_not_candidate_values():
    facts = facts_for(
        {"framework": "patchwork", "model-prefix": "agentic-x", "precision": "eval-only"},
        frozenset({"patchwork", "agentic", "eval-only"}),
    )

    assert facts == {**BLANK_FACTS, "node-count": 1}


@pytest.mark.parametrize("criteria", [{"unknown"}, {"patchwork", "unknown"}, {"dsv41", "fp4"}])
def test_unknown_criteria_credit_nothing_without_raising(criteria):
    entry = {"framework": "vllm", "model-prefix": "dsv4", "precision": "fp8", "spec-decoding": "mtp"}

    assert facts_for(entry, frozenset(criteria)) == {**BLANK_FACTS, "node-count": 1}


@pytest.mark.parametrize("node_count", [0, -1, 1.5, True, None, "2"])
def test_node_count_must_be_a_positive_integer(node_count):
    with pytest.raises(ValueError, match="positive integer"):
        facts_for({"framework": "trt", "node-count": node_count})


@pytest.mark.parametrize(("criteria", "patchwork", "checklist"), [
    (None, False, False),
    (frozenset(), False, False),
    (frozenset({"patchwork"}), True, False),
    (frozenset({"checklist-complete", "sglang"}), False, True),
    (frozenset({"patchwork", "checklist-complete"}), True, True),
])
def test_run_facts_report_patchwork_and_checklist(criteria, patchwork, checklist):
    facts = run_facts(SchedulingContext(event_name="push", criteria=criteria), {})

    assert facts == {
        "version": 1, "event": "push", "patchwork": patchwork,
        "checklist-complete": checklist, "jobs": {},
    }


def test_catalog_covers_the_master_configs():
    criteria = supported_criteria(repository_root())

    assert {"multi-node", "agentic", "eval-only", "checklist-complete", "patchwork"} <= criteria
    assert {"dsr1", "mtp", "fp4", "sglang", "dynamo-vllm"} <= criteria


def test_catalog_reads_values_nested_anywhere(root):
    assert supported_criteria(root) == {
        "multi-node", "agentic", "eval-only", "checklist-complete", "patchwork",
        "dsv4", "glm5.3", "glm5.3flash", "fp4", "fp8", "mtp", "sglang", "sglang-disagg", "vllm",
    }


def run_cli(monkeypatch, capsys, payload, *args):
    monkeypatch.setattr(sys, "argv", ["scheduler_facts", *args])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    assert scheduler_facts.main() == 0
    return json.loads(capsys.readouterr().out)


def test_invocations_merge_into_one_facts_file(monkeypatch, capsys, tmp_path):
    facts_file = tmp_path / "scheduling-facts.json"
    common = ["--event-name", "pull_request", "--criteria-json", '["patchwork", "fp4"]',
              "--facts-file", str(facts_file)]
    single = run_cli(monkeypatch, capsys, [{"runner": "b200", "framework": "vllm", "precision": "fp4"}],
                     "--queue-namespace", "1:1:single", *common)
    multi = run_cli(monkeypatch, capsys, [{"runner": "gb200", "framework": "sglang", "prefill": {},
                                           "node-count": 2}],
                    "--queue-namespace", "1:1:multi", *common)

    assert json.loads(facts_file.read_text()) == {
        "version": 1,
        "event": "pull_request",
        "patchwork": True,
        "checklist-complete": False,
        "jobs": {
            single[0]["queue-token"]: {**BLANK_FACTS, "precision": "fp4", "node-count": 1},
            multi[0]["queue-token"]: {**BLANK_FACTS, "node-count": 2},
        },
    }


def test_empty_matrices_still_write_the_facts_file(monkeypatch, capsys, tmp_path):
    facts_file = tmp_path / "scheduling-facts.json"

    assert run_cli(monkeypatch, capsys, [], "--facts-file", str(facts_file)) == []
    assert json.loads(facts_file.read_text())["jobs"] == {}


@pytest.mark.parametrize("change", [
    {"event": "push"}, {"patchwork": True}, {"checklist-complete": True}, {"version": 2},
])
def test_merging_rejects_mismatched_run_facts(tmp_path, change):
    path = tmp_path / "scheduling-facts.json"
    first = run_facts(SchedulingContext(event_name="pull_request"), {"a" * 32: FULL_FACTS})
    merge_facts(path, first)

    with pytest.raises(ValueError, match=f"different run facts: {next(iter(change))}$"):
        merge_facts(path, {**first, **change, "jobs": {}})
    assert json.loads(path.read_text()) == first


def test_merging_rejects_conflicting_facts_for_one_token(tmp_path):
    path = tmp_path / "scheduling-facts.json"
    first = run_facts(SchedulingContext(), {"a" * 32: FULL_FACTS})
    merge_facts(path, first)
    merge_facts(path, first)

    with pytest.raises(ValueError, match="different facts for queue tokens"):
        merge_facts(path, {**first, "jobs": {"a" * 32: {**FULL_FACTS, "node-count": 4}}})


def test_list_criteria_reads_the_given_root(monkeypatch, capsys, root):
    assert run_cli(monkeypatch, capsys, None, "--list-criteria", "--root", str(root)) == sorted(
        supported_criteria(root)
    )


def test_queue_tokens_change_between_run_attempts():
    entry = {"runner": "b200", "framework": "sglang"}

    assert queue_token(entry, "123:1", ("0",)) != queue_token(entry, "123:2", ("0",))


def test_queue_tokens_are_stable_for_reordered_keys_but_distinct_for_duplicate_jobs():
    entry = {"runner": "example", "framework": "vllm"}
    reordered = {"framework": "vllm", "runner": "example"}

    assert queue_token(entry, "run", ("0",)) == queue_token(reordered, "run", ("0",))
    assert queue_token(entry, "run", ("0",)) != queue_token(entry, "run", ("1",))
