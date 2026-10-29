"""Launch-request validation that rejects a job before any work."""

import re

import pytest

from infx.launch.request import LaunchRequest, RequestError

MULTINODE_AGENTX = {"RUNNER_NAME": "r_0", "IS_MULTINODE": "true", "IS_AGENTIC": "1"}


@pytest.mark.parametrize(
    ("overrides", "concurrencies"),
    [
        ({"CONC": "4", "CONC_LIST": "4"}, [4]),
        ({"EVAL_ONLY": "true", "CONC": "4", "CONC_LIST": "4 8"}, [4, 8]),
        ({"IS_AGENTIC": "0", "CONC_LIST": "4 8"}, [4, 8]),
        ({"IS_MULTINODE": "false", "CONC": "4"}, []),
    ],
    ids=["one-concurrency", "eval-only-batches", "fixed-sequence-batches", "single-node"],
)
def test_one_agentx_deployment_or_batched_points_are_accepted(overrides, concurrencies):
    assert LaunchRequest.from_env({**MULTINODE_AGENTX, **overrides}).conc_list == concurrencies


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"CONC": "4", "CONC_LIST": "4 8"}, "exactly one positive concurrency"),
        ({"CONC": "0", "CONC_LIST": "0"}, "exactly one positive concurrency"),
        ({"CONC_LIST": "4"}, "not set for AgentX throughput: CONC$"),
    ],
    ids=["two-concurrencies", "zero", "no-conc"],
)
def test_multinode_agentx_throughput_needs_one_concurrency_per_deployment(overrides, error):
    with pytest.raises(RequestError, match=error):
        LaunchRequest.from_env({**MULTINODE_AGENTX, **overrides})


DATASET = "al_gate_dataset/speedbench_coding.jsonl"
PROMPTS = '{"question_id": "0", "prompt": "Reverse a singly linked list in place."}\n'


def gate_env(tmp_path, **overrides: str) -> dict[str, str]:
    """A multi-node AgentX MTP point's gate whose job downloaded the prompts into its workspace."""
    workspace = tmp_path / "ws"
    (workspace / DATASET).parent.mkdir(parents=True)
    (workspace / DATASET).write_text(PROMPTS)
    (tmp_path / "speedbench_coding.jsonl").write_text(PROMPTS)
    return {
        **MULTINODE_AGENTX, "GITHUB_WORKSPACE": str(workspace), "CONC": "4", "CONC_LIST": "4", "SPEC_DECODING": "mtp",
        "AL_GATE": "true", "AL_GATE_CURVE": "dsv4_mtp", "AL_GATE_DRAFT_TOKENS": "3", "AL_GATE_DATASET": DATASET,
        **overrides,
    }  # fmt: skip


def test_a_gate_reads_the_prompts_its_job_downloaded(tmp_path):
    request = LaunchRequest.from_env(gate_env(tmp_path))
    assert (request.al_gate_curve, request.al_gate_draft_tokens, request.al_gate_dataset) == ("dsv4_mtp", 3, DATASET)


@pytest.mark.parametrize(
    "overrides",
    [{"AL_GATE_CURVE": ""}, {"AL_GATE_DRAFT_TOKENS": "0"}, {"IS_AGENTIC": "0"}, {"SPEC_DECODING": "none"},
     {"RUN_EVAL": "true"}, {"BENCH_SCRIPT_OVERRIDE": "collect.sh"}, {"FRAMEWORK": "dynamo-trt"}],
    ids=["no-curve", "zero-drafts", "fixed-sequence", "no-drafting", "eval", "script", "exempt"],
)  # fmt: skip
def test_a_gate_measures_one_agentx_speculative_srt_point(tmp_path, overrides):
    with pytest.raises(RequestError, match="AL_GATE needs AL_GATE_CURVE"):
        LaunchRequest.from_env(gate_env(tmp_path, **overrides))


@pytest.mark.parametrize(
    "dataset", ["speedbench_coding.jsonl", "../speedbench_coding.jsonl", "{workspace}/" + DATASET],
    ids=["not-downloaded", "outside-the-workspace", "absolute"],
)  # fmt: skip
def test_a_gate_needs_its_prompts_at_a_relative_path_in_the_workspace(tmp_path, dataset):
    # The container resolves AL_GATE_DATASET in the workspace it mounts at /infmax-workspace.
    workspace = (tmp_path / "ws").resolve()
    dataset = dataset.format(workspace=workspace)
    with pytest.raises(RequestError, match=re.escape(f"AL_GATE_DATASET {dataset} is not a file in {workspace}")):
        LaunchRequest.from_env(gate_env(tmp_path, AL_GATE_DATASET=dataset))
