"""Launch-request validation that rejects a job before any work."""

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
