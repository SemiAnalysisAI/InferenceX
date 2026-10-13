"""DWDP topology must survive the producer boundary without becoming TP."""

import pytest
from pydantic import ValidationError

from infx.bench.eval.meta import build
from infx.launch.drivers.srt.submit import eval_args
from infx.matrix.generate import worker_gpus_per_node, worker_node_count
from infx.matrix.validation import WorkerConfig
from infx.results.agentic import _gpu_shape
from infx.results.artifacts import agentic_key, benchmark_key
from infx.results.eval_artifacts import eval_key
from infx.results.topology import Parallelism, validate_parallelism


def worker(**overrides):
    return {"num-worker": 2, "tp": 1, "ep": 1, "dp-attn": False, "dwdp-size": 8, **overrides}


def test_group_allocation():
    config = WorkerConfig.model_validate(worker())
    assert config.dwdp_size == 8
    assert (
        worker_node_count(
            worker(), "prefill", "cluster:test", {"clusters": {"test": {"gpus-per-node": 8}}}
        )
        == 2
    )
    assert worker_gpus_per_node(worker(), 8) == 8
    layout = Parallelism(tp=1, ep=1, dwdp_size=8)
    assert layout.gpus_per_worker == 8
    assert layout.fields("prefill_")["prefill_dwdp_size"] == 8
    assert "prefill_dwdp_size" not in Parallelism(tp=1, ep=1).fields("prefill_")


@pytest.mark.parametrize(
    "override",
    [
        {"dwdp-size": 0},
        {"dwdp-size": -1},
        {"dwdp-size": True},
        {"dwdp-size": "8"},
        {"dwdp-size": 1.5},
        {"tp": 2},
        {"ep": 2},
        {"pp": 2},
        {"pcp-size": 2},
        {"dcp-size": 2},
        {"dp-attn": True},
        {"num-worker": 0},
    ],
)
def test_rejects_incompatible_worker_topology(override):
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate(worker(**override))


def test_result_layout_rejects_tp_combination():
    with pytest.raises(ValueError, match="DWDP"):
        validate_parallelism(Parallelism(tp=2, ep=1, dwdp_size=8))


def test_agentic_metadata_counts_groups():
    fields, total, _, _, _ = _gpu_shape(
        {
            "IS_MULTINODE": "true",
            "DISAGG": "true",
            "PREFILL_TP": "1",
            "PREFILL_EP": "1",
            "PREFILL_DWDP_SIZE": "8",
            "PREFILL_NUM_WORKERS": "2",
            "DECODE_TP": "8",
            "DECODE_EP": "8",
            "DECODE_NUM_WORKERS": "1",
        }
    )
    assert (fields["num_prefill_gpu"], fields["num_decode_gpu"], total) == (16, 8, 24)
    assert fields["prefill_tp"] == 1
    assert fields["prefill_dwdp_size"] == 8


def test_eval_metadata_keeps_dwdp():
    env = {"IS_MULTINODE": "true"}
    assert build({**env, "PREFILL_DWDP_SIZE": "8"}, conc=1, suite="test")["prefill_dwdp_size"] == 8
    assert "prefill_dwdp_size" not in build(env, conc=1, suite="test")


def test_eval_passthrough_keeps_dwdp():
    args = eval_args({"PREFILL_DWDP_SIZE": "8"}, "run-eval")
    assert 'post_eval.passthrough_env=["PREFILL_DWDP_SIZE"]' in args


@pytest.mark.parametrize("key", [benchmark_key, agentic_key, eval_key])
def test_artifact_identity_separates_dwdp(key):
    base = {"is_multinode": True, "prefill_tp": 1, "prefill_ep": 1}
    assert key(base) == key({**base, "prefill_dwdp_size": 1})
    assert key(base) != key({**base, "prefill_dwdp_size": 8})
    assert key({**base, "prefill_dwdp_size": 4}) != key({**base, "prefill_dwdp_size": 8})
