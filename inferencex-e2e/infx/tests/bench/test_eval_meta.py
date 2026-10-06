"""meta_env.json: the workflow's topology names mapped onto the fields consumers read."""

import pytest

from infx.bench.eval import meta

TOPOLOGY = (
    "tp", "ep", "dp_attention",
    "prefill_tp", "prefill_ep", "prefill_dp_attention", "prefill_num_workers",
    "decode_tp", "decode_ep", "decode_dp_attention", "decode_num_workers",
)  # fmt: skip


@pytest.mark.parametrize(("inputs", "expected"), [
    # srt-slurm names: explicit per-phase DP attention wins over a stale DP_ATTENTION.
    (
        {"PREFILL_TP": "4", "PREFILL_EP": "4", "PREFILL_DP_ATTN": "true",
         "DECODE_TP": "8", "DECODE_EP": "8", "DECODE_DP_ATTN": "false", "DP_ATTENTION": "false"},
        (4, 4, True, 4, 4, True, 1, 8, 8, False, 1),
    ),
    # Decode inherits prefill's TP and EP; worker counts default to one each.
    (
        {"PREFILL_TP": "8", "PREFILL_EP": "2", "DECODE_DP_ATTN": "true"},
        (8, 2, False, 8, 2, False, 1, 8, 2, True, 1),
    ),
])  # fmt: skip
def test_disaggregated_topology_maps_onto_metadata(inputs, expected):
    document = meta.build({"IS_MULTINODE": "true", **inputs}, conc=4, suite="gsm8k")

    assert tuple(document[key] for key in TOPOLOGY) == expected


@pytest.mark.parametrize(("inputs", "expected"), [
    ({}, ("sglang", "fp8")),
    ({"FRAMEWORK": "dynamo-sglang"}, ("dynamo-sglang", "fp8")),
])  # fmt: skip
def test_framework_and_precision_fall_back_to_the_result_filename(inputs, expected):
    environ = {"IS_MULTINODE": "false", "RESULT_FILENAME": "dsr1_1k1k_fp8_sglang_tp8-ep1_conc4"}
    document = meta.build({**environ, **inputs}, conc=4, suite="gsm8k")

    assert (document["framework"], document["precision"]) == expected
