"""Historical identity formats shared by sweep validation and eval deduplication."""

import pytest

from infx.results.artifacts import agentic_key, benchmark_key
from infx.results.eval_artifacts import eval_key


@pytest.mark.parametrize(
    ("kind", "topology", "expected", "expected_agentic"),
    [
        (
            "single",
            {
                "tp": "4",
                "pp": "2",
                "dcp_size": "3",
                "pcp_size": "5",
                "ep": "8",
                "dp_attention": "TRUE",
            },
            (4, 2, 3, 5, 8, True),
            (
                "single", "B200", "example", "vllm", "fp8",
                4, 2, 3, 5, 8, True,
                24,
                ("cpu", (("name", "cache"), ("version", "2"))),
            ),
        ),
        (
            "multi",
            {
                "is_multinode": "true",
                "prefill_tp": "4",
                "prefill_pp": "2",
                "prefill_dcp_size": "3",
                "prefill_pcp_size": "5",
                "prefill_ep": "8",
                "prefill_dp_attention": True,
                "prefill_num_workers": "6",
                "decode_tp": "16",
                "decode_pp": "7",
                "decode_dcp_size": "9",
                "decode_pcp_size": "11",
                "decode_ep": "32",
                "decode_dp_attention": "false",
                "decode_num_workers": "12",
            },
            (4, 2, 3, 5, 8, True, 6, 16, 7, 9, 11, 32, False, 12),
            (
                "multi", "B200", "example", "vllm", "fp8", "mtp", True,
                4, 2, 3, 5, 8, True, 6,
                16, 7, 9, 11, 32, False, 12,
                24,
                ("cpu", (("name", "cache"), ("version", "2"))),
            ),
        ),
    ],
)
def test_result_identities_preserve_topology_and_format_specific_fields(
    kind, topology, expected, expected_agentic
):
    row = {
        "hw": "B200",
        "infmax_model_prefix": "example",
        "framework": "vllm",
        "precision": "fp8",
        "spec_decoding": "mtp",
        "disagg": "true",
        "isl": "2048",
        "osl": "512",
        "conc": "24",
        "eval_suite": "tools",
        "kv_offloading": "cpu",
        "kv_offload_backend": {"version": "2", "name": "cache"},
        **topology,
    }
    assert benchmark_key(row) == (
        kind, "B200", "example", "vllm", "fp8", "mtp", True,
        2048, 512, *expected, 24,
    )
    assert eval_key(row) == (
        kind, "b200", "example", "vllm", "fp8", "tools", "mtp",
        2048, 512, *expected, 24,
    )
    assert agentic_key(row) == expected_agentic


@pytest.mark.parametrize(
    ("kind", "topology", "expected", "expected_agentic"),
    [
        (
            "single",
            {},
            (0, 1, 1, 1, 1, False),
            ("single", None, None, None, None, 0, 1, 1, 1, 1, False, 0, "none"),
        ),
        (
            "single",
            {
                "tp": "bad",
                "pp": None,
                "dcp_size": "bad",
                "pcp_size": "",
                "ep": None,
                "dp_attention": "yes",
            },
            (0, 1, 1, 1, 0, False),
            ("single", None, None, None, None, 0, 1, 1, 1, 0, False, 0, "none"),
        ),
        (
            "multi",
            {
                "is_multinode": True,
                "prefill_ep": None,
                "decode_tp": "bad",
                "decode_pp": "",
                "decode_num_workers": "bad",
            },
            (0, 1, 1, 1, 0, False, 0, 0, 1, 1, 1, 1, False, 0),
            (
                "multi", None, None, None, None, "none", False,
                0, 1, 1, 1, 0, False, 0,
                0, 1, 1, 1, 1, False, 0,
                0,
            ),
        ),
    ],
)
def test_legacy_identities_retain_missing_and_malformed_value_defaults(
    kind, topology, expected, expected_agentic
):
    assert benchmark_key(topology) == (
        kind, None, None, None, None, "none", False,
        0, 0, *expected, 0,
    )
    assert eval_key(topology) == (
        kind, "", None, None, None, "<legacy-eval-suite>", "none",
        8192, 1024, *expected, 0,
    )
    assert agentic_key(topology) == expected_agentic
