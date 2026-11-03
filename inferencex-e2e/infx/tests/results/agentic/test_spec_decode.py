"""Realized speculative acceptance from two Prometheus scrapes of the engines' counters."""

from __future__ import annotations

import pytest

from infx.results.agentic.spec_decode import READERS, increase, parse

WORKER = "http://10.0.0.1:8000/metrics"
OTHER = "http://10.0.0.2:8000/metrics"


def test_counter_samples_are_read_under_their_family_names() -> None:
    # prometheus_client's text format: samples carry _total, and _created timestamps come along.
    # Label values stay escaped; their order does not matter.
    text = r"""
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 1100.0
vllm:spec_decode_num_drafts_created{engine="0",model_name="m"} 1.7598e+09
trtllm_spec_decode_drafted_tokens_total{token_position="0", path="a\"b,c}\n"} 1e3 1712345678000
"""
    assert parse(text) == {
        "vllm:spec_decode_num_drafts": {frozenset({("engine", "0"), ("model_name", "m")}): 1100},
        "trtllm_spec_decode_drafted_tokens": {
            frozenset({("token_position", "0"), ("path", r"a\"b,c}\n")}): 1000
        },
    }


def test_text_that_is_not_prometheus_exposition_is_rejected() -> None:
    # trtllm-serve's /metrics is JSON iteration stats.
    with pytest.raises(ValueError, match="line 2 is not Prometheus text exposition"):
        parse('# comment\n[{"iterationCounter": 3}]')


def test_each_series_increase_survives_resets_new_series_and_shared_labels() -> None:
    before = {
        WORKER: 'x_total{a="1"} 900\nx_total{a="2"} 500\n',
        OTHER: 'x_total{a="1"} 10\n',
    }
    after = {
        WORKER: 'x_total{a="1"} 1000\nx_total{a="2"} 40\nx_total{a="3"} 700\n',
        OTHER: 'x_total{a="1"} 15\n',
    }

    # a=2 went backwards, so its engine restarted: Prometheus counts the new value. a=3 is new.
    assert increase(before, after) == {
        "x": [({"a": "1"}, 100), ({"a": "2"}, 40), ({"a": "3"}, 700), ({"a": "1"}, 5)]
    }


@pytest.mark.parametrize(
    ("engine", "after", "expected"),
    [
        (
            "vllm",
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 2840\n'
            'vllm:spec_decode_num_drafts_total{engine="0"} 1000\n',
            (2840, 1000),
        ),
        # Decode emitted 4100 tokens for 100 requests in 1000 verifications: 4100 - 100 - 1000
        # accepted drafts. Prefill workers' series never verify.
        (
            "sglang",
            'sglang:spec_verify_calls_total{engine_type="decode"} 1000\n'
            'sglang:spec_verify_calls_total{engine_type="prefill"} 0\n'
            'sglang:generation_tokens_total{engine_type="decode"} 4100\n'
            'sglang:generation_tokens_total{engine_type="prefill"} 500\n'
            'sglang:num_requests_total{engine_type="decode"} 100\n'
            'sglang:num_requests_total{engine_type="prefill"} 100\n',
            (3000, 1000),
        ),
        # Each verification drafts position 0; accepted tokens count at every position.
        (
            "trtllm",
            'trtllm_spec_decode_drafted_tokens_total{token_position="0"} 1000\n'
            'trtllm_spec_decode_drafted_tokens_total{token_position="1"} 800\n'
            'trtllm_spec_decode_accepted_tokens_total{token_position="0"} 900\n'
            'trtllm_spec_decode_accepted_tokens_total{token_position="1"} 700\n',
            (1600, 1000),
        ),
        (
            "atom",
            'atom:mtp_decode_steps_total{accepted_tokens="0"} 100\n'
            'atom:mtp_decode_steps_total{accepted_tokens="1"} 300\n'
            "atom:mtp_accepted_tokens_total 500\n",
            (500, 400),
        ),
        ("atom", "atom_mtp_decode_steps_total 40\natom_mtp_accepted_tokens_total 70\n", (70, 40)),
    ],
    ids=["vllm", "sglang-disagg", "trtllm", "atom", "atom-underscore"],
)
def test_each_engine_reads_its_accepted_drafts_and_verifications(engine, after, expected) -> None:
    assert READERS[engine](increase({WORKER: ""}, {WORKER: after})) == expected


@pytest.mark.parametrize("engine", ["vllm", "sglang", "trtllm", "atom"])
def test_an_engine_without_its_counters_reads_nothing(engine) -> None:
    # Another engine's spec-decode families never give this one's reading.
    other = (
        "sglang:spec_verify_calls_total 5\n"
        if engine == "vllm"
        else "vllm:spec_decode_num_drafts_total 5\n"
    )

    assert READERS[engine](increase({WORKER: ""}, {WORKER: other})) is None


def test_sglang_without_decode_workers_tokens_reads_nothing() -> None:
    # Deriving it anyway would read 1 + (0 - 100 - 1000) / 1000 = -0.1.
    after = (
        'sglang:spec_verify_calls_total{engine_type="decode"} 1000\n'
        'sglang:generation_tokens_total{engine_type="prefill"} 100\n'
        'sglang:num_requests_total{engine_type="decode"} 100\n'
    )

    assert READERS["sglang"](increase({WORKER: ""}, {WORKER: after})) is None
