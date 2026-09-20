import pytest

from infx.bench_serving.speedbench_acceptance import acceptance, collect_cell, read_counters


def counters(drafts, proposed, accepted):
    return (
        f'vllm:spec_decode_num_drafts_total{{engine="0"}} {drafts}\n'
        f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {proposed}\n'
        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {accepted}\n'
    )


def test_counter_deltas_use_actual_proposals_and_include_bonus():
    result = collect_cell(counters(10, 30, 20), counters(14, 40, 26), {"completed": 2}, 2, 3)
    assert result == {
        "num_drafts": 4, "num_draft_tokens": 10, "num_accepted_tokens": 6,
        "al": 2.5, "ar": 0.6,
    }


def test_sum_engines_and_parse_scientific_notation():
    text = counters("1e2", "3e2", "2e2") + counters(10, 30, 20).replace('"0"', '"1"')
    assert read_counters(text) == {
        "num_drafts": 110, "num_draft_tokens": 330, "num_accepted_tokens": 220,
    }


@pytest.mark.parametrize("text", ["", counters(1, 3, "NaN"), counters(1, 3, -1)])
def test_invalid_counters_fail(text):
    with pytest.raises(ValueError, match="Missing or invalid counter"):
        read_counters(text)


@pytest.mark.parametrize("after", [counters(0, 0, 0), counters(1, 4, 2), counters(1, 3, 4)])
def test_invalid_deltas_fail(after):
    with pytest.raises(ValueError, match="Invalid speculative counter deltas"):
        acceptance(counters(0, 0, 0), after, 3)


@pytest.mark.parametrize("result", [{"completed": 1}, {"completed": 2, "errors": ["HTTP 500"]}])
def test_incomplete_or_failed_benchmarks_fail(result):
    with pytest.raises(ValueError):
        collect_cell(counters(0, 0, 0), counters(4, 10, 6), result, 2, 3)
