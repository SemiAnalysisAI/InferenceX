import pytest

from infx.bench.env import BenchError
from infx.bench.speedbench import measure_al, read_counters


def test_acceptance_uses_counter_deltas_across_workers():
    before = '''vllm:spec_decode_num_accepted_tokens_total{worker="0"} 100
vllm:spec_decode_num_accepted_tokens_total{worker="1"} 200
vllm:spec_decode_num_drafts_total{worker="0"} 40
vllm:spec_decode_num_drafts_total{worker="1"} 60
'''
    after = '''vllm:spec_decode_num_accepted_tokens_total{worker="0"} 130
vllm:spec_decode_num_accepted_tokens_total{worker="1"} 245
vllm:spec_decode_num_drafts_total{worker="0"} 50
vllm:spec_decode_num_drafts_total{worker="1"} 75
'''
    assert measure_al(before, after, {"completed": 80, "failed": 0}, 5) == 4.0


@pytest.mark.parametrize("completed,failed", [(79, 0), (80, 1)])
def test_incomplete_benchmark_rejected(completed, failed):
    with pytest.raises(BenchError, match="80 coding prompts"):
        measure_al("", "", {"completed": completed, "failed": failed}, 5)


@pytest.mark.parametrize("sample", ["NaN", "Inf", "-1"])
def test_invalid_counter_rejected(sample):
    with pytest.raises(BenchError, match="Invalid acceptance counter"):
        read_counters(f"vllm:spec_decode_num_accepted_tokens_total {sample}\n")


def test_missing_counter_rejected():
    with pytest.raises(BenchError, match="Missing acceptance counter"):
        read_counters("# unrelated metric\n")


@pytest.mark.parametrize("accepted,drafts", [(9, 11), (10, 10), (16, 11), (11, 9)])
def test_invalid_deltas_rejected(accepted, drafts):
    before = 'vllm:spec_decode_num_accepted_tokens_total 10\nvllm:spec_decode_num_drafts_total 10\n'
    after = f'vllm:spec_decode_num_accepted_tokens_total {accepted}\nvllm:spec_decode_num_drafts_total {drafts}\n'
    with pytest.raises(BenchError, match="Invalid acceptance deltas"):
        measure_al(before, after, {"completed": 80}, 5)
