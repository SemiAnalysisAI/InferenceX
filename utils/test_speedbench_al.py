import pytest

from speedbench_al import measure_al, read_counters


def metrics(accepted: float, drafts: float) -> str:
    return (f'vllm:spec_decode_num_accepted_tokens_total{{model_name="fixture"}} {accepted}\n'
            f'vllm:spec_decode_num_drafts_total{{model_name="fixture"}} {drafts}\n')


def test_al_uses_delta_and_includes_bonus_token() -> None:
    assert measure_al(metrics(10, 5), metrics(235, 105), {"completed": 80}) == 3.25


def test_counter_reader_sums_engine_series_and_accepts_scientific_notation() -> None:
    text = metrics(10, 5).replace("10", "1e1") + metrics(20, 10).replace("fixture", "second")
    assert read_counters(text) == (30, 15)


@pytest.mark.parametrize("after", [metrics(10, 5), metrics(9, 10), metrics(36, 10), metrics(float('nan'), 10), ''])
def test_invalid_metrics_cannot_emit_golden_al(after: str) -> None:
    with pytest.raises(ValueError):
        measure_al(metrics(10, 5), after, {"completed": 80})


@pytest.mark.parametrize("result", [{"completed": 79}, {"completed": 80, "failed": 1}, {}])
def test_partial_benchmark_cannot_emit_golden_al(result: dict) -> None:
    with pytest.raises(ValueError):
        measure_al(metrics(10, 5), metrics(235, 105), result)
