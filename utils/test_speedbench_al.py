import pytest

from speedbench_al import measure_al, read_counters


def metrics(accepted: float, drafts: float) -> str:
    return (f'vllm:spec_decode_num_accepted_tokens_total{{model_name="fixture"}} {accepted}\n'
            f'vllm:spec_decode_num_drafts_total{{model_name="fixture"}} {drafts}\n')


def test_al_uses_delta_and_includes_bonus_token() -> None:
    assert measure_al(metrics(10, 5), metrics(235, 105), {"completed": 80}, 5) == 3.25


def test_counter_reader_sums_engine_series_and_accepts_scientific_notation() -> None:
    text = metrics(10, 5).replace("10", "1e1") + metrics(20, 10).replace("fixture", "second")
    assert read_counters(text) == (30, 15)


@pytest.mark.parametrize("after", [metrics(10, 5), metrics(9, 10), metrics(36, 10), metrics(float('nan'), 10), ''])
def test_invalid_metrics_cannot_emit_golden_al(after: str) -> None:
    with pytest.raises(ValueError):
        measure_al(metrics(10, 5), after, {"completed": 80}, 5)


@pytest.mark.parametrize("result", [{"completed": 79}, {"completed": 80, "failed": 1}, {}])
def test_partial_benchmark_cannot_emit_golden_al(result: dict) -> None:
    with pytest.raises(ValueError):
        measure_al(metrics(10, 5), metrics(235, 105), result, 5)


def test_acceptance_bound_uses_each_cells_draft_length() -> None:
    before, after = metrics(0, 0), metrics(20, 10)
    assert measure_al(before, after, {"completed": 80}, 2) == 3.0
    with pytest.raises(ValueError):
        measure_al(before, after, {"completed": 80}, 1)


def test_emitter_keeps_all_draft_lengths_and_thinking_modes(tmp_path, monkeypatch) -> None:
    import json
    from speedbench_al import main

    for mode in ("on", "off"):
        for length, accepted in ((1, 5), (3, 20)):
            cell = f"{mode}_mtp{length}"
            (tmp_path / f"before_{cell}.prom").write_text(metrics(0, 0))
            (tmp_path / f"after_{cell}.prom").write_text(metrics(accepted, 10))
            (tmp_path / f"speedbench_{cell}.json").write_text(json.dumps({"completed": 80}))
    output = tmp_path / "curve.yaml"
    monkeypatch.setattr("sys.argv", ["speedbench_al", "--results-dir", str(tmp_path),
        "--output", str(output), "--modes", "on", "off", "--draft-lengths", "1", "3",
        "--thinking-kwargs", "{}", "--model", "fixture", "--image", "fixture",
        "--tp", "4", "--category", "coding", "--output-len", "4096"])
    main()
    assert output.read_text().endswith(
        "deepseek-v4.1-flash:\n  thinking_on:\n    1: 1.50\n    3: 3.00\n"
        "  thinking_off:\n    1: 1.50\n    3: 3.00\n")
