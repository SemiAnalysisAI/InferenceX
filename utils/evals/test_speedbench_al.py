import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_eval_results import build_row, extract_speedbench_al_metrics, score_cell
from speedbench_al import build_result, load_reference, lookup_reference
from validate_scores import validate_speedbench_al


def test_lookup_reference_uses_model_prefix_alias(tmp_path: Path) -> None:
    ref = tmp_path / "speedbench-reference-al.yaml"
    ref.write_text(
        """
deepseek-v4-pro:
  thinking_on:
    2: 2.75
  thinking_off:
    2: 2.40
"""
    )

    data = load_reference(ref)
    model_key, mode_key, value = lookup_reference(
        data,
        model="deepseek-ai/DeepSeek-V4-Pro",
        model_prefix="dsv4",
        thinking_mode="on",
        num_speculative_tokens=2,
    )

    assert model_key == "deepseek-v4-pro"
    assert mode_key == "thinking_on"
    assert value == 2.75


def test_build_result_records_threshold_pass(tmp_path: Path) -> None:
    ref = tmp_path / "speedbench-reference-al.yaml"
    ref.write_text(
        """
deepseek-v4-pro:
  thinking_on:
    2: 2.50
"""
    )
    args = argparse.Namespace(
        reference_yaml=str(ref),
        model="deepseek-ai/DeepSeek-V4-Pro",
        model_prefix="dsv4",
        thinking_mode="on",
        num_speculative_tokens=2,
        category="coding",
        output_len=4096,
        temperature=1.0,
        threshold_ratio=0.90,
        acceptance_length="2.30",
        accepted_tokens="13",
        draft_tokens="10",
        error=None,
    )

    result = build_result(args)

    assert result["reference_acceptance_length"] == 2.50
    assert result["min_acceptance_length"] == 2.25
    assert result["passed"] is True


def test_validate_speedbench_al_fails_below_minimum() -> None:
    ok, checked = validate_speedbench_al(
        {
            "speedbench_al_eval_version": 1,
            "task": "speedbench_al",
            "thinking_mode": "thinking_on",
            "num_speculative_tokens": 2,
            "acceptance_length": 2.0,
            "min_acceptance_length": 2.25,
            "passed": False,
        },
        "results_speedbench_al.json",
    )

    assert checked == 1
    assert ok is False


def test_collect_eval_results_formats_speedbench_row(tmp_path: Path) -> None:
    result_path = tmp_path / "results_speedbench_al_thinking_on_mtp2.json"
    result_path.write_text(
        json.dumps(
            {
                "speedbench_al_eval_version": 1,
                "task": "speedbench_al",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "thinking_mode": "thinking_on",
                "num_speculative_tokens": 2,
                "acceptance_length": 2.3,
                "reference_acceptance_length": 2.5,
                "min_acceptance_length": 2.25,
                "threshold_ratio": 0.9,
                "passed": True,
            }
        )
    )
    metrics = extract_speedbench_al_metrics(result_path)
    row = build_row(
        {
            "infmax_model_prefix": "dsv4",
            "hw": "b300",
            "framework": "vllm",
            "precision": "fp4",
            "spec_decoding": "mtp",
        },
        metrics[0],
    )

    assert row["task"] == "speedbench_al/thinking_on/mtp2"
    assert row["score_name"] == "acceptance_length"
    assert score_cell(row) == "2.30 >= 2.25 (PASS)"
