import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_eval_results import (
    build_row,
    detect_eval_jsons,
    extract_speedbench_al_metrics,
    score_cell,
)
from speedbench_al import build_result, load_reference, lookup_reference
from speedbench_client import (
    _chat_payload,
    _completion_payload,
    _load_speedbench_requests,
)
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
        verify_steps="10",
        proposed_draft_tokens="20",
        framework="vllm",
        metric_source="vllm-prometheus-counters-endpoints1",
        error=None,
    )

    result = build_result(args)

    assert result["reference_acceptance_length"] == 2.50
    assert result["min_acceptance_length"] == 2.25
    assert result["framework"] == "vllm"
    assert result["metric_source"] == "vllm-prometheus-counters-endpoints1"
    assert result["verify_steps"] == 10
    assert result["proposed_draft_tokens"] == 20
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
                "framework": "sglang",
                "metric_source": "sglang-prometheus-gauge-endpoints1+derived-token-counters",
                "accepted_tokens": 13,
                "verify_steps": 10,
                "proposed_draft_tokens": 20,
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
    assert row["speedbench_framework"] == "sglang"
    assert row["speedbench_metric_source"] == "sglang-prometheus-gauge-endpoints1+derived-token-counters"
    assert row["speedbench_accepted_tokens"] == 13
    assert row["speedbench_verify_steps"] == 10
    assert row["speedbench_proposed_draft_tokens"] == 20
    assert score_cell(row) == "2.30 >= 2.25 (PASS)"


def test_detect_eval_jsons_dedupes_flat_speedbench_result(tmp_path: Path) -> None:
    result_path = tmp_path / "results_speedbench_al_thinking_on_mtp2.json"
    result_path.write_text(
        json.dumps(
            {
                "speedbench_al_eval_version": 1,
                "task": "speedbench_al",
                "thinking_mode": "thinking_on",
                "num_speculative_tokens": 2,
                "acceptance_length": 2.3,
                "min_acceptance_length": 2.25,
                "passed": True,
            }
        )
    )

    lm_path, speedbench_paths = detect_eval_jsons(tmp_path)

    assert lm_path is None
    assert speedbench_paths == [result_path]


def test_speedbench_client_loads_coding_and_builds_dsv4_payloads(tmp_path: Path) -> None:
    dataset = tmp_path / "speed_bench_data"
    dataset.mkdir()
    (dataset / "qualitative.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "category": "coding",
                        "messages": [{"role": "user", "content": "Write fizzbuzz."}],
                    }
                ),
                json.dumps(
                    {
                        "category": "math",
                        "messages": [{"role": "user", "content": "Solve 2+2."}],
                    }
                ),
            ]
        )
    )

    prompts = _load_speedbench_requests(dataset, "coding", -1)
    chat = _chat_payload(
        prompts[0],
        model="deepseek-ai/DeepSeek-V4-Pro",
        output_len=4096,
        temperature=1.0,
        thinking_mode="on",
        thinking_kwargs={"thinking": True, "reasoning_effort": "high"},
    )
    completions = _completion_payload(
        prompts[0],
        model="deepseek-ai/DeepSeek-V4-Pro",
        output_len=4096,
        temperature=1.0,
        thinking_mode="on",
        thinking_kwargs={"thinking": True, "reasoning_effort": "high"},
        dsv4=True,
    )

    assert len(prompts) == 1
    assert chat["chat_template_kwargs"]["thinking"] is True
    assert chat["reasoning_effort"] == "high"
    assert "<think>" in completions["prompt"]
    assert completions["max_tokens"] == 4096
