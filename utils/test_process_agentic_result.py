"""Tests for process_agentic_result.py."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent / "process_agentic_result.py"


def write_detailed_results(result_dir: Path) -> None:
    trace_dir = result_dir / "trace_replay"
    trace_dir.mkdir(parents=True)
    (trace_dir / "detailed_results.csv").write_text(
        "\n".join([
            "success,request_start_time,request_complete_time,ttft,ttlt,itl,input_tokens,output_tokens_expected,output_tokens_actual,cache_hit_blocks,cache_miss_blocks",
            "True,0.0,1.0,0.10,1.00,0.02,100,20,20,2,8",
            "True,1.0,3.0,0.20,2.00,0.03,200,40,40,4,6",
        ])
        + "\n"
    )


def run_script(tmp_path: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result_dir = tmp_path / "results"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    write_detailed_results(result_dir)

    full_env = {
        **env,
        "RESULT_DIR": str(result_dir),
        "AGENTIC_OUTPUT_DIR": str(output_dir),
        "RESULT_FILENAME": "agentic_result",
        "RUNNER_TYPE": "gb200",
        "IMAGE": "test-image",
        "MODEL": "test-model",
        "MODEL_PREFIX": "test",
        "FRAMEWORK": "dynamo-trt",
        "PRECISION": "fp4",
        "USERS": "2",
    }

    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        env=full_env,
        capture_output=True,
        text=True,
    )


def load_output(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "out" / "agentic_result.json").read_text())


def test_single_node_agentic_result(tmp_path):
    result = run_script(tmp_path, {
        "TP": "8",
        "EP_SIZE": "1",
        "DP_ATTENTION": "false",
        "OFFLOADING": "none",
    })
    assert result.returncode == 0, result.stderr

    output = load_output(tmp_path)
    assert output["is_multinode"] is False
    assert output["tp"] == 8
    assert output["dp_attention"] == "false"
    assert output["num_requests_successful"] == 2
    assert output["tput_per_gpu"] == pytest.approx(120 / 8)


def test_multinode_agentic_result_skips_missing_metrics(tmp_path):
    result = run_script(tmp_path, {
        "IS_MULTINODE": "true",
        "SPEC_DECODING": "none",
        "DISAGG": "true",
        "PREFILL_NUM_WORKERS": "5",
        "PREFILL_TP": "4",
        "PREFILL_EP": "4",
        "PREFILL_DP_ATTN": "true",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "8",
        "DECODE_EP": "8",
        "DECODE_DP_ATTN": "true",
    })
    assert result.returncode == 0, result.stderr

    output = load_output(tmp_path)
    assert output["is_multinode"] is True
    assert output["prefill_num_workers"] == 5
    assert output["num_prefill_gpu"] == 20
    assert output["num_decode_gpu"] == 8
    assert output["prefill_dp_attention"] == "true"
    assert output["decode_dp_attention"] == "true"
    assert output["disagg"] is True
    assert output["tput_per_gpu"] == pytest.approx(120 / 28)
