"""CPU-only gates for the Kimi K3 MI300X native multi-node AgentX path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "utils" / "matrix_logic"))

from generate_sweep_configs import generate_test_config_sweep  # noqa: E402
from validation import load_config_files, load_runner_file  # noqa: E402

CONFIG_KEY = "kimik3-fp4-mi300x-vllm-agentic"

def generate_kimik3_matrix() -> list[dict]:
    configs = load_config_files([str(REPO_ROOT / "configs" / "amd-master.yaml")])
    runners = load_runner_file(str(REPO_ROOT / "configs" / "runners.yaml"))
    args = argparse.Namespace(
        config_keys=[CONFIG_KEY],
        seq_lens=None,
        conc=None,
        scenario_type=["agentic-coding"],
        runner_node_filter=None,
    )
    return generate_test_config_sweep(args, configs, runners)

def test_kimik3_matrix_is_exactly_four_tp8_pp2_aggregate_jobs() -> None:
    rows = generate_kimik3_matrix()

    assert [row["conc"] for row in rows] == [[1], [2], [4], [8]]
    assert {row["runner"] for row in rows} == {"cluster:mi300x-amds"}
    assert {row["framework"] for row in rows} == {"vllm"}
    assert {row["disagg"] for row in rows} == {False}
    assert {
        (
            row["prefill"]["num-worker"],
            row["prefill"]["tp"],
            row["prefill"]["pp"],
            row["prefill"]["ep"],
            row["prefill"]["dp-attn"],
            row["decode"]["num-worker"],
            row["decode"]["tp"],
            row["decode"]["pp"],
            row["decode"]["ep"],
            row["decode"]["dp-attn"],
        )
        for row in rows
    } == {(1, 8, 2, 1, False, 0, 8, 2, 1, False)}
    settings = rows[0]["prefill"]["additional-settings"]
    assert settings == ["NATIVE_MULTINODE=1"]
    assert all("AITER_SITUV2_A8W4" not in setting for setting in settings)
