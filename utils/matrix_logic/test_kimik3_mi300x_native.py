"""CPU-only gates for the Kimi K3 MI300X native multi-node AgentX path."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "utils" / "matrix_logic"))

from generate_sweep_configs import generate_test_config_sweep  # noqa: E402
from validation import load_config_files, load_runner_file  # noqa: E402

CONFIG_KEY = "kimik3-fp4-mi300x-vllm-agentic"
SERVER_SCRIPT = (
    REPO_ROOT / "benchmarks" / "multi_node" / "agentic" / "kimik3_fp4_mi300x_vllm.sh"
)

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

def server_env(rank: int = 0) -> dict[str, str]:
    env = {
        **os.environ,
        "MODEL": "moonshotai/Kimi-K3",
        "MODEL_PATH": "/models/Kimi-K3",
        "PORT": "8888",
        "CONC_LIST": "4",
        "KV_OFFLOADING": "none",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "8",
        "PREFILL_PP_SIZE": "2",
        "PREFILL_EP": "1",
        "PREFILL_DP_ATTN": "false",
        "DECODE_NUM_WORKERS": "0",
        "MULTINODE_NODE_COUNT": "2",
        "MULTINODE_GPUS_PER_NODE": "8",
        "MULTINODE_NODE_RANK": str(rank),
        "MULTINODE_MASTER_ADDR": "node-a",
        "KIMIK3_VLLM_DRY_RUN": "1",
    }
    env.pop("AITER_SITUV2_A8W4", None)
    return env

def run_server(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SERVER_SCRIPT)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

def test_rank_zero_serves_tp8_pp2_without_headless() -> None:
    result = run_server(server_env(0))
    assert result.returncode == 0, result.stderr
    assert "--tensor-parallel-size 8" in result.stdout
    assert "--pipeline-parallel-size 2" in result.stdout
    assert "--nnodes 2" in result.stdout
    assert "--node-rank 0" in result.stdout
    assert "--master-addr node-a" in result.stdout
    assert "--headless" not in result.stdout
    assert "FLASHMLA" not in result.stdout
    assert "FLASHINFER" not in result.stdout

def test_rank_one_is_headless() -> None:
    result = run_server(server_env(1))
    assert result.returncode == 0, result.stderr
    assert "--node-rank 1" in result.stdout
    assert "--headless" in result.stdout

@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PREFILL_TP", "16", "TP8 x PP2"),
        ("PREFILL_PP_SIZE", "1", "TP8 x PP2"),
        ("PREFILL_EP", "8", "EP1"),
        ("DECODE_NUM_WORKERS", "1", "aggregated"),
        ("CONC_LIST", "4 8", "one concurrency"),
        ("CONC_LIST", "16", "1, 2, 4, or 8"),
        ("AITER_SITUV2_A8W4", "auto", "0 or 1"),
    ],
)
def test_server_rejects_out_of_contract_values(
    name: str, value: str, message: str
) -> None:
    env = server_env()
    env[name] = value
    result = run_server(env)
    assert result.returncode != 0
    assert message in result.stderr

def test_aiter_mode_is_not_defaulted_and_accepts_both_modes() -> None:
    unset_result = run_server(server_env())
    assert "AITER_SITUV2_A8W4=unset" in unset_result.stdout
    for value in ("0", "1"):
        env = server_env()
        env["AITER_SITUV2_A8W4"] = value
        result = run_server(env)
        assert result.returncode == 0
        assert f"AITER_SITUV2_A8W4={value}" in result.stdout
