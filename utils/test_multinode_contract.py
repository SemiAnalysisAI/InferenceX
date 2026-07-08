"""Tests for the cluster-agnostic multi-node benchmark contract."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
MULTI_NODE_DIR = REPO_ROOT / "benchmarks" / "multi_node"


def contract_env(tmp_path: Path) -> dict[str, str]:
    launcher = tmp_path / "launcher.sh"
    launcher.write_text(
        """#!/usr/bin/env bash
printf '%s\n' \
  "image=$CONTAINER_IMAGE" \
  "prefill_ep=$PREFILL_ENABLE_EP" \
  "prefill_dp=$PREFILL_ENABLE_DP" \
  "decode_ep=$DECODE_ENABLE_EP" \
  "decode_dp=$DECODE_ENABLE_DP" \
  "node_list=$NODE_LIST" \
  "spec=$SPEC_DECODING" \
  "mtp=$DECODE_MTP_SIZE" \
  "block=${BLOCK_SIZE:-}" \
  "max_model_len=${MAX_MODEL_LEN:-}"
""",
        encoding="utf-8",
    )

    return {
        **os.environ,
        "MULTINODE_LAUNCHER": str(launcher),
        "CONC_LIST": "8 16",
        "ISL": "1024",
        "OSL": "1024",
        "IMAGE": "example.com/inference:latest",
        "MODEL": "org/model",
        "SPEC_DECODING": "none",
        "PREFILL_NUM_WORKERS": "2",
        "PREFILL_TP": "8",
        "PREFILL_EP": "1",
        "PREFILL_DP_ATTN": "true",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "8",
        "DECODE_EP": "8",
        "DECODE_DP_ATTN": "false",
        "PREFILL_NODES": "2",
        "DECODE_NODES": "1",
        "RANDOM_RANGE_RATIO": "0.8",
        "FRAMEWORK": "sglang-disagg",
        "NODELIST": "node-a,node-b,node-c",
    }


def run_recipe(recipe: str, env: dict[str, str]) -> dict[str, str]:
    result = subprocess.run(
        ["bash", str(MULTI_NODE_DIR / recipe)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_recipe_normalizes_workload_for_runner(tmp_path: Path) -> None:
    values = run_recipe(
        "dsr1_fp8_mi355x_sglang-disagg.sh",
        contract_env(tmp_path),
    )

    assert values == {
        "image": "example.com/inference:latest",
        "prefill_ep": "false",
        "prefill_dp": "true",
        "decode_ep": "true",
        "decode_dp": "false",
        "node_list": "node-a,node-b,node-c",
        "spec": "none",
        "mtp": "0",
        "block": "",
        "max_model_len": "",
    }


def test_model_recipe_only_adds_model_tuning(tmp_path: Path) -> None:
    env = contract_env(tmp_path)
    env.pop("SPEC_DECODING")

    values = run_recipe("minimaxm3_fp4_mi355x_atom-disagg.sh", env)

    assert values["spec"] == "none"
    assert values["mtp"] == "0"
    assert values["block"] == "128"
    assert values["max_model_len"] == "32768"


def test_deprecated_mi355x_recipe_uses_portable_launcher(tmp_path: Path) -> None:
    env = contract_env(tmp_path)
    env["FRAMEWORK"] = "vllm-disagg"

    values = run_recipe(
        "deprecated/minimaxm2.5_fp8_mi355x_vllm-disagg.sh",
        env,
    )

    assert values["image"] == "example.com/inference:latest"
    assert values["node_list"] == "node-a,node-b,node-c"


def test_active_benchmarks_do_not_contain_known_cluster_literals() -> None:
    forbidden = (
        "/it-share",
        "/nfsdata",
        "mia1-p",
        "SLURM_ACCOUNT",
        "SLURM_PARTITION",
        "sbatch ",
        "sudo docker",
    )
    offenders: list[str] = []

    for script in MULTI_NODE_DIR.rglob("*.sh"):
        if "deprecated" in script.parts:
            continue
        contents = script.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in contents:
                offenders.append(f"{script.relative_to(REPO_ROOT)}: {marker}")

    assert not offenders, "cluster policy leaked into benchmark code:\n" + "\n".join(offenders)


def test_srt_launchers_only_use_canonical_checkout_helper() -> None:
    helper = REPO_ROOT / "runners" / "lib" / "srt_slurm.sh"
    offenders: list[str] = []

    for script in (REPO_ROOT / "runners").glob("launch_*.sh"):
        contents = script.read_text(encoding="utf-8")
        if "srt-slurm" not in contents:
            continue
        for marker in ("git clone", "git checkout"):
            if marker in contents:
                offenders.append(f"{script.relative_to(REPO_ROOT)}: {marker}")

    helper_contents = helper.read_text(encoding="utf-8")
    assert "clone_srt_slurm()" in helper_contents
    assert "git clone" in helper_contents
    assert not offenders, "launcher bypasses canonical srt-slurm helper:\n" + "\n".join(offenders)


def test_srt_benchmark_uses_runtime_frontend_endpoint() -> None:
    contents = (
        REPO_ROOT / "benchmarks" / "multi_node" / "srt_benchmark.sh"
    ).read_text(encoding="utf-8")

    assert "SRT_FRONTEND_HOST" in contents
    assert "SRT_FRONTEND_PORT" in contents
    assert '--host "$SERVER_HOST"' in contents


def test_shared_benchmark_helper_defaults_unset_eval_only() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -u; source "$1"; unset EVAL_ONLY; '
            "run_benchmark_serving 2>&1",
            "bash",
            str(REPO_ROOT / "benchmarks" / "benchmark_lib.sh"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--model is required" in result.stdout
    assert "unbound variable" not in result.stdout


def test_benchmark_workflows_bootstrap_runner_registry_dependency() -> None:
    workflows = (
        REPO_ROOT / ".github" / "workflows" / "benchmark-tmpl.yml",
        REPO_ROOT / ".github" / "workflows" / "benchmark-multinode-tmpl.yml",
    )
    helper = (
        REPO_ROOT / "runners" / "lib" / "runner_config.sh"
    ).read_text(encoding="utf-8")

    for workflow_path in workflows:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "Bootstrap runner registry dependency" in workflow
        assert "INFERENCEX_RUNNER_PYTHON" in workflow
    assert "INFERENCEX_RUNNER_PYTHON" in helper
    assert "cannot import PyYAML" in helper


def test_multinode_launchers_require_successful_slurm_exit() -> None:
    launchers = (
        "launch_h100-dgxc-slurm.sh",
        "launch_h200-dgxc-slurm.sh",
        "launch_b200-dgxc.sh",
        "launch_b300-nv.sh",
        "launch_gb200-nv.sh",
        "launch_gb300-nv.sh",
        "launch_mi355x-amds.sh",
    )

    for launcher in launchers:
        contents = (REPO_ROOT / "runners" / launcher).read_text(encoding="utf-8")
        assert "require_slurm_job_succeeded" in contents, launcher


def test_mi355x_workload_exit_code_survives_cleanup() -> None:
    job = (
        REPO_ROOT / "runners" / "mi355x-amds" / "job.slurm"
    ).read_text(encoding="utf-8")
    launcher = (
        REPO_ROOT / "runners" / "launch_mi355x-amds.sh"
    ).read_text(encoding="utf-8")

    assert "WORKLOAD_RC=$?" in job
    assert 'exit "$WORKLOAD_RC"' in job
    assert "invalid Slurm job ID" in launcher
    assert 'grep -Fxq "$JOB_ID"' in launcher
