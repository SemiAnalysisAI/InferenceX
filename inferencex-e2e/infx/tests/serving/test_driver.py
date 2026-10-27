import json
import shutil
from types import SimpleNamespace

import pytest

from infx.launch.policy import LaunchPath, launch_path
from infx.launch.request import LaunchRequest
from infx.tests.launch.fake_slurm import (
    ROOT,
    base_env,
    install_fakes,
    launch,
    lines,
    runner_for,
    sandbox_runner_config,
)

SRUN_LOGGER = '#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_LOG_DIR/srun.log"\nexit 0\n'


def serving_json(**extra) -> str:
    return json.dumps({
        "framework": "sglang", "serve_cmd": "python3 -m sglang.launch_server --model-path org/model",
        "scenario_yaml": "schemaVersion: '2.0'\n", "scenario_label": "agentx-cc-062126-smoke",
        "dataset_kind": "public", "dataset": "semianalysis_cc_traces_weka_062126", "search": False,
        "yaml": "name: x\n", **extra,
    })


@pytest.fixture
def harness(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    workspace = tmp_path / "ws"
    serving = workspace / "benchmarks/single_node/serving"
    serving.mkdir(parents=True)
    shutil.copy(ROOT / "benchmarks/single_node/serving/job.sh", serving / "job.sh")
    fakes = install_fakes(tmp_path / "bin")
    (fakes / "srun").write_text(SRUN_LOGGER)
    logs = tmp_path / "logs"
    env = base_env(fakes=fakes, logs=logs, workspace=workspace, sandbox=sandbox)
    env.update(
        RUNNER_NAME=runner_for("hcm-hgx02"), IMAGE="lmsysorg/sglang:v0.5.22", MODEL="org/model",
        GPU_COUNT="2", SALLOC_TIME_LIMIT="30", ENROOT_IMPORT_TIME_LIMIT="10",
        HF_HUB_CACHE="/mnt/hf_hub_cache/", SERVING_JSON=serving_json(),
    )
    return SimpleNamespace(config=sandbox_runner_config(sandbox), workspace=workspace, logs=logs, env=env)


def test_policy_routes_a_serving_point():
    request = LaunchRequest.from_env({"RUNNER_NAME": "hcm-hgx02_00", "SERVING_JSON": "{}"})
    assert launch_path("hcm-hgx02", request) == LaunchPath.SERVING


def test_point_is_one_right_sized_shared_batch_job_on_its_node(harness):
    result = launch(harness.env, harness.config, harness.workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    [submit] = lines(harness.logs, "sbatch")
    flags = set(submit.split())
    assert {"--gres=gpu:nvidia_h200:2", "--nodelist=hgx-h200-02", "--cpus-per-gpu=16",
            "--mem-per-gpu=240000M", f"--chdir={harness.workspace}"} <= flags
    assert "--exclusive" not in flags
    steps = lines(harness.logs, "srun")
    assert len(steps) == 2  # DCGM sidecar + engine step
    assert "bench.sh" in steps[-1] and "--overlap" not in steps[-1]


def test_router_runs_as_an_overlap_sidecar(harness):
    env = {**harness.env, "SERVING_JSON": serving_json(
        router_image="lmsysorg/sglang-router:v0.2", router_cmd="echo router", router_profile="generic", replicas=2)}
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    [router] = [step for step in lines(harness.logs, "srun") if "sidecar.sh router" in step]
    assert "--overlap" in router


def test_bad_serving_json_fails_before_any_slurm_call(harness):
    result = launch({**harness.env, "SERVING_JSON": "{}"}, harness.config, harness.workspace)
    assert result.returncode == 1
    assert "SERVING_JSON" in result.stdout + result.stderr
    assert lines(harness.logs, "sbatch") == []


def test_remote_point_is_a_cpu_only_job_with_one_client_step(harness):
    shutil.copy(ROOT / "benchmarks/single_node/serving/remote.sh",
                harness.workspace / "benchmarks/single_node/serving/remote.sh")
    env = {**harness.env, "IMAGE": "ghcr.io/astral-sh/uv:python3.12-bookworm-slim", "GPU_COUNT": "1",
           "SERVING_API_KEY": "sk-secret-123",
           "SERVING_JSON": serving_json(framework="remote", serve_cmd="", remote_url="https://api.example.com",
                                        tokenizer="builtin", api_key_secret="OPENAI_API_KEY")}
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    [submit] = lines(harness.logs, "sbatch")
    flags = submit.split()
    assert {"--cpus-per-task=16", "--mem=64G", "--nodelist=hgx-h200-02"} <= set(flags)
    assert not [f for f in flags if f.startswith(("--gres", "--cpus-per-gpu", "--mem-per-gpu"))]
    assert submit.endswith("remote.sh")
    steps = "\n".join(lines(harness.logs, "srun"))  # the bash -c script spans several log lines
    assert steps.count("--container-image=") == 1  # one client step, no DCGM sidecar
    assert "infx.serving.client prepare" in steps and "infx.serving.client run" in steps
    assert "sk-secret-123" not in result.stdout + result.stderr


def test_remote_sbatch_args_keep_the_node_pin_only():
    from infx.launch.drivers.serving import remote_sbatch_args

    assert remote_sbatch_args(["--nodelist=n1", "--cpus-per-gpu=16", "--mem-per-gpu=110000M"]) == [
        "--ntasks=1", "--cpus-per-task=16", "--mem=64G", "--nodelist=n1"]
