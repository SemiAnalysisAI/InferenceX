"""The srt-slurm driver end to end: ``python -m infx.launch run`` on the fakes.

Most points run on the sandboxed cluster records; the lane test runs synthetic lanes
in-process, so the lane table it patches applies.
"""

import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infx.launch.__main__ import main
from infx.launch.drivers.srt import lanes, models
from infx.launch.drivers.srt.checkout import SRT_FORKS
from infx.launch.drivers.srt.lanes import LaneMount, SrtLane
from infx.launch.drivers.srt.models import Override
from infx.launch.policy import LaunchPath, Match
from infx.tests.launch.fake_slurm import (
    base_env,
    install_fakes,
    launch,
    lines,
    make_workspace,
    runner_for,
    sandbox_runner_config,
    srtctl_calls,
)

POINT_RECIPE = {
    "engine": "sglang",
    "resources": {"gpus_per_node": 8},
    "model": {"path": "hf:test/model", "container": "test:tag", "precision": "fp8"},
    "roles": {"agg": {
        "nodes": 1, "workers": 1, "gpus": 4,
        "args": {"tensor-parallel-size": 4, "data-parallel-size": 1, "max-running-requests": 32},
    }},
    "benchmark": {"type": "custom", "env": {
        "MODEL": "test/model", "ISL": "256", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5",
        "USE_CHAT_TEMPLATE": "false",
    }},
}  # fmt: skip

POINT_ENV = {
    "FRAMEWORK": "sglang", "MODEL": "test/model", "IMAGE": "test:tag", "PRECISION": "fp8",
    "TP": "4", "GPU_COUNT": "4", "PP_SIZE": "1", "DCP_SIZE": "1", "PCP_SIZE": "1",
    "EP_SIZE": "1", "DP_ATTENTION": "false", "SPEC_DECODING": "none", "IS_AGENTIC": "0",
    "ISL": "256", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5", "CONC": "2", "MODEL_PREFIX": "dsr1",
    "IS_MULTINODE": "false", "SRT_RECIPE": "recipe.yaml", "FAKE_RESULTS": "single",
}  # fmt: skip

LANE_RECIPE = """name: "fixture"
model:
  path: "alias"
  container: "test:tag"
  precision: "fp8"
roles:
  prefill:
    args:
      tensor-parallel-size: 8
      watchdog-timeout: 600
  decode:
    args:
      tensor-parallel-size: 8
      watchdog-timeout: 600
health_check:
  max_attempts: 100
  interval_seconds: 5
benchmark:
  type: sa-bench
  concurrencies: [4]
"""
POWER_TELEMETRY = "telemetry:\n  dcgm_exporter:\n    image: dcgm\n  enabled: true\n"


@pytest.fixture
def harness(tmp_path):
    """Sandbox, fake binaries, workspace and sandboxed cluster records for one launch."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    config = sandbox_runner_config(sandbox)
    workspace = make_workspace(tmp_path / "ws")
    logs = tmp_path / "logs"
    env = base_env(
        fakes=install_fakes(tmp_path / "bin"), logs=logs, workspace=workspace, sandbox=sandbox
    )
    return SimpleNamespace(tmp=tmp_path, config=config, workspace=workspace, logs=logs, env=env)


def single_node_env(harness, cluster_id: str, **overrides: str) -> dict[str, str]:
    """Environment of a fixed-sequence single-node point with its recipe in the workspace."""
    recipe = {"base": POINT_RECIPE, "zip_override_conc": {"benchmark": {"env": {"CONC": ["2", "4"]}}}}
    (harness.workspace / "recipe.yaml").write_text(yaml.safe_dump(recipe))
    return {**harness.env, **POINT_ENV, "RUNNER_NAME": runner_for(cluster_id), **overrides}


def lane_env(harness, cluster_id: str, recipe: str = LANE_RECIPE, **overrides: str) -> dict[str, str]:
    """Environment of a multi-node point whose recipe lives in the workspace mirror.

    RUNNER_NAME defaults to the cluster's first runner.
    """
    mirror = harness.workspace / "benchmarks/multi_node/srt-slurm-recipes/test/lane.yaml"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(recipe)
    env = {
        **harness.env, "RUNNER_NAME": overrides.pop("RUNNER_NAME", None) or runner_for(cluster_id),
        "IS_MULTINODE": "true", "CONFIG_FILE": "recipes/test/lane.yaml", "IMAGE": "test:tag",
        "CONC_LIST": "4", "SPEC_DECODING": "none", "IS_AGENTIC": "0", "ISL": "1024", "OSL": "1024",
        "FAKE_RESULTS": "fixed",
    }  # fmt: skip
    return {**env, **overrides}


def srtslurm(root: Path) -> dict:
    """The srtslurm.yaml the launch rendered under ``root``."""
    [path] = root.glob("**/srtslurm.yaml")
    return yaml.safe_load(path.read_text())


WORKDIR = 'srun_options.container-workdir="/infmax-workspace"'


def assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    """Fail with the launcher's output when it did not exit 0."""
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-8000:]


def test_single_node_point_stages_workflow_artifacts(harness):
    workspace = harness.workspace
    env = single_node_env(harness, "h200-cw")
    assert_ok(launch(env, harness.config, workspace))

    assert json.loads((workspace / "point-identity.json").read_text()) == {"completed": 2}
    assert (workspace / "gpu_metrics.csv").read_text() == "gpu,power\n0,300\n"
    assert json.loads((workspace / "gpu_metrics_context.json").read_text()) == {"device_count": 4}
    assert (workspace / "srt-single-node-logs.tar.gz").stat().st_size > 0
    assert (workspace / "srt-slurm-sha.txt").read_text() == harness.env["FAKE_SRT_COMMIT"] + "\n"
    [call] = srtctl_calls(harness.logs)
    argv = call["argv"]
    assert argv[argv.index("--file") + 1] == f"{workspace}/recipe.yaml:zip_override_conc[0]"
    assert {"--json", "--yes", "--output", WORKDIR} <= set(argv)
    [callback] = [json.loads(arg.split("=", 1)[1]) for arg in argv
                  if arg.startswith("post_eval.command=")]  # fmt: skip
    assert callback == [
        "bash", "{infmax_workspace}/benchmarks/single_node/srt_eval.sh",
        "{endpoint}", "/logs/infx-eval-exit-code", "none",
    ]
    assert (call["env"]["INFMAX_WORKSPACE"], call["env"]["VIRTUAL_ENV"]) == (str(workspace), None)
    assert call["env"]["RUNNER_NAME"] == f"inferencex-{env['RUNNER_NAME']}"
    assert "/hf" in srtslurm(workspace)["default_mounts"].values()
    applied = [line.split()[-1] for line in lines(harness.logs, "git") if line.split()[2:3] == ["apply"]]
    assert applied == [str(workspace / "runners/srt-slurm/patches/001-fixture.patch")]
    assert lines(harness.logs, "scancel") == []


def test_single_node_eval_requires_a_successful_eval(harness):
    env = single_node_env(harness, "h100-cw", RUN_EVAL="true", MAX_MODEL_LEN="1024")
    assert_ok(launch(env, harness.config, harness.workspace))
    exit_file = next(harness.workspace.glob("srt-single.*/outputs/42/logs/infx-eval-exit-code"))
    harness.logs.joinpath("srtctl.jsonl").unlink()
    srtctl = Path(harness.env["PATH"].split(os.pathsep)[0]) / "srtctl"
    srtctl.write_text(srtctl.read_text().replace('write_text("0\\n")', 'write_text("3\\n")'))
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 1
    assert "eval did not succeed" in result.stderr
    assert exit_file.read_text() == "0\n"


def test_single_node_failed_allocation_fails_the_launch(harness):
    env = single_node_env(harness, "b200-nb", FAKE_STATE="FAILED|1:0")
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 1
    assert (harness.workspace / "point-identity.json").is_file()


LABS = {
    "lab-a": dict(
        lane=SrtLane(
            setup_scripts={"dynamo-sglang": "setup.sh"},
            mounts=(LaneMount(Match(), "cache", "/cache"),), time_limit="2:00:00",
        ),
        env=dict(FRAMEWORK="dynamo-sglang"),
        model="nvme/model", preflight=False, tag="lab,dsr1,fp8,1024x1024,", setup_script="setup.sh",
        served="served-model", dist_timeout=True, time="2:00:00", mounts=("/cache",), staging="import",
    ),
    "lab-b": dict(
        lane=SrtLane(shared_run_root=(Match(),)),
        env=dict(FRAMEWORK="dynamo-vllm", IS_AGENTIC="1", ISL="0", OSL="0", FAKE_RESULTS="agentic"),
        model="models/model", preflight=True, tag=None, setup_script=None, served=None,
        dist_timeout=False, time="10", mounts=(), staging="registry", shared_checkout=True,
    ),
}  # fmt: skip


def lab_config(tmp: Path) -> Path:
    """runners.yaml of the synthetic clusters, their storage under ``tmp``."""
    common = {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm"}
    clusters = {
        "lab-a": {**common, "models": {"entries": {"Model": {"root": "nvme", "dir": "model"}}}, "slurm": {
            "partition": "p", "exclusive": True,
            "volumes": {"nvme": {"path": str(tmp / "nvme"), "visibility": "node-local"},
                        "cache": {"path": str(tmp / "cache")}},
            "squash": {"dir": str(tmp / "squash"), "import": "submit-host"},
            "srt-slurm": {"network-interface": "", "job-tag": "lab", "dist-timeout-s": 1800},
        }},
        "lab-b": {**common, "models": {"entries": {"Model": {"root": "models", "dir": "model"}}}, "slurm": {
            "partition": "p", "exclusive": False,
            "volumes": {"models": {"path": str(tmp / "models")}},
            "srt-slurm": {"network-interface": "", "outputs": str(tmp / "outputs"),
                          "shared-run-root": str(tmp / "runs")},
        }},
    }  # fmt: skip
    path = tmp / "lab-runners.yaml"
    labels = {f"cluster:{cluster_id}": [f"{cluster_id}_00"] for cluster_id in clusters}
    path.write_text(yaml.safe_dump({"labels": labels, "clusters": clusters}))
    return path


def launch_here(monkeypatch, env: dict[str, str], config: Path, cwd: Path) -> int:
    """``python -m infx.launch run`` in this process, with exactly ``env`` as its environment."""
    for name in os.environ.keys() - env.keys():
        monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(cwd)
    return main(["--runner-config", str(config), "run"])


@pytest.mark.parametrize("cluster_id", LABS)
def test_multinode_lane_stages_workflow_artifacts(harness, monkeypatch, cluster_id):
    lab = LABS[cluster_id]
    for other, spec in LABS.items():
        monkeypatch.setitem(lanes.SRT_LANES, (other, LaunchPath.SRT_MULTI), spec["lane"])
    monkeypatch.setitem(models.OVERRIDES, "lab-a", (Override(Match(), served_name="served-model"),))
    env = lane_env(harness, cluster_id, RUNNER_NAME=f"{cluster_id}_00", MODEL_PREFIX="dsr1",
                   PRECISION="fp8", MODEL="org/Model", **lab["env"])  # fmt: skip
    runner, tmp, workspace = env["RUNNER_NAME"], harness.tmp, harness.workspace
    assert launch_here(monkeypatch, env, lab_config(tmp), workspace) == 0

    if env["FAKE_RESULTS"] == "agentic":
        assert json.loads((workspace / "point-identity_conc4.json").read_text()) == {"conc": 4}
    else:
        [point] = workspace.glob("point-identity_*.json")
        assert point.name == "point-identity_sweep_isl_1024_osl_1024_conc4_gpus_16_ctx_8_gen_8.json"
    with tarfile.open(workspace / "multinode_server_logs.tar.gz") as bundle:
        assert "./sweep_42.log" in bundle.getnames()
    assert (workspace / "srt-slurm-sha.txt").read_text() == harness.env["FAKE_SRT_COMMIT"] + "\n"
    assert (workspace / "LOGS/sweep_42.log").is_file()

    [call] = srtctl_calls(harness.logs)
    argv = call["argv"]
    checkout = Path(call["cwd"]).resolve()
    if lab.get("shared_checkout"):
        assert checkout.name.startswith(f"srt-slurm-9001-1-{runner}-")
        assert checkout.parent == (tmp / "runs").resolve()
    else:
        assert checkout.parent == workspace.resolve()
        assert checkout.name.startswith("srt-slurm-9001-1-") and len(checkout.name) == len("srt-slurm-9001-1-") + 12
    assert ("--no-preflight" in argv) is not lab["preflight"]
    assert argv[argv.index("--file") + 1] == "recipes/test/lane.yaml"
    assert {"--json", "--yes", "benchmark.stream_output=true", WORKDIR} <= set(argv)
    if lab["tag"] is None:
        assert "--tags" not in argv
    else:
        assert argv[argv.index("--tags") + 1].startswith(lab["tag"] + "infmax-")
    if lab["setup_script"] is None:
        assert "--setup-script" not in argv
    else:
        assert argv[argv.index("--setup-script") + 1] == lab["setup_script"]
    assert call["env"]["SERVED_MODEL_NAME"] == lab["served"]

    staged = yaml.safe_load((checkout / "recipes/test/lane.yaml").read_text())
    assert staged["name"] == call["env"]["RUNNER_NAME"] == f"inferencex-{runner}"
    dist = {"dist-timeout": 1800} if lab["dist_timeout"] else {}
    assert staged["roles"]["prefill"]["args"] == {"tensor-parallel-size": 8, "watchdog-timeout": 600, **dist}

    config = srtslurm(checkout)
    assert config["model_paths"] == {"alias": str(tmp / lab["model"])}
    assert config["default_time_limit"] == lab["time"]
    assert set(lab["mounts"]) <= set(config.get("default_mounts", {}).values())
    imported = [line.split()[-1] for line in lines(harness.logs, "enroot")]
    if lab["staging"] == "import":
        assert config["containers"][env["IMAGE"]].endswith(".sqsh") and imported == ["docker://test:tag"]
    else:
        assert config["containers"][env["IMAGE"]] == env["IMAGE"] and imported == []
    outputs = Path(json.loads((workspace / "srt-submission.json").read_text())["output_dir"])
    assert outputs.exists() is not outputs.is_relative_to(checkout)


@pytest.mark.parametrize(("model_prefix", "precision", "framework", "model", "require_power", "lane"), [
    ("kimik3", "fp4", "vllm", "moonshotai/Kimi-K3", "0", "agentx"),
    ("glm5.2", "fp8", "dynamo-sglang", "zai-org/GLM-5.2-FP8", "1", "adapter"),
    ("glm5.2", "fp8", "dynamo-sglang", "zai-org/GLM-5.2-FP8", "0", "adapter"),
])  # fmt: skip
def test_power_lane_stages_provenance_and_validates_each_concurrency(
    harness, model_prefix, precision, framework, model, require_power, lane
):
    adapter = harness.tmp / "results-python"
    adapter.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_LOG_DIR/adapter.log\"\n"
    )
    adapter.chmod(0o755)
    env = lane_env(
        harness, "h200-dgxc", LANE_RECIPE + POWER_TELEMETRY,
        MODEL_PREFIX=model_prefix, PRECISION=precision, FRAMEWORK=framework, MODEL=model,
        IS_AGENTIC="1", ISL="0", OSL="0", CONC_LIST="4 8", FAKE_RESULTS="agentic",
        REQUIRE_POWER=require_power, INFERENCEX_RESULTS_PYTHON=str(adapter),
    )  # fmt: skip
    assert_ok(launch(env, harness.config, harness.workspace))

    workspace = harness.workspace
    commit = harness.env["FAKE_SRT_COMMIT"]
    assert (workspace / "power-producer-sha.txt").read_text() == commit + "\n"
    [checkout] = workspace.glob("srt-slurm-9001-*")
    exporter = srtslurm(checkout)["containers"]["dcgm-exporter"]
    provenance = (workspace / "exporter-image.sha256").read_text()
    assert provenance.endswith(f"  {exporter}\n")
    assert (workspace / "LOGS/power/exporter-image.sha256").read_text() == provenance
    assert (workspace / "LOGS/power/power-producer-sha.txt").read_text() == commit + "\n"
    staged = yaml.safe_load((checkout / "recipes/test/lane.yaml").read_text())
    assert staged["benchmark"]["concurrencies"] == [4, 8]

    runs = lines(harness.logs, "adapter")
    assert [run.split("--result-dir ")[1].split()[0].rsplit("/", 1)[1] for run in runs] == ["conc_4", "conc_8"]
    assert all(f"--expected-producer-sha {commit}" in run for run in runs)
    required = lane == "agentx" or require_power == "1"
    assert all(run.endswith("--require-power") is required for run in runs)
    if lane == "agentx":
        status = (workspace / "LOGS/power/native-job-status.txt").read_text()
        assert status == "42|COMPLETED|0:0\n"


@pytest.mark.parametrize("shape", ["single", "multi"])
def test_submission_failure_code_propagates_and_cancels_the_job(harness, shape):
    active = harness.tmp / "active"
    if shape == "single":
        env = single_node_env(harness, "mi355x-amds", FAKE_SRTCTL_RC="7", FAKE_ACTIVE=str(active))
    else:
        env = lane_env(harness, "b300-dsxe", MODEL_PREFIX="dsr1", PRECISION="fp4", FRAMEWORK="dynamo-trt",
                       MODEL="deepseek-r1-fp4", FAKE_SRTCTL_RC="7", FAKE_ACTIVE=str(active))  # fmt: skip
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 7, result.stderr[-4000:]
    assert lines(harness.logs, "scancel") == ["42"]
    if shape == "single":
        assert (harness.workspace / "point-identity.json").is_file()


@pytest.mark.parametrize("shape", ["single", "multi"])
def test_sigterm_while_streaming_cancels_the_job_and_exits_143(harness, shape):
    active, tailing = harness.tmp / "active", harness.tmp / "tailing"
    extra = {"FAKE_ACTIVE": str(active), "FAKE_TAIL_MARKER": str(tailing)}
    if shape == "single":
        env = single_node_env(harness, "h200-cw", **extra)
    else:
        env = lane_env(harness, "b300-dsxe", MODEL_PREFIX="dsr1", PRECISION="fp4", FRAMEWORK="dynamo-trt",
                       MODEL="deepseek-r1-fp4", **extra)  # fmt: skip
    command = [sys.executable, "-m", "infx.launch", "--runner-config", str(harness.config), "run"]
    process = subprocess.Popen(
        command, cwd=harness.workspace, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    deadline = time.monotonic() + 120
    while not tailing.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "the launcher never started streaming"
        time.sleep(0.1)
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=60)
    assert process.returncode == 143, stdout[-2000:] + stderr[-4000:]
    assert lines(harness.logs, "scancel") == ["42"]
    staged = "srt-single-node-logs.tar.gz" if shape == "single" else "multinode_server_logs.tar.gz"
    assert (harness.workspace / staged).stat().st_size > 0


def test_b300_flash_agentx_reenters_inside_a_batch_allocation(harness):
    runner_temp = harness.tmp / "runner-temp"
    runner_temp.mkdir()
    recipe = json.loads(json.dumps(POINT_RECIPE))
    recipe["benchmark"]["command"] = "bash /infmax-workspace/benchmarks/srt_agentic.sh"
    (harness.workspace / "recipe.yaml").write_text(yaml.safe_dump({"base": recipe}))
    env = {
        **harness.env, **POINT_ENV, "RUNNER_NAME": runner_for("b300-dsxe"), "MODEL_PREFIX": "dsv41flash",
        "PRECISION": "fp8", "IS_AGENTIC": "1", "DURATION": "600", "RUNNER_TEMP": str(runner_temp),
        "SRT_RECIPE": "recipe.yaml:base",
    }  # fmt: skip
    assert_ok(launch(env, harness.config, harness.workspace))

    [submit] = lines(harness.logs, "sbatch")
    assert {"--nodes=1", "--ntasks=1", f"--chdir={harness.workspace}", "--time=10"} <= set(submit.split())
    assert (harness.logs / "batch-rc").read_text() == "0"
    assert json.loads((harness.workspace / "point-identity.json").read_text()) == {"completed": 2}
    assert len(srtctl_calls(harness.logs)) == 1
    assert "4242" in lines(harness.logs, "scancel")
    assert list(runner_temp.glob("srt-batch.*.sh")) == []


def test_tilert_native_lane_runs_on_its_fork_without_patches(harness):
    env = lane_env(
        harness, "b200-nscale", MODEL_PREFIX="glm5.1", PRECISION="fp8", FRAMEWORK="tilert",
        MODEL="zai-org/GLM-5.1-FP8", SPEC_DECODING="mtp", IS_AGENTIC="1", ISL="0", OSL="0",
        FAKE_RESULTS="agentic", PREFILL_IMAGE="prefill:tag",
        FAKE_SRT_COMMIT=SRT_FORKS["tilert"].commit,
    )  # fmt: skip
    assert_ok(launch(env, harness.config, harness.workspace))

    assert not any(" apply " in f" {line} " for line in lines(harness.logs, "git"))
    [call] = srtctl_calls(harness.logs)
    assert not {"--json", "--no-preflight", "benchmark.stream_output=true"} & set(call["argv"])
    config = srtslurm(Path(call["cwd"]))
    assert "default_health_check" not in config
    assert config["containers"]["tilert-decode"].endswith("/test_tag.sqsh")
    assert config["containers"]["tilert-prefill"].endswith("/prefill_tag.sqsh")
    assert config["default_mounts"][str(harness.workspace)] == "/infmax-workspace"
    assert json.loads((harness.workspace / "point-identity_conc4.json").read_text()) == {"conc": 4}


def test_eval_only_runs_the_eval_recipe_with_real_verification(harness):
    mirror = harness.workspace / "benchmarks/multi_node/srt-slurm-recipes/test"
    (mirror / "trtllm").mkdir(parents=True)
    (mirror / "trtllm/forced.yaml").write_text(
        "roles:\n  decode:\n    env:\n      TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS: 2\n      KEEP: 1\n"
    )
    (mirror / "eval.yaml").write_text(LANE_RECIPE)
    env = lane_env(
        harness, "gb300-nv", MODEL_PREFIX="dsv4", PRECISION="fp4", FRAMEWORK="dynamo-trt",
        MODEL="deepseek-ai/DeepSeek-V4-Pro", IS_AGENTIC="1", SPEC_DECODING="mtp", ISL="0", OSL="0",
        EVAL_ONLY="true", EVAL_CONFIG_FILE="recipes/test/eval.yaml", FAKE_RESULTS="eval",
        EVAL_CONC="4 8",
    )  # fmt: skip
    assert_ok(launch(env, harness.config, harness.workspace))

    [call] = srtctl_calls(harness.logs)
    argv = call["argv"]
    assert argv[argv.index("--file") + 1] == "recipes/test/eval.yaml"
    assert "frontend.placement.node=head" in argv
    checkout = Path(call["cwd"])
    assert (checkout / "recipes/test/trtllm/forced.yaml").read_text() == (
        "roles:\n  decode:\n    env:\n      KEEP: 1\n"
    )
    assert srtslurm(checkout)["default_time_limit"] == "8:00:00"
    workspace = harness.workspace
    assert (workspace / "results_gsm8k.json").read_text() == "{}"
    assert json.loads((workspace / "meta_env.json").read_text()) == {"conc": "4 8"}
    assert list(workspace.glob("point-identity_*.json")) == []


def test_setup_retries_only_after_discarding_a_truncated_archive(harness):
    env = lane_env(
        harness, "h200-dgxc", MODEL_PREFIX="dsr1", PRECISION="fp8", FRAMEWORK="dynamo-sglang",
        MODEL="deepseek-ai/DeepSeek-R1-0528", FAKE_TRUNCATED_NATS="1",
    )  # fmt: skip
    assert_ok(launch(env, harness.config, harness.workspace))
    assert len(lines(harness.logs, "make")) == 2
    assert not list(harness.workspace.glob("srt-slurm-9001-*/configs/nats-server-*.deb"))


def test_a_setup_failure_without_a_bad_archive_is_not_retried(harness):
    env = lane_env(
        harness, "h200-dgxc", MODEL_PREFIX="dsr1", PRECISION="fp8", FRAMEWORK="dynamo-sglang",
        MODEL="deepseek-ai/DeepSeek-R1-0528", FAKE_MAKE_RC="2",
    )  # fmt: skip
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 2
    assert "make setup broke" in result.stderr
    assert len(lines(harness.logs, "make")) == 1
    assert srtctl_calls(harness.logs) == [] and lines(harness.logs, "scancel") == []


@pytest.mark.parametrize(("cluster_id", "env", "message"), [
    ("h100-dgxc", dict(MODEL_PREFIX="dsr1", PRECISION="fp8", FRAMEWORK="dynamo-vllm"), "Unsupported framework"),
    ("b200-nscale", dict(MODEL_PREFIX="dsv4", PRECISION="fp4", FRAMEWORK="dynamo-trt"), "only dynamo-vllm"),
    ("gb300-nv", dict(MODEL_PREFIX="llama", PRECISION="fp8", FRAMEWORK="dynamo-sglang"), "stages no checkpoint"),
    ("h200-dgxc", dict(MODEL_PREFIX="dsr1", PRECISION="fp8", FRAMEWORK="dynamo-sglang", CONFIG_FILE=""),
     "CONFIG_FILE is not set"),
])  # fmt: skip
def test_unsupported_multinode_requests_fail_before_any_setup(harness, cluster_id, env, message):
    result = launch(lane_env(harness, cluster_id, MODEL="m", **env), harness.config, harness.workspace)
    assert result.returncode == 1
    assert message in result.stderr
    assert lines(harness.logs, "git") == []


@pytest.mark.parametrize(("shape", "overrides", "missing"), [
    ("single", dict(IS_AGENTIC="1", SPEC_DECODING="mtp", THINKING_MODE=""), "THINKING_MODE"),
    ("multi", dict(SPEC_DECODING=None), "SPEC_DECODING"),
    ("batch", dict(SRT_RECIPE=None), "SRT_RECIPE"),
])  # fmt: skip
def test_a_missing_input_is_named_before_any_setup(harness, shape, overrides, missing):
    if shape == "multi":
        env = lane_env(harness, "h200-dgxc", MODEL_PREFIX="dsr1", PRECISION="fp8",
                       FRAMEWORK="dynamo-sglang", MODEL="deepseek-ai/DeepSeek-R1-0528")  # fmt: skip
    else:
        env = single_node_env(harness, "b300-dsxe" if shape == "batch" else "h200-cw")
    if shape == "batch":
        env.update(MODEL_PREFIX="dsv41flash", PRECISION="fp8", IS_AGENTIC="1", DURATION="600")
    for name, value in overrides.items():
        env.pop(name) if value is None else env.update({name: value})
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 1
    assert missing in result.stderr
    assert lines(harness.logs, "git") == [] and lines(harness.logs, "sbatch") == []


def test_post_eval_is_handed_the_workload_contract_and_no_other_secret(harness):
    handed = {"SWEBENCH_NEW_KNOB": "7", "AIPERF_NEW": "1", "MODAL_TOKEN_ID": "modal-id"}
    withheld = {
        "HF_TOKEN": "hf_fixture", "GITHUB_TOKEN": "ghs_fixture", "PORT": "8888",
        "EVAL_EMPTY": "", "EVAL_NOT-A-NAME": "x", "SLURM_JOB_ID": "99",
    }  # fmt: skip
    env = single_node_env(harness, "h200-cw", **handed, **withheld)
    assert_ok(launch(env, harness.config, harness.workspace))
    [call] = srtctl_calls(harness.logs)
    [names] = [json.loads(arg.split("=", 1)[1]) for arg in call["argv"]
               if arg.startswith("post_eval.passthrough_env=")]  # fmt: skip
    assert handed.keys() <= set(names)
    assert not {*withheld, "PATH", "HF_HUB_CACHE"} & set(names)
