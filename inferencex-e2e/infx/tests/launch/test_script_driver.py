"""ScriptDriver (SPEED-Bench collectors) on the Slurm backend, against fake Slurm/Pyxis binaries."""

import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from infx.clusters import load_inventory
from infx.launch.__main__ import launch
from infx.launch.request import LaunchRequest

ROOT = Path(__file__).resolve().parents[3]

# Every fake appends ``[name, *argv]`` to $FAKE_CALLS. srun emulates Pyxis for
# container steps: the command runs in the host directory bound to the container
# workdir, and path-valued variables are translated through the bind mounts, so a
# script writing $OUT_YAML=/workspace/... lands in the mounted workspace.
FAKE = r"""#!{python}
import json, os, subprocess, sys

name = os.path.basename(sys.argv[0])
args = sys.argv[1:]
with open(os.environ["FAKE_CALLS"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if name == "salloc":
    print("salloc: Granted job allocation 42", file=sys.stderr)
elif name == "unsquashfs":
    sys.exit(1)
elif name == "srun":
    options = {{}}
    while args and args[0].startswith("--"):
        key, _, value = args.pop(0).partition("=")
        options[key] = value
    if "--container-image" not in options:
        sys.exit(int(os.environ.get("FAKE_PROBE_RC", "0")) if args[0] == "test" else 0)
    mounts = [mount.split(":", 1) for mount in options["--container-mounts"].split(",")]
    if os.environ.get("FAKE_HANG_MARKER"):
        open(os.environ["FAKE_HANG_MARKER"], "w").close()
        import time
        time.sleep(60)

    def host(path):
        for source, target in sorted(mounts, key=lambda mount: -len(mount[1])):
            if path == target or path.startswith(target + "/"):
                return source + path[len(target):]
        return path

    env = dict(os.environ)
    exported = options["--export"].split(",")
    assert exported[0] == "ALL", exported
    env.update(item.split("=", 1) for item in exported[1:])
    env = {{key: host(value) for key, value in env.items()}}
    sys.exit(subprocess.run(args, cwd=host(options["--container-workdir"]), env=env).returncode)
"""

COLLECTOR = """#!/usr/bin/env bash
set -eo pipefail
printf 'cwd=%s\\nMODEL_PATH=%s\\nPORT=%s\\n' "$PWD" "$MODEL_PATH" "$PORT" > seen.txt
echo 'kimi-k3: {}' > "$OUT_YAML"
exit "${COLLECTOR_RC:-0}"
"""
COLLECTOR_PATH = "benchmarks/single_node/speedbench/fixture_fp4_b300_vllm.sh"


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """Install fake salloc/srun/scancel/squeue/unsquashfs; return a reader of their calls."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("salloc", "srun", "scancel", "squeue", "unsquashfs"):
        binary = binaries / name
        binary.write_text(FAKE.format(python=sys.executable))
        binary.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("PATH", f"{binaries}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CALLS", str(log))
    # speedbench-al.yml points the collector at the container workspace.
    monkeypatch.setenv("OUT_YAML", "/workspace/speedbench-reference-al.yaml")

    def calls() -> list[list[str]]:
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return calls


@pytest.fixture
def workspace(tmp_path):
    """Measured checkout holding one collector script."""
    root = tmp_path / "workspace"
    collector = root / COLLECTOR_PATH
    collector.parent.mkdir(parents=True)
    collector.write_text(COLLECTOR)
    return root


def launch_env(workspace: Path, **overrides: str) -> dict[str, str]:
    """A SPEED-Bench launch environment (speedbench-al.yml) for a staged model."""
    return {
        "RUNNER_NAME": "fixture_00",
        "GITHUB_WORKSPACE": str(workspace),
        "MODEL": "moonshotai/Kimi-K3",
        "MODEL_PREFIX": "kimik3",
        "IMAGE": "vllm/vllm-openai:v0.21.0",
        "FRAMEWORK": "vllm",
        "PRECISION": "fp4",
        "GPU_COUNT": "8",
        "IS_MULTINODE": "false",
        "IS_AGENTIC": "0",
        "EVAL_ONLY": "false",
        "BENCH_SCRIPT_OVERRIDE": COLLECTOR_PATH,
        "SALLOC_TIME_LIMIT": "480",
        "ENROOT_IMPORT_TIME_LIMIT": "120",
        # speedbench-al.yml points the collector at the container workspace.
        "OUT_YAML": "/workspace/speedbench-reference-al.yaml",
        **overrides,
    }


def request_for(workspace: Path, **overrides: str) -> LaunchRequest:
    """The parsed launch request for :func:`launch_env`."""
    return LaunchRequest.from_env(launch_env(workspace, **overrides))


def inventory_for(tmp_path: Path, slurm: dict | None = None, squash: dict | None = None) -> dict:
    """Runner config whose one cluster keeps squash, cache and model volumes under ``tmp_path``."""
    config = {
        "gpus-per-node": 8,
        "arch": "x86_64",
        "env": {"UCX_NET_DEVICES": "eth0"},
        "models": {
            "download-root": "downloads",
            "entries": {"Kimi-K3": {"root": "scratch", "dir": "Kimi-K3"}},
        },
        "scheduler": "slurm",
        "slurm": {
            "partition": "batch",
            "account": "bench",
            "exclusive": True,
            "exclude": ["bad-node"],
            "gres": "gpu:{gpus}",
            "salloc-args": ["--mem=0"],
            "volumes": {
                "hf-home": {"path": str(tmp_path / "hf")},
                "scratch": {"path": str(tmp_path / "scratch"), "visibility": "node-local"},
                "downloads": {"path": str(tmp_path / "downloads")},
            },
            "squash": {"dir": str(tmp_path / "squash"), "visibility": "shared", "import": "compute"},
        },
    }
    config["slurm"].update(slurm or {})
    config["slurm"]["squash"].update(squash or {})
    return {"labels": {"cluster:fixture": ["fixture_00"]}, "clusters": {"fixture": copy.deepcopy(config)}}


def cluster_for(tmp_path: Path, slurm: dict | None = None, squash: dict | None = None):
    """The fixture cluster of :func:`inventory_for`."""
    return load_inventory(inventory_for(tmp_path, slurm, squash)).clusters["fixture"]


def option(argv: list[str], name: str) -> str:
    """Value of ``--name=value`` in ``argv``."""
    [value] = [arg.split("=", 1)[1] for arg in argv if arg.startswith(f"{name}=")]
    return value


def container_step(calls: list[list[str]]) -> list[str]:
    """The one srun call that starts a container."""
    [step] = [call for call in calls if any(arg.startswith("--container-image=") for arg in call)]
    return step


def test_collector_writes_the_reference_yaml_into_the_workspace(fakes, workspace, tmp_path):
    assert launch(cluster_for(tmp_path), request_for(workspace)) == 0

    # The workflow's success check: the collector output lands in GITHUB_WORKSPACE.
    assert (workspace / "speedbench-reference-al.yaml").read_text() == "kimi-k3: {}\n"
    assert (workspace / "seen.txt").read_text().splitlines() == [
        f"cwd={workspace}", f"MODEL_PATH={tmp_path}/scratch/Kimi-K3", "PORT=8888",
    ]
    assert (tmp_path / "hf").is_dir()

    # Cold shared cache: a one-node import step runs before the GPU allocation.
    importer, allocation, probe, container, cancel = fakes()
    assert importer[0] == "srun" and not any(arg.startswith("--jobid") for arg in importer)
    assert {"--partition=batch", "--account=bench", "--time=120", "--exclude=bad-node",
            "--nodes=1"} <= set(importer)
    assert allocation[0] == "salloc"
    assert {"--partition=batch", "--account=bench", "--nodes=1", "--gres=gpu:8", "--exclusive",
            "--mem=0", "--time=480", "--job-name=fixture_00", "--exclude=bad-node",
            "--no-shell"} <= set(allocation)
    # The staged root is node-local, so readiness is checked on the allocated node.
    assert probe == ["srun", "--jobid=42", "--export=ALL", "test", "-r",
                     f"{tmp_path}/scratch/Kimi-K3/config.json"]
    # Only the volume holding the checkpoint is mounted, not the download root.
    assert option(container, "--container-mounts").split(",") == [
        f"{workspace}:/workspace", f"{tmp_path}/scratch:/models", f"{tmp_path}/hf:/hf_hub_cache",
    ]
    assert option(container, "--container-image") == f"{tmp_path}/squash/vllm_vllm-openai_v0.21.0.sqsh"
    # The cluster's workload env, then the driver's own settings.
    assert option(container, "--export").split(",") == [
        "ALL", "UCX_NET_DEVICES=eth0", "PORT=8888", "MODEL_PATH=/models/Kimi-K3", "HF_HOME=/hf_hub_cache",
        "HF_HUB_CACHE=/hf_hub_cache/hub", "HF_XET_CACHE=/hf_hub_cache/xet",
    ]
    assert {"--jobid=42", "--container-workdir=/workspace", "--no-container-mount-home",
            "--container-remap-root", "--no-container-entrypoint", "--mpi=none"} <= set(container)
    assert container[-2:] == ["bash", COLLECTOR_PATH]
    assert cancel == ["scancel", "42"]


def test_collector_failure_propagates_and_still_cancels_the_allocation(
    fakes, workspace, tmp_path, monkeypatch
):
    monkeypatch.setenv("COLLECTOR_RC", "7")

    assert launch(cluster_for(tmp_path), request_for(workspace)) == 7
    assert fakes()[-1] == ["scancel", "42"]


def test_unavailable_staged_checkpoint_blocks_before_the_container_starts(
    fakes, workspace, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("FAKE_PROBE_RC", "1")

    assert launch(cluster_for(tmp_path), request_for(workspace)) == 1
    assert "readiness-blocked" in capsys.readouterr().err
    assert not any(arg.startswith("--container-image=") for call in fakes() for arg in call)
    assert fakes()[-1] == ["scancel", "42"]


def test_unstaged_model_resolves_under_the_download_root_without_a_node_probe(
    fakes, workspace, tmp_path
):
    assert launch(cluster_for(tmp_path), request_for(workspace, MODEL="org/Unstaged-Model")) == 0

    downloads = tmp_path / "downloads"
    assert downloads.is_dir()
    container = container_step(fakes())
    assert "MODEL_PATH=/models/Unstaged-Model" in option(container, "--export").split(",")
    assert f"{downloads}:/models" in option(container, "--container-mounts").split(",")
    assert not any(call[0] == "srun" and "test" in call for call in fakes())


def test_a_result_outside_the_container_workspace_fails_before_any_slurm_call(
    fakes, workspace, tmp_path, capsys
):
    request = request_for(workspace, OUT_YAML="/tmp/speedbench-reference-al.yaml")

    assert launch(cluster_for(tmp_path), request) == 1
    assert "lies outside the container workspace" in capsys.readouterr().err
    assert fakes() == []


def test_editable_install_images_mount_the_workspace_at_ix(
    fakes, workspace, tmp_path, monkeypatch
):
    monkeypatch.setenv("OUT_YAML", "/ix/speedbench-reference-al.yaml")
    request = request_for(
        workspace, IMAGE="lmsysorg/sglang:deepseek-v4-b300-dev", OUT_YAML="/ix/speedbench-reference-al.yaml"
    )

    assert launch(cluster_for(tmp_path), request) == 0

    # /workspace belongs to the image's editable install; the checkout is at /ix.
    assert (workspace / "speedbench-reference-al.yaml").exists()
    container = container_step(fakes())
    assert option(container, "--container-workdir") == "/ix"
    assert f"{workspace}:/ix" in option(container, "--container-mounts").split(",")


def test_node_local_images_are_imported_inside_the_allocation(fakes, workspace, tmp_path):
    cluster = cluster_for(
        tmp_path,
        slurm={"exclusive": False, "account": None, "exclude": []},
        squash={"visibility": "node-local", "import": "all-nodes"},
    )

    assert launch(cluster, request_for(workspace)) == 0

    allocation, importer = fakes()[:2]
    assert allocation[0] == "salloc"
    assert not {arg.split("=")[0] for arg in allocation} & {"--exclusive", "--account", "--exclude"}
    assert importer[0] == "srun" and {"--jobid=42", "--ntasks-per-node=1"} <= set(importer)


def test_salloc_exclude_escape_hatch_extends_the_cluster_exclusions(fakes, workspace, tmp_path):
    request = request_for(workspace, SALLOC_EXCLUDE="node[1,3]")

    assert launch(cluster_for(tmp_path), request) == 0

    allocation = next(call for call in fakes() if call[0] == "salloc")
    assert option(allocation, "--exclude") == "bad-node,node[1,3]"


def test_missing_gpu_count_fails_before_any_slurm_call(fakes, workspace, tmp_path, capsys):
    assert launch(cluster_for(tmp_path), request_for(workspace, GPU_COUNT="")) == 1
    assert "GPU_COUNT" in capsys.readouterr().err
    assert fakes() == []


def test_sigterm_during_the_collector_cancels_the_allocation_and_exits_143(
    fakes, workspace, tmp_path
):
    config = tmp_path / "runners.yaml"
    config.write_text(yaml.safe_dump(inventory_for(tmp_path)))
    started = tmp_path / "container-started"
    launcher = subprocess.Popen(
        [sys.executable, "-m", "infx.launch", "--runner-config", str(config), "run"],
        env={**os.environ, **launch_env(workspace), "PYTHONPATH": str(ROOT),
             "FAKE_HANG_MARKER": str(started)},
        cwd=workspace, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while not started.exists() and launcher.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert started.exists(), "the container step never started"

    launcher.send_signal(signal.SIGTERM)

    assert launcher.wait(timeout=60) == 143
    assert fakes()[-1] == ["scancel", "42"]
