import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "runners" / "launch_b300-dsxe.sh"
BASH = Path(shutil.which("bash") or "/usr/bin/bash")
DIGEST = "sha256:" + "7" * 64


def test_single_node_pyxis_imports_pinned_image_inside_final_allocation(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun_args = tmp_path / "srun-args"

    commands = {
        "mkdir": "exit 0\n",
        "salloc": "exit 0\n",
        "squeue": "printf '777\\n'\n",
        "unsquashfs": "exit 0\n",
        "srun": 'printf "%s\\n" "$@" > "$B300_SRUN_ARGS"\n',
    }
    for name, body in commands.items():
        command = bin_dir / name
        command.write_text(f"#!/bin/sh\n{body}")
        command.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "B300_SRUN_ARGS": str(srun_args),
            "GITHUB_WORKSPACE": str(ROOT),
            "IS_MULTINODE": "false",
            "IMAGE": f"lmsysorg/sglang:v0.5.16-cu130@{DIGEST}",
            "MODEL": "Qwen/Qwen3.5-397B-A17B-FP8",
            "FRAMEWORK": "sglang",
            "PRECISION": "fp8",
            "SPEC_DECODING": "none",
            "SCENARIO_SUBDIR": "agentic/",
            "EXP_NAME": "qwen3.5_tp4_conc1_kvnone",
            "TP": "4",
            "RUNNER_NAME": "b300-test",
            "USER": "runner",
            "HF_HUB_CACHE": "/mnt/hf_hub_cache",
        }
    )

    result = subprocess.run(
        [BASH, LAUNCHER],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = srun_args.read_text().splitlines()
    assert (
        f"--container-image=docker://registry-1.docker.io#lmsysorg/sglang:{DIGEST}"
        in args
    )
    mounts = next(arg for arg in args if arg.startswith("--container-mounts="))
    assert "/scratch/models:/scratch/models" not in mounts
    assert "/data/home/sa-gha-runner/models:/data/home/sa-gha-runner/models" in mounts
