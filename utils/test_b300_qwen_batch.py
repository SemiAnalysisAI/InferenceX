"""Exercise B300 batch routing without submitting a Slurm job."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("overrides", "batch"),
    [
        ({}, True),
        ({"EVAL_ONLY": "true"}, True),
        ({"MODEL_PREFIX": "dsv41flash", "IS_AGENTIC": "1", "SPEC_DECODING": "mtp"}, True),
        ({"MODEL_PREFIX": "other"}, False),
        ({"PRECISION": "fp4"}, False),
        ({"SPEC_DECODING": "mtp"}, False),
        ({"IS_AGENTIC": "1"}, False),
        ({"IS_MULTINODE": "true"}, False),
        ({"FRAMEWORK": "vllm"}, False),
        ({"B300_AGENTX_BATCH": "1"}, False),
        ({"PRECISION": None}, False),
        ({"SPEC_DECODING": None}, False),
        ({"PRECISION": ""}, False),
        ({"SPEC_DECODING": ""}, False),
    ],
)
def test_batch_allocation_routing(
    tmp_path: Path, overrides: dict[str, str | None], batch: bool
) -> None:
    workspace = tmp_path / "workspace"
    for name in (
        "runners/launch_b300-dsxe.sh",
        "runners/slurm_utils.sh",
        "benchmarks/benchmark_lib.sh",
    ):
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    commands = tmp_path / "commands"
    commands.mkdir()
    for name in (
        "sbatch",
        "squeue",
        "sacct",
        "scancel",
        "mkdir",
        "tail",
        "sleep",
        "salloc",
        "srun",
        "enroot",
        "unsquashfs",
    ):
        stub = commands / name
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\nfrom pathlib import Path\n"
            "name = Path(sys.argv[0]).name\nargs = sys.argv[1:]\n"
            "capture = Path(os.environ['CAPTURE'])\n"
            "if name == 'sbatch':\n"
            "    capture.write_text(json.dumps(args))\n"
            "    log = next(a.removeprefix('--output=') for a in args if a.startswith('--output='))\n"
            "    Path(log).write_text('batch log fixture\\n')\n"
            "    print('12345;fixture')\n"
            "elif name == 'sacct': print('COMPLETED|0:0')\n"
            "elif name == 'scancel': capture.with_suffix('.cleanup').write_text(json.dumps(args))\n"
            "elif name in ('salloc', 'srun', 'enroot', 'unsquashfs'): sys.exit(97)\n"
        )
        stub.chmod(0o755)
    capture = tmp_path / "submitted.json"
    result = subprocess.run(
        ["bash", str(workspace / "runners/launch_b300-dsxe.sh")],
        cwd=workspace,
        env={
            key: value
            for key, value in {
                "PATH": f"{commands}:/usr/bin:/bin",
                "HOME": os.environ["HOME"],
                "ENROOT_IMPORT_TIME_LIMIT": "120",
                "SALLOC_TIME_LIMIT": "480",
                "EVAL_ONLY": "false",
                "RUN_EVAL": "false",
                "IS_MULTINODE": "false",
                "IS_AGENTIC": "0",
                "MODEL_PREFIX": "qwen3.5",
                "FRAMEWORK": "sglang",
                "PRECISION": "fp8",
                "SPEC_DECODING": "none",
                "B300_AGENTX_BATCH": "0",
                "GITHUB_WORKSPACE": str(workspace),
                "GPU_COUNT": "4",
                "RUNNER_NAME": "fixture_01",
                "SALLOC_EXCLUDE": "excluded-node",
                "CAPTURE": str(capture),
                **overrides,
            }.items()
            if value is not None
        },
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    assert capture.exists() is batch, result.stdout + result.stderr
    if batch:
        assert result.returncode == 0, result.stderr
        args = json.loads(capture.read_text())
        assert {
            "--partition=batch_1",
            "--account=benchmark",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:4",
            "--exclusive",
            "--mem=0",
            "--time=480",
            "--export=ALL",
            "--exclude=excluded-node",
            f"--chdir={workspace}",
        } <= set(args)
        assert json.loads(capture.with_suffix(".cleanup").read_text()) == ["12345"]
        assert not Path(args[-1]).exists()
    else:
        assert result.returncode == 1
        missing = [name for name, value in overrides.items() if value is None or value == ""]
        if missing:
            assert "required environment variables are not set" in result.stdout
            assert all(name in result.stdout for name in missing)
        else:
            assert (
                "B300_HF_CACHE_HOST_DIR" in result.stdout
                or "Unsupported framework" in result.stdout
            )
