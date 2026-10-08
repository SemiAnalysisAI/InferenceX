"""Setup scripts that install a master-versioned component, run with a recording python3."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infx.srt_slurm.workload import INSTALLERS, VERSION_ENV
from infx.tests.bench.stubs import executable

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "benchmarks/multi_node/srt-slurm-recipes/configs"
# The container's python3: records each call; its pip accepts --break-system-packages.
PYTHON3 = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["STUB_PYTHON3_TRACE"], "a") as trace:
    trace.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:] == ["-m", "pip", "install", "--help"]:
    print("  --break-system-packages")
"""
# The container mounts this checkout at /infmax-workspace.
MOUNT = 'source() { builtin source "${1/#\\/infmax-workspace/$STUB_WORKSPACE}" "${@:2}"; }\n'


def run(tmp_path: Path, script: str, env: dict[str, str]) -> tuple[subprocess.CompletedProcess, list]:
    """Run ``script`` as srtctl does; return the result and the pip installs it ran."""
    (tmp_path / "bin").mkdir()
    executable(tmp_path / "bin/python3", PYTHON3)
    (tmp_path / "mount.sh").write_text(MOUNT)
    trace = tmp_path / "python3.jsonl"
    result = subprocess.run(
        ["bash", str(CONFIGS / script)],
        env={
            "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "BASH_ENV": str(tmp_path / "mount.sh"),
            "STUB_WORKSPACE": str(ROOT),
            "STUB_PYTHON3_TRACE": str(trace),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    return result, [call for call in calls if call[:3] == ["-m", "pip", "install"] and "--help" not in call]


@pytest.mark.parametrize("script", sorted(INSTALLERS))
def test_an_installer_fails_without_the_version_the_binder_writes(tmp_path, script):
    result, installs = run(tmp_path, script, {})

    assert result.returncode == 1
    assert f"  - {VERSION_ENV[INSTALLERS[script][0]]}\n" in result.stdout
    assert installs == []


@pytest.mark.parametrize(("script", "env", "expected"), [
    ("vllm-router.sh", {"ROUTER_VERSION": "9.8.7"}, {"vllm-router==9.8.7"}),
    ("vllm-mooncake.sh", {"KV_OFFLOAD_BACKEND_VERSION": "9.8.7"},
     {"mooncake-transfer-engine-cuda13==9.8.7"}),
    ("lmcache-mp-rocm.sh", {"KV_OFFLOAD_BACKEND_VERSION": "9.8.7"},
     {"lmcache==9.8.7", "https://github.com/LMCache/LMCache/releases/expanded_assets/v9.8.7-rocm"}),
    ("glm5.3-tilert-rocm.sh", {"ROUTER_VERSION": "9.8.7", "TILERT_ROLE": "prefill"}, {"tilert==9.8.7"}),
])  # fmt: skip
def test_an_installer_installs_exactly_the_bound_version(tmp_path, script, env, expected):
    result, installs = run(tmp_path, script, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert expected <= set(installs[0])
