import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model_prefix", ["kimik3", "dsv4"])
def test_b300_staged_target_keeps_kimi_draft_in_persistent_mount(
    tmp_path: Path, model_prefix: str,
) -> None:
    log = tmp_path / "launch.jsonl"
    result = subprocess.run(
        ["bash", "-c", '''
        mkdir() { :; }
        unsquashfs() { return 0; }
        salloc() { echo 'salloc: Granted job allocation 123' >&2; }
        scancel() { :; }
        srun() {
            python3 -c 'import json,os,sys; open(sys.argv[1], "a").write(json.dumps({"args":sys.argv[2:], "draft_root":os.environ.get("WRITABLE_MODELS_DIR")})+"\\n")' "$SRUN_LOG" "$@"
        }
        unset WRITABLE_MODELS_DIR
        export MODEL_PREFIX="$3" PRECISION=fp4 FRAMEWORK=vllm
        export MODEL=moonshotai/Kimi-K3 IS_MULTINODE=false
        export SPEC_DECODING=mtp TP=8 RUNNER_NAME=b300-test IS_AGENTIC=1
        export SCENARIO_SUBDIR=agentic/ EXP_NAME="${MODEL_PREFIX}_tp8_conc1"
        export IMAGE=vllm/test:fixture GITHUB_WORKSPACE="$1" SRUN_LOG="$2"
        cd "$GITHUB_WORKSPACE"
        source runners/launch_b300-dsxe.sh
        ''', "bash", str(REPO_ROOT), str(log), model_prefix],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stderr
    serve = json.loads(log.read_text().splitlines()[-1])
    mounts = next(arg for arg in serve["args"] if arg.startswith("--container-mounts="))
    assert "/scratch/models:/scratch/models" in mounts
    if model_prefix == "kimik3":
        draft_root = serve["draft_root"]
        assert draft_root
        assert f"{draft_root}:{draft_root}" in mounts
    else:
        assert serve["draft_root"] is None
        assert "/data/home/sa-gha-runner/models" not in mounts


@pytest.mark.parametrize("first_exit", [0, 23])
def test_kimi_shared_draft_waits_for_download_completion(tmp_path: Path, first_exit: int) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text("{}")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    hf = binaries / "hf"
    hf.write_text('''#!/usr/bin/env python3
import os, pathlib, sys, time
root = pathlib.Path(os.environ["TEST_ROOT"])
draft = pathlib.Path(sys.argv[-1])
draft.mkdir(exist_ok=True)
(draft / "config.json").write_text("{}")
(root / (os.environ["CELL"] + ".started")).touch()
while not (root / "release").exists():
    time.sleep(0.01)
if os.environ["CELL"] == "first" and os.environ["FIRST_EXIT"] != "0":
    raise SystemExit(int(os.environ["FIRST_EXIT"]))
(draft / "weights.safetensors").write_text("complete")
''')
    hf.chmod(0o755)
    smi = binaries / "nvidia-smi"
    smi.write_text('''#!/usr/bin/env python3
import os, pathlib
complete = pathlib.Path(os.environ["WRITABLE_MODELS_DIR"], "Kimi-K3-DSpark", "weights.safetensors").is_file()
pathlib.Path(os.environ["TEST_ROOT"], os.environ["CELL"] + ".serving").touch()
raise SystemExit(77 if complete else 66)
''')
    smi.chmod(0o755)
    lock = binaries / "flock"
    native_flock = shutil.which("flock")
    lock.write_text(f'''#!/usr/bin/env python3
import fcntl, os, pathlib, subprocess, sys
pathlib.Path(os.environ["TEST_ROOT"], os.environ["CELL"] + ".locking").touch()
if {native_flock!r}:
    raise SystemExit(subprocess.run([{native_flock!r}, *sys.argv[1:]]).returncode)
# macOS lacks the Linux CLI; use the same OS advisory lock for this fixture.
with open(sys.argv[3], "a") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    raise SystemExit(subprocess.run(sys.argv[4:]).returncode)
''')
    lock.chmod(0o755)
    env = {
        **os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
        "FIRST_EXIT": str(first_exit), "TEST_ROOT": str(tmp_path), "MODEL": "moonshotai/Kimi-K3", "TP": "8", "CONC": "1",
        "KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "mooncake", "TOTAL_CPU_DRAM_GB": "1024", "DURATION": "1",
        "RESULT_DIR": str(tmp_path / "result"), "MODEL_PATH": str(target),
        "WRITABLE_MODELS_DIR": str(cache), "DRAFT_MODEL": "Inferact/Kimi-K3-DSpark",
    }
    command = ["bash", str(REPO_ROOT / "benchmarks/single_node/agentic/kimik3_fp4_b300_vllm_mtp.sh")]
    processes = []
    try:
        first = subprocess.Popen(command, env={**env, "CELL": "first"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(first)
        deadline = time.monotonic() + 5
        while not (tmp_path / "first.started").exists():
            assert first.poll() is None, "first downloader exited before the barrier"
            assert time.monotonic() < deadline
            time.sleep(0.01)
        second = subprocess.Popen(command, env={**env, "CELL": "second"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(second)
        deadline = time.monotonic() + 5
        while not any((tmp_path / f"second.{stage}").exists() for stage in ("locking", "started", "serving")):
            assert second.poll() is None, "second cell exited before staging"
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert not (tmp_path / "second.serving").exists(), "another cell accepted the nonempty partial draft"
        assert not (tmp_path / "second.started").exists(), "downloaders must serialize"
        (tmp_path / "release").touch()
        assert first.wait(timeout=5) == (first_exit or 77)
        assert second.wait(timeout=5) == 77
        assert (tmp_path / "second.started").exists()
    finally:
        (tmp_path / "release").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
