"""Exercise CPU control paths with tiny inputs and external Slurm/GPU substitutes."""

import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operatorx import ci
from operatorx import main as benchmark
from operatorx.core import Result, UnsupportedOpError


def test_plan_chunks_and_preserves_moe_groups():
    ordinary = {"type": "gemm", "args": {"m": 2}}
    moe = {"type": "moe_forward", "args": {"world_size": 4, "expert_parallel_size": 2}}
    multi = {"type": "allreduce", "args": {"world_size": 16}}
    result = ci.plan(
        "h100-dgxc",
        ["a", "b"],
        {"small": [ordinary, ordinary, moe, multi]},
        {"a": {"image": "same:1"}, "b": {"image": "same:1"}},
        [1, 4],
        1,
    )
    cells = result["include"]
    assert [(c["world_size"], c["moe"], len(c["cases"])) for c in cells] == [
        (1, (), 1),
        (1, (), 1),
        (4, (2, 1, 1), 1),
    ]
    assert result["excluded_shapes"] == 1
    assert cells[2]["backends"] == ["a", "b"]
    assert len({c["id"] for c in cells}) == 3


@pytest.mark.parametrize(
    "pool,backends,worlds,shapes,chunk",
    [
        ("b200", ["a"], [1], [{"type": "gemm", "args": {}}], 1),
        ("h100-dgxc", ["missing"], [1], [{"type": "gemm", "args": {}}], 1),
        ("h100-dgxc", ["a"], [16], [{"type": "gemm", "args": {}}], 1),
        ("h100-dgxc", ["a"], [1], [], 1),
        ("h100-dgxc", ["a"], [1], [{"type": "gemm", "args": {}}] * 257, 1),
    ],
)
def test_plan_rejects_unexecutable_selection(pool, backends, worlds, shapes, chunk):
    with pytest.raises(ValueError):
        ci.plan(
            pool, backends, {"tiny": shapes}, {"a": {"image": "image:1"}}, worlds, chunk
        )


@pytest.mark.parametrize(
    "outcome,expected_rc", [("ok", 0), ("error", 1), ("unsupported", 1)]
)
def test_strict_benchmark_writes_actual_status(
    tmp_path, monkeypatch, outcome, expected_rc
):
    # The GPU kernel is an external collaborator; selection, exception handling,
    # checkpointing, serialization and exit decisions execute the real main().
    backend = types.ModuleType("operatorx.runners.testgpu.backends.kernel")
    backend.IMPLS = [types.SimpleNamespace(op_type="gemm")]
    runner = types.ModuleType("operatorx.runners.testgpu.runner")

    def kernel(op):
        if outcome == "error":
            raise RuntimeError("device failed")
        if outcome == "unsupported":
            raise UnsupportedOpError("dtype unsupported")
        return Result(op=op, metrics={"latency_us": 12.5})

    runner.run = kernel
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("OPERATORX_MOE_PARALLELISM", raising=False)
    (tmp_path / "tiny.json").write_text(
        json.dumps([{"type": "gemm", "args": {"m": 2}}])
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "operatorx",
            "--platform",
            "testgpu",
            "--backends",
            "kernel",
            "--testlist-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "output"),
            "--strict",
        ],
    )
    assert benchmark.main() == expected_rc
    body = json.loads(next((tmp_path / "output").rglob("*.json")).read_text())
    assert body["rows"][0]["status"] == outcome
    assert body["rows"][0]["metrics"] == (
        {"latency_us": 12.5} if outcome == "ok" else {}
    )


def test_testlist_loading_and_unknown_selection(tmp_path):
    (tmp_path / "one.json").write_text('[{"type":"gemm","args":{"m":7}}]')
    assert benchmark._load_testlists(["one"], tmp_path) == {
        "one": [{"type": "gemm", "args": {"m": 7}}]
    }
    with pytest.raises(SystemExit, match="unknown testlist"):
        benchmark._load_testlists(["two"], tmp_path)


@pytest.mark.parametrize("exit_code,cancel", [(0, False), (3, False), (0, True)])
def test_allocation_completion_failure_and_cancellation(tmp_path, exit_code, cancel):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    stub = """#!/usr/bin/env python3
import json, os, pathlib, sys, time
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['TRACE'], 'a') as f: f.write(name + '\\n')
if name == 'salloc': print('salloc: Granted job allocation 12345')
if name == 'srun' and sys.argv[-1] == 'rank':
    mount = next(x for x in sys.argv if x.startswith('--container-mounts=')).split('=',1)[1].split(':')[0]
    out = pathlib.Path(mount) / 'results' / 'partial.json'
    out.write_text('{"rows":[{"status":"ok"}]}')
    pathlib.Path(os.environ['READY']).touch()
    if os.environ['CANCEL'] == '1': time.sleep(60)
    sys.exit(int(os.environ['EXIT_CODE']))
"""
    for name in ("salloc", "srun", "scancel", "squeue"):
        path = binaries / name
        path.write_text(stub)
        path.chmod(0o755)
    (tmp_path / "shared").mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "platforms": {
                    "h100-dgxc": {
                        "operator": {
                            "partition": "test",
                            "squash_dir": str(tmp_path / "shared/squash"),
                        }
                    }
                }
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    control = ci.plan(
        "h100-dgxc",
        ["torch"],
        {"tiny": [{"type": "gemm", "args": {"m": 2}}]},
        {"torch": {"image": "image:1"}},
        [1],
        1,
    )
    control.update(source_sha="abc", run_id="12")
    control["include"][0]["digest"] = "sha256:" + "a" * 64
    manifest.write_text(json.dumps(control))
    output = tmp_path / "output"
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        TRACE=str(tmp_path / "trace"),
        READY=str(tmp_path / "ready"),
        EXIT_CODE=str(exit_code),
        CANCEL=str(int(cancel)),
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(ci.ROOT / "ci.py"),
            "execute",
            "--platform-config",
            str(profile),
            "--manifest",
            str(manifest),
            "--shard",
            control["include"][0]["id"],
            "--output",
            str(output),
            "--run-id",
            "12",
            "--attempt",
            "1",
            "--source-sha",
            "abc",
            "--runner-name",
            "test_runner",
            "--time-minutes",
            "1",
        ],
        env=env,
    )
    try:
        if cancel:
            deadline = time.monotonic() + 10
            while (
                not (tmp_path / "ready").exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            assert (tmp_path / "ready").exists()
            process.send_signal(signal.SIGTERM)
        rc = process.wait(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert (rc == 0) == (exit_code == 0 and not cancel)
    assert json.loads((output / "results/partial.json").read_text())["rows"] == [
        {"status": "ok"}
    ]
    assert "scancel" in (tmp_path / "trace").read_text().splitlines()
    stage = Path(json.loads((output / "execution.json").read_text())["stage"])
    assert not stage.exists()
