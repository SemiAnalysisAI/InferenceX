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


def platforms(pool="h100-dgxc", gpus=8, architecture="linux/amd64"):
    return {pool: {"gpus_per_node": gpus, "image_platform": architecture}}


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
        platforms(),
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
            pool,
            backends,
            {"tiny": shapes},
            {"a": {"image": "image:1"}},
            worlds,
            chunk,
            platforms(),
        )


def test_plan_bounds_world_size_to_physical_arm_node():
    shapes = {
        "tiny": [
            {"type": "gemm", "args": {"m": 2}},
            {"type": "allreduce", "args": {"world_size": 8}},
        ]
    }
    hardware = platforms("gb200", 4, "linux/arm64")
    result = ci.plan(
        "gb200", ["torch"], shapes, {"torch": {"image": "image:1"}}, [1], 50, hardware
    )
    assert result["excluded_shapes"] == 1
    assert result["include"][0]["cases"] == [
        {"testlist": "tiny", "shape": shapes["tiny"][0]}
    ]
    assert result["include"][0]["image_platform"] == "linux/arm64"
    with pytest.raises(ValueError, match="4-GPU"):
        ci.plan(
            "gb200",
            ["torch"],
            shapes,
            {"torch": {"image": "image:1"}},
            [1, 8],
            50,
            hardware,
        )


def test_image_import_separates_architectures_and_refuses_wrong_host(
    tmp_path, monkeypatch
):
    trace = []
    scratch_paths = []

    def external(argv, **kwargs):
        trace.append(argv)
        if argv[0] == "enroot":
            Path(argv[3]).write_bytes(b"validated squash fixture")
            env = kwargs["env"]
            assert Path(env["ENROOT_TEMP_PATH"]).is_dir()
            scratch = Path(env["TMPDIR"])
            (scratch / "parallel-buffer").write_text("importer scratch")
            scratch_paths.append(scratch)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(ci.subprocess, "run", external)
    digest = "sha256:" + "a" * 64
    # Only registry traffic is replaced; use the production reference parser.
    probe = ci.probe_module()
    monkeypatch.setattr(probe, "resolve_image_digest", lambda image: digest)
    monkeypatch.setattr(ci, "probe_module", lambda: probe)
    args = types.SimpleNamespace(
        image="nvcr.io/nvidia/pytorch:test",
        digest=digest,
        cache=tmp_path / "cache",
        image_platform="linux/arm64",
    )
    monkeypatch.setattr(ci.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="import host"):
        ci.import_image(args)
    assert not args.cache.exists()

    for architecture, machine in [
        ("linux/amd64", "x86_64"),
        ("linux/arm64", "aarch64"),
    ]:
        args.image_platform = architecture
        monkeypatch.setattr(ci.platform, "machine", lambda: machine)
        ci.import_image(args)
        ci.import_image(args)  # Reuse a validated cache on the same architecture.
    images = list(args.cache.glob("*.sqsh"))
    assert len(images) == 2
    assert all(image.read_bytes() == b"validated squash fixture" for image in images)
    imports = [argv for argv in trace if argv[0] == "enroot"]
    assert len(imports) == 2
    assert imports[0][-1] == "docker://nvcr.io#nvidia/pytorch:test"
    assert all(not path.exists() for path in scratch_paths)


def test_shared_storage_uses_only_configured_writable_roots(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    assert ci.shared_base(
        {"storage_roots": [str(tmp_path / "absent"), str(shared)]}, "gb200"
    ) == (shared / f".operatorx-{os.getuid()}")
    monkeypatch.setenv("HOME", str(tmp_path / "runner-local-sandbox"))
    monkeypatch.setattr(
        ci.pwd, "getpwuid", lambda uid: types.SimpleNamespace(pw_dir=str(shared))
    )
    assert ci.shared_base({}, "b300") == shared / f".operatorx-{os.getuid()}"
    with pytest.raises(ValueError, match="shared storage"):
        ci.shared_base({"storage_roots": [str(tmp_path / "absent")]}, "gb200")


def test_amd_staging_uses_shared_runner_root(tmp_path, monkeypatch):
    runner = tmp_path / "runner"
    (runner / "_work/_temp").mkdir(parents=True)
    monkeypatch.setenv("RUNNER_TEMP", str(runner / "_work/_temp"))
    monkeypatch.setenv("HOME", str(tmp_path / "private-home"))
    assert ci.shared_base({}, "mi300x") == runner / f".operatorx-{os.getuid()}"
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "unrelated"))
    with pytest.raises(ValueError, match="shared runner"):
        ci.shared_base({}, "mi355x")


def test_platform_overlay_preserves_base_and_replaces_explicit_profile(tmp_path):
    base = tmp_path / "base.json"
    ci.write_json(
        base,
        {
            "platforms": {
                "fixture": {"gpus_per_node": 4, "operator": {"partition": "old"}}
            }
        },
    )
    child = tmp_path / "child.json"
    ci.write_json(
        child,
        {
            "base": "base.json",
            "platforms": {"fixture": {"operator": {"partition": "new"}}},
        },
    )
    assert ci.load_platforms(child) == {
        "fixture": {"gpus_per_node": 4, "operator": {"partition": "new"}}
    }


@pytest.mark.parametrize(
    "backend,worlds,kind",
    [("flashinfer", [1], "gemm"), ("torch", [2], "gemm"), ("torch", [1], "allreduce")],
)
def test_amd_plan_rejects_unimplemented_execution(backend, worlds, kind):
    with pytest.raises(ValueError, match="single-GPU torch GEMM"):
        ci.plan(
            "mi300x",
            [backend],
            {"tiny": [{"type": kind, "args": {}}]},
            {backend: {"image": "rocm:1"}},
            worlds,
            50,
            platforms("mi300x"),
        )


@pytest.mark.parametrize(
    "outcome,expected_rc",
    [("ok", 0), ("error", 1), ("unsupported", 1), ("unclaimed", 1)],
)
def test_strict_benchmark_writes_actual_status(
    tmp_path, monkeypatch, outcome, expected_rc
):
    # The GPU kernel is an external collaborator; selection, exception handling,
    # checkpointing, serialization and exit decisions execute the real main().
    backend = types.ModuleType("operatorx.runners.testgpu.backends.kernel")
    backend.IMPLS = (
        [] if outcome == "unclaimed" else [types.SimpleNamespace(op_type="gemm")]
    )
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
    assert body["rows"][0]["status"] == (
        "unsupported" if outcome == "unclaimed" else outcome
    )
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


@pytest.mark.parametrize(
    "exit_code,cancel,pool,gpus,architecture",
    [
        (0, False, "h100-dgxc", 8, "linux/amd64"),
        (3, False, "h100-dgxc", 8, "linux/amd64"),
        (0, True, "h100-dgxc", 8, "linux/amd64"),
        (0, "queued", "h100-dgxc", 8, "linux/amd64"),
        (0, False, "gb200", 4, "linux/arm64"),
        (0, False, "gb300", 4, "linux/arm64"),
        (0, False, "b300", 8, "linux/amd64"),
        (0, False, "mi300x", 8, "linux/amd64"),
        (0, False, "mi355x", 8, "linux/amd64"),
    ],
)
def test_allocation_completion_failure_and_cancellation(
    tmp_path, exit_code, cancel, pool, gpus, architecture
):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    stub = """#!PYTHON
import json, os, pathlib, sys, time
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['TRACE'], 'a') as f: f.write(name + '\\n')
with open(os.environ['TRACE_ARGS'], 'a') as f:
    row = {'argv': sys.argv, 'cache': os.environ.get('ENROOT_CACHE_PATH')}
    f.write(json.dumps(row) + '\\n')
if name == 'salloc':
    if os.environ['QUEUED'] == '1':
        print('salloc: Pending job allocation 12345', flush=True)
        pathlib.Path(os.environ['READY']).touch()
        time.sleep(60)
    else: print('salloc: Granted job allocation 12345')
if name == 'squeue':
    if '-j' in sys.argv:
        print('slurm_load_jobs error: Invalid job id specified', file=sys.stderr)
        sys.exit(1)
    print('99999')
if name == 'srun' and sys.argv[-1] == 'rank':
    mount_arg = next(x for x in sys.argv if x.startswith('--container-mounts='))
    mount = mount_arg.split('=',1)[1].split(':')[0]
    out = pathlib.Path(mount) / 'results' / 'partial.json'
    out.write_text('{"rows":[{"status":"ok"}]}')
    pathlib.Path(os.environ['READY']).touch()
    if os.environ['CANCEL'] == '1': time.sleep(60)
    sys.exit(int(os.environ['EXIT_CODE']))
"""
    stub = stub.replace("#!PYTHON", f"#!{sys.executable}")
    for name in ("salloc", "srun", "scancel", "squeue", "python3"):
        path = binaries / name
        path.write_text(stub)
        path.chmod(0o755)
    (tmp_path / "shared").mkdir()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "platforms": {
                    pool: {
                        "gpus_per_node": gpus,
                        "image_platform": architecture,
                        "operator": {
                            "partition": "test",
                            "stage_dir": str(tmp_path / "shared"),
                            "account": "fixture",
                            "cpus_per_node": 128,
                            **({"qos": "fixture-qos"} if pool != "b300" else {}),
                            "exclude_nodes": "quarantined",
                            "enroot_cache_path": str(tmp_path / "shared/enroot"),
                            **(
                                {"storage_roots": [str(tmp_path / "shared")]}
                                if pool == "gb200"
                                else {"squash_dir": str(tmp_path / "shared/squash")}
                            ),
                        },
                    }
                }
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    control = ci.plan(
        pool,
        ["torch"],
        {"tiny": [{"type": "gemm", "args": {"m": 2}}]},
        {"torch": {"image": "image:1"}},
        [1],
        1,
        platforms(pool, gpus, architecture),
    )
    control.update(source_sha="abc", run_id="12")
    control["include"][0]["digest"] = "sha256:" + "a" * 64
    manifest.write_text(json.dumps(control))
    output = tmp_path / "output"
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        TRACE=str(tmp_path / "trace"),
        TRACE_ARGS=str(tmp_path / "trace-args"),
        READY=str(tmp_path / "ready"),
        EXIT_CODE=str(exit_code),
        CANCEL=str(int(bool(cancel))),
        QUEUED=str(int(cancel == "queued")),
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
            "--cleanup-seconds",
            "2",
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
    if cancel == "queued":
        assert not (output / "results/partial.json").exists()
    else:
        assert json.loads((output / "results/partial.json").read_text())["rows"] == [
            {"status": "ok"}
        ]
    assert "scancel" in (tmp_path / "trace").read_text().splitlines()
    stage = Path(json.loads((output / "execution.json").read_text())["stage"])
    assert not stage.exists()
    calls = [
        json.loads(line) for line in (tmp_path / "trace-args").read_text().splitlines()
    ]
    allocation = next(c["argv"] for c in calls if Path(c["argv"][0]).name == "salloc")
    assert f"--gres=gpu:{gpus}" in allocation
    assert "--nodes=1" in allocation
    assert "--account=fixture" in allocation
    if pool == "b300":
        assert not any(arg.startswith("--qos=") for arg in allocation)
    else:
        assert "--qos=fixture-qos" in allocation
    assert "--exclude=quarantined" in allocation
    if pool in {"mi300x", "mi355x"}:
        assert "--cpus-per-task=16" in allocation
    if not cancel:
        imported = next(c for c in calls if "import" in c["argv"])
        assert imported["cache"] == str(tmp_path / "shared/enroot")
        assert imported["argv"][-2:] == ["--image-platform", architecture]
        assert Path(imported["argv"][0]).name == "srun"
        launched = next(c["argv"] for c in calls if c["argv"][-1] == "rank")
        assert "--ntasks=1" in launched
        if pool in ("gb200", "gb300", "b300", "mi300x", "mi355x"):
            assert "--container-remap-root" in launched
        if pool == "mi300x":
            mounts = next(
                arg for arg in launched if arg.startswith("--container-mounts=")
            )
            assert mounts.endswith(",/dev/kfd:/dev/kfd,/dev/dri:/dev/dri")


def test_summary_uses_latest_attempt_and_reports_missing_coverage(tmp_path):
    manifest = {
        "run_id": "12",
        "source_sha": "abc",
        "include": [{"id": "one", "cases": [{}, {}]}, {"id": "two", "cases": [{}]}],
    }
    for attempt, status in [(1, "error"), (2, "ok")]:
        root = tmp_path / str(attempt)
        ci.write_json(
            root / "execution.json",
            {
                "run_id": "12",
                "source_sha": "abc",
                "attempt": str(attempt),
                "cell": {"id": "one"},
            },
        )
        ci.write_json(root / "status.json", {"exit_code": 0 if attempt == 2 else 1})
        ci.write_json(
            root / "results/run.json",
            {"rows": [{"status": status}, {"status": "unsupported"}]},
        )
    report = ci.summarize(manifest, tmp_path)
    assert report == {
        "success": False,
        "shards": [
            {
                "shard": "one",
                "requested_shapes": 2,
                "status": "success",
                "ok": 1,
                "unsupported": 1,
                "error": 0,
                "attempt": 2,
            },
            {
                "shard": "two",
                "requested_shapes": 1,
                "status": "missing",
                "ok": 0,
                "unsupported": 0,
                "error": 0,
            },
        ],
    }
    manifest["source_sha"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        ci.summarize(manifest, tmp_path)


def test_recovery_refuses_unrelated_pool_or_storage(tmp_path):
    artifacts = tmp_path / "artifacts"
    (tmp_path / "shared").mkdir()
    profile = tmp_path / "platforms.json"
    ci.write_json(
        profile,
        {
            "platforms": {
                "h100-dgxc": {
                    "operator": {"squash_dir": str(tmp_path / "shared/squash")}
                }
            }
        },
    )
    data = {
        "run_id": "12",
        "cell": {"pool": "h200-dgxc"},
        "stage": str(tmp_path / "unrelated"),
    }
    ci.write_json(artifacts / "execution.json", data)
    with pytest.raises(ValueError, match="run/pool"):
        ci.recover(artifacts, "12", "h100-dgxc", profile, 2)
    data["cell"]["pool"] = "h100-dgxc"
    ci.write_json(artifacts / "execution.json", data)
    with pytest.raises(ValueError, match="pool/user"):
        ci.recover(artifacts, "12", "h100-dgxc", profile, 2)


@pytest.mark.parametrize("release", [True, False])
def test_cleanup_waits_for_delayed_release_but_stays_bounded(
    tmp_path, monkeypatch, release
):
    (tmp_path / "allocation.log").write_text("Granted job allocation 12345\n")
    clock = [0]
    monkeypatch.setattr(ci.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        ci.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )

    def slurm(argv, **kwargs):
        active = "12345\n" if not release or clock[0] < 30 else ""
        return subprocess.CompletedProcess(argv, 0, active, "")

    monkeypatch.setattr(ci.subprocess, "run", slurm)
    if release:
        ci.cleanup(tmp_path, 180)
        assert (tmp_path / "cleanup.log").read_text().splitlines()[-1] == (
            "job=12345 rc=0 active='' error=''"
        )
        assert clock[0] == 30
    else:
        with pytest.raises(RuntimeError, match="retaining staged evidence"):
            ci.cleanup(tmp_path, 180)
        assert clock[0] == 180


def test_amd_plan_keeps_requested_moe_shard_factors_with_one_gpu():
    result = ci.plan(
        "mi300x", ["vllm"],
        {"tiny": [
            {"type": "moe_gemm", "args": {"num_tokens": 16, "expert_parallel_size": 8}},
            {"type": "moe_gemm", "args": {"num_tokens": 32, "world_size": 2}},
        ]},
        {"vllm": {"image": "rocm:fixture"}}, [1], 50, platforms("mi300x"),
    )
    cell = result["include"][0]
    assert cell["world_size"] == cell["nodes"] == 1
    assert result["excluded_shapes"] == 1
    assert len(result["include"]) == len(cell["cases"]) == 1
    assert cell["cases"][0]["shape"]["args"]["num_tokens"] == 16
