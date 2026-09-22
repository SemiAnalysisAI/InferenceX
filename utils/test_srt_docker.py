"""Execute native command generation and the Docker server/client lifecycle."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
from infx.srt_slurm.docker import prepare


@pytest.mark.parametrize("local_model,eval_only,context", [("", "false", "256"), ("/cache/model with space", "true", "1024")])
def test_native_docker_commands_preserve_model_flags_and_literal_environment(tmp_path, local_model, eval_only, context):
    recipe = {
        "schema": 2, "name": "test", "engine": "sglang",
        "model": {"path": "hf:test/model", "container": "test:tag", "precision": "fp8"},
        "resources": {"gpu_type": "h200", "gpus_per_node": 8},
        "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 2,
            "args": {"tensor-parallel-size": 2, "context-length": 256, "served-model-name": "test/model"},
            "env": {"SGLANG_LITERAL": "$(touch injected); 'literal'", "SGLANG_SIMULATE_ACC_LEN": "2.5"}}},
        "benchmark": {"type": "custom", "command": "python3 capture-client",
            "env": {"MODEL": "test/model", "ISL": "128", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5", "USE_CHAT_TEMPLATE": "false"}},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    env = {
        "FRAMEWORK": "sglang", "MODEL": "test/model", "MODEL_PREFIX": "test", "MODEL_PATH": local_model,
        "IMAGE": "test:tag", "PRECISION": "fp8", "TP": "2", "GPU_COUNT": "2",
        "PP_SIZE": "1", "DCP_SIZE": "1", "PCP_SIZE": "1", "EP_SIZE": "1", "DP_ATTENTION": "false",
        "SPEC_DECODING": "none", "IS_AGENTIC": "0", "RUN_EVAL": "false", "EVAL_ONLY": eval_only,
        "MAX_MODEL_LEN": "1024", "ISL": "128", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5", "CONC": "3",
        "RESULT_FILENAME": "controlled", "GPU_MONITOR_INTERVAL": "1", "PORT": "9019",
    }
    server, client = prepare(str(path), env)
    stub = tmp_path / "python3"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\nprint(json.dumps({{'argv': sys.argv[1:], 'env': dict(os.environ)}}))\n")
    stub.chmod(0o755)
    runtime_env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    run = subprocess.run(["bash", "-c", server], cwd=tmp_path, env=runtime_env, capture_output=True, text=True, check=True)
    observed = json.loads(run.stdout)
    argv = observed["argv"]
    assert argv[:2] == ["-m", "sglang.launch_server"]
    assert argv[argv.index("--model-path") + 1] == (local_model or "test/model")
    assert argv[argv.index("--port") + 1] == "9019"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert argv[argv.index("--context-length") + 1] == context
    assert observed["env"]["SGLANG_LITERAL"] == "$(touch injected); 'literal'"
    assert "SGLANG_SIMULATE_ACC_LEN" not in observed["env"]
    assert not (tmp_path / "injected").exists()
    run = subprocess.run(["bash", "-c", client], cwd=tmp_path, env=runtime_env, capture_output=True, text=True, check=True)
    observed = json.loads(run.stdout)
    assert observed["argv"] == ["capture-client"]
    assert {key: observed["env"][key] for key in ("CONC", "RESULT_FILENAME", "SRT_FRONTEND_HOST", "SRT_FRONTEND_PORT")} == {
        "CONC": "3", "RESULT_FILENAME": "controlled", "SRT_FRONTEND_HOST": "127.0.0.1", "SRT_FRONTEND_PORT": "9019",
    }


@pytest.mark.parametrize("client_exit,ready", [(0, True), (7, True), (0, False)])
def test_docker_client_failure_and_readiness_clean_up_owned_server(tmp_path, client_exit, ready):
    workspace = tmp_path / "repo"
    scripts = workspace / "benchmarks/single_node"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "benchmarks/benchmark_lib.sh", scripts.parent / "benchmark_lib.sh")
    shutil.copyfile(ROOT / "benchmarks/single_node/srt_docker.sh", scripts / "srt_docker.sh")
    (workspace / "infx").symlink_to(ROOT / "infx", target_is_directory=True)
    (workspace / "srt-docker-server.sh").write_text('echo $$ > "$INFMAX_CONTAINER_WORKSPACE/server.pid"\nexec sleep 120\n' if ready else 'exit 12\n')
    (workspace / "srt-docker-client.sh").write_text(f'touch "$INFMAX_CONTAINER_WORKSPACE/client-ran"\nexit {client_exit}\n')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {"hf": 'printf "%s\\n" "$@" > "$INFMAX_CONTAINER_WORKSPACE/download"', "curl": f"exit {0 if ready else 1}", "sleep": "exit 0"}.items():
        # The server uses the real sleep; only readiness retry sleeps are accelerated.
        binary = binaries / name
        binary.write_text(f"#!/bin/bash\n{body}\n")
        binary.chmod(0o755)
    if ready:
        (workspace / "srt-docker-server.sh").write_text('echo $$ > "$INFMAX_CONTAINER_WORKSPACE/server.pid"\nexec /bin/sleep 120\n')
    else:
        # A log follower can take time to stop. Detach its output so EOF alone
        # cannot hide a missing wait in the wrapper's failed-readiness path.
        follower = binaries / "tail"
        follower.write_text(f"#!{sys.executable}\n" +
            "import os, pathlib, signal, time\n"
            "workspace = pathlib.Path(os.environ['INFMAX_CONTAINER_WORKSPACE'])\n"
            "null = os.open(os.devnull, os.O_WRONLY)\n"
            "os.dup2(null, 1); os.dup2(null, 2); os.close(null)\n"
            "def stop(*_):\n"
            "    time.sleep(0.1)\n"
            "    (workspace / 'follower-stopped').touch()\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "(workspace / 'follower-ready').touch()\n"
            "signal.pause()\n")
        follower.chmod(0o755)
        (binaries / "curl").write_text(
            '#!/bin/bash\nwhile [[ ! -e "$INFMAX_CONTAINER_WORKSPACE/follower-ready" ]]; do /bin/sleep 0.01; done\nexit 1\n')
    env = {**os.environ, "PATH": f"{binaries}:{Path(sys.executable).parent}:{os.environ['PATH']}",
           "INFMAX_CONTAINER_WORKSPACE": str(workspace), "MODEL": "test/model", "PORT": "9019",
           "RUN_EVAL": "false", "EVAL_ONLY": "false", "MODEL_PATH": ""}
    result = subprocess.run(["bash", str(scripts / "srt_docker.sh")], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == (client_exit if ready else 1), result.stdout + result.stderr
    assert (workspace / "client-ran").exists() is ready
    assert (workspace / "download").read_text() == "download\ntest/model\n"
    if not ready:
        assert (workspace / "follower-stopped").exists()
    if ready:
        pid = int((workspace / "server.pid").read_text())
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("Docker wrapper left its server process alive")


def test_rtx_launcher_binds_eval_model_and_preserves_container_failure(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (tmp_path / "benchmarks").symlink_to(ROOT / "benchmarks", target_is_directory=True)
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({
        "schema": 2, "name": "fixture", "engine": "sglang",
        "model": {"path": "hf:test/model", "container": "test:tag", "precision": "fp4"},
        "resources": {"gpu_type": "rtx6000pro", "gpus_per_node": 8},
        "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4,
                           "args": {"tensor-parallel-size": 4, "served-model-name": "test/model"}}},
        "benchmark": {"type": "custom", "command": "bash /infmax-workspace/benchmarks/single_node/srt_fixed_sequence.sh",
            "env": {"MODEL": "test/model", "ISL": "128", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5", "USE_CHAT_TEMPLATE": "false"}},
    }))
    scripts = {
        "git": 'if [[ "$1" == clone ]]; then mkdir -p "${@: -1}/configs"; else echo test-commit; fi',
        "uv": 'if [[ "$1" == venv ]]; then mkdir -p .venv/bin; echo ":" > .venv/bin/activate; fi',
    }
    for name, body in scripts.items():
        binary = binaries / name
        binary.write_text(f"#!/usr/bin/env bash\n{body}\n")
        binary.chmod(0o755)
    docker = binaries / "docker"
    docker.write_text(f"#!{sys.executable}\n" +
        "import json, os, pathlib, sys\n"
        "with pathlib.Path(os.environ['CAPTURE']).open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "sys.exit(7 if sys.argv[1] == 'run' else 0)\n")
    docker.chmod(0o755)
    env = {**os.environ, "PATH": f"{binaries}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "PYTHONPATH": f"{ROOT}:{ROOT / 'utils/srt-slurm/src'}", "GITHUB_WORKSPACE": str(tmp_path),
        "SRT_RECIPE": path.name, "FRAMEWORK": "sglang", "MODEL": "test/model", "MODEL_PREFIX": "test",
        "IMAGE": "test:tag", "PRECISION": "fp4", "TP": "4", "GPU_COUNT": "4", "PP_SIZE": "1",
        "DCP_SIZE": "1", "PCP_SIZE": "1", "EP_SIZE": "1", "DP_ATTENTION": "false", "SPEC_DECODING": "none",
        "IS_AGENTIC": "0", "RUN_EVAL": "true", "EVAL_ONLY": "true", "MAX_MODEL_LEN": "1024",
        "ISL": "128", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5", "CONC": "3", "RESULT_FILENAME": "point",
        "GPU_MONITOR_INTERVAL": "1", "PORT": "9019", "IS_MULTINODE": "false", "HF_HUB_CACHE_MOUNT": str(tmp_path / 'cache'),
        "HF_HUB_CACHE": "/hf", "NCCL_IB_DISABLE": "1", "EXP_NAME": "test_8k1k", "SCENARIO_SUBDIR": "fixed_seq_len/",
        "RUNNER_NAME": "fixture_00", "INFERENCEX_RUNTIME_ENV_VARS": "REQUIRE_POWER", "REQUIRE_POWER": "1",
        "CAPTURE": str(tmp_path / "docker.jsonl"), "MODEL_PATH": ""}
    result = subprocess.run(["bash", str(ROOT / 'runners/launch_rtx6000pro-lat.sh')], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 7, result.stdout + result.stderr
    calls = [json.loads(line) for line in Path(env["CAPTURE"]).read_text().splitlines()]
    run = next(call for call in calls if call[0] == 'run')
    assert "MODEL_NAME=test/model" in [run[i+1] for i,v in enumerate(run[:-1]) if v == '--env']
    assert run[-2:] == ["test:tag", "benchmarks/single_node/srt_docker.sh"]
    assert calls[-1] == ["rm", "-f", "bmk-server-fixture_00"]
    # Run the emitted script against an external Python stub to verify the
    # launcher's actual CLI path emitted the eval context and requested model.
    stub = binaries / "python3"
    stub.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    stub.chmod(0o755)
    observed = subprocess.run(["bash", str(tmp_path / "srt-docker-server.sh")], env=env, capture_output=True, text=True, check=True)
    argv = json.loads(observed.stdout)
    assert argv[argv.index('--context-length')+1] == '1024'
    assert argv[argv.index('--served-model-name')+1] == 'test/model'
