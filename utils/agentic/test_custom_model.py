"""GPU-free checks of custom metadata and the real Bash/launch/replay boundary."""

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from utils.agentic.custom_model import native_context_length, verify_model_config
from utils.agentic.server_launch import apply_launch_args, digest, validate_overrides

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"max_position_embeddings": 40960}, 40960),
        ({"text_config": {"max_position_embeddings": 32768}}, 32768),
        ({"max_position_embeddings": 40960, "seq_length": 8192}, 8192),
    ],
)
def test_native_context(metadata, expected):
    assert native_context_length(metadata) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"max_position_embeddings": True},
        {"max_position_embeddings": -1},
        {"text_config": []},
    ],
)
def test_unknown_context_is_rejected(metadata):
    with pytest.raises(ValueError):
        native_context_length(metadata)


def test_verifier_binds_metadata_bytes_and_context(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"max_position_embeddings":40960}')
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_model_config(path, checksum, 40960, 8192)
    with pytest.raises(ValueError, match="within"):
        verify_model_config(path, checksum, 40960, 50000)
    with pytest.raises(ValueError, match="does not match"):
        verify_model_config(path, checksum, 32768, 8192)
    path.write_text('{"max_position_embeddings":32768}')
    with pytest.raises(ValueError, match="changed"):
        verify_model_config(path, checksum, 40960, 8192)


def test_pure_argv_contract_requires_no_source_or_executable_access():
    base = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--tp",
        "1",
        "--context-length",
        "8192",
        "--foo",
        "old",
    ]
    overrides = validate_overrides(
        {
            "version": 1,
            "executable": "/not/installed/python",
            "source_files": {"/not/present.py": "0" * 64},
            "replace_args": True,
            "append_args": ["--foo", "new"],
        }
    )
    assert apply_launch_args(base, overrides, "sglang") == [
        "/not/installed/python"
    ] + base[1:7] + ["--foo", "new"]


def test_generic_manifest_declares_only_supported_single_node_launchers():
    generic = json.loads((ROOT / "configs/agentx-launchers.json").read_text())[
        "generic"
    ]
    assert len(generic) == 6
    for framework in ("sglang", "vllm"):
        for runner in ("mi300x", "mi325x", "mi355x"):
            row = generic[f"custom-{framework}-{runner}"]
            assert row == {
                "framework": framework,
                "runner_type": runner,
                "benchmark_script": f"single_node/agentic/generic_{framework}.sh",
                "launch_overrides_version": 1,
                "max_gpus": 8,
            }
            path = ROOT / "benchmarks" / row["benchmark_script"]
            subprocess.run(["bash", "-n", str(path)], check=True)
            assert f"run_generic_agentic_server {framework}" in path.read_text()


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_generic_bash_builds_canonical_replay_and_verified_launch(tmp_path, framework):
    model = tmp_path / "model files with spaces"
    model.mkdir()
    metadata = model / "config.json"
    metadata.write_text('{"max_position_embeddings":40960}')
    output = tmp_path / "results with spaces"
    output.mkdir()
    fake_server = tmp_path / "fake server"
    fake_server.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n")
    fake_server.chmod(0o755)
    overrides = validate_overrides(
        {
            "version": 1,
            "executable": str(fake_server),
            "append_args": ["--test-value", "literal space"],
            "env": {"SGLANG_TEST": "verified"},
        }
    )
    request = output / "request.json"
    request.write_text(json.dumps(overrides))
    evidence = output / "server_launch.json"
    env = dict(
        os.environ,
        MODEL="Qwen/Qwen3-0.6B",
        MODEL_PATH=str(model),
        TP="1",
        EP_SIZE="1",
        CONC="64",
        PRECISION="bf16",
        FRAMEWORK=framework,
        DURATION="3600",
        RESULT_DIR=str(output),
        KV_OFFLOADING="none",
        IS_AGENTIC="1",
        AGENTX_CUSTOM_RECIPE="1",
        AGENTX_NATIVE_CONTEXT_LENGTH="40960",
        AGENTX_MAX_MODEL_LEN="8192",
        AGENTX_MODEL_CONFIG_SHA256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        AGENTX_LAUNCH_OVERRIDES_FILE=str(request),
        AGENTX_LAUNCH_OVERRIDES_SHA256=digest(overrides),
        AGENTX_SERVER_LAUNCH_FILE=str(evidence),
        MAX_MODEL_LEN="123",
        AIPERF_EXPERIMENTAL_FAST="0",
    )
    # Stub external services only. The real library verifies metadata, composes
    # server argv, applies launch edits, and composes the canonical replay CLI.
    script = f"""
set -eo pipefail
source {shlex.quote(str(ROOT / "benchmarks/benchmark_lib.sh"))}
install_agentic_deps() {{ AIPERF_CLI=aiperf; }}
resolve_trace_source() {{ :; }}
wait_for_server_ready() {{ :; }}
run_agentic_replay_and_write_outputs() {{ printf '%s\\n' "$REPLAY_CMD" > "$RESULT_DIR/replay.txt"; }}
run_benchmark_serving() {{ echo "wrong fixed-length client" >&2; exit 91; }}
run_generic_agentic_server {framework}
"""
    completed = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(evidence.read_text())
    assert receipt["effective_argv"][-2:] == ["--test-value", "literal space"]
    assert receipt["effective_env"] == {"SGLANG_TEST": "verified"}
    assert str(model) in receipt["base_argv"]
    context_arg = "--context-length" if framework == "sglang" else "--max-model-len"
    assert receipt["base_argv"][receipt["base_argv"].index(context_arg) + 1] == "8192"
    replay = (output / "replay.txt").read_text()
    assert "--scenario inferencex-agentx-mvp" in replay
    assert "--benchmark-duration 3600" in replay
    assert "--max-context-length 8192" in replay
    assert "--concurrency 64" in replay
    assert "--random-seed 42" in replay
    assert "--warmup-requests-per-lane 10" in replay
