"""GPU-free tests for the real AgentX server launch boundary."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from utils.agentic.server_launch import (
    PROTECTED_ARGS,
    digest,
    prepare_launch,
    validate_overrides,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = [
    sys.executable,
    "-m",
    "sglang.launch_server",
    "--model-path",
    "/models/a b",
    "--host",
    "0.0.0.0",
    "--port",
    "8888",
    "--tp",
    "4",
    "--mem-fraction-static",
    "0.85",
    "--enable-metrics",
]


def request(**kwargs):
    return validate_overrides({"version": 1, **kwargs})


def test_empty_extension_preserves_every_command_token_and_environment():
    command, receipt = prepare_launch(
        BASE, request(), "sglang", {"PATH": os.environ["PATH"]}
    )
    assert command == BASE
    assert receipt["effective_argv"] == BASE
    assert receipt["effective_env"] == {}
    assert receipt["evidence_sha256"] == digest(
        {k: v for k, v in receipt.items() if k != "evidence_sha256"}
    )


def test_remove_then_append_keeps_quoted_values_and_protocol():
    command, receipt = prepare_launch(
        BASE,
        request(
            remove_args=["--mem-fraction-static"],
            append_args=["--mem-fraction-static", "0.8", "--json", '{"spaces": "a b"}'],
        ),
        "sglang",
    )
    assert "0.85" not in command
    assert command[-4:] == [
        "--mem-fraction-static",
        "0.8",
        "--json",
        '{"spaces": "a b"}',
    ]
    assert receipt["base_argv"] == BASE
    assert command[:11] == BASE[:11]


def test_remove_equals_form_and_negative_multivalue_arguments():
    command, _ = prepare_launch(
        BASE + ["--test=-1"], request(remove_args=["--test"]), "sglang"
    )
    assert command == BASE
    command, _ = prepare_launch(
        BASE + ["--test", "-1", "-2.5"], request(remove_args=["--test"]), "sglang"
    )
    assert command == BASE


def test_replace_preserves_protocol_options_and_executable_positionals():
    command, _ = prepare_launch(
        BASE, request(replace_args=True, append_args=["--foo", "bar"]), "sglang"
    )
    assert command == BASE[:11] + ["--foo", "bar"]


@pytest.mark.parametrize("name", sorted(PROTECTED_ARGS))
def test_protocol_options_cannot_be_added_or_removed(name):
    with pytest.raises(ValueError, match="protocol option"):
        request(remove_args=[name])
    with pytest.raises(ValueError, match="cannot change"):
        prepare_launch(BASE, request(append_args=[f"{name}=different"]), "sglang")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remove_args": ["--missing"]},
        {"append_args": ["positional"]},
        {"append_args": ["--foo", "-p", "123"]},
        {"append_args": ["--foo=bar", "orphan"]},
        {"append_args": ["--"]},
    ],
)
def test_ambiguous_or_unmatched_edits_fail_closed(kwargs):
    with pytest.raises(ValueError):
        prepare_launch(BASE, request(**kwargs), "sglang")


def test_duplicate_base_option_cannot_be_removed_ambiguously():
    with pytest.raises(ValueError, match="exactly one"):
        prepare_launch(
            BASE + ["--enable-metrics"],
            request(remove_args=["--enable-metrics"]),
            "sglang",
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"env": {"PORT": "9999"}},
        {"unset_env": ["AGENTIC_TRACE_SOURCE"]},
        {"env": {"SGLANG_SIMULATE_ACC_LEN": "100"}},
        {"env": {"A-B": "x"}},
        {"env": {"SAFE": "x"}, "unset_env": ["SAFE"]},
        {"source_files": {"relative.py": "0" * 64}},
        {"absent_source_files": ["relative.py"]},
        {"executable": "python3"},
        {"replace_args": "false"},
    ],
)
def test_invalid_requests_fail_closed(kwargs):
    with pytest.raises(ValueError):
        request(**kwargs)


def test_server_env_is_scoped_and_executes_literal_values():
    code = "import json,os;print(json.dumps([os.getenv('TEST_LITERAL'),os.getenv('REMOVE_ME')]))"
    original = dict(os.environ, REMOVE_ME="old")
    command, evidence = prepare_launch(
        [sys.executable, "-c", code],
        request(
            env={"TEST_LITERAL": "$(touch should-not-exist)"}, unset_env=["REMOVE_ME"]
        ),
        "sglang",
        original,
    )
    completed = subprocess.run(
        command, env=original, check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == ["$(touch should-not-exist)", None]
    assert original["REMOVE_ME"] == "old"
    assert evidence["effective_env"] == {
        "REMOVE_ME": None,
        "TEST_LITERAL": "$(touch should-not-exist)",
    }


def test_candidate_sources_are_verified_after_launcher_setup(tmp_path):
    source = tmp_path / "framework kernel.py"
    source.write_text("candidate\n")
    deleted = tmp_path / "deleted.py"
    overrides = request(
        source_files={str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
        absent_source_files=[str(deleted)],
    )
    _, evidence = prepare_launch(BASE, overrides, "sglang")
    assert evidence["source_files"] == overrides["source_files"]
    source.write_text("installer overwrote candidate\n")
    with pytest.raises(ValueError, match="changed before server launch"):
        prepare_launch(BASE, overrides, "sglang")
    source.write_text("candidate\n")
    deleted.symlink_to(tmp_path / "nonexistent")
    with pytest.raises(ValueError, match="reappeared"):
        prepare_launch(BASE, overrides, "sglang")


def test_runtime_executable_override(tmp_path):
    candidate = tmp_path / "python candidate"
    candidate.symlink_to(sys.executable)
    command, evidence = prepare_launch(
        BASE, request(executable=str(candidate)), "sglang"
    )
    assert command[0] == str(candidate)
    assert evidence["resolved_executable"] == str(Path(sys.executable).resolve())


def test_runtime_environment_records_final_controls_without_credentials():
    _, evidence = prepare_launch(
        BASE,
        request(env={"SGLANG_USE_AITER": "0"}, unset_env=["OMP_NUM_THREADS"]),
        "sglang",
        {
            "PATH": os.environ["PATH"],
            "SGLANG_USE_AITER": "1",
            "OMP_NUM_THREADS": "4",
            "VLLM_API_KEY": "private",
            "SGLANG_SECRET": "private",
            "UNRELATED": "ignored",
        },
    )
    assert evidence["runtime_environment"] == {
        "PATH": os.environ["PATH"],
        "SGLANG_USE_AITER": "0",
    }


def test_real_bash_hook_preserves_spaces_and_writes_verified_receipt(tmp_path):
    work = tmp_path / "run with spaces"
    work.mkdir()
    overrides = request(
        append_args=["--new", "value with spaces"], env={"CUSTOM_LITERAL": "$not_shell"}
    )
    config = work / "request.json"
    config.write_text(json.dumps(overrides))
    receipt = work / "receipt.json"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1"; shift; agentic_apply_server_launch sglang "$@"; printf "%s\\0" "${AGENTX_SERVER_COMMAND[@]}"',
            "test",
            str(ROOT / "benchmarks/benchmark_lib.sh"),
            *BASE,
        ],
        env={
            **os.environ,
            "AGENTX_LAUNCH_OVERRIDES_FILE": str(config),
            "AGENTX_LAUNCH_OVERRIDES_SHA256": digest(overrides),
            "AGENTX_SERVER_LAUNCH_FILE": str(receipt),
            "PYTHONPYCACHEPREFIX": str(work / "pycache"),
        },
        capture_output=True,
        check=True,
    )
    argv = result.stdout.decode().split("\0")[:-1]
    assert argv == [
        "env",
        "CUSTOM_LITERAL=$not_shell",
        *BASE,
        "--new",
        "value with spaces",
    ]
    evidence = json.loads(receipt.read_text())
    assert evidence["effective_argv"] == BASE + ["--new", "value with spaces"]


def test_manifest_launchers_all_call_hook_before_recording_and_spawning():
    manifest = json.loads((ROOT / "configs/agentx-launchers.json").read_text())
    assert len(manifest["recipes"]) == 11
    for row in manifest["recipes"].values():
        path = ROOT / "benchmarks" / row["benchmark_script"]
        script = path.read_text()
        var = "SGLANG_CMD" if row["framework"] == "sglang" else "VLLM_CMD"
        hook = f'agentic_apply_server_launch {row["framework"]} "${{{var}[@]}}"'
        assert script.count(hook) == 1
        assert script.index(hook) < script.index(f'"${{{var}[@]}}" >')
        assert row["launch_overrides_version"] == 1


def test_every_launcher_mapping_matches_the_actual_recipe_identity():
    manifest = json.loads((ROOT / "configs/agentx-launchers.json").read_text())
    recipes = yaml.safe_load((ROOT / "configs/amd-master.yaml").read_text())
    runners = yaml.safe_load((ROOT / "configs/runners.yaml").read_text())["hardware"]
    for name, row in manifest["recipes"].items():
        recipe = recipes[name]
        assert row["model"] == recipe["model"]
        assert row["precision"] == recipe["precision"]
        assert row["framework"] == recipe["framework"]
        assert recipe["runner"] in runners
        assert (
            recipe["runner"].removeprefix("cluster:").split("-")[0]
            == row["runner_type"]
        )
        assert "agentic-coding" in recipe["scenarios"]
        assert not recipe.get("multinode", False)
        assert Path(row["benchmark_script"]).name.startswith(
            recipe["model-prefix"] + "_"
        )


def test_supported_launchers_bash_syntax():
    major = subprocess.check_output(
        ["bash", "-c", 'printf %s "${BASH_VERSINFO[0]}"'], text=True
    )
    if int(major) < 4:
        pytest.skip("The existing InferenceX launchers require Bash 4+ ([[ -v ]])")
    manifest = json.loads((ROOT / "configs/agentx-launchers.json").read_text())
    for row in manifest["recipes"].values():
        subprocess.run(
            ["bash", "-n", str(ROOT / "benchmarks" / row["benchmark_script"])],
            check=True,
        )
