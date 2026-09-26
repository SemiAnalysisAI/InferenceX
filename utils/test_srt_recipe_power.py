"""Exercise native recipe power routing and submission through the shell entrypoints."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))

from srtctl.core.config import generate_override_configs
from srtctl.core.overrides import (
    apply_overrides_to_recipe,
    parse_overrides,
)


@pytest.fixture
def agentx_recipe() -> dict:
    """A controlled custom benchmark with two physical GPU roles."""
    return {
        "schema": 2,
        "name": "controlled-agentx",
        "model": {"path": "weights", "container": "serving", "precision": "fp4"},
        "resources": {"gpu_type": "gb200", "gpus_per_node": 4},
        "engine": "vllm",
        "roles": {
            "prefill": {"nodes": 1, "workers": 1, "gpus": 4, "args": {}},
            "decode": {
                "nodes": 1,
                "workers": 1,
                "gpus": 4,
                "args": {
                    "speculative-config": json.dumps(
                        {"method": "mtp", "num_speculative_tokens": 3}
                    )
                },
            },
        },
        "telemetry": {
            "enabled": True,
            "required": True,
            "collect_interval_ms": 1000,
            "storage_subdir": "power",
            "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
        },
        "benchmark": {
            "type": "custom",
            "concurrencies": [99],
            "command": "bash /infmax-workspace/benchmarks/srt_agentic.sh",
            "env": {
                "RESULT_DIR": "/logs/agentic",
                "INFMAX_CONTAINER_WORKSPACE": "/infmax-workspace",
                "IS_MULTINODE": "true",
                "CONC_LIST": "99",
            },
        },
    }


def _executable(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o755)


def _submit(
    tmp_path: Path,
    recipe: dict,
    *,
    selector="",
    overrides=(),
    framework="dynamo-vllm",
    **environment,
):
    (tmp_path / "srt-slurm-sha.txt").write_text("a" * 40 + "\n")
    recipe_path = tmp_path / "recipe with spaces.yaml"
    original = yaml.safe_dump(recipe)
    recipe_path.write_text(original)
    _executable(
        tmp_path / "srtctl",
        "import json,os,sys\n"
        "with open(os.environ['SUBMISSIONS'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "print('✅ Job 12345 submitted')\n"
        "sys.exit(int(os.environ.get('SUBMISSION_RC', '0')))\n",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1" || exit $?; config="$2"; framework="$3"; shift 3; '
                'SRTCTL_EVAL_ARGS+=("$@"); '
                'prepare_srt_power "$config" "$framework" || exit $?; '
                'printf \'{"dcgm":%s,"agentx":%s}\\n\' '
                '"$USES_DCGM_POWER" "$USES_AGENTX_POWER" > "$LANE"; '
                'apply_srt_recipe "$config" "$framework" '
                '-f "$config" --tags "ordinary AgentX submission" "${SRTCTL_RECIPE_ARGS[@]}"'
            ),
            "bash",
            str(ROOT / "runners/slurm_utils.sh"),
            f"{recipe_path}{':' + selector if selector else ''}",
            framework,
            *overrides,
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(ROOT / "utils/srt-slurm/src")]
            ),
            "GITHUB_WORKSPACE": str(tmp_path),
            "MODEL_PREFIX": "dsv4",
            "IS_AGENTIC": "1",
            "EVAL_ONLY": "false",
            "SPEC_DECODING": "mtp",
            "THINKING_MODE": "thinking_on",
            "CONC_LIST": "1 4",
            "SUBMISSIONS": str(tmp_path / "submissions.jsonl"),
            "LANE": str(tmp_path / "lane.json"),
            **environment,
        },
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert recipe_path.read_text() == original
    recorded = tmp_path / "submissions.jsonl"
    commands = (
        [json.loads(line) for line in recorded.read_text().splitlines()]
        if recorded.exists()
        else []
    )
    lane_path = tmp_path / "lane.json"
    lane = json.loads(lane_path.read_text()) if lane_path.exists() else None
    return result, commands, lane


def _resolved_submission(raw: dict, command: list[str]) -> dict:
    """Decode recorded native CLI arguments with the pinned SRT implementation."""
    recipe = copy.deepcopy(raw)
    sets = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--set"]
    unsets = [
        command[i + 1] for i, value in enumerate(command[:-1]) if value == "--unset"
    ]
    apply_overrides_to_recipe(recipe, parse_overrides(sets, unsets))
    config = command[command.index("--file") + 1]
    _, _, selector = config.partition(":")
    if "base" in recipe:
        return generate_override_configs(recipe, selector=selector or None)[0][1]
    return recipe


def test_normal_submission_injects_matrix_power_without_changing_serving(
    tmp_path, agentx_recipe
):
    result, commands, lane = _submit(tmp_path, agentx_recipe)
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": 1, "agentx": 1}
    assert len(commands) == 1
    resolved = _resolved_submission(agentx_recipe, commands[0])
    assert resolved["benchmark"]["concurrencies"] == [1, 4]
    assert resolved["benchmark"]["env"]["CONC_LIST"] == "1 4"
    assert resolved["benchmark"]["env"]["REQUIRE_POWER"] == "1"
    assert resolved["benchmark"]["env"]["ENABLE_AGENTX_POWER"] == "1"
    assert resolved["telemetry"]["required"] is True
    assert (tmp_path / "power-producer-sha.txt").read_text() == "a" * 40 + "\n"
    for field in ("model", "resources", "engine"):
        assert resolved[field] == agentx_recipe[field]
    before = copy.deepcopy(agentx_recipe["roles"])
    after = copy.deepcopy(resolved["roles"])
    for role in ("decode",):
        original_spec = json.loads(before[role]["args"].pop("speculative-config"))
        generated_spec = json.loads(after[role]["args"].pop("speculative-config"))
        assert generated_spec.pop("rejection_sample_method") == "synthetic"
        assert generated_spec.pop("synthetic_acceptance_length") > 1
        assert generated_spec == original_spec
    assert after == before


@pytest.mark.parametrize(
    "selector, expected",
    [
        ("base", 0),
        ("override_power", 1),
        ("zip_override_power[0]", 0),
        ("zip_override_power[1]", 1),
    ],
)
def test_native_selector_controls_the_power_lane(
    tmp_path, agentx_recipe, selector, expected
):
    agentx_recipe["telemetry"]["enabled"] = False
    raw = {
        "schema": 2,
        "base": agentx_recipe,
        "override_power": {
            "telemetry": {
                "enabled": True,
                "dcgm_exporter": {
                    "command": "dcgm-exporter --address :{port} --collectors /configs/power.csv"
                },
            }
        },
        "zip_override_power": {
            "name": ["disabled", "enabled"],
            "telemetry": {"enabled": [False, True]},
        },
    }
    result, commands, lane = _submit(tmp_path, raw, selector=selector)
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": expected, "agentx": expected}
    assert len(commands) == 1
    resolved = _resolved_submission(raw, commands[0])
    assert resolved["telemetry"]["enabled"] is bool(expected)
    assert resolved["benchmark"]["concurrencies"] == [1, 4]
    expected_exporter = {"container_image": "dcgm-exporter", "port": 9401}
    if selector == "override_power":
        expected_exporter["command"] = (
            "dcgm-exporter --address :{port} --collectors /configs/power.csv"
        )
    assert resolved["telemetry"]["dcgm_exporter"] == expected_exporter


@pytest.mark.parametrize(
    "overrides", [("--set", "telemetry.enabled=false"), ("--unset", "telemetry")]
)
def test_native_caller_override_disables_power(tmp_path, agentx_recipe, overrides):
    result, commands, lane = _submit(tmp_path, agentx_recipe, overrides=overrides)
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": 0, "agentx": 0}
    resolved = _resolved_submission(agentx_recipe, commands[0])
    assert not resolved.get("telemetry", {}).get("enabled", False)
    assert "REQUIRE_POWER" not in resolved["benchmark"]["env"]


def test_eval_uses_real_verification_with_the_same_recipe_contract(
    tmp_path, agentx_recipe
):
    result, commands, lane = _submit(tmp_path, agentx_recipe, EVAL_ONLY="true")
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": 1, "agentx": 1}
    resolved = _resolved_submission(agentx_recipe, commands[0])
    assert resolved["roles"] == agentx_recipe["roles"]
    assert resolved["benchmark"]["concurrencies"] == [1, 4]


def test_multiple_selected_recipes_fail_before_submission(tmp_path, agentx_recipe):
    raw = {
        "schema": 2,
        "base": agentx_recipe,
        "zip_override_power": {"name": ["first", "second"]},
    }
    result, commands, lane = _submit(tmp_path, raw, selector="zip_override_power")
    assert result.returncode != 0
    assert "exactly one selected recipe" in result.stderr
    assert commands == []
    assert lane is None


def test_non_agentx_disabled_telemetry_preserves_multiple_submissions(
    tmp_path, agentx_recipe
):
    agentx_recipe["telemetry"]["enabled"] = False
    raw = {
        "schema": 2,
        "base": agentx_recipe,
        "zip_override_legacy": {"name": ["first", "second"]},
    }
    result, commands, lane = _submit(
        tmp_path, raw, selector="zip_override_legacy", IS_AGENTIC="0"
    )
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": 0, "agentx": 0}
    assert len(commands) == 2
    resolved = [_resolved_submission(raw, command) for command in commands]
    assert [recipe["name"] for recipe in resolved] == ["first", "second"]
    for recipe in resolved:
        assert recipe["telemetry"]["enabled"] is False
        assert recipe["benchmark"]["concurrencies"] == [99]
        assert recipe["roles"] == agentx_recipe["roles"]


def test_non_agentx_mixed_telemetry_fails_before_any_submission(
    tmp_path, agentx_recipe
):
    raw = {
        "schema": 2,
        "base": agentx_recipe,
        "zip_override_mixed": {
            "name": ["disabled", "enabled"],
            "telemetry": {"enabled": [False, True]},
        },
    }
    result, commands, lane = _submit(
        tmp_path, raw, selector="zip_override_mixed", IS_AGENTIC="0"
    )
    assert result.returncode != 0
    assert "exactly one selected recipe" in result.stderr
    assert commands == []
    assert lane is None


@pytest.mark.parametrize("concurrencies", ["1 1", "0 4", "1 nan"])
def test_invalid_matrix_concurrencies_fail_before_submission(
    tmp_path, agentx_recipe, concurrencies
):
    result, commands, _ = _submit(tmp_path, agentx_recipe, CONC_LIST=concurrencies)
    assert result.returncode != 0
    assert commands == []


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (("--set", "telemetry.required=false"), "telemetry.required"),
        (("--set", 'telemetry.storage_subdir="other"'), "storage_subdir"),
        (("--set", 'benchmark.env.RESULT_DIR="/logs/other"'), "window/result contract"),
    ],
)
def test_incompatible_power_contract_fails_before_submission(
    tmp_path, agentx_recipe, overrides, reason
):
    result, commands, _ = _submit(tmp_path, agentx_recipe, overrides=overrides)
    assert result.returncode != 0
    assert reason in result.stderr
    assert commands == []


def test_submission_failure_is_returned_even_with_a_job_id(tmp_path, agentx_recipe):
    result, commands, lane = _submit(tmp_path, agentx_recipe, SUBMISSION_RC="7")
    assert result.returncode == 7
    assert "Job 12345" in result.stdout
    assert len(commands) == 1
    assert lane == {"dcgm": 1, "agentx": 1}


@pytest.mark.parametrize("disaggregated", [False, True])
def test_sglang_power_admission_preserves_physical_role_topology(
    tmp_path, agentx_recipe, disaggregated
):
    agentx_recipe["engine"] = "sglang"
    worker = {
        "nodes": 2,
        "workers": 1,
        "gpus": 8,
        "env": {"SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE": "0"},
        "args": {"tensor-parallel-size": 8, "expert-parallel-size": 1},
    }
    if disaggregated:
        agentx_recipe["roles"] = {
            "prefill": worker,
            "decode": {
                "nodes": 4,
                "workers": 4,
                "gpus": 4,
                "args": {"tensor-parallel-size": 4},
            },
        }
    else:
        agentx_recipe["roles"] = {"agg": worker}
    result, commands, lane = _submit(
        tmp_path,
        agentx_recipe,
        framework="dynamo-sglang",
        MODEL_PREFIX="new-model",
        SPEC_DECODING="none",
    )
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": 1, "agentx": 1}
    resolved = _resolved_submission(agentx_recipe, commands[0])
    assert resolved["roles"] == agentx_recipe["roles"]
    assert resolved["benchmark"]["env"]["REQUIRE_POWER"] == "1"
    assert resolved["benchmark"]["concurrencies"] == [1, 4]
