"""Exercise GB200 recipe power routing and submission through the shell entrypoints."""

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

from srtctl.core.config import generate_override_configs  # noqa: E402
from srtctl.core.overrides import (  # noqa: E402
    apply_overrides_to_recipe,
    parse_overrides,
)


@pytest.fixture
def dsv4_recipe() -> dict:
    """Use the supported serving recipe, with a controlled telemetry contract."""
    path = ROOT / (
        "benchmarks/multi_node/srt-slurm-recipes/"
        "dsv4/vllm/gb200-fp4/agentx/agg-tp8-mtp.yaml"
    )
    recipe = yaml.safe_load(path.read_text())
    recipe["telemetry"] = {
        "enabled": True,
        "required": True,
        "collect_interval_ms": 1000,
        "storage_subdir": "power",
        "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 9401},
    }
    recipe["benchmark"]["concurrencies"] = [99]
    recipe["benchmark"]["env"]["CONC_LIST"] = "99"
    return recipe


def _executable(path: Path, source: str) -> None:
    path.write_text(f"#!{sys.executable}\n{source}")
    path.chmod(0o755)


def _submit(tmp_path: Path, recipe: dict, *, selector="", overrides=(), **environment):
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
            'source "$1" || exit $?; config="$2"; shift 2; '
            'SRTCTL_EVAL_ARGS+=("$@"); '
            'prepare_gb200_srt_power "$config" dynamo-vllm || exit $?; '
            'printf \'{"dcgm":%s,"agentx":%s}\\n\' '
            '"$USES_DCGM_POWER" "$USES_AGENTX_POWER" > "$LANE"; '
            'apply_srt_recipe "$config" dynamo-vllm '
            '-f "$config" --tags "ordinary AgentX submission" "${SRTCTL_RECIPE_ARGS[@]}"',
            "bash",
            str(ROOT / "runners/slurm_utils.sh"),
            f"{recipe_path}{':' + selector if selector else ''}",
            *overrides,
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), str(ROOT / "utils/srt-slurm/src")]
            ),
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


def test_dsv4_normal_submission_injects_matrix_power_without_changing_serving(
    tmp_path, dsv4_recipe
):
    result, commands, lane = _submit(tmp_path, dsv4_recipe)
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": 1, "agentx": 1}
    assert len(commands) == 1
    resolved = _resolved_submission(dsv4_recipe, commands[0])
    assert resolved["benchmark"]["concurrencies"] == [1, 4]
    assert resolved["benchmark"]["env"]["CONC_LIST"] == "1 4"
    assert resolved["benchmark"]["env"]["REQUIRE_POWER"] == "1"
    assert resolved["benchmark"]["env"]["ENABLE_AGENTX_POWER"] == "1"
    assert resolved["telemetry"]["required"] is True
    for field in ("model", "identity", "resources", "frontend", "engine"):
        assert resolved[field] == dsv4_recipe[field]
    before = copy.deepcopy(dsv4_recipe["roles"])
    after = copy.deepcopy(resolved["roles"])
    for role in before:
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
    tmp_path, dsv4_recipe, selector, expected
):
    dsv4_recipe["telemetry"]["enabled"] = False
    raw = {
        "schema": 2,
        "base": dsv4_recipe,
        "override_power": {"telemetry": {"enabled": True}},
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


@pytest.mark.parametrize(
    "overrides", [("--set", "telemetry.enabled=false"), ("--unset", "telemetry")]
)
def test_native_caller_override_disables_power(tmp_path, dsv4_recipe, overrides):
    result, commands, lane = _submit(tmp_path, dsv4_recipe, overrides=overrides)
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": 0, "agentx": 0}
    resolved = _resolved_submission(dsv4_recipe, commands[0])
    assert not resolved.get("telemetry", {}).get("enabled", False)
    assert "REQUIRE_POWER" not in resolved["benchmark"]["env"]


def test_eval_uses_real_verification_with_the_same_recipe_contract(
    tmp_path, dsv4_recipe
):
    result, commands, lane = _submit(tmp_path, dsv4_recipe, EVAL_ONLY="true")
    assert result.returncode == 0, result.stderr
    assert lane == {"dcgm": 1, "agentx": 1}
    resolved = _resolved_submission(dsv4_recipe, commands[0])
    assert resolved["roles"] == dsv4_recipe["roles"]
    assert resolved["benchmark"]["concurrencies"] == [1, 4]


def test_multiple_selected_recipes_fail_before_submission(tmp_path, dsv4_recipe):
    raw = {
        "schema": 2,
        "base": dsv4_recipe,
        "zip_override_power": {"name": ["first", "second"]},
    }
    result, commands, lane = _submit(tmp_path, raw, selector="zip_override_power")
    assert result.returncode != 0
    assert "exactly one selected recipe" in result.stderr
    assert commands == []
    assert lane is None


def test_non_agentx_disabled_telemetry_preserves_multiple_submissions(
    tmp_path, dsv4_recipe
):
    dsv4_recipe["telemetry"]["enabled"] = False
    raw = {
        "schema": 2,
        "base": dsv4_recipe,
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
        assert recipe["roles"] == dsv4_recipe["roles"]


def test_non_agentx_mixed_telemetry_fails_before_any_submission(tmp_path, dsv4_recipe):
    raw = {
        "schema": 2,
        "base": dsv4_recipe,
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
    tmp_path, dsv4_recipe, concurrencies
):
    result, commands, _ = _submit(tmp_path, dsv4_recipe, CONC_LIST=concurrencies)
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
    tmp_path, dsv4_recipe, overrides, reason
):
    result, commands, _ = _submit(tmp_path, dsv4_recipe, overrides=overrides)
    assert result.returncode != 0
    assert reason in result.stderr
    assert commands == []


def test_submission_failure_is_returned_even_with_a_job_id(tmp_path, dsv4_recipe):
    result, commands, lane = _submit(tmp_path, dsv4_recipe, SUBMISSION_RC="7")
    assert result.returncode == 7
    assert "Job 12345" in result.stdout
    assert len(commands) == 1
    assert lane == {"dcgm": 1, "agentx": 1}


@pytest.mark.parametrize(
    "job_status, adapter_rc, expected_rc",
    [("COMPLETED|0:0", 9, 9), ("FAILED|1:0", 0, 1)],
)
def test_collection_retains_results_and_audits_when_a_collaborator_fails(
    tmp_path, job_status, adapter_rc, expected_rc
):
    """Check shell failure propagation; the adapter here is deliberately a stub."""
    source, workspace, logs = [
        tmp_path / name for name in ("source", "workspace", "logs")
    ]
    for path in (source, workspace, logs):
        path.mkdir()
    for concurrency in (1, 4):
        (source / f"point_conc{concurrency}.json").write_text(
            json.dumps({"conc": concurrency})
        )
    _executable(tmp_path / "sacct", f"print('12345|{job_status}')\n")
    _executable(
        tmp_path / "python3",
        "import json,os,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "directory=Path(args[args.index('--result-dir')+1]); directory.mkdir(parents=True,exist_ok=True)\n"
        "code=int(os.environ['ADAPTER_RC']) if directory.name=='conc_1' else 0\n"
        "(directory/'power_validation.json').write_text(json.dumps({'stub_exit_code':code}))\n"
        "sys.exit(code)\n",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; collect_agentic_power_results 12345 "$2" "$3" "$4" point producer-sha 1 4',
            "bash",
            str(ROOT / "runners/slurm_utils.sh"),
            str(logs),
            str(source),
            str(workspace),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "ADAPTER_RC": str(adapter_rc),
        },
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected_rc, result.stderr
    assert (
        logs / "power/native-job-status.txt"
    ).read_text().strip() == f"12345|{job_status}"
    for concurrency, expected_adapter_rc in ((1, adapter_rc), (4, 0)):
        assert json.loads(
            (workspace / f"point_conc{concurrency}.json").read_text()
        ) == {"conc": concurrency}
        assert json.loads(
            (logs / f"agentic/conc_{concurrency}/power_validation.json").read_text()
        ) == {"stub_exit_code": expected_adapter_rc}


def test_failed_submission_cleanup_cancels_only_returned_job_ids(tmp_path):
    _executable(
        tmp_path / "scancel",
        "import json,os,sys\n"
        "with open(os.environ['CANCELLED'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "sys.exit(1)\n",
    )
    output = "✅ Job 12345 submitted\n✅ Job 12346 submitted\nJob 12345\nERROR: next submission failed\n"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; cancel_submitted_srt_jobs "$2"; exit 7',
            "bash",
            str(ROOT / "runners/slurm_utils.sh"),
            output,
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CANCELLED": str(tmp_path / "cancelled.jsonl"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7
    assert [
        json.loads(line)
        for line in (tmp_path / "cancelled.jsonl").read_text().splitlines()
    ] == [["12345"], ["12346"]]
