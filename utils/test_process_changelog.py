"""Tests for changelog-driven sweep generation."""

import json
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import process_changelog
from matrix_logic.generate_sweep_configs import generate_test_config_sweep
from matrix_logic.validation import validate_master_config


@pytest.fixture
def generation_repo(tmp_path, monkeypatch):
    """An isolated history containing the real generator and controlled inputs."""
    source = Path(__file__).resolve().parents[1]
    for directory in ("utils/matrix_logic", "infx"):
        if (source / directory).exists():
            shutil.copytree(source / directory, tmp_path / directory)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/amd-master.yaml").write_text("{}\n")
    (tmp_path / "configs/runners.yaml").write_text("labels: {fixture: [node-a]}\nhardware: {}\n")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    for revision, conc in (("older", 2), ("newer", 6)):
        master = {"fixture": {
            "image": "example/image:stable", "model": revision, "model-prefix": "dsr1",
            "precision": "fp8", "framework": "sglang", "runner": "fixture",
            "multinode": False,
            "scenarios": {"fixed-seq-len": [{
                "isl": 1024, "osl": 1024, "search-space": [{"tp": 1, "conc-list": [conc]}],
            }]},
        }}
        (tmp_path / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(master))
        git("add", ".")
        git("commit", "-qm", revision)
        git("tag", revision)

    # Neither uncommitted source nor inputs may leak into historical generation.
    (tmp_path / "configs/nvidia-master.yaml").write_text("invalid working tree\n")
    for directory in ("utils/matrix_logic", "infx"):
        for path in (tmp_path / directory).rglob("*.py"):
            path.write_text('raise RuntimeError("working tree source was used")\n')
    monkeypatch.chdir(tmp_path)
    return tmp_path, git


@pytest.mark.parametrize("revision,expected", [("older", ("older", 2)), ("newer", ("newer", 6))])
def test_historical_generation_uses_committed_source_and_inputs(generation_repo, revision, expected):
    with process_changelog.generation_inputs_at_ref(revision) as inputs:
        result = subprocess.run(
            [sys.executable, inputs.generator_script, "test-config", "--config-files",
             *inputs.config_files, "--runner-config", inputs.runner_config,
             "--config-keys", "fixture", "--no-evals"],
            capture_output=True, text=True, check=True,
        )
        rows = json.loads(result.stdout)
        assert [(row["model"], row["conc"]) for row in rows] == [expected]
        assert result.stderr == ""
        extracted_script = Path(inputs.generator_script)
    assert not extracted_script.exists()


def test_historical_generation_rejects_missing_inputs(generation_repo):
    _, git = generation_repo
    git("rm", "-f", "configs/runners.yaml")
    git("commit", "-qm", "missing runner inventory")

    with pytest.raises(ValueError, match="missing generation inputs.*configs/runners.yaml"):
        with process_changelog.generation_inputs_at_ref("HEAD"):
            pytest.fail("an incomplete snapshot must not be used for generation")


def test_historical_generation_supports_legacy_script_layout(generation_repo):
    root, git = generation_repo
    git("rm", "-rf", "--ignore-unmatch", "infx")
    # The historical command is an external collaborator: exercise extraction
    # and sibling imports without freezing a past copy of the matrix algorithm.
    script = root / "utils/matrix_logic/generate_sweep_configs.py"
    schema = script.with_name("validation.py")
    script.write_text("from validation import revision\nprint(revision)\n")
    schema.write_text('revision = "legacy snapshot"\n')
    git("add", str(script), str(schema))
    git("commit", "-qm", "legacy generator")
    schema.write_text('raise RuntimeError("wrong revision")\n')

    with process_changelog.generation_inputs_at_ref("HEAD") as inputs:
        result = subprocess.run(
            [sys.executable, inputs.generator_script],
            capture_output=True, text=True, check=True,
        )
        assert result.stdout == "legacy snapshot\n"
        assert result.stderr == ""


def _fixed_matrix_row(
    conc,
    *,
    image="vllm/vllm-openai:v0.16.0",
    tp=8,
    duration=None,
):
    return {
        "image": image,
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "model-prefix": "dsv4",
        "precision": "fp4",
        "framework": "vllm",
        "spec-decoding": "mtp",
        "runner": "cluster:b300-nv",
        "isl": 8192,
        "osl": 1024,
        "tp": tp,
        "pp": 1,
        "dcp-size": 1,
        "pcp-size": 1,
        "ep": 8,
        "dp-attn": True,
        "conc": conc,
        "max-model-len": 10240,
        "exp-name": f"dsv4_tp{tp}_conc{conc}",
        "disagg": False,
        "run-eval": False,
        "eval-only": False,
    } | ({"duration": duration} if duration is not None else {})


def _scenario_values(command):
    if "--scenario-type" not in command:
        return []
    index = command.index("--scenario-type") + 1
    return command[index:]


def test_trim_conc_supports_nested_backend_metadata():
    common = {
        "model": "moonshotai/Kimi-K3",
        "kv-offloading": "dram",
        "kv-offload-backend": {
            "name": "vllm-simple",
            "settings": {"tiers": ["cpu", "gpu"]},
        },
    }
    entries = [
        {**common, "conc": 8, "exp-name": "kimi_tp8_conc8_kvdram"},
        {**common, "conc": 2, "exp-name": "kimi_tp8_conc2_kvdram"},
        {
            **common,
            "kv-offload-backend": {"name": "lmcache"},
            "conc": 4,
            "exp-name": "kimi_tp8_conc4_lmcache",
        },
    ]

    trimmed = process_changelog.trim_conc(entries)

    assert [entry["conc"] for entry in trimmed] == [2, 4]
    assert [entry["kv-offload-backend"]["name"] for entry in trimmed] == [
        "vllm-simple",
        "lmcache",
    ]


def test_config_key_expansion_is_deterministic_and_deduplicated():
    master_config = {
        "config-b": {},
        "config-a": {},
        "other": {},
    }

    result = process_changelog.get_config_keys_from_master(
        ["config-*", "config-a"],
        master_config,
    )

    assert result == ["config-b", "config-a"]


def test_append_only_delta_keeps_only_new_single_node_points():
    base = [_fixed_matrix_row(4), _fixed_matrix_row(8)]
    head = [*base, _fixed_matrix_row(12)]

    delta = process_changelog.append_only_delta(base, head)

    assert [entry["conc"] for entry in delta] == [12]


def test_append_only_delta_slices_multinode_concurrency_lists():
    common = {
        "image": "lmsysorg/sglang:v0.5.7",
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "model-prefix": "dsv4",
        "precision": "fp4",
        "framework": "dynamo-sglang",
        "conc": [8, 16],
        "exp-name": "dsv4-disagg",
    }

    delta = process_changelog.append_only_delta(
        [common],
        [{**common, "conc": [8, 16, 24]}],
    )

    assert delta == [{**common, "conc": [24]}]


def test_append_only_delta_deduplicates_new_single_node_points():
    base = [_fixed_matrix_row(4)]
    head = [base[0], _fixed_matrix_row(8), _fixed_matrix_row(8)]

    delta = process_changelog.append_only_delta(base, head)

    assert [entry["conc"] for entry in delta] == [8]


def test_append_only_delta_deduplicates_multinode_concurrency_lists():
    common = {
        "image": "lmsysorg/sglang:v0.5.7",
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "framework": "dynamo-sglang",
        "conc": [8, 16],
        "exp-name": "dsv4-disagg",
    }

    delta = process_changelog.append_only_delta(
        [common],
        [{**common, "conc": [8, 16, 24, 24]}],
    )

    assert delta == [{**common, "conc": [24]}]


def test_append_only_delta_rejects_image_changes():
    base = [_fixed_matrix_row(4)]
    head = [
        _fixed_matrix_row(4, image="vllm/vllm-openai:v0.16.1"),
        _fixed_matrix_row(8, image="vllm/vllm-openai:v0.16.1"),
    ]

    try:
        process_changelog.append_only_delta(base, head)
    except ValueError as error:
        assert "remove or modify" in str(error)
    else:
        raise AssertionError("image mutation should reject append-only mode")


def test_append_only_delta_allows_new_parallelism_with_its_points():
    base = [
        _fixed_matrix_row(1, tp=4),
        _fixed_matrix_row(4, tp=4),
        _fixed_matrix_row(8, tp=4),
    ]
    head = [
        *base,
        _fixed_matrix_row(12, tp=8),
        _fixed_matrix_row(16, tp=8),
    ]

    delta = process_changelog.append_only_delta(base, head)

    assert [(entry["tp"], entry["conc"]) for entry in delta] == [
        (8, 12),
        (8, 16),
    ]


def test_append_only_delta_allows_any_new_recipe_while_preserving_old_recipe():
    base = [_fixed_matrix_row(4, duration=3600)]
    head = [*base, _fixed_matrix_row(6, duration=300)]

    delta = process_changelog.append_only_delta(base, head)

    assert [(entry["duration"], entry["conc"]) for entry in delta] == [(300, 6)]


def test_append_only_delta_rejects_head_only_image_variant():
    base = [_fixed_matrix_row(4)]
    head = [
        *base,
        _fixed_matrix_row(8, image="vllm/vllm-openai:v0.16.1", tp=16),
    ]

    try:
        process_changelog.append_only_delta(base, head)
    except ValueError as error:
        assert "unchanged non-null image" in str(error)
    else:
        raise AssertionError("an append cannot fork the target curve's image")


def test_recipe_fingerprint_ignores_concurrency_and_experiment_name():
    first = _fixed_matrix_row(4)
    second = _fixed_matrix_row(16)

    assert process_changelog.recipe_fingerprint(first) == (
        process_changelog.recipe_fingerprint(second)
    )


def test_recipe_fingerprint_changes_for_any_recipe_variant():
    base = _fixed_matrix_row(4, tp=4, duration=3600)
    changed_parallelism = _fixed_matrix_row(4, tp=8, duration=3600)
    changed_duration = _fixed_matrix_row(4, tp=4, duration=300)

    fingerprints = {
        process_changelog.recipe_fingerprint(entry)
        for entry in (base, changed_parallelism, changed_duration)
    }

    assert len(fingerprints) == 3


def test_append_only_delta_rejects_removed_parallelism_recipe():
    tp4 = _fixed_matrix_row(4, tp=4)
    tp8 = _fixed_matrix_row(8, tp=8)

    try:
        process_changelog.append_only_delta([tp4, tp8], [tp4])
    except ValueError as error:
        assert "remove or modify" in str(error)
    else:
        raise AssertionError("removing a parallelism recipe should reject append-only mode")


def test_append_only_delta_rejects_modified_existing_recipe():
    base = [_fixed_matrix_row(4, duration=3600)]
    head = [_fixed_matrix_row(4, duration=300)]

    try:
        process_changelog.append_only_delta(base, head)
    except ValueError as error:
        assert "remove or modify" in str(error)
    else:
        raise AssertionError("modifying an existing recipe should reject append-only mode")


def test_append_only_delta_rejects_removed_existing_point():
    base = [_fixed_matrix_row(4), _fixed_matrix_row(8)]
    head = [_fixed_matrix_row(8), _fixed_matrix_row(12)]

    try:
        process_changelog.append_only_delta(base, head)
    except ValueError as error:
        assert "remove existing concurrency" in str(error)
    else:
        raise AssertionError("removing an existing point should reject append-only mode")


def test_append_only_scope_defers_selected_scenario_changes_to_matrix_comparison():
    base = {
        "test-config": {
            "image": "vllm/vllm-openai:v0.16.0",
            "scenarios": {
                "agentic-coding": {
                    "duration": 3600,
                    "search-space": [{"tp": 8, "conc-list": [1, 4]}],
                }
            },
        }
    }
    head = {
        "test-config": {
            "image": "vllm/vllm-openai:v0.16.0",
            "scenarios": {
                "agentic-coding": {
                    "duration": 1800,
                    "search-space": [{"tp": 8, "conc-list": [1, 4, 8]}],
                }
            },
        }
    }
    process_changelog.validate_append_only_scope(
        base, head, {"test-config": {"agentic-coding"}}
    )


def test_append_only_scope_allows_additive_top_level_restructuring():
    router_a = {"name": "router-a", "version": "1"}
    router_b = {"name": "router-b", "version": "2"}
    base = {
        "test-config": {
            "image": "img",
            "model": "m",
            "model-prefix": "m",
            "precision": "fp4",
            "framework": "vllm",
            "runner": "b200",
            "multinode": False,
            "router": router_a,
            "scenarios": {
                "fixed-seq-len": [
                    {
                        "isl": 8192,
                        "osl": 1024,
                        "search-space": [{"tp": 4, "conc-list": [1, 4, 8]}],
                    }
                ]
            },
        }
    }
    head = json.loads(json.dumps(base))
    head["test-config"].pop("router")
    search_space = head["test-config"]["scenarios"]["fixed-seq-len"][0][
        "search-space"
    ]
    search_space[0]["router"] = router_a
    search_space.append(
        {"tp": 8, "conc-list": [12, 16], "router": router_b}
    )

    validate_master_config(base)
    validate_master_config(head)
    args = SimpleNamespace(
        config_keys=["test-config"],
        seq_lens=None,
        conc=None,
        scenario_type=["fixed-seq-len"],
        runner_node_filter=None,
    )
    base_rows = generate_test_config_sweep(args, base)
    head_rows = generate_test_config_sweep(args, head)

    process_changelog.validate_append_only_scope(
        base, head, {"test-config": {"fixed-seq-len"}}
    )
    delta = process_changelog.append_only_delta(base_rows, head_rows)

    assert [(row["tp"], row["conc"], row["router"]) for row in delta] == [
        (8, 12, router_b),
        (8, 16, router_b),
    ]


def test_append_only_scope_rejects_global_change_with_unselected_scenario():
    base = {
        "test-config": {
            "router": {"name": "dynamo-router", "version": "0.8.1"},
            "scenarios": {
                "fixed-seq-len": {"search-space": [{"tp": 4, "conc-list": [1]}]},
                "agentic-coding": {"search-space": [{"tp": 4, "conc-list": [1]}]},
            },
        }
    }
    head = {
        "test-config": {
            "router": {"name": "dynamo-router", "version": "0.8.2"},
            "scenarios": base["test-config"]["scenarios"],
        }
    }

    try:
        process_changelog.validate_append_only_scope(
            base, head, {"test-config": {"fixed-seq-len"}}
        )
    except ValueError as error:
        assert "config-wide fields" in str(error)
    else:
        raise AssertionError("global changes may not affect an unselected scenario")


def test_append_only_scope_rejects_changes_to_unselected_scenario():
    base = {
        "test-config": {
            "scenarios": {
                "fixed-seq-len": {"search-space": [{"tp": 4, "conc-list": [1]}]},
                "agentic-coding": {"search-space": [{"tp": 4, "conc-list": [1]}]},
            }
        }
    }
    head = {
        "test-config": {
            "scenarios": {
                "fixed-seq-len": {"search-space": [{"tp": 4, "conc-list": [1]}]},
                "agentic-coding": {
                    "search-space": [{"tp": 4, "conc-list": [1, 4]}]
                },
            }
        }
    }

    try:
        process_changelog.validate_append_only_scope(
            base, head, {"test-config": {"fixed-seq-len"}}
        )
    except ValueError as error:
        assert "outside its changelog scope" in str(error)
    else:
        raise AssertionError("unselected scenario changes should reject append-only mode")


def test_append_only_scope_allows_range_to_list_expansion():
    base = {
        "test-config": {
            "image": "vllm/vllm-openai:v0.16.0",
            "scenarios": {
                "fixed-seq-len": {
                    "search-space": [{"tp": 8, "conc-start": 4, "conc-end": 64}],
                }
            },
        }
    }
    head = {
        "test-config": {
            "image": "vllm/vllm-openai:v0.16.0",
            "scenarios": {
                "fixed-seq-len": {
                    "search-space": [{"tp": 8, "conc-list": [4, 16, 32, 64]}],
                }
            },
        }
    }
    process_changelog.validate_append_only_scope(
        base, head, {"test-config": {"fixed-seq-len"}}
    )


@pytest.mark.parametrize("trim", [False, True])
def test_append_only_main_runs_only_added_points_and_skips_evals(
    monkeypatch,
    capsys,
    trim,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Add one concurrency point without rerunning the curve
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  append-only: true
"""
    base_rows = [_fixed_matrix_row(4)]
    head_rows = [*base_rows, _fixed_matrix_row(8)]
    commands = []

    monkeypatch.setattr(process_changelog, "get_added_lines", lambda *_: added_yaml)
    monkeypatch.setattr(
        process_changelog,
        "generation_inputs_at_ref",
        lambda *_: nullcontext(
            process_changelog.GenerationInputs(
                config_files=["base-nvidia.yaml", "base-amd.yaml"],
                generator_script="base-generate-sweep-configs.py",
                runner_config="base-runners.yaml",
            )
        ),
    )
    monkeypatch.setattr(
        process_changelog,
        "load_config_files",
        lambda _: {"test-config": {"image": "vllm/vllm-openai:v0.16.0"}},
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        rows = base_rows if "base-nvidia.yaml" in command else head_rows
        return SimpleNamespace(stdout=json.dumps(rows))

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "process_changelog.py",
        "--base-ref", "base",
        "--head-ref", "head",
        "--changelog-file", "perf-changelog.yaml", *(["--trim-conc"] if trim else []),
    ])

    process_changelog.main()

    output = json.loads(capsys.readouterr().out)
    assert [row["conc"] for row in output["single_node"]["8k1k"]] == [8]
    assert len(output["single_node"]["8k1k"][0]["recipe-fingerprint"]) == 64
    assert output["evals"] == []
    assert output["changelog_metadata"]["entries"][0]["append-only"] is True
    assert len(commands) == 2
    assert commands[0][1] == process_changelog.GENERATE_SWEEPS_PY_SCRIPT
    assert commands[1][1] == "base-generate-sweep-configs.py"
    assert commands[0][commands[0].index("--runner-config") + 1] == "configs/runners.yaml"
    assert commands[1][commands[1].index("--runner-config") + 1] == "base-runners.yaml"


def test_cli_evals_only_generates_agentic_eval(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Agentic-only work with the evals-only PR modifier
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  scenario-type:
    - agentic-coding
"""
    commands = []

    monkeypatch.setattr(
        process_changelog,
        "get_added_lines",
        lambda *_: added_yaml,
    )
    monkeypatch.setattr(
        process_changelog,
        "load_config_files",
        lambda _: {"test-config": {}},
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="[]")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "process_changelog.py",
        "--base-ref", "base",
        "--head-ref", "head",
        "--changelog-file", "perf-changelog.yaml",
        "--evals-only",
    ])

    process_changelog.main()

    assert len(commands) == 1
    assert "--evals-only" in commands[0]
    assert "--no-evals" not in commands[0]
    assert _scenario_values(commands[0]) == ["agentic-coding"]
    json.loads(capsys.readouterr().out)


def test_cli_all_evals_generates_agentic_eval(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Agentic-only work with the all-evals PR modifier
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  scenario-type:
    - agentic-coding
"""
    commands = []

    monkeypatch.setattr(
        process_changelog,
        "get_added_lines",
        lambda *_: added_yaml,
    )
    monkeypatch.setattr(
        process_changelog,
        "load_config_files",
        lambda _: {"test-config": {}},
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="[]")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "process_changelog.py",
        "--base-ref", "base",
        "--head-ref", "head",
        "--changelog-file", "perf-changelog.yaml",
        "--all-evals",
    ])

    process_changelog.main()

    assert len(commands) == 2
    assert "--no-evals" in commands[0]
    assert _scenario_values(commands[0]) == ["agentic-coding"]
    assert "--evals-only" in commands[1]
    assert "--all-evals" in commands[1]
    assert _scenario_values(commands[1]) == ["agentic-coding"]
    json.loads(capsys.readouterr().out)


def test_all_evals_takes_precedence_for_duplicate_configs(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Regular benchmark entry appears first
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1

- config-keys:
    - test-config
  description:
    - Expand the same config to all evals
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  all-evals: true
"""
    commands = []

    monkeypatch.setattr(
        process_changelog,
        "get_added_lines",
        lambda *_: added_yaml,
    )
    monkeypatch.setattr(
        process_changelog,
        "load_config_files",
        lambda _: {"test-config": {}},
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="[]")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "process_changelog.py",
        "--base-ref", "base",
        "--head-ref", "head",
        "--changelog-file", "perf-changelog.yaml",
    ])

    process_changelog.main()

    assert len(commands) == 2
    assert "--all-evals" in commands[0]
    assert "--evals-only" in commands[0]
    assert "--no-evals" in commands[1]
    json.loads(capsys.readouterr().out)


def test_disjoint_scenario_entries_for_same_config_are_not_deduplicated(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Fixed sequence jobs
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  scenario-type:
    - fixed-seq-len

- config-keys:
    - test-config
  description:
    - Agentic jobs
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  scenario-type:
    - agentic-coding
"""
    commands = []

    monkeypatch.setattr(
        process_changelog,
        "get_added_lines",
        lambda *_: added_yaml,
    )
    monkeypatch.setattr(
        process_changelog,
        "load_config_files",
        lambda _: {"test-config": {}},
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="[]")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "process_changelog.py",
        "--base-ref", "base",
        "--head-ref", "head",
        "--changelog-file", "perf-changelog.yaml",
    ])

    process_changelog.main()

    assert len(commands) == 4
    assert "--no-evals" in commands[0]
    assert _scenario_values(commands[0]) == ["fixed-seq-len"]
    assert "--evals-only" in commands[1]
    assert _scenario_values(commands[1]) == ["fixed-seq-len"]
    assert "--no-evals" in commands[2]
    assert _scenario_values(commands[2]) == ["agentic-coding"]
    assert "--evals-only" in commands[3]
    assert _scenario_values(commands[3]) == ["agentic-coding"]
    json.loads(capsys.readouterr().out)


def test_agentic_only_all_evals_does_not_suppress_later_fixed_evals(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Agentic-only all-evals entry
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  scenario-type:
    - agentic-coding
  all-evals: true

- config-keys:
    - test-config
  description:
    - Fixed sequence jobs
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  scenario-type:
    - fixed-seq-len
"""
    commands = []

    monkeypatch.setattr(
        process_changelog,
        "get_added_lines",
        lambda *_: added_yaml,
    )
    monkeypatch.setattr(
        process_changelog,
        "load_config_files",
        lambda _: {"test-config": {}},
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="[]")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "process_changelog.py",
        "--base-ref", "base",
        "--head-ref", "head",
        "--changelog-file", "perf-changelog.yaml",
    ])

    process_changelog.main()

    assert len(commands) == 3
    assert "--evals-only" in commands[0]
    assert "--all-evals" in commands[0]
    assert _scenario_values(commands[0]) == ["agentic-coding"]
    assert "--no-evals" in commands[1]
    assert _scenario_values(commands[1]) == ["fixed-seq-len"]
    assert "--evals-only" in commands[2]
    assert "--all-evals" not in commands[2]
    assert _scenario_values(commands[2]) == ["fixed-seq-len"]
    json.loads(capsys.readouterr().out)


def test_eval_rows_split_into_fixed_and_agentic_buckets(
    monkeypatch,
    capsys,
):
    """Realistic eval rows must pass final validation and land in the bucket
    matching their dispatch job: fixed-seq-len rows in `evals`, agentic
    GSM8K rows in `agentic_evals`."""
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Mixed fixed-seq-len and agentic eval selection
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
"""
    common = {
        "image": "vllm/vllm-openai:v0.11.0",
        "model": "deepseek-ai/DeepSeek-V4-Pro", "model-prefix": "dsv4",
        "precision": "fp4", "framework": "vllm", "spec-decoding": "mtp",
        "runner": "cluster:b300-nv", "tp": 8, "pp": 1, "dcp-size": 1,
        "pcp-size": 1, "ep": 8, "dp-attn": True, "conc": 224,
        "run-eval": True, "eval-only": True,
    }
    fixed_eval_row = {
        **common, "isl": 8192, "osl": 1024, "max-model-len": 10240,
        "disagg": False, "exp-name": "fixed_eval",
    }
    agentic_eval_row = {
        **common, "kv-offloading": "none", "total-cpu-dram-gb": 0,
        "duration": 3600, "scenario-type": "agentic-coding",
        "exp-name": "agentic_eval",
    }

    monkeypatch.setattr(
        process_changelog, "get_added_lines", lambda *_: added_yaml)
    monkeypatch.setattr(
        process_changelog, "load_config_files", lambda _: {"test-config": {}})

    def fake_run(command, **kwargs):
        is_evals = "--evals-only" in command
        rows = [fixed_eval_row, agentic_eval_row] if is_evals else []
        return SimpleNamespace(stdout=json.dumps(rows))

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "process_changelog.py", "--base-ref", "base", "--head-ref", "head",
        "--changelog-file", "perf-changelog.yaml",
    ])

    process_changelog.main()

    output = json.loads(capsys.readouterr().out)
    assert [r["exp-name"] for r in output["evals"]] == ["fixed_eval"]
    assert [r["exp-name"] for r in output["agentic_evals"]] == ["agentic_eval"]
    assert output["multinode_evals"] == []


def test_eval_rows_split_into_multinode_fixed_and_agentic_buckets(
    monkeypatch,
    capsys,
):
    """Multi-node eval rows must split the same way single-node rows do:
    fixed-seq-len rows in `multinode_evals`, agentic (SWE-bench) rows in
    `multinode_agentic_evals`."""
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Mixed multi-node fixed-seq-len and agentic eval selection
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
"""
    common = {
        "image": "lmsysorg/sglang-rocm:v0.5.15", "model": "deepseek-ai/DeepSeek-V4-Pro",
        "model-prefix": "dsv4", "precision": "fp4", "framework": "sglang-disagg",
        "spec-decoding": "none", "runner": "cluster:mi355x-amds",
        "node-count": 2,
        "prefill": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
        "decode": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
        "disagg": True, "kv-p2p-transfer": "mori",
        "run-eval": True, "eval-only": True,
    }
    multinode_fixed_eval_row = {
        **common, "isl": 8192, "osl": 1024, "max-model-len": 10240,
        "conc": [64], "eval-conc": 64, "exp-name": "multinode_fixed_eval",
    }
    multinode_agentic_eval_row = {
        **common, "kv-offloading": "dram",
        "kv-offload-backend": {"name": "hicache"},
        "total-cpu-dram-gb": 2399, "duration": 3600,
        "scenario-type": "agentic-coding",
        "conc": [32], "eval-conc": 32, "exp-name": "multinode_agentic_eval",
    }

    monkeypatch.setattr(
        process_changelog, "get_added_lines", lambda *_: added_yaml)
    monkeypatch.setattr(
        process_changelog, "load_config_files", lambda _: {"test-config": {}})

    def fake_run(command, **kwargs):
        is_evals = "--evals-only" in command
        rows = (
            [multinode_fixed_eval_row, multinode_agentic_eval_row]
            if is_evals else []
        )
        return SimpleNamespace(stdout=json.dumps(rows))

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "process_changelog.py", "--base-ref", "base", "--head-ref", "head",
        "--changelog-file", "perf-changelog.yaml",
    ])

    process_changelog.main()

    output = json.loads(capsys.readouterr().out)
    assert [r["exp-name"] for r in output["multinode_evals"]] == ["multinode_fixed_eval"]
    assert [r["exp-name"] for r in output["multinode_agentic_evals"]] == ["multinode_agentic_eval"]
    assert output["evals"] == []
    assert output["agentic_evals"] == []


@pytest.fixture
def changelog_run(monkeypatch, capsys):
    """Exercise main and final validation; replace only Git/config/subprocess I/O."""
    def run(entries, cli_flags=(), generated=None, master=None):
        entries = [{"config-keys": ["config-a"], "description": ["Controlled change"],
                    "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
                    **entry} for entry in entries]
        commands = []
        monkeypatch.setattr(process_changelog, "get_added_lines", lambda *_: json.dumps(entries))
        monkeypatch.setattr(process_changelog, "load_config_files",
                            lambda _: master if master is not None else {"config-b": {}, "config-a": {}})
        monkeypatch.setattr(sys, "argv", ["process_changelog.py", "--base-ref", "base",
                            "--head-ref", "head", "--changelog-file", "perf-changelog.yaml", *cli_flags])

        def generate(command, **kwargs):
            commands.append(command)
            result = generated(command) if generated else []
            if isinstance(result, subprocess.CalledProcessError):
                if kwargs.get("check"):
                    raise result
                result = []
            return SimpleNamespace(stdout=result if isinstance(result, str) else json.dumps(result))

        monkeypatch.setattr(subprocess, "run", generate)
        process_changelog.main()
        return json.loads(capsys.readouterr().out), commands
    return run


# The table is the contract: CLI expansion preserves throughput, whereas an
# entry requesting all evals is eval-only. Trimming affects throughput alone.
@pytest.mark.parametrize("cli_flags,expected_modes", [
    ([], ("benchmark-subset", "subset", "all", "all")),
    (["--all-evals"], ("benchmark-all", "all", "all", "all")),
    (["--evals-only"], ("subset", "subset", "all", "all")),
    (["--all-evals", "--evals-only"], ("all", "all", "all", "all")),
])
@pytest.mark.parametrize("entry_flags,mode_index", [
    ({}, 0), ({"evals-only": True}, 1), ({"all-evals": True}, 2),
    ({"all-evals": True, "evals-only": True}, 3),
])
@pytest.mark.parametrize("trim", [False, True])
def test_cli_entry_mode_truth_table(changelog_run, cli_flags, expected_modes, entry_flags, mode_index, trim):
    def generate(command):
        concs = [8, 4] if "--no-evals" in command or "--all-evals" in command else [4]
        return [_fixed_matrix_row(conc) for conc in concs]

    output, commands = changelog_run([entry_flags], cli_flags + (["--trim-conc"] if trim else []), generate)
    mode = expected_modes[mode_index]
    expected_benchmarks = ([4] if trim else [8, 4]) if mode.startswith("benchmark-") else []
    assert [row["conc"] for row in output["single_node"].get("8k1k", [])] == expected_benchmarks
    assert [row["conc"] for row in output["evals"]] == ([8, 4] if mode.endswith("all") else [4])
    assert ["--no-evals" in command for command in commands] == (
        [True, False] if mode.startswith("benchmark-") else [False])
    assert ("--all-evals" in commands[-1]) == mode.endswith("all")
    assert "--evals-only" in commands[-1]
    assert _scenario_values(commands[-1]) == ["fixed-seq-len", "agentic-coding"]
    if mode.startswith("benchmark-"):
        assert "--all-evals" not in commands[0]
    assert output["changelog_metadata"]["base_ref"] == "base"
    for flag, value in entry_flags.items():
        assert output["changelog_metadata"]["entries"][0][flag] is value


def test_overlapping_wildcard_scenarios_keep_priority_and_group_order(changelog_run):
    entries = [
        {"scenario-type": ["fixed-seq-len"]},
        {"config-keys": ["config-*", "config-a"], "scenario-type": ["agentic-coding", "fixed-seq-len"]},
        {"config-keys": ["config-b"], "all-evals": True, "scenario-type": ["agentic-coding"]},
        {"config-keys": ["config-*"], "evals-only": True, "scenario-type": ["fixed-seq-len"]},
    ]
    _, commands = changelog_run(entries)
    trace = [("benchmark" if "--no-evals" in c else "all" if "--all-evals" in c else "subset",
              c[c.index("--config-keys") + 1:c.index("--config-files")], _scenario_values(c))
             for c in commands]
    assert trace == [
        ("all", ["config-b"], ["agentic-coding"]),
        ("benchmark", ["config-a"], ["fixed-seq-len"]),
        ("subset", ["config-a"], ["fixed-seq-len"]),
        ("benchmark", ["config-b"], []),
        ("benchmark", ["config-a"], ["agentic-coding"]),
        ("subset", ["config-b"], ["fixed-seq-len"]),
        ("subset", ["config-a"], ["agentic-coding"]),
    ]


@pytest.mark.parametrize("cli_flags", [["--all-evals"], ["--evals-only"], ["--all-evals", "--evals-only"]])
@pytest.mark.parametrize("trim", [False, True])
def test_append_only_rejects_cli_eval_modifiers_before_generation(changelog_run, cli_flags, trim):
    with pytest.raises(ValueError, match="append-only sweeps cannot use"):
        changelog_run([{"append-only": True}], cli_flags + (["--trim-conc"] if trim else []),
                      lambda _: pytest.fail("invalid sweep must not invoke the generator"))


@pytest.mark.parametrize("entries", [
    [{"append-only": True}, {}],
    [{"append-only": True, "all-evals": True}],
    [{"append-only": True, "evals-only": True}],
    [{"append-only": True, "eval-min-prefill-ep": 2}],
])
def test_append_only_rejects_mixed_or_entry_eval_modes(changelog_run, entries):
    with pytest.raises(ValueError, match="append-only"):
        changelog_run(entries, generated=lambda _: pytest.fail("invalid sweep must not invoke the generator"))


@pytest.mark.parametrize("stage", ["--no-evals", "--evals-only", "append-head", "append-base"])
@pytest.mark.parametrize("failure", ["exit", "json"])
def test_generator_failure_never_publishes_partial_matrix(changelog_run, monkeypatch, capsys, stage, failure):
    error = subprocess.CalledProcessError(23, ["generator"], stderr="controlled generator failure")
    append = stage.startswith("append-")
    if append:
        monkeypatch.setattr(process_changelog, "generation_inputs_at_ref", lambda _: nullcontext(
            process_changelog.GenerationInputs(["base-config"], "base-generator", "base-runners")))

    def generate(command):
        revision_stage = "append-base" if command[1] == "base-generator" else "append-head"
        if stage in command or (append and stage == revision_stage):
            return error if failure == "exit" else "not-json"
        return [_fixed_matrix_row(4), _fixed_matrix_row(8)]

    with pytest.raises(subprocess.CalledProcessError if failure == "exit" else json.JSONDecodeError) as caught:
        changelog_run([{"append-only": append}], generated=generate)
    if failure == "exit":
        assert caught.value is error
    assert capsys.readouterr().out == ("controlled generator failure\n" if failure == "exit" else "")


@pytest.mark.parametrize("keys,message", [
    (["config-a", "missing"], "not found"),
    (["config-a", "missing-*"], "No config keys matched"),
])
def test_invalid_key_after_valid_key_rejects_entire_selection(changelog_run, keys, message):
    with pytest.raises(ValueError, match=message):
        changelog_run([{"config-keys": keys}], generated=lambda _: pytest.fail("keys must resolve before generation"))


@pytest.mark.parametrize("threshold,expected", [
    (None, ["single", "default", "low", "high", "null", "invalid"]),
    (1, ["single", "default", "low", "high"]),
    (2, ["single", "low", "high"]),
    (4, ["single"]),
])
def test_eval_prefill_ep_filter_preserves_single_node_and_order(threshold, expected):
    rows = [
        {"label": "single"}, {"label": "default", "prefill": {}},
        {"label": "low", "prefill": {"ep": 2}},
        {"label": "high", "prefill": {"ep": "3"}},
        {"label": "null", "prefill": {"ep": None}},
        {"label": "invalid", "prefill": {"ep": "bad"}},
    ]
    assert [row["label"] for row in process_changelog.filter_eval_rows_by_prefill_ep(rows, threshold)] == expected
