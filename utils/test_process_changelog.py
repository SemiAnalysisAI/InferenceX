"""Tests for changelog-driven sweep generation."""

import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import process_changelog


def _fixed_matrix_row(conc, *, image="vllm/vllm-openai:v0.16.0"):
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
        "tp": 8,
        "pp": 1,
        "dcp-size": 1,
        "pcp-size": 1,
        "ep": 8,
        "dp-attn": True,
        "conc": conc,
        "max-model-len": 10240,
        "exp-name": f"dsv4_conc{conc}",
        "disagg": False,
        "run-eval": False,
        "eval-only": False,
    }


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
        assert "curve logic" in str(error)
    else:
        raise AssertionError("image mutation should reject append-only mode")


def test_append_only_scope_rejects_non_concurrency_recipe_changes():
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
    try:
        process_changelog.validate_append_only_scope(
            base, head, {"test-config": {"agentic-coding"}}
        )
    except ValueError as error:
        assert "duration" in str(error)
    else:
        raise AssertionError("recipe mutation should reject append-only mode")


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


def test_append_only_main_runs_only_added_points_and_skips_evals(
    monkeypatch,
    capsys,
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
        "config_files_at_ref",
        lambda *_: nullcontext(["base-nvidia.yaml", "base-amd.yaml"]),
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
        "--changelog-file", "perf-changelog.yaml",
    ])

    process_changelog.main()

    output = json.loads(capsys.readouterr().out)
    assert [row["conc"] for row in output["single_node"]["8k1k"]] == [8]
    assert output["evals"] == []
    assert output["changelog_metadata"]["entries"][0]["append-only"] is True
    assert len(commands) == 2


def test_all_evals_skips_benchmarks_and_uses_all_evals_generator_flag(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Run every eval configuration
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

    assert len(commands) == 1
    assert "--all-evals" in commands[0]
    assert "--evals-only" in commands[0]
    assert "--no-evals" not in commands[0]
    assert _scenario_values(commands[0]) == ["fixed-seq-len", "agentic-coding"]

    output = json.loads(capsys.readouterr().out)
    assert output["changelog_metadata"]["entries"][0]["all-evals"] is True


def test_regular_changelog_entry_keeps_benchmark_and_subset_eval_commands(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Run benchmarks and selected evals
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
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
    assert "--no-evals" in commands[0]
    assert "--evals-only" in commands[1]
    assert "--all-evals" not in commands[1]
    assert _scenario_values(commands[1]) == ["fixed-seq-len", "agentic-coding"]
    json.loads(capsys.readouterr().out)


def test_cli_all_evals_expands_evals_and_preserves_benchmarks(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Run every eval configuration through a PR label
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
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
    assert "--all-evals" not in commands[0]
    assert "--all-evals" in commands[1]
    assert "--evals-only" in commands[1]
    assert _scenario_values(commands[1]) == ["fixed-seq-len", "agentic-coding"]
    json.loads(capsys.readouterr().out)


def test_cli_all_evals_expands_evals_only_entry_without_benchmarks(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Expand an eval-only entry through a PR label
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  evals-only: true
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

    assert len(commands) == 1
    assert "--all-evals" in commands[0]
    assert "--evals-only" in commands[0]
    assert "--no-evals" not in commands[0]
    json.loads(capsys.readouterr().out)


def test_cli_evals_only_suppresses_benchmarks_and_keeps_default_subset(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Run only the default eval subset through a PR label
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
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
    assert "--all-evals" not in commands[0]
    assert "--no-evals" not in commands[0]
    assert _scenario_values(commands[0]) == ["fixed-seq-len", "agentic-coding"]
    json.loads(capsys.readouterr().out)


def test_cli_eval_modifiers_compose_as_all_evals_without_benchmarks(
    monkeypatch,
    capsys,
):
    added_yaml = """
- config-keys:
    - test-config
  description:
    - Run every eval and no throughput through PR labels
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
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
        "--evals-only",
    ])

    process_changelog.main()

    assert len(commands) == 1
    assert "--evals-only" in commands[0]
    assert "--all-evals" in commands[0]
    assert "--no-evals" not in commands[0]
    json.loads(capsys.readouterr().out)


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
