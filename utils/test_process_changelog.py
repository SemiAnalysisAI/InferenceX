"""Tests for changelog-driven sweep generation."""

import json
import subprocess
import sys
from types import SimpleNamespace

import process_changelog


def _scenario_values(command):
    if "--scenario-type" not in command:
        return []
    index = command.index("--scenario-type") + 1
    return command[index:]


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
    assert _scenario_values(commands[0]) == ["fixed-seq-len"]

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
    assert _scenario_values(commands[1]) == ["fixed-seq-len"]
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

    assert len(commands) == 3
    assert "--no-evals" in commands[0]
    assert _scenario_values(commands[0]) == ["fixed-seq-len"]
    assert "--evals-only" in commands[1]
    assert _scenario_values(commands[1]) == ["fixed-seq-len"]
    assert "--no-evals" in commands[2]
    assert _scenario_values(commands[2]) == ["agentic-coding"]
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

    assert len(commands) == 2
    assert "--no-evals" in commands[0]
    assert "--evals-only" in commands[1]
    assert "--all-evals" not in commands[1]
    assert _scenario_values(commands[1]) == ["fixed-seq-len"]
    json.loads(capsys.readouterr().out)
