"""Tests for changelog-driven sweep generation."""

import json
import subprocess
import sys
from types import SimpleNamespace

import process_changelog


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
    assert "--evals-only" not in commands[0]
    assert "--no-evals" not in commands[0]

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
    assert "--no-evals" in commands[1]
    assert all("--evals-only" not in command for command in commands)
    json.loads(capsys.readouterr().out)
