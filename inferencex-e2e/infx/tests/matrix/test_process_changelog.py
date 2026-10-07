"""Tests for changelog-driven sweep generation."""

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infx.matrix import plan as process_changelog
from infx.matrix import revision
from infx.matrix.generate import generate_test_config_sweep
from infx.matrix.validation import validate_master_config
from infx.workflows import benchmark_schema


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


def test_append_only_scope_allows_additive_top_level_restructuring(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(tmp_path))
    recipe = tmp_path / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("{}\n")
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
            "srt-recipe-dir": "fixture",
            "router": router_a,
            "scenarios": {
                "fixed-seq-len": [
                    {
                        "isl": 8192,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 4, "conc-list": [1, 4, 8], "srt-recipe": "recipe.yaml"}
                        ],
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
        {"tp": 8, "conc-list": [12, 16], "router": router_b, "srt-recipe": "recipe.yaml"}
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


def planning_inputs() -> tuple[dict, dict]:
    runners = {"labels": {"cluster:fixture": ["node-a"]}, "clusters": {"fixture": {
        "gpus-per-node": 8, "available-cpu-dram-mib": 1024000, "arch": "x86_64",
        "scheduler": "slurm", "slurm": {"partition": "batch", "exclusive": True},
    }}}
    master = {}
    for key, multinode in (("single", False), ("multi", True)):
        shape = ({role: {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False}
                  for role in ("prefill", "decode")} if multinode else {"tp": 8})
        master[key] = {
            "image": "example/image:stable", "model": key, "model-prefix": "dsr1",
            "precision": "fp8", "framework": "sglang", "runner": "cluster:fixture",
            "multinode": multinode, "disagg": multinode,
            **({"kv-p2p-transfer": "nixl"} if multinode else {}),
            "srt-recipe-dir": "fixture",
            "scenarios": {
                "fixed-seq-len": [{"isl": 8192, "osl": 1024, "search-space": [
                    {**shape, "conc-list": [16, 32, 64], "srt-recipe": "recipe.yaml"},
                ]}],
                "agentic-coding": [{"search-space": [
                    {**shape, "conc-list": [16, 32], "srt-recipe": "recipe.yaml",
                     **({} if multinode else {"kv-offloading": "none"})},
                ]}],
            },
        }
    return master, runners


@pytest.fixture
def planning_repo(tmp_path, monkeypatch):
    """Real CLI/config/generator wiring with small, independent input recipes."""
    source = Path(__file__).resolve().parents[3]
    shutil.copytree(source / "infx", tmp_path / "infx", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/amd-master.yaml").write_text("{}\n")
    master, runners = planning_inputs()
    (tmp_path / "configs/runners.yaml").write_text(yaml.safe_dump(runners))
    (tmp_path / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(master, sort_keys=False))
    for node in ("single_node", "multi_node"):
        recipe = tmp_path / f"benchmarks/{node}/srt-slurm-recipes/fixture/recipe.yaml"
        recipe.parent.mkdir(parents=True)
        recipe.write_text("schema: 2\nroles:\n  prefill: {nodes: 1}\n  decode: {nodes: 1}\n")
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path, master, runners


@pytest.fixture
def committed_planning_repo(planning_repo):
    root, _, _ = planning_repo
    entry = {"config-keys": ["single"], "description": ["Fixture change"],
             "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
             "scenario-type": ["fixed-seq-len"], "no-evals": True}
    (root / "perf-changelog.yaml").write_text("")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (root / "perf-changelog.yaml").write_text(yaml.safe_dump([entry]))
    git("add", "perf-changelog.yaml")
    git("commit", "-qm", "head")
    head = git("rev-parse", "HEAD")
    return root, base, head


@pytest.mark.parametrize("ambient", [False, True])
def test_recovery_plans_with_the_checkouts_own_planner_and_recipes(
    committed_planning_repo, monkeypatch, tmp_path_factory, ambient
):
    from infx.workflows.recover_failed_ingest import build_config

    root, base, head = committed_planning_repo
    master_path = root / "configs/nvidia-master.yaml"
    master = yaml.safe_load(master_path.read_text())
    master["multi"]["scenarios"]["fixed-seq-len"][0]["search-space"][0][
        "srt-recipe"
    ] = "recovery-fixture.yaml"
    master_path.write_text(yaml.safe_dump(master, sort_keys=False))
    recipe = root / "benchmarks/multi_node/srt-slurm-recipes/fixture/recovery-fixture.yaml"
    recipe.write_text("schema: 2\nroles:\n  prefill: {nodes: 3}\n  decode: {nodes: 4}\n")
    changelog = root / "perf-changelog.yaml"
    entries = yaml.safe_load(changelog.read_text())
    entries[0]["config-keys"] = ["multi"]
    changelog.write_text(yaml.safe_dump(entries, sort_keys=False))
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "recipe fixture"], cwd=root, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if ambient:
        shadow = tmp_path_factory.mktemp("ambient")
        (shadow / "infx").mkdir()
        (shadow / "infx/__init__.py").write_text("raise RuntimeError('ambient tooling imported')\n")
        monkeypatch.setenv("PYTHONPATH", str(shadow))
        monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(tmp_path_factory.mktemp("other")))
    changelog.write_text("\n" + yaml.safe_dump(entries, sort_keys=False))
    output, metadata = root / "config.json", root / "metadata.json"

    result = build_config(root, base, head, 1, "perf-changelog.yaml", output, metadata)

    assert result["fixed_rows"] == 1
    assert result["agentic_rows"] == result["eval_jobs"] == 0
    config = json.loads(output.read_text())
    assert [row["conc"] for row in config["multi_node"]["8k1k"]] == [[16, 32, 64]]
    assert [row["node-count"] for row in config["multi_node"]["8k1k"]] == [7]
    written_metadata = json.loads(metadata.read_text())
    assert written_metadata["base_ref"] == base
    assert written_metadata["head_ref"] == head
    assert [entry["pr-link"] for entry in written_metadata["entries"]] == [
        "https://github.com/SemiAnalysisAI/InferenceX/pull/1"
    ]


@pytest.mark.parametrize("nested_layout", [False, True])
def test_historical_generator_uses_snapshot_recipes_not_inherited_recovery_root(
    planning_repo, monkeypatch, nested_layout
):
    root, master, _ = planning_repo
    master["multi"]["scenarios"]["fixed-seq-len"][0]["search-space"][0][
        "srt-recipe"
    ] = "snapshot.yaml"
    (root / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(master, sort_keys=False))
    recipe = root / "benchmarks/multi_node/srt-slurm-recipes/fixture/snapshot.yaml"
    recipe.write_text("schema: 2\nroles:\n  prefill: {nodes: 3}\n  decode: {nodes: 4}\n")
    if nested_layout:
        project = root / "inferencex-e2e"
        project.mkdir()
        for name in ("infx", "configs", "benchmarks"):
            shutil.move(root / name, project / name)
        recipe = project / "benchmarks/multi_node/srt-slurm-recipes/fixture/snapshot.yaml"
    for command in (
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@example.com"],
        ["add", "."],
        ["commit", "-qm", "committed recipe allocation"],
    ):
        subprocess.run(["git", *command], cwd=root, check=True)
    recipe.write_text("schema: 2\nroles:\n  prefill: {nodes: 20}\n  decode: {nodes: 30}\n")
    unrelated = root / "other-revision"
    unrelated.mkdir()
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(unrelated))
    with revision.snapshot("HEAD") as inputs:
        rows = inputs.generate(["multi"], ["--no-evals", "--scenario-type", "fixed-seq-len"])

    assert [row["node-count"] for row in rows] == [7]
    assert rows[0]["conc"] == [16, 32, 64]


def test_validator_uses_trusted_entrypoints_while_reading_another_checkout(committed_planning_repo):
    root, base, head = committed_planning_repo
    tooling = root / ".tooling"
    tooling.mkdir()
    shutil.move(root / "infx", tooling / "infx")
    shutil.rmtree(root / "utils", ignore_errors=True)
    (root / "infx").mkdir()
    (root / "infx/__init__.py").write_text("raise RuntimeError('wrong tooling checkout')\n")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONPATH"] = str(tooling)
    result = subprocess.run(
        [sys.executable, "-P", "-m", "infx.workflows.validate_perf_changelog",
         "--base-ref", base, "--head-ref", head],
        cwd=root, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "Validated perf-changelog.yaml: final newline present and matrix generated\n"
    assert result.stderr == ""


@pytest.fixture
def changelog_run(planning_repo, monkeypatch, capsys):
    def run(entries, cli_flags=()):
        entries = [{"config-keys": ["single"], "description": ["Controlled change"],
                    "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
                    **entry} for entry in entries]
        monkeypatch.setattr(process_changelog, "get_added_lines", lambda *_: json.dumps(entries))
        monkeypatch.setattr(sys, "argv", ["infx.matrix.plan", "--base-ref", "base",
                            "--head-ref", "head", "--changelog-file", "perf-changelog.yaml", *cli_flags])
        process_changelog.main()
        captured = capsys.readouterr()
        assert captured.err == ""
        monkeypatch.setattr(sys, "argv", ["benchmark_schema", "--plan"])
        monkeypatch.setattr(sys, "stdin", io.StringIO(captured.out))
        benchmark_schema.main()
        validated = capsys.readouterr()
        assert validated.err == ""
        assert validated.out == captured.out
        return json.loads(validated.out)
    return run


# Hand-worked policy: entry all-evals suppresses throughput; the CLI modifier
# expands evals while retaining throughput. Trimming affects throughput alone.
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
    output = changelog_run([entry_flags], cli_flags + (["--trim-conc"] if trim else []))
    mode = expected_modes[mode_index]
    expected_benchmarks = ([16] if trim else [16, 32, 64]) if mode.startswith("benchmark-") else []
    assert [r["conc"] for r in output["single_node"].get("8k1k", [])] == expected_benchmarks
    assert [r["conc"] for r in output["evals"]] == ([16, 32, 64] if mode.endswith("all") else [32, 64])
    assert [r["conc"] for r in output["agentic_evals"]] == ([16, 32] if mode.endswith("all") else [32])
    assert output["changelog_metadata"]["base_ref"] == "base"
    for flag, value in entry_flags.items():
        assert output["changelog_metadata"]["entries"][0][flag] is value


def test_overlapping_scenarios_preserve_eval_precedence_and_buckets(changelog_run):
    output = changelog_run([
        {"config-keys": ["*"], "scenario-type": ["fixed-seq-len"]},
        {"config-keys": ["single"], "all-evals": True, "scenario-type": ["agentic-coding"]},
        {"config-keys": ["single", "multi"], "scenario-type": ["agentic-coding", "fixed-seq-len"]},
        {"config-keys": ["multi"], "all-evals": True, "scenario-type": ["fixed-seq-len"]},
    ])
    assert [r["conc"] for r in output["evals"]] == [32, 64]
    assert [r["conc"] for r in output["agentic_evals"]] == [16, 32]
    assert [r["conc"] for r in output["multinode_evals"]] == [[16, 32, 64]]
    assert [r["conc"] for r in output["multinode_agentic_evals"]] == [[32]]
    assert [r["conc"] for r in output["single_node"]["8k1k"]] == [16, 32, 64]
    assert [r["conc"] for r in output["multi_node"]["8k1k"]] == [[16, 32, 64]]
    assert [r["conc"] for r in output["single_node"]["agentic"]] == [16, 32]
    assert [r["conc"] for r in output["multi_node"]["agentic"]] == [[16], [32]]
    # Generated rows are independent across throughput and eval passes.
    assert all(r.get("run-eval", False) is False for bucket in output["single_node"].values() for r in bucket)
    assert all(r["eval-only"] is True for r in output["evals"] + output["agentic_evals"])


@pytest.mark.parametrize("skip", [False, True])
def test_no_evals_preserves_throughput_and_other_entries(changelog_run, skip):
    output = changelog_run([{"no-evals": skip}, {"config-keys": ["multi"]}])
    assert [r["conc"] for r in output["single_node"]["8k1k"]] == [16, 32, 64]
    assert [r["conc"] for r in output["evals"]] == ([] if skip else [32, 64])
    assert [r["conc"] for r in output["multinode_evals"]] == [[16, 32, 64]]
    assert output["changelog_metadata"]["entries"][0]["no-evals"] is skip


@pytest.mark.parametrize("flags", [{"evals-only": True}, {"all-evals": True}, {"eval-min-prefill-ep": 2}])
def test_no_evals_rejects_conflicting_entry_options(changelog_run, flags):
    with pytest.raises(ValueError, match="no-evals cannot be combined"):
        changelog_run([{"no-evals": True, **flags}])


@pytest.mark.parametrize("flag", ["--all-evals", "--evals-only"])
def test_no_evals_rejects_conflicting_pr_modifiers(changelog_run, flag):
    with pytest.raises(ValueError, match="no-evals entries cannot use"):
        changelog_run([{"no-evals": True}], [flag])


@pytest.mark.parametrize("cli_flags", [["--all-evals"], ["--evals-only"], ["--all-evals", "--evals-only"]])
@pytest.mark.parametrize("trim", [False, True])
def test_append_only_rejects_cli_eval_modifiers_before_generation(changelog_run, cli_flags, trim):
    with pytest.raises(ValueError, match="append-only sweeps cannot use"):
        changelog_run([{"append-only": True}], cli_flags + (["--trim-conc"] if trim else []))


@pytest.mark.parametrize("entries", [
    [{"append-only": True}, {}], [{"append-only": True, "all-evals": True}],
    [{"append-only": True, "evals-only": True}], [{"append-only": True, "eval-min-prefill-ep": 2}],
])
def test_append_only_rejects_mixed_or_entry_eval_modes(changelog_run, entries):
    with pytest.raises(ValueError, match="append-only"):
        changelog_run(entries)


@pytest.mark.parametrize("keys,message", [
    (["single", "missing"], "not found"), (["single", "missing-*"], "No config keys matched"),
])
def test_invalid_key_after_valid_key_rejects_entire_selection(changelog_run, keys, message, planning_repo):
    # Key errors precede runner loading, even when that file is invalid.
    (planning_repo[0] / "configs/runners.yaml").write_text("invalid\n")
    with pytest.raises(ValueError, match=message):
        changelog_run([{"config-keys": keys}])


@pytest.mark.parametrize("trim", [False, True])
def test_append_only_main_runs_only_added_points_and_skips_evals(planning_repo, changelog_run, trim):
    root, master, _ = planning_repo
    git = lambda *args: subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("add", ".")
    git("commit", "-qm", "base")
    git("tag", "base")
    master["single"]["scenarios"]["fixed-seq-len"][0]["search-space"][0]["conc-list"].append(128)
    (root / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(master, sort_keys=False))
    output = changelog_run([{"append-only": True, "scenario-type": ["fixed-seq-len"]}],
                           ["--trim-conc"] if trim else [])
    rows = output["single_node"]["8k1k"]
    assert [r["conc"] for r in rows] == [128]
    assert len(rows[0]["recipe-fingerprint"]) == 64
    assert output["evals"] == output["agentic_evals"] == []


def test_generation_failure_does_not_publish_a_partial_matrix(planning_repo, changelog_run, capsys):
    # Single-node generation succeeds before multinode scheduling fails.
    (planning_repo[0] / "benchmarks/multi_node/srt-slurm-recipes/fixture/recipe.yaml").write_text(
        "schema: 2\n"
    )
    with pytest.raises(subprocess.CalledProcessError):
        changelog_run([{"config-keys": ["single", "multi"]}])
    captured = capsys.readouterr()
    assert "Recipe has no worker roles" in captured.out
    assert '\"single_node\":' not in captured.out


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


def test_current_plan_loads_inputs_once_and_uses_no_generator_process(planning_repo, monkeypatch):
    import builtins
    from collections import Counter
    from infx.matrix.plan import build_plan

    reads = Counter()
    real_open = builtins.open

    def counted_open(file, *args, **kwargs):
        reads[str(file)] += 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counted_open)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("current generation launched a child"))
    entries = [{"config-keys": [key], "description": ["Controlled change"],
                "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1"}
               for key in ("single", "multi")]
    result = build_plan(entries, base_ref="base", head_ref="head").model_dump(by_alias=True, exclude_none=True)
    assert [r["conc"] for r in result["single_node"]["8k1k"]] == [16, 32, 64]
    assert result["multi_node"]["8k1k"][0]["node-count"] == 2
    assert {path: reads[path] for path in (
        "configs/amd-master.yaml", "configs/nvidia-master.yaml", "configs/runners.yaml",
    )} == {"configs/amd-master.yaml": 1, "configs/nvidia-master.yaml": 1, "configs/runners.yaml": 1}


def test_generation_api_preserves_inputs_and_returns_independent_nested_rows(planning_repo):
    import copy
    from infx.matrix.generate import generate_config_matrix

    _, master, runners = planning_repo
    master["multi"]["router"] = {"name": "fixture-router", "version": "1.0"}
    original = copy.deepcopy((master, runners))
    rows = generate_config_matrix(["multi"], master, runners, eval_mode="all")
    assert [(r["conc"], r.get("eval-conc")) for r in rows] == [([16, 32, 64], None), ([16, 32], 32)]
    rows[0]["router"]["name"] = "mutated"
    rows[0]["prefill"]["tp"] = 999
    rows[0]["conc"].append(999)
    assert (master, runners) == original
    repeated = generate_config_matrix(["multi"], master, runners, eval_mode="none")
    assert repeated[0]["prefill"]["tp"] == 8
    assert repeated[0]["conc"] == [16, 32, 64]


def test_generation_api_rejects_unknown_eval_mode(planning_repo):
    from infx.matrix.generate import generate_config_matrix
    _, master, runners = planning_repo
    with pytest.raises(ValueError, match="Unknown eval mode"):
        generate_config_matrix(["single"], master, runners, eval_mode="typo")


def test_plan_failure_after_throughput_generation_publishes_nothing(planning_repo, changelog_run, monkeypatch, capsys):
    from infx.matrix import plan
    generate = plan.generate_config_matrix

    def fail_eval(*args, **kwargs):
        if kwargs["eval_mode"] == "subset":
            raise ValueError("controlled eval selection failure")
        return generate(*args, **kwargs)

    monkeypatch.setattr(plan, "generate_config_matrix", fail_eval)
    with pytest.raises(subprocess.CalledProcessError):
        changelog_run([{}])
    output = capsys.readouterr().out
    assert "controlled eval selection failure" in output
    assert '"single_node":' not in output


def test_plan_rejects_empty_changelog_before_reading_inputs():
    from infx.matrix.plan import build_plan
    with pytest.raises(ValueError, match="No valid YAML entries"):
        build_plan([], base_ref="base", head_ref="head", config_files=["missing.yaml"])


def test_generation_api_preserves_json_rejection_for_yaml_sets(planning_repo):
    from infx.matrix.generate import generate_config_matrix
    _, master, runners = planning_repo
    # Pydantic accepts this YAML set as a list, but validation returns the raw
    # config. The generator CLI has always rejected it at JSON serialization.
    worker = master["multi"]["scenarios"]["fixed-seq-len"][0]["search-space"][0]["prefill"]
    worker["additional-settings"] = {"A=1"}
    validate_master_config(master)
    with pytest.raises(TypeError, match="set is not JSON serializable"):
        generate_config_matrix(["multi"], master, runners, eval_mode="none")


def test_changelog_move_preserves_history_and_selects_only_additions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "perf-changelog.yaml").write_text("historical entry\n")
    git("add", ".")
    git("commit", "-qm", "old layout")
    base = git("rev-parse", "HEAD")
    (tmp_path / "inferencex-e2e").mkdir()
    git("mv", "perf-changelog.yaml", "inferencex-e2e/")
    git("commit", "-qm", "move changelog")
    assert process_changelog.get_added_lines(base, "HEAD", "perf-changelog.yaml") == ""
    (tmp_path / "inferencex-e2e/perf-changelog.yaml").write_text(
        "historical entry\nnew entry\n"
    )
    git("commit", "-qam", "append entry")
    monkeypatch.chdir(tmp_path / "inferencex-e2e")
    assert process_changelog.get_added_lines(base, "HEAD", "perf-changelog.yaml") == "new entry"
