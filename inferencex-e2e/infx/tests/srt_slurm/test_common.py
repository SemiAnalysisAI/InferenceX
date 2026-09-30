"""Behavioral checks for InferenceX composition before native SRT expansion."""

from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.common import load_recipe, merge_blocks


def test_merge_keeps_tuning_and_replaces_lists_without_mutating_inputs():
    common = {"model": {"container": "shared", "precision": "fp8"}, "items": [1, 2]}
    tuning = {"model": {"container": "special"}, "items": [3]}
    merged = merge_blocks(common, tuning)
    assert merged == {"model": {"container": "special", "precision": "fp8"}, "items": [3]}
    merged["items"].append(4)
    assert common["items"] == [1, 2]
    assert tuning["items"] == [3]


@pytest.fixture
def source(tmp_path, monkeypatch):
    sources = tmp_path / "configs/srt-recipes"
    sources.mkdir(parents=True)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("stale: export\n")
    (sources / "sources.yaml").write_text(
        "recipe.yaml: {common: fixed.yaml, tuning: tuning.yaml}\n"
    )
    (sources / "fixed.yaml").write_text(
        "base:\n  model: {path: 'hf:${MODEL}', container: '${IMAGE}', precision: '${PRECISION}'}\n"
        "  benchmark: {type: custom, env: {ISL: '${ISL}', OSL: '${OSL}'}}\n"
    )
    (sources / "tuning.yaml").write_text(
        "base:\n  roles: {agg: {args: {tokenizer-path: '${MODEL}', quantization: fp8}}}\n"
        "zip_override_conc:\n  benchmark: {env: {CONC: ['4', '8']}}\n"
    )
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(tmp_path))
    env = {
        "MODEL": "org/new-model", "IMAGE": "registry/image:next", "PRECISION": "bf16",
        "ISL": "8192", "OSL": "1024", "IS_AGENTIC": "0",
    }
    return recipe, env


def test_registered_source_renders_data_and_preserves_native_variants(source):
    recipe, env = source
    actual = load_recipe(recipe, env)
    assert actual["base"]["model"] == {
        "path": "hf:org/new-model", "container": "registry/image:next", "precision": "bf16",
    }
    assert actual["base"]["benchmark"] == {
        "type": "custom", "env": {"ISL": "8192", "OSL": "1024"},
    }
    assert actual["base"]["roles"]["agg"]["args"] == {
        "tokenizer-path": "org/new-model", "quantization": "fp8",
    }
    assert actual["zip_override_conc"] == {"benchmark": {"env": {"CONC": ["4", "8"]}}}
    # Values resembling YAML remain a scalar through serialization.
    injected = load_recipe(recipe, {**env, "IMAGE": "image\nroles: changed"})
    decoded = yaml.safe_load(yaml.safe_dump(injected))
    assert decoded["base"]["model"]["container"] == "image\nroles: changed"
    assert decoded["base"]["roles"]["agg"]["args"]["quantization"] == "fp8"


@pytest.mark.parametrize("name,value", [("MODEL", ""), ("IS_AGENTIC", "1")])
def test_source_requires_its_workload_parameters(source, name, value):
    recipe, env = source
    with pytest.raises(ValueError, match="Missing recipe parameter|require IS_AGENTIC"):
        load_recipe(recipe, {**env, name: value})


def test_unregistered_native_recipe_is_not_interpolated(source):
    recipe, env = source
    native = Path(recipe.parent / "native.yaml")
    native.write_text("roles: {agg: {env: {PATH: '${PATH}'}}}\n")
    assert load_recipe(native, env) == {"roles": {"agg": {"env": {"PATH": "${PATH}"}}}}


def test_multinode_staging_composes_before_applying_job_settings(source):
    from infx.launch.drivers.srt.recipe import compose_recipe, prepare_recipe

    recipe, env = source
    root = recipe.parent
    sources = root / "configs/srt-recipes"
    relative = "benchmarks/multi_node/srt-slurm-recipes/demo.yaml"
    original = root / relative
    original.parent.mkdir(parents=True)
    original.write_text("name: stale-export\n")
    (sources / "sources.yaml").write_text(
        yaml.safe_dump({relative: {"common": "fixed.yaml", "tuning": "tuning.yaml"}})
    )
    (sources / "fixed.yaml").write_text(
        "name: shared\nmodel: {container: '${IMAGE}'}\n"
        "benchmark: {concurrencies: '${CONCURRENCIES}'}\n"
    )
    (sources / "tuning.yaml").write_text(
        "roles:\n  decode:\n    args: {watchdog-timeout: 30}\n"
        "health_check: {max_attempts: 4}\n"
    )
    checkout = root / "job-local-checkout"
    (checkout / "recipes").mkdir(parents=True)
    compose_recipe(root, checkout, "recipes/demo.yaml", {**env, "CONC_LIST": "8 16"})
    prepare_recipe(checkout, "recipes/demo.yaml", "job-42", 900, "8")
    actual = yaml.safe_load((checkout / "recipes/demo.yaml").read_text())
    assert actual["name"] == "job-42"
    assert actual["model"] == {"container": "registry/image:next"}
    assert actual["benchmark"]["concurrencies"] == [8]
    assert actual["health_check"]["max_attempts"] == 720
    assert actual["roles"]["decode"]["args"] == {"watchdog-timeout": 30, "dist-timeout": 900}
    assert original.read_text() == "name: stale-export\n"
