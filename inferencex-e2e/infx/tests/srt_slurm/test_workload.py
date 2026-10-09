"""Composing fixed-sequence fragments with their shared block, and binding the matrix point."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.workload import bind_workload, compose_recipe, parse_concurrencies

ROOT = Path(__file__).resolve().parents[3]
MULTI_ENV = {
    "IMAGE": "registry/image:2", "MODEL": "org/model", "PRECISION": "fp8",
    "ISL": "8192", "OSL": "1024", "CONC_LIST": "4 16",
}  # fmt: skip


@pytest.fixture
def project(tmp_path):
    """A project whose shared blocks set the client and a default env value."""
    for lane in ("single", "multi"):
        shared = tmp_path / f"configs/srt-recipes/fixed-sequence-{lane}.yaml"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text(yaml.safe_dump({"benchmark": {
            "type": "custom", "command": f"bash {lane}.sh", "env": {"TOKENIZER": "/shared"},
        }, "health_check": {"max_attempts": 10, "interval_seconds": 5}}))
    return tmp_path


def fragment(project, data, name="fragment.yaml"):
    path = project / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_fragment_values_win_over_shared_defaults_and_lists_replace(project):
    path = fragment(project, {
        "name": "plain",
        "benchmark": {"command": "bash own.sh --trust-remote-code", "env": {"HF_HOME": "/hf"}},
        "health_check": {"max_attempts": 720},
        "services": [{"name": "etcd"}],
    })

    assert compose_recipe(path, multinode=True, root=project) == {
        "name": "plain",
        "benchmark": {
            "type": "custom", "command": "bash own.sh --trust-remote-code",
            "env": {"TOKENIZER": "/shared", "HF_HOME": "/hf"},
        },
        "health_check": {"max_attempts": 720, "interval_seconds": 5},
        "services": [{"name": "etcd"}],
    }  # fmt: skip


def test_bundles_take_the_shared_block_under_base_and_keep_their_variants(project):
    shared = project / "configs/srt-recipes/fixed-sequence-single.yaml"
    shared.write_text(yaml.safe_dump({"benchmark": {"type": "custom"}, "args": [1, 2]}))
    path = fragment(project, {
        "base": {"name": "bundle", "args": [3]},
        "zip_override_conc": {"benchmark": {"env": {"CONC": ["2", "4"]}}},
    })

    assert compose_recipe(path, multinode=False, root=project) == {
        "base": {"name": "bundle", "args": [3], "benchmark": {"type": "custom"}},
        "zip_override_conc": {"benchmark": {"env": {"CONC": ["2", "4"]}}},
    }


@pytest.mark.parametrize(("data", "multinode", "reported"), [
    ({"model": {"container": "other/image:1"}}, True, "model.container (= 'other/image:1')"),
    ({"base": {"benchmark": {"env": {"ISL": "8192"}}}}, False, "base.benchmark.env.ISL (= '8192')"),
    ({"base": {}, "zip_override_c": {"benchmark": {"env": {"CONC": ["4"]}}}}, True,
     "zip_override_c.benchmark.env.CONC (= ['4'])"),
    ({"base": {}, "override_c": {"benchmark": {"concurrencies": [4]}}}, False,
     "override_c.benchmark.concurrencies (= [4])"),
])  # fmt: skip
def test_a_fragment_that_sets_a_bound_key_is_rejected(project, data, multinode, reported):
    path = fragment(project, data)
    with pytest.raises(ValueError) as error:
        compose_recipe(path, multinode=multinode, root=project)
    assert str(error.value) == (
        f"{path}: remove {reported} from the fragment; they are bound from the matrix point"
    )


def test_multinode_binding_writes_the_point_and_leaves_the_client_its_job_environment():
    recipe = {
        "schema": 2, "name": "job", "engine": "sglang",
        "model": {"stage_dir": "/raid"},
        "identity": {"container": {}, "frameworks": {"sglang": "0.5"}},
        "benchmark": {"type": "custom", "env": {"TOKENIZER": "/model"}},
    }  # fmt: skip

    bound = bind_workload(recipe, MULTI_ENV, multinode=True)

    assert bound == {
        "schema": 2, "name": "job",
        "model": {
            "stage_dir": "/raid", "path": "hf:org/model", "container": "registry/image:2",
            "precision": "fp8",
        },
        "engine": "sglang",
        "identity": {"container": {"image": "registry/image:2"}, "frameworks": {"sglang": "0.5"}},
        "benchmark": {"type": "custom", "env": {"TOKENIZER": "/model", "ISL": "8192", "OSL": "1024"}},
    }  # fmt: skip
    assert "identity" not in bind_workload({}, MULTI_ENV, multinode=True)


@pytest.mark.parametrize(("telemetry", "concurrencies"), [
    ({"enabled": True, "dcgm_exporter": {"port": 9401}}, [4, 16]),
    ({"enabled": False}, None),
])  # fmt: skip
def test_telemetry_recipes_get_the_points_concurrencies(telemetry, concurrencies):
    bound = bind_workload({"telemetry": telemetry}, MULTI_ENV, multinode=True)
    assert bound["benchmark"].get("concurrencies") == concurrencies


@pytest.mark.parametrize(("environment", "message"), [
    ({**MULTI_ENV, "IMAGE": ""}, "Missing workload input: IMAGE"),
    ({**MULTI_ENV, "OSL": "1k"}, "OSL must be a positive integer: '1k'"),
    ({**MULTI_ENV, "CONC_LIST": "4 0"}, "CONC_LIST entries must be canonical positive integers: '0'"),
])  # fmt: skip
def test_a_malformed_point_is_rejected_before_binding(environment, message):
    with pytest.raises(ValueError, match=message):
        bind_workload({}, environment, multinode=True)


def test_conc_list_must_be_canonical_positive_integers():
    assert parse_concurrencies(" 4 8\t16 ") == [4, 8, 16]
    for bad in ("", "08", "0", "-4", "4.0", "4 4", "+4"):
        with pytest.raises(ValueError):
            parse_concurrencies(bad)


def test_the_multinode_binder_writes_the_one_selected_variant(project, tmp_path):
    path = fragment(project, {
        "base": {"name": "bundle", "roles": {"decode": {"nodes": 1}}},
        "override_wide": {"roles": {"decode": {"nodes": 4}}},
        "override_narrow": {"roles": {"decode": {"nodes": 2}}},
    })
    output = tmp_path / "bound.yaml"
    env = {**os.environ, **MULTI_ENV, "INFERENCEX_REPOSITORY_ROOT": str(project),
           "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "utils/srt-slurm/src")])}  # fmt: skip

    def bind(selector: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "infx.srt_slurm.workload", f"{path}{selector}", str(output)],
            env=env, capture_output=True, text=True, check=False,
        )

    assert bind(":override_wide").returncode == 0
    bound = yaml.safe_load(output.read_text())
    assert (bound["name"], bound["roles"], bound["benchmark"]["command"]) == (
        "bundle_wide", {"decode": {"nodes": 4}}, "bash multi.sh",
    )
    output.unlink()
    result = bind(":override_*")
    assert result.returncode == 2
    assert "selects 2 variants, not one" in result.stderr
    assert not output.exists()
