"""Composing recipe fragments with their shared blocks, and binding the matrix point."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.synthetic_acceptance import selected_recipes
from infx.srt_slurm.workload import bind_workload, compose_recipe, parse_concurrencies

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
MULTI_ENV = {
    "IMAGE": "registry/image:2", "MODEL": "org/model", "PRECISION": "fp8",
    "ISL": "8192", "OSL": "1024", "CONC_LIST": "4 16",
}  # fmt: skip
AGENTX_ENV = {"IMAGE": "nvcr.io#org/image:3", "MODEL": "org/model", "PRECISION": "fp4", "CONC_LIST": "8"}
CLIENT_ENV = {"RESULT_DIR": "/logs/agentic", "HF_HUB_CACHE": "/hf_hub_cache"}


@pytest.fixture
def project(tmp_path):
    """A project whose shared blocks set the client and a default env value."""
    for block in ("fixed-sequence-single", "fixed-sequence-multi", "agentic"):
        shared = tmp_path / f"configs/srt-recipes/{block}.yaml"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text(yaml.safe_dump({"benchmark": {
            "type": "custom", "command": f"bash {block}.sh", "env": {"TOKENIZER": "/shared"},
        }, "health_check": {"max_attempts": 10, "interval_seconds": 5}}))
    (tmp_path / "configs/srt-recipes/telemetry-dcgm.yaml").write_text(yaml.safe_dump({"telemetry": {
        "enabled": True, "required": True, "dcgm_exporter": {"container_image": "dcgm-exporter"},
    }}))  # fmt: skip
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

    assert compose_recipe(path, agentic=False, multinode=True, root=project) == {
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

    assert compose_recipe(path, agentic=False, multinode=False, root=project) == {
        "base": {"name": "bundle", "args": [3], "benchmark": {"type": "custom"}},
        "zip_override_conc": {"benchmark": {"env": {"CONC": ["2", "4"]}}},
    }


@pytest.mark.parametrize(("data", "agentic", "multinode", "reported"), [
    ({"model": {"container": "other/image:1"}}, False, True, "model.container (= 'other/image:1')"),
    ({"base": {"benchmark": {"env": {"ISL": "8192"}}}}, False, False, "base.benchmark.env.ISL (= '8192')"),
    ({"base": {}, "zip_override_c": {"benchmark": {"env": {"CONC": ["4"]}}}}, False, True,
     "zip_override_c.benchmark.env.CONC (= ['4'])"),
    ({"base": {}, "override_c": {"benchmark": {"concurrencies": [4]}}}, False, False,
     "override_c.benchmark.concurrencies (= [4])"),
    ({"telemetry": {"enabled": True}}, False, True, "telemetry.enabled (= True)"),
    ({"benchmark": {"env": {"HF_HUB_CACHE": "/elsewhere"}}}, True, True,
     "benchmark.env.HF_HUB_CACHE (= '/elsewhere')"),
    ({"base": {"benchmark": {"env": {"RESULT_DIR": "/logs"}}}}, True, False,
     "base.benchmark.env.RESULT_DIR (= '/logs')"),
    ({"base": {}, "override_tp8": {"benchmark": {"env": {
        "IS_MULTINODE": "false", "ENABLE_AGENTX_POWER": "0",
    }}}}, True, True,
     "override_tp8.benchmark.env.IS_MULTINODE (= 'false'), "
     "override_tp8.benchmark.env.ENABLE_AGENTX_POWER (= '0')"),
    # The workflow exports the matrix point; a fragment pins none of it.
    ({"benchmark": {"env": {"REQUIRE_POWER": "1", "MODEL": "/model"}}}, True, True,
     "benchmark.env.MODEL (= '/model'), benchmark.env.REQUIRE_POWER (= '1')"),
])  # fmt: skip
def test_a_fragment_that_sets_a_bound_key_is_rejected(project, data, agentic, multinode, reported):
    path = fragment(project, data)
    with pytest.raises(ValueError, match=re.escape(f"remove {reported} from the fragment")):
        compose_recipe(path, agentic=agentic, multinode=multinode, root=project)


def test_multinode_binding_writes_the_point_and_leaves_the_client_its_job_environment():
    recipe = {
        "schema": 2, "name": "job", "engine": "sglang",
        "model": {"stage_dir": "/raid"},
        "identity": {"container": {}, "frameworks": {"sglang": "0.5"}},
        "benchmark": {"type": "custom", "env": {"TOKENIZER": "/model"}},
    }  # fmt: skip

    bound = bind_workload(recipe, MULTI_ENV, agentic=False, multinode=True)

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
    assert "identity" not in bind_workload({}, MULTI_ENV, agentic=False, multinode=True)


def test_agentx_binding_writes_the_launchers_client_paths_without_lengths():
    recipe = {"identity": {"container": {}}, "benchmark": {"env": {"AIPERF_X": "1"}}}

    bound = bind_workload(recipe, AGENTX_ENV, agentic=True, multinode=True, client_env=CLIENT_ENV)

    assert bound == {
        "model": {"path": "hf:org/model", "container": "nvcr.io#org/image:3", "precision": "fp4"},
        "identity": {"container": {"image": "nvcr.io/org/image:3"}},
        "benchmark": {"env": {"AIPERF_X": "1", **CLIENT_ENV}},
    }
    own_layout = {"benchmark": {"env": {"HF_HOME": "/logs/hf"}}}
    bound = bind_workload(own_layout, AGENTX_ENV, agentic=True, multinode=True, client_env=CLIENT_ENV)
    assert bound["benchmark"]["env"] == {"HF_HOME": "/logs/hf", "RESULT_DIR": "/logs/agentic"}


def test_a_power_point_gets_telemetry_and_its_concurrencies_on_a_bundle_variant(project):
    path = fragment(project, {
        "base": {"name": "bundle", "telemetry": {"collector_join_timeout_seconds": 12}},
        "override_c8": {"roles": {"agg": {"nodes": 2}}},
    })

    composed = compose_recipe(path, agentic=True, multinode=True, root=project, power_port=19401)
    [(_, variant)] = selected_recipes(composed, "override_c8")
    bound = bind_workload(variant, AGENTX_ENV, agentic=True, multinode=True)

    assert bound["telemetry"] == {
        "collector_join_timeout_seconds": 12, "enabled": True, "required": True,
        "dcgm_exporter": {"container_image": "dcgm-exporter", "port": 19401},
    }  # fmt: skip
    assert bound["benchmark"]["concurrencies"] == [8]
    unpowered = compose_recipe(path, agentic=True, multinode=True, root=project)["base"]
    assert "concurrencies" not in bind_workload(unpowered, AGENTX_ENV, agentic=True, multinode=True)[
        "benchmark"
    ]


@pytest.mark.parametrize(("environment", "message"), [
    ({**MULTI_ENV, "IMAGE": ""}, "Missing workload input: IMAGE"),
    ({**MULTI_ENV, "OSL": "1k"}, "OSL must be a positive integer: '1k'"),
    ({**MULTI_ENV, "CONC_LIST": "4 0"}, "CONC_LIST entries must be canonical positive integers: '0'"),
])  # fmt: skip
def test_a_malformed_point_is_rejected_before_binding(environment, message):
    with pytest.raises(ValueError, match=message):
        bind_workload({}, environment, agentic=False, multinode=True)


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
    env = {**os.environ, **MULTI_ENV, "IS_AGENTIC": "0", "INFERENCEX_REPOSITORY_ROOT": str(project),
           "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "utils/srt-slurm/src")])}  # fmt: skip

    def bind(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "infx.srt_slurm.workload", *arguments],
            env=env, capture_output=True, text=True, check=False,
        )

    assert bind(f"{path}:override_wide", str(output)).returncode == 0
    bound = yaml.safe_load(output.read_text())
    assert (bound["name"], bound["roles"], bound["benchmark"]["command"]) == (
        "bundle_wide", {"decode": {"nodes": 4}}, "bash fixed-sequence-multi.sh",
    )
    output.unlink()
    result = bind(f"{path}:override_*", str(output))
    assert result.returncode == 2
    assert "selects 2 variants, not one" in result.stderr
    assert not output.exists()
    env["IS_AGENTIC"] = "1"
    assert bind(f"{path}:override_narrow", str(output)).returncode == 0
    assert yaml.safe_load(output.read_text())["benchmark"]["command"] == "bash agentic.sh"
