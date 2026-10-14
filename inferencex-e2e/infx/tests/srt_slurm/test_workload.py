"""Composing recipe fragments with their shared blocks, and binding the matrix point."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.clusters.slurm import Fabric
from infx.srt_slurm.synthetic_acceptance import selected_recipes
from infx.srt_slurm.workload import (
    bind_workload,
    check_setup_script,
    compose_recipe,
    parse_concurrencies,
    resolve_fabric,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
MULTI_ENV = {
    "IMAGE": "registry/image:2", "MODEL": "org/model", "PRECISION": "fp8",
    "ISL": "8192", "OSL": "1024", "CONC_LIST": "4 16",
}  # fmt: skip
AGENTX_ENV = {"IMAGE": "nvcr.io#org/image:3", "MODEL": "org/model", "PRECISION": "fp4", "CONC_LIST": "8"}
CLIENT_ENV = {"RESULT_DIR": "/logs/agentic", "HF_HUB_CACHE": "/hf_hub_cache"}
MOONCAKE = '{"name": "mooncake", "version": "0.3.11.post1"}'
ROUTER = '{"name": "vllm-router", "version": "0.1.14"}'


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
    ({"environment": {"ROUTER_VERSION": "0.1.14"}}, True, False, "environment.ROUTER_VERSION (= '0.1.14')"),
    ({"base": {}, "override_c": {"services": [{"name": "m", "env": {"KV_OFFLOAD_BACKEND_VERSION": "1"}}]}},
     True, True, "override_c.services[0].env.KV_OFFLOAD_BACKEND_VERSION (= '1')"),
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


def test_a_component_a_script_installs_gets_the_master_version_where_the_script_runs():
    recipe = {
        "setup_script": "vllm-mooncake.sh",
        "environment": {"NCCL_DEBUG": "WARN"},
        "services": [
            {"name": "mooncake-master", "preamble": "bash /configs/vllm-mooncake.sh"},
            {"name": "etcd", "env": {"ETCD_QUOTA": "1"}},
        ],
    }
    environment = {**AGENTX_ENV, "KV_OFFLOAD_BACKEND_METADATA": MOONCAKE}

    bound = bind_workload(recipe, environment, agentic=True, multinode=True)

    assert bound["environment"] == {"NCCL_DEBUG": "WARN", "KV_OFFLOAD_BACKEND_VERSION": "0.3.11.post1"}
    assert bound["services"] == [
        {"name": "mooncake-master", "preamble": "bash /configs/vllm-mooncake.sh",
         "env": {"KV_OFFLOAD_BACKEND_VERSION": "0.3.11.post1"}},
        {"name": "etcd", "env": {"ETCD_QUOTA": "1"}},
    ]  # fmt: skip


@pytest.mark.parametrize("metadata", ["", '{"name": "mooncake"}', '{"name": "lmcache", "version": "1"}'])
def test_a_point_that_does_not_declare_an_installed_components_version_is_rejected(metadata):
    recipe = {"services": [{"name": "mooncake-master", "preamble": "bash /configs/vllm-mooncake.sh"}]}
    environment = {**AGENTX_ENV, "KV_OFFLOAD_BACKEND_METADATA": metadata}
    message = "^vllm-mooncake.sh installs mooncake, so the master config must declare"

    with pytest.raises(ValueError, match=message):
        bind_workload(recipe, environment, agentic=True, multinode=True)


def test_a_router_pinned_in_setup_pip_packages_must_be_the_master_router():
    recipe = {
        "frontend": {"type": "vllm-router", "env": {"SETUP_PIP_PACKAGES": "vllm-router==0.1.13"}},
        "roles": {"agg": {"env": {"SETUP_PIP_PACKAGES": "Pillow fastapi"}}},
    }
    environment = {**AGENTX_ENV, "ROUTER_METADATA": ROUTER}

    with pytest.raises(ValueError) as error:
        bind_workload(recipe, environment, agentic=True, multinode=True)
    assert str(error.value) == "SETUP_PIP_PACKAGES vllm-router==0.1.13 is not the master router 0.1.14"
    recipe["frontend"]["env"]["SETUP_PIP_PACKAGES"] = "vllm-router==0.1.14"
    bind_workload(recipe, environment, agentic=True, multinode=True)


def test_a_setup_script_srtctl_would_not_find_fails(project):
    source = Path("fragment.yaml")
    patches = project / "utils/srt-slurm/configs/patches"
    patches.mkdir(parents=True)
    (patches / "upstream.sh").write_text("true\n")
    check_setup_script({"setup_script": "upstream.sh"}, source, project)
    with pytest.raises(ValueError, match="^fragment.yaml: setup_script own.sh is in none of"):
        check_setup_script({"setup_script": "own.sh"}, source, project)
    configs = project / "benchmarks/multi_node/srt-slurm-recipes/configs"
    configs.mkdir(parents=True)
    (configs / "own.sh").write_text("true\n")
    check_setup_script({"setup_script": "own.sh"}, source, project)


def test_conc_list_must_be_canonical_positive_integers():
    assert parse_concurrencies(" 4 8\t16 ") == [4, 8, 16]
    for bad in ("", "08", "0", "-4", "4.0", "4 4", "+4"):
        with pytest.raises(ValueError):
            parse_concurrencies(bad)


def test_the_multinode_binder_writes_the_one_selected_variant(project, tmp_path):
    path = fragment(project, {
        "base": {"name": "bundle", "roles": {"decode": {"nodes": 1}}},
        "override_wide": {"roles": {"decode": {"nodes": 4}}},
        "override_narrow": {"roles": {"decode": {"nodes": 2, "env": {"UCX": "@fabric.rdma-device-ports"}}}},
    })
    output = tmp_path / "bound.yaml"
    env = {**os.environ, **MULTI_ENV, "IS_AGENTIC": "0", "INFERENCEX_REPOSITORY_ROOT": str(project),
           "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "utils/srt-slurm/src")])}  # fmt: skip

    def bind(*arguments: str) -> subprocess.CompletedProcess[str]:
        fabric = json.dumps({"rdma-device-ports": "mlx5_0:1,mlx5_1:1", "rdma-devices": None})
        return subprocess.run(
            [sys.executable, "-m", "infx.srt_slurm.workload", *arguments, "--fabric", fabric],
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
    bound = yaml.safe_load(output.read_text())
    assert bound["benchmark"]["command"] == "bash agentic.sh"
    assert bound["roles"]["decode"]["env"] == {"UCX": "mlx5_0:1,mlx5_1:1"}


def test_fabric_references_take_the_clusters_rendering_in_env_args_and_services():
    fabric = Fabric.model_validate({
        "rdma-device-ports": ["mlx5_0:1", "mlx5_1:1"], "rdma-devices": ["rdma0", "rdma1"],
        "rdma-gid-index": 3,
    })  # fmt: skip
    recipe = {
        "roles": {"prefill": {
            "env": {
                "UCX_NET_DEVICES": "@fabric.rdma-device-ports",
                "MC_GID_INDEX": "@fabric.rdma-gid-index",
                "OWN": "mlx5_9:1",
            },
            "args": {"disaggregation-ib-device": "@fabric.rdma-devices", "tp-size": 8},
        }},
        "services": [{"name": "etcd"}, {"name": "mooncake-master", "options": {"store_config": {
            "device_name": "@fabric.rdma-devices", "protocol": "rdma",
        }}}],
    }  # fmt: skip

    assert resolve_fabric(recipe, fabric.rendered()) == {
        "roles": {"prefill": {
            "env": {"UCX_NET_DEVICES": "mlx5_0:1,mlx5_1:1", "MC_GID_INDEX": "3", "OWN": "mlx5_9:1"},
            "args": {"disaggregation-ib-device": "rdma0,rdma1", "tp-size": 8},
        }},
        "services": [{"name": "etcd"}, {"name": "mooncake-master", "options": {"store_config": {
            "device_name": "rdma0,rdma1", "protocol": "rdma",
        }}}],
    }  # fmt: skip


@pytest.mark.parametrize("value", ["@fabric.rdma-device-port", "IBDEVICES=@fabric.rdma-devices bash setup.sh"])
def test_an_unknown_or_embedded_fabric_reference_fails(value):
    recipe = {"services": [{"command": [value]}]}
    with pytest.raises(ValueError) as error:
        resolve_fabric(recipe, Fabric().rendered())
    assert str(error.value).startswith(f"services[0].command[0]: {value!r} is not a whole '@fabric.")


def test_a_fabric_field_the_cluster_does_not_set_fails():
    fabric = Fabric.model_validate({"rdma-gid-index": 1}).rendered()
    recipe = {"roles": {"agg": {"env": {"MC_GID_INDEX": "@fabric.rdma-gid-index", "MORI_RDMA_TC": "@fabric.rdma-tc"}}}}
    with pytest.raises(ValueError, match=r"^roles\.agg\.env\.MORI_RDMA_TC: this cluster sets no srt-slurm"):
        resolve_fabric(recipe, fabric)
