"""Composing recipe fragments with their shared blocks, and binding the matrix point."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.bench.agentic.run import Plan
from infx.srt_slurm.synthetic_acceptance import selected_recipes
from infx.srt_slurm.workload import (
    bind_workload,
    compose_recipe,
    dram_budget,
    parse_concurrencies,
    resolve_dram,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
MULTI_ENV = {
    "IMAGE": "registry/image:2", "MODEL": "org/model", "PRECISION": "fp8",
    "ISL": "8192", "OSL": "1024", "CONC_LIST": "4 16",
}  # fmt: skip
AGENTX_ENV = {
    "IMAGE": "nvcr.io#org/image:3", "MODEL": "org/model", "PRECISION": "fp4", "CONC_LIST": "8",
    "KV_OFFLOADING": "none",
}  # fmt: skip
# A single-node TP8 point offloading KV to a 1731 GB budget: 216.375 GB per GPU.
DRAM_ENV = {
    "IMAGE": "org/image:1", "MODEL": "org/model", "PRECISION": "fp8", "CONC": "4", "GPU_COUNT": "8",
    "KV_OFFLOADING": "dram", "TOTAL_CPU_DRAM_GB": "1731",
}  # fmt: skip
CLIENT_ENV = {"RESULT_DIR": "/logs/agentic", "HF_HUB_CACHE": "/hf_hub_cache"}


@pytest.fixture
def project(tmp_path):
    """A project whose shared blocks set the client and a default env value."""
    for workload in ("fixed-sequence", "agentic"):
        for lane in ("single", "multi"):
            shared = tmp_path / f"configs/srt-recipes/{workload}-{lane}.yaml"
            shared.parent.mkdir(parents=True, exist_ok=True)
            shared.write_text(yaml.safe_dump({"benchmark": {
                "type": "custom", "command": f"bash {workload}-{lane}.sh", "env": {"TOKENIZER": "/shared"},
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
    # A single-node variant may name its point's KV offloading, never the budget it gets.
    ({"base": {}, "override_c": {"benchmark": {"env": {"TOTAL_CPU_DRAM_GB": "1731"}}}}, True, False,
     "override_c.benchmark.env.TOTAL_CPU_DRAM_GB (= '1731')"),
    ({"base": {"benchmark": {"env": {"KV_OFFLOADING": "dram"}}}}, True, False,
     "base.benchmark.env.KV_OFFLOADING (= 'dram')"),
])  # fmt: skip
def test_a_fragment_that_sets_a_bound_key_is_rejected(project, data, agentic, multinode, reported):
    path = fragment(project, data)
    with pytest.raises(ValueError) as error:
        compose_recipe(path, agentic=agentic, multinode=multinode, root=project)
    assert str(error.value) == f"{path}: remove {reported} from the fragment; the launcher binds them"


@pytest.mark.parametrize(("data", "reported"), [
    ({"roles": {"agg": {"env": {"LMCACHE_MAX_LOCAL_CPU_SIZE": "128"}}}},
     "roles.agg.env.LMCACHE_MAX_LOCAL_CPU_SIZE (= '128')"),
    ({"base": {}, "override_c": {"roles": {"agg": {"args": {
        "kv-transfer-config": '{"kv_connector_extra_config":{"cpu_bytes_to_use":1000}}',
    }}}}}, "override_c.roles.agg.args.kv-transfer-config.kv_connector_extra_config.cpu_bytes_to_use (= 1000)"),
    ({"services": [{"args": ["--l1-size-gb", "1296"]}]}, "services[0].args[1] (= '1296')"),
])  # fmt: skip
def test_a_fragment_that_sizes_host_dram_with_a_literal_is_rejected(project, data, reported):
    path = fragment(project, data)
    with pytest.raises(ValueError) as error:
        compose_recipe(path, agentic=True, multinode=False, root=project)
    assert str(error.value) == (
        f"{path}: set {reported} to a '@dram.<name>' value; the launcher binds the point's DRAM"
        " budget"
    )


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
        "benchmark": {"env": {"AIPERF_X": "1", "KV_OFFLOADING": "none", **CLIENT_ENV}},
    }
    own_layout = {"benchmark": {"env": {"HF_HOME": "/logs/hf"}}}
    bound = bind_workload(own_layout, AGENTX_ENV, agentic=True, multinode=True, client_env=CLIENT_ENV)
    assert bound["benchmark"]["env"] == {
        "HF_HOME": "/logs/hf", "KV_OFFLOADING": "none", "RESULT_DIR": "/logs/agentic",
    }  # fmt: skip


def test_a_dram_point_binds_its_budget_and_every_backend_size_derived_from_it(project):
    path = fragment(project, {
        "roles": {"agg": {
            "args": {
                "hicache-size": "@dram.per-gpu-gb",
                "kv-transfer-config": '{"kv_connector":"SimpleCPUOffloadConnector",'
                '"kv_connector_extra_config":{"cpu_bytes_to_use":"@dram.total-bytes",'
                '"cpu_bytes_to_use_per_rank":"@dram.per-gpu-bytes","lazy_offload":true}}',
                "kv_cache_config": {"host_cache_size": "@dram.per-gpu-bytes"},
            },
            "env": {"LMCACHE_MAX_LOCAL_CPU_SIZE": "@dram.per-gpu-gb"},
        }},
        "services": [{
            "args": ["--l1-size-gb", "@dram.total-gb"],
            "options": {"store_config": {"global_segment_size": "@dram.per-gpu-bytes"}},
        }],
    })
    composed = compose_recipe(path, agentic=True, multinode=False, root=project)

    bound = bind_workload(composed, DRAM_ENV, agentic=True, multinode=False)
    resolved = resolve_dram(bound, dram_budget(DRAM_ENV, multinode=False))

    assert resolved["benchmark"]["env"] == {
        "TOKENIZER": "/shared", "MODEL": "org/model", "CONC": "4", "KV_OFFLOADING": "dram",
        "TOTAL_CPU_DRAM_GB": "1731",
    }  # fmt: skip
    assert resolved["roles"]["agg"] == {
        "args": {
            "hicache-size": 216,
            "kv-transfer-config": '{"kv_connector":"SimpleCPUOffloadConnector",'
            '"kv_connector_extra_config":{"cpu_bytes_to_use":1731000000000,'
            '"cpu_bytes_to_use_per_rank":216375000000,"lazy_offload":true}}',
            "kv_cache_config": {"host_cache_size": 216375000000},
        },
        "env": {"LMCACHE_MAX_LOCAL_CPU_SIZE": "216"},
    }
    assert resolved["services"] == [{
        "args": ["--l1-size-gb", "1731"],
        "options": {"store_config": {"global_segment_size": 216375000000}},
    }]


@pytest.mark.parametrize(("value", "environment", "message"), [
    ("@dram.total-gb", {**DRAM_ENV, "KV_OFFLOADING": "none", "TOTAL_CPU_DRAM_GB": "0"},
     "roles.agg.args.size: '@dram.total-gb' sizes host DRAM on a point without a DRAM budget"),
    ("@dram.per-rank-gb", DRAM_ENV, "'@dram.per-rank-gb' is not a whole '@dram.<name>' value"),
    ("x@dram.total-gb", DRAM_ENV, "'x@dram.total-gb' is not a whole '@dram.<name>' value"),
])  # fmt: skip
def test_a_point_without_a_budget_or_an_unknown_reference_fails(value, environment, message):
    recipe = {"roles": {"agg": {"args": {"size": value}}}}
    bound = bind_workload(recipe, environment, agentic=True, multinode=False)
    with pytest.raises(ValueError, match=message):
        resolve_dram(bound, dram_budget(environment, multinode=False))
    if environment["KV_OFFLOADING"] == "none":
        assert bound["benchmark"]["env"] == {"MODEL": "org/model", "CONC": "4", "KV_OFFLOADING": "none"}


@pytest.mark.parametrize(("prefill", "per_gpu_gb"), [
    # TP8 x PP2 spans four-GPU nodes and fills each; TP2 covers half of one.
    ({"PREFILL_TP": "8", "PREFILL_PP_SIZE": "2", "PREFILL_PCP_SIZE": "1"}, 180),
    ({"PREFILL_TP": "2", "PREFILL_PP_SIZE": "1", "PREFILL_PCP_SIZE": "1"}, 360),
])  # fmt: skip
def test_a_multinode_budget_covers_the_prefill_workers_gpus_on_a_node(prefill, per_gpu_gb):
    environment = {"KV_OFFLOADING": "dram", "TOTAL_CPU_DRAM_GB": "721", **prefill}
    assert dram_budget(environment, multinode=True, gpus_per_node=4)["per-gpu-gb"] == per_gpu_gb
    with pytest.raises(ValueError, match="needs its cluster's gpus-per-node"):
        dram_budget(environment, multinode=True)


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


def test_a_bound_agentx_point_satisfies_the_benchmark_client(project, tmp_path):
    path = fragment(project, {"benchmark": {"env": {"AIPERF_LIVE_FAILED_REQUEST_THRESHOLD": "0.25"}}})
    composed = compose_recipe(path, agentic=True, multinode=True, root=project)
    bound = bind_workload(composed, AGENTX_ENV, agentic=True, multinode=True, client_env=CLIENT_ENV)
    job = {
        "RESULT_FILENAME": "r", "EVAL_ONLY": "false", "IS_MULTINODE": "true", "PRECISION": "fp4",
        "MODEL": "org/model", "MODEL_PREFIX": "dsv4", "FRAMEWORK": "dynamo-vllm", "CONC": "8",
        "CONC_LIST": "8", "DURATION": "3600", "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS": "3600",
        "AIPERF_EXPERIMENTAL_FAST": "0", "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID": "false",
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING": "1", "ENABLE_AGENTX_POWER": "0", "REQUIRE_POWER": "0",
        "IS_AGENTIC": "1", "KV_OFFLOADING": "none", "SRT_FRONTEND_HOST": "10.0.0.1",
        "SRT_FRONTEND_PORT": "8000",
    }  # fmt: skip

    plan = Plan.from_env({**job, **bound["benchmark"]["env"]})

    assert plan.result_dir == Path("/logs/agentic/conc_8")
    assert plan.replay.url == "http://10.0.0.1:8000"
    assert plan.replay.live_failed_request_threshold == "0.25"


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
        "override_narrow": {
            "roles": {"decode": {"nodes": 2}},
            "services": [{"options": {"store_config": {"global_segment_size": "@dram.per-gpu-bytes"}}}],
        },
    })
    output = tmp_path / "bound.yaml"
    env = {**os.environ, **MULTI_ENV, "INFERENCEX_REPOSITORY_ROOT": str(project),
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
    # A TP8 x PP2 prefill worker on four-GPU nodes: its 721 GB budget covers four GPUs.
    env.update(IS_AGENTIC="1", KV_OFFLOADING="dram", TOTAL_CPU_DRAM_GB="721", PREFILL_TP="8",
               PREFILL_PP_SIZE="2", PREFILL_PCP_SIZE="1")  # fmt: skip
    arguments = ("--power-port", "9401", "--client-env", "RESULT_DIR=/logs/agentic",
                 "--gpus-per-node", "4")  # fmt: skip
    assert bind(f"{path}:override_narrow", str(output), *arguments).returncode == 0
    bound = yaml.safe_load(output.read_text())
    assert (bound["benchmark"]["command"], bound["benchmark"]["concurrencies"]) == (
        "bash agentic-multi.sh", [4, 16],
    )
    assert bound["benchmark"]["env"]["RESULT_DIR"] == "/logs/agentic"
    assert bound["benchmark"]["env"]["TOTAL_CPU_DRAM_GB"] == "721"
    assert bound["services"] == [{"options": {"store_config": {"global_segment_size": 180250000000}}}]
    assert bound["telemetry"]["dcgm_exporter"]["port"] == 9401
