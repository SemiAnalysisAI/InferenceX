import sys
from pathlib import Path

import pytest

from infx.launch.resources import WorkerResources
from infx.srt_slurm.memory import bind_memory, bind_worker_topology

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))


def recipe(roles, engine="sglang", **extra):
    return {
        "schema": 2,
        "name": "memory-share",
        "model": {"path": "hf:org/model", "container": "engine:tag", "precision": "fp8"},
        "resources": {"gpus_per_node": 8},
        "engine": engine,
        "roles": roles,
        "benchmark": {"type": "custom", "command": "true"},
        **extra,
    }


def test_master_request_drives_topology_and_preserves_helpers():
    item = recipe(
        {
            "agg": {
                "nodes": 9,
                "workers": 9,
                "gpus": 8,
                "srun_options": {"mem-per-cpu": "20M", "cpu-bind": "cores"},
            }
        },
        srun_options={"mem": "64M", "container-remap-root": ""},
        sbatch_directives={"time": "00:10:00", "mem-per-gpu": "100M"},
        telemetry={"enabled": True},
    )
    result = bind_memory(item, 8000, 8, {"agg": WorkerResources(1, 4)})
    assert result is item
    assert result["roles"]["agg"] == {
        "nodes": 1, "workers": 1, "gpus": 4,
        "srun_options": {"mem": "4000M", "cpu-bind": "cores"},
    }
    assert result["srun_options"] == {"mem": "64M", "container-remap-root": ""}
    assert result["sbatch_directives"] == {"time": "00:10:00", "mem-per-gpu": "100M"}
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "4000"
    assert result["telemetry"] == {"enabled": True}


@pytest.mark.parametrize("engine", ["sglang", "trtllm"])
def test_colocated_roles_have_independent_caps(engine):
    result = bind_memory(
        recipe({"prefill": {}, "decode": {"nodes": "colocate"}}, engine=engine),
        8000, 8,
        {"prefill": WorkerResources(1, 2), "decode": WorkerResources(2, 3)},
    )
    assert result["roles"]["prefill"]["nodes"] == 1
    assert result["roles"]["decode"]["nodes"] == "colocate"
    assert result["roles"]["prefill"]["srun_options"]["mem"] == "2000M"
    assert result["roles"]["decode"]["srun_options"]["mem"] == "3000M"
    assert "sbatch_directives" not in result
    assert result["benchmark"]["env"] == {
        "INFERENCEX_WORKER_MEMORY_MIB": "8000",
        "INFERENCEX_DECODE_MEMORY_MIB": "6000",
    }


def test_colocated_roles_reserve_enough_nodes_for_both_roles():
    result = bind_memory(
        recipe({"prefill": {}, "decode": {"nodes": "colocate"}}), 8000, 8,
        {"prefill": WorkerResources(2, 4), "decode": WorkerResources(2, 4)},
    )
    assert result["roles"]["prefill"]["nodes"] == 2
    assert result["roles"]["prefill"]["srun_options"]["mem"] == "4000M"
    assert result["roles"]["decode"]["srun_options"]["mem"] == "4000M"
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "16000"


@pytest.mark.parametrize("engine", ["sglang", "trtllm"])
def test_dep16_worker_gets_one_full_budget_on_each_hgx8_node(engine):
    result = bind_memory(
        recipe({"agg": {}}, engine=engine), 8000, 8, {"agg": WorkerResources(1, 16)}
    )
    assert result["roles"]["agg"] == {
        "nodes": 2, "workers": 1, "gpus": 16, "srun_options": {"mem": "8000M"},
    }
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "16000"


@pytest.mark.parametrize("engine", ["sglang", "trtllm"])
def test_partial_multinode_worker_is_rejected(engine):
    item = recipe({"agg": {}}, engine=engine)
    with pytest.raises(ValueError, match="spanning nodes must be a multiple"):
        bind_memory(item, 8000, 8, {"agg": WorkerResources(1, 12)})
    assert "srun_options" not in item["roles"]["agg"]


@pytest.mark.parametrize("available", [None, 0, True])
def test_missing_or_invalid_memory_fails_instead_of_leaving_workers_unlimited(available):
    with pytest.raises(ValueError, match="available-cpu-dram-mib"):
        bind_memory(recipe({"agg": {}}), available, 8, {"agg": WorkerResources(1, 4)})


def test_global_per_cpu_memory_cannot_conflict_with_worker_memory():
    item = recipe({"agg": {}}, srun_options={"mem-per-cpu": "5G"})
    with pytest.raises(ValueError, match="Global srun_options"):
        bind_memory(item, 8000, 8, {"agg": WorkerResources(1, 4)})


@pytest.mark.parametrize("tp,dp,limit,total", [(1, 4, "384924M", "1539696"), (2, 2, "769849M", "1539698")])
def test_vllm_dp_steps_split_the_logical_worker_budget(tp, dp, limit, total):
    item = recipe(
        {"agg": {"args": {
            "tensor-parallel-size": tp,
            "data-parallel-size": dp,
            "enable-expert-parallel": True,
        }}},
        engine={"type": "vllm", "dp_launch_mode": "per_gpu"},
        frontend={"type": "dynamo"},
    )
    result = bind_memory(item, 3079397, 8, {"agg": WorkerResources(1, 4)})
    assert result["roles"]["agg"]["srun_options"]["mem"] == limit
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == total


def test_logical_gpu_capacity_does_not_inflate_physical_memory_share():
    item = recipe({"agg": {}})
    item["resources"]["gpus_per_node"] = 2
    result = bind_memory(item, 8000, 4, {"agg": WorkerResources(3, 1)})
    assert result["resources"]["gpus_per_node"] == 2
    assert result["roles"]["agg"]["nodes"] == 2
    assert result["roles"]["agg"]["srun_options"]["mem"] == "2000M"
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "6000"


def test_logical_gpu_capacity_cannot_exceed_the_physical_node():
    with pytest.raises(ValueError, match="must fit the allocated cluster"):
        bind_memory(recipe({"agg": {}}), 8000, 4, {"agg": WorkerResources(1, 4)})


def test_logical_capacity_cannot_split_a_physical_node_worker():
    item = recipe({"agg": {}})
    item["resources"]["gpus_per_node"] = 2
    with pytest.raises(ValueError, match="would split the requested node-local worker"):
        bind_memory(item, 8000, 4, {"agg": WorkerResources(1, 4)})


def test_partial_workers_are_packed_without_splitting_them():
    item = recipe({"agg": {}})
    result = bind_worker_topology(item, {"agg": WorkerResources(5, 3)}, 8)
    assert result["roles"]["agg"]["nodes"] == 3


@pytest.mark.parametrize("roles", [None, {}, {"agg": None}, {"agg": {}, "decode": {}}])
def test_every_recipe_worker_needs_an_explicit_request(roles):
    with pytest.raises(ValueError, match="roles must match"):
        bind_memory(recipe(roles), 8000, 8, {"agg": WorkerResources(1, 4)})


def test_shadow_engines_cannot_each_duplicate_the_logical_worker_budget():
    item = recipe(
        {"agg": {"args": {"tensor-parallel-size": 4}}},
        engine={"type": "vllm", "failover": {"shadow_engines": 1}},
        frontend={"type": "dynamo"},
    )
    with pytest.raises(ValueError, match="must partition"):
        bind_memory(item, 8000, 8, {"agg": WorkerResources(1, 4)})
