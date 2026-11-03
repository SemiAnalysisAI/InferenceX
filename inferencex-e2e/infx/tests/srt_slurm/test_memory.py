import sys
from pathlib import Path

import pytest

from infx.srt_slurm.memory import bind_memory

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


def test_half_node_worker_cap_preserves_helpers_and_overrides_worker_memory():
    item = recipe(
        {
            "agg": {
                "nodes": 1,
                "workers": 1,
                "gpus": 4,
                "srun_options": {"mem-per-cpu": "20M", "cpu-bind": "cores"},
            }
        },
        srun_options={"mem": "64M", "container-remap-root": ""},
        sbatch_directives={"time": "00:10:00", "mem-per-gpu": "100M"},
        telemetry={"enabled": True},
    )
    result = bind_memory(item, 8000, 8)
    assert result is item
    assert result["roles"]["agg"]["srun_options"] == {"mem": "4000M", "cpu-bind": "cores"}
    assert result["srun_options"] == {"mem": "64M", "container-remap-root": ""}
    assert result["sbatch_directives"] == {"time": "00:10:00", "mem-per-gpu": "100M"}
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "4000"
    assert result["telemetry"] == {"enabled": True}


def test_colocated_roles_have_independent_caps():
    result = bind_memory(
        recipe(
            {
                "prefill": {"nodes": 1, "workers": 1, "gpus": 2},
                "decode": {"nodes": "colocate", "workers": 2, "gpus": 3},
            }
        ),
        8000,
        8,
    )
    assert result["roles"]["prefill"]["srun_options"]["mem"] == "2000M"
    assert result["roles"]["decode"]["srun_options"]["mem"] == "3000M"
    assert "sbatch_directives" not in result
    assert result["benchmark"]["env"] == {
        "INFERENCEX_WORKER_MEMORY_MIB": "8000",
        "INFERENCEX_DECODE_MEMORY_MIB": "6000",
    }


def test_partial_multinode_worker_uses_its_actual_local_gpu_count():
    result = bind_memory(recipe({"agg": {"nodes": 2, "workers": 1, "gpus": 12}}), 8000, 8)
    assert result["roles"]["agg"]["srun_options"]["mem"] == "6000M"
    assert "sbatch_directives" not in result
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "12000"


def test_packed_mpi_with_unequal_local_gpu_counts_fails_before_submission():
    item = recipe({"agg": {"nodes": 2, "workers": 1, "gpus": 12}}, engine="trtllm")
    with pytest.raises(ValueError, match="different local GPU counts"):
        bind_memory(item, 8000, 8)
    assert "sbatch_directives" not in item


@pytest.mark.parametrize("available", [None, 0, -1, True])
def test_missing_or_invalid_memory_fails_instead_of_leaving_workers_unlimited(available):
    with pytest.raises(ValueError, match="available-cpu-dram-mib"):
        bind_memory(recipe({"agg": {"nodes": 1, "workers": 1, "gpus": 4}}), available, 8)


def test_global_per_cpu_memory_cannot_conflict_with_worker_memory():
    item = recipe(
        {"agg": {"nodes": 1, "workers": 1, "gpus": 4}}, srun_options={"mem-per-cpu": "5G"}
    )
    with pytest.raises(ValueError, match="Global srun_options"):
        bind_memory(item, 8000, 8)


def test_per_gpu_vllm_processes_split_the_endpoint_memory_between_steps():
    item = recipe(
        {
            "agg": {
                "nodes": 1,
                "workers": 1,
                "gpus": 4,
                "args": {
                    "tensor-parallel-size": 1,
                    "data-parallel-size": 4,
                    "enable-expert-parallel": True,
                },
            }
        },
        engine={"type": "vllm", "dp_launch_mode": "per_gpu"},
        frontend={"type": "dynamo"},
    )
    result = bind_memory(item, 3079397, 8)
    assert result["roles"]["agg"]["srun_options"]["mem"] == "384924M"
    assert "sbatch_directives" not in result
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "1539696"


def test_uniform_multinode_mpi_caps_each_node_without_dividing_by_task_count():
    item = recipe({"agg": {"nodes": 2, "workers": 1, "gpus": 16}}, engine="trtllm")
    result = bind_memory(item, 8000, 8)
    assert result["roles"]["agg"]["srun_options"]["mem"] == "8000M"
    assert "sbatch_directives" not in result
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "16000"


def test_logical_gpu_capacity_does_not_inflate_physical_memory_share():
    item = recipe({"agg": {"nodes": 1, "workers": 2, "gpus": 1}})
    item["resources"]["gpus_per_node"] = 2
    result = bind_memory(item, 8000, 4)
    assert result["resources"]["gpus_per_node"] == 2
    assert result["roles"]["agg"]["srun_options"]["mem"] == "2000M"
    assert "sbatch_directives" not in result
    assert result["benchmark"]["env"]["INFERENCEX_WORKER_MEMORY_MIB"] == "4000"


def test_logical_gpu_capacity_cannot_exceed_the_physical_node():
    item = recipe({"agg": {"nodes": 1, "workers": 1, "gpus": 4}})
    with pytest.raises(ValueError, match="must fit the allocated cluster"):
        bind_memory(item, 8000, 4)
