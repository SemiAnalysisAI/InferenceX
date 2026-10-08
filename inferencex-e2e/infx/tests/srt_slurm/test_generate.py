"""``infx generate``: bound srt-slurm recipes of master-config points, validated by srtctl."""

import json
from pathlib import Path

import pytest
import yaml

from infx.cli import main
from infx.launch.drivers.srt import lanes, power
from infx.launch.policy import LaunchPath, Match

ROOT = Path(__file__).resolve().parents[3]
FRAGMENT = {
    "base": {
        "schema": 2, "name": "fixture", "engine": "sglang",
        "resources": {"gpu_type": "h200", "gpus_per_node": 8},
        "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4, "args": {
            "tensor-parallel-size": 4, "speculative-algorithm": "EAGLE", "speculative-num-steps": 2,
        }, "env": {"SGLANG_SIMULATE_ACC_LEN": "2.5"}}},
    },
    "zip_override_conc": {"benchmark": {"env": {"CONC": ["2", "4"]}}},
}  # fmt: skip
RECIPE = {
    "schema": 2, "name": "agentic", "engine": "sglang",
    "resources": {"gpu_type": "h200", "gpus_per_node": 8},
}  # fmt: skip
ROLE = {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False}
MASTER = {
    "fixture-sglang": {
        "image": "org/image:1", "model": "org/model", "model-prefix": "dsr1", "precision": "fp8",
        "framework": "sglang", "runner": "fixture", "multinode": False, "srt-recipe-dir": "fixture",
        "scenarios": {"fixed-seq-len": [{"isl": 1024, "osl": 128, "search-space": [
            {"tp": 4, "conc-start": 2, "conc-end": 4, "spec-decoding": "mtp", "srt-recipe": "recipe.yaml"},
        ]}]},
    },
    "agentx-single": {
        "image": "org/image:1", "model": "org/model", "model-prefix": "dsr1", "precision": "fp8",
        "framework": "sglang", "runner": "cluster:fixture", "multinode": False,
        "srt-recipe-dir": "fixture", "scenarios": {"agentic-coding": [{"search-space": [
            {"tp": 4, "conc-list": [2], "kv-offloading": "none", "srt-recipe": "agentic.yaml"},
        ]}]},
    },
    "agentx-multi": {
        "image": "org/image:1", "model": "org/model", "model-prefix": "dsr1", "precision": "fp8",
        "framework": "dynamo-sglang", "runner": "cluster:fixture", "multinode": True,
        "disagg": True, "kv-p2p-transfer": "nixl", "srt-recipe-dir": "fixture",
        "scenarios": {"agentic-coding": [{"search-space": [{
            "prefill": ROLE, "decode": ROLE, "conc-list": [8], "srt-recipe": "agentic.yaml",
            "power": True,
        }]}]},
    },
    "agentx-multi-dram": {
        "image": "org/image:1", "model": "org/model", "model-prefix": "dsr1", "precision": "fp8",
        "framework": "dynamo-sglang", "runner": "cluster:fixture", "multinode": True,
        "disagg": True, "kv-p2p-transfer": "nixl", "srt-recipe-dir": "fixture",
        "scenarios": {"agentic-coding": [{"dram-utilization": 0.5, "search-space": [{
            "prefill": ROLE, "decode": ROLE, "conc-list": [8], "srt-recipe": "agentic-dram.yaml",
            "kv-offloading": "dram", "kv-offload-backend": {"name": "hicache"},
        }]}]},
    },
}  # fmt: skip
SHARED = {
    "fixed-sequence-single.yaml": {"benchmark": {"type": "custom", "command": "bash client.sh"}},
    "agentic-single.yaml": {"benchmark": {"type": "custom", "command": "bash srt_agentic.sh"}},
    "agentic-multi.yaml": {"benchmark": {"type": "custom", "command": "bash srt_agentic.sh"}},
    "telemetry-dcgm.yaml": {"telemetry": {"enabled": True, "dcgm_exporter": {
        "container_image": "dcgm-exporter", "command": "dcgm-exporter --address :{port}",
    }}},
}  # fmt: skip


@pytest.fixture
def project(tmp_path):
    for name, block in SHARED.items():
        (tmp_path / "configs/srt-recipes" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "configs/srt-recipes" / name).write_text(yaml.safe_dump(block))
    (tmp_path / "configs/master.yaml").write_text(yaml.safe_dump(MASTER))
    (tmp_path / "configs/runners.yaml").write_text(yaml.safe_dump({
        "labels": {"fixture": ["fixture_0"], "cluster:fixture": ["fixture_0"]},
        "clusters": {"fixture": {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm", "slurm": {
            "partition": "p", "exclusive": True,
            "volumes": {"hf-hub-cache": {"path": "/srv/hf"}, "aiperf-cache": {"path": "/srv/aiperf"}},
            "srt-slurm": {
                "network-interface": "eth0", "power-exporter-port": 9555,
                "volume-mounts": {"hf-hub-cache": "/hf"},
                "agentic-volume-mounts": {"aiperf-cache": "/aiperf"},
            },
        }, "available-cpu-dram-mib": 1_000_000}},
    }))  # fmt: skip
    recipes = {
        "single_node/srt-slurm-recipes/fixture/recipe.yaml": FRAGMENT,
        "single_node/srt-slurm-recipes/fixture/agentic.yaml": {
            **RECIPE, "frontend": {"type": "sglang", "enable_multiple_frontends": False},
            "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4, "args": {"tensor-parallel-size": 4}}},
        },
        "multi_node/srt-slurm-recipes/fixture/agentic.yaml": {
            **RECIPE, "frontend": {"type": "sglang-router"}, "roles": {
                role: {"nodes": 1, "workers": 1, "gpus": 8, "args": {"tensor-parallel-size": 8}}
                for role in ("prefill", "decode")
            },
        },
        "multi_node/srt-slurm-recipes/fixture/agentic-dram.yaml": {
            **RECIPE, "frontend": {"type": "sglang-router"}, "roles": {
                "prefill": {"nodes": 1, "workers": 1, "gpus": 8, "args": {
                    "tensor-parallel-size": 8, "hicache-size": "@dram.per-gpu-gb",
                }},
                "decode": {"nodes": 1, "workers": 1, "gpus": 8, "args": {"tensor-parallel-size": 8}},
            },
        },
    }  # fmt: skip
    for path, data in recipes.items():
        (tmp_path / "benchmarks" / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "benchmarks" / path).write_text(yaml.safe_dump(data))
    (tmp_path / "utils/srt-slurm").mkdir(parents=True)
    (tmp_path / "utils/srt-slurm/src").symlink_to(ROOT / "utils/srt-slurm/src")
    return tmp_path


@pytest.fixture
def power_lane(monkeypatch):
    """The fixture cluster's multi-node lane, measuring AgentX power."""
    monkeypatch.setitem(
        lanes.SRT_LANES, ("fixture", LaunchPath.SRT_MULTI), lanes.SrtLane(agentic_result_dir="/results")
    )
    rules = (power.PowerRule(Match(agentic=True), agentx=True),)
    monkeypatch.setitem(
        power.POWER_LANES, ("fixture", LaunchPath.SRT_MULTI), power.PowerLane(rules, error="no power")
    )


def generate(project: Path, output: Path, key: str = "fixture-*") -> int:
    return main([
        "generate", "--config-key", key, "--output-dir", str(output),
        "--config-file", str(project / "configs/master.yaml"),
        "--runner-config", str(project / "configs/runners.yaml"),
    ])  # fmt: skip


def written(output: Path) -> list[tuple[dict, dict]]:
    """Each manifest record with the recipe it names."""
    records = json.loads((output / "manifest.json").read_text())["recipes"]
    return [(record, yaml.safe_load((output / record["file"]).read_text())) for record in records]


def test_each_point_gets_its_bound_variant_and_a_manifest_entry(project, tmp_path):
    output = tmp_path / "out"
    assert generate(project, output) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert [(r["config-key"], r["variant"], r["matrix"]["conc"]) for r in manifest["recipes"]] == [
        ("fixture-sglang", "zip_override_conc[0]", 2),
        ("fixture-sglang", "zip_override_conc[1]", 4),
    ]
    recipe = yaml.safe_load((output / manifest["recipes"][1]["file"]).read_text())
    assert recipe["model"] == {"path": "hf:org/model", "container": "org/image:1", "precision": "fp8"}
    assert recipe["benchmark"] == {"type": "custom", "command": "bash client.sh", "env": {
        "CONC": "4", "ISL": "1024", "OSL": "128", "MODEL": "org/model",
        "RANDOM_RANGE_RATIO": "0.8", "USE_CHAT_TEMPLATE": "true",
    }}  # fmt: skip
    # Fixed-sequence runs verify real drafts; simulated acceptance is unset.
    assert recipe["roles"]["agg"]["env"] == {}


def test_a_single_node_agentx_point_gets_the_agentic_client_and_no_sequence_lengths(
    project, tmp_path
):
    output = tmp_path / "out"
    assert generate(project, output, "agentx-single") == 0

    [(record, recipe)] = written(output)
    assert (record["variant"], record["cluster"]) == (None, "fixture")
    assert recipe["model"] == {"path": "hf:org/model", "container": "org/image:1", "precision": "fp8"}
    assert recipe["benchmark"] == {"type": "custom", "command": "bash srt_agentic.sh", "env": {
        "MODEL": "org/model", "CONC": "2", "KV_OFFLOADING": "none",
    }}  # fmt: skip


def test_a_multinode_agentx_power_point_gets_its_clusters_exporter_port_and_client_paths(
    project, power_lane, tmp_path
):
    output = tmp_path / "out"
    assert generate(project, output, "agentx-multi") == 0

    [(record, recipe)] = written(output)
    assert record["cluster"] == "fixture"
    assert recipe["telemetry"] == {"enabled": True, "dcgm_exporter": {
        "container_image": "dcgm-exporter", "command": "dcgm-exporter --address :{port}", "port": 9555,
    }}  # fmt: skip
    # The lane's result directory, the AgentX-only aiperf mount and the cluster-wide Hub cache.
    assert recipe["benchmark"]["env"] == {
        "KV_OFFLOADING": "none", "RESULT_DIR": "/results", "AIPERF_DATASET_MMAP_CACHE_DIR": "/aiperf",
        "HF_HUB_CACHE": "/hf",
    }  # fmt: skip
    assert recipe["benchmark"]["concurrencies"] == [8]


def test_a_multinode_dram_point_sizes_host_dram_from_its_budget_per_prefill_gpu(
    project, power_lane, tmp_path
):
    output = tmp_path / "out"
    assert generate(project, output, "agentx-multi-dram") == 0

    # 1,000,000 MiB at 0.5 over all eight GPUs of a node: 524 GB, 65 GB per GPU.
    [(record, recipe)] = written(output)
    assert record["matrix"]["total-cpu-dram-gb"] == 524
    assert recipe["roles"]["prefill"]["args"]["hicache-size"] == 65
    assert recipe["benchmark"]["env"]["TOTAL_CPU_DRAM_GB"] == "524"


def test_a_power_point_its_lane_refuses_fails_without_writing(
    project, power_lane, monkeypatch, tmp_path, capsys
):
    refusing = power.PowerLane(rules=(), error="this lane measures no AgentX power")
    monkeypatch.setitem(power.POWER_LANES, ("fixture", LaunchPath.SRT_MULTI), refusing)
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        generate(project, output, "agentx-multi")
    assert "this lane measures no AgentX power" in capsys.readouterr().err
    assert not output.exists()


def test_output_must_be_new_or_empty(project, tmp_path, capsys):
    output = tmp_path / "out"
    output.mkdir()
    (output / "stale.yaml").write_text("{}\n")
    with pytest.raises(SystemExit) as exit_info:
        generate(project, output)
    assert exit_info.value.code == 2
    assert "Output directory must be empty" in capsys.readouterr().err


def test_a_recipe_srtctl_rejects_fails_without_writing(project, tmp_path, capsys):
    recipe = project / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
    recipe.write_text(yaml.safe_dump({**FRAGMENT, "base": {**FRAGMENT["base"], "no_such_field": 1}}))
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        generate(project, output)
    assert "srtctl rejects the bound recipe" in capsys.readouterr().err
    assert not output.exists()
