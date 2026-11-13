"""``infx generate``: bound srt-slurm recipes of master-config points, validated by srtctl."""

import json
from pathlib import Path

import pytest
import yaml

from infx.cli import main
from infx.launch.drivers.srt import lanes
from infx.launch.policy import LaunchPath
from infx.tests.srt_recipes import (
    VIDEO_MODEL,
    VIDEO_SPEC,
    video_entry,
    video_fragment,
    write_shared_blocks,
    write_video_specs,
)

ROOT = Path(__file__).resolve().parents[3]
FRAGMENT = {
    "base": {
        "schema": 2, "name": "fixture", "engine": "sglang",
        "resources": {"gpu_type": "h200", "gpus_per_node": 8},
        "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4, "args": {
            "tensor-parallel-size": 4, "speculative-algorithm": "EAGLE", "speculative-num-steps": 2,
        }, "env": {"SGLANG_SIMULATE_ACC_LEN": "2.5", "NCCL_SOCKET_IFNAME": "@fabric.socket-ifname"}}},
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
    "video": video_entry(),
}  # fmt: skip
SHARED = {
    "telemetry-native.yaml": {"telemetry": {"enabled": True, "required": False}},
    "fixed-sequence-single.yaml": {"benchmark": {"type": "custom", "command": "bash client.sh"}},
    "agentic.yaml": {"benchmark": {"type": "custom", "command": "bash srt_agentic.sh"}},
    "telemetry-dcgm.yaml": {"telemetry": {"enabled": True, "dcgm_exporter": {
        "container_image": "dcgm-exporter", "command": "dcgm-exporter --address :{port}",
    }}},
}  # fmt: skip


@pytest.fixture
def project(tmp_path):
    """The end-to-end project beside the video harness, as the repository lays them out."""
    project = tmp_path / "inferencex-e2e"
    write_shared_blocks(project)
    write_video_specs(tmp_path)
    for name, block in SHARED.items():
        (project / "configs/srt-recipes" / name).write_text(yaml.safe_dump(block))
    (project / "configs/master.yaml").write_text(yaml.safe_dump(MASTER))
    (project / "configs/runners.yaml").write_text(yaml.safe_dump({
        "labels": {"fixture": ["fixture_0"], "cluster:fixture": ["fixture_0"]},
        "clusters": {"fixture": {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm", "slurm": {
            "partition": "p", "exclusive": True,
            "volumes": {"hf-hub-cache": {"path": "/srv/hf"}, "aiperf-cache": {"path": "/srv/aiperf"}},
            "srt-slurm": {
                "network-interface": "eth0", "power-exporter-port": 9555,
                "extra": {"default_gpu_exporter": {"container_image": "exporter:test", "port": 9499}},
                "volume-mounts": {"hf-hub-cache": "/hf"},
                "agentic-volume-mounts": {"aiperf-cache": "/aiperf"},
                "fabric": {"socket-ifname": ["eth7"]},
            },
        }}},
    }))  # fmt: skip
    recipes = {
        "single_node/srt-slurm-recipes/fixture/recipe.yaml": FRAGMENT,
        "single_node/srt-slurm-recipes/fixture/agentic.yaml": {
            **RECIPE, "frontend": {"type": "sglang", "enable_multiple_frontends": False},
            "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4, "args": {"tensor-parallel-size": 4}}},
        },
        "single_node/srt-slurm-recipes/fixture/video.yaml": video_fragment(),
        "multi_node/srt-slurm-recipes/fixture/agentic.yaml": {
            **RECIPE, "frontend": {"type": "sglang-router"}, "roles": {
                role: {"nodes": 1, "workers": 1, "gpus": 8, "args": {"tensor-parallel-size": 8}}
                for role in ("prefill", "decode")
            },
        },
    }  # fmt: skip
    for path, data in recipes.items():
        (project / "benchmarks" / path).parent.mkdir(parents=True, exist_ok=True)
        (project / "benchmarks" / path).write_text(yaml.safe_dump(data))
    (project / "utils/srt-slurm").mkdir(parents=True)
    (project / "utils/srt-slurm/src").symlink_to(ROOT / "utils/srt-slurm/src")
    return project


@pytest.fixture
def power_lane(monkeypatch):
    """The fixture cluster's multi-node lane."""
    monkeypatch.setitem(
        lanes.SRT_LANES, ("fixture", LaunchPath.SRT_MULTI), lanes.SrtLane(agentic_result_dir="/results")
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
    assert [
        (r["config-key"], r["variant"], r["cluster"], r["matrix"]["conc"]) for r in manifest["recipes"]
    ] == [
        ("fixture-sglang", "zip_override_conc[0]", "fixture", 2),
        ("fixture-sglang", "zip_override_conc[1]", "fixture", 4),
    ]
    recipe = yaml.safe_load((output / manifest["recipes"][1]["file"]).read_text())
    assert recipe["model"] == {"path": "hf:org/model", "container": "org/image:1", "precision": "fp8"}
    assert recipe["benchmark"] == {"type": "custom", "command": "bash client.sh", "env": {
        "CONC": "4", "ISL": "1024", "OSL": "128", "MODEL": "org/model",
        "RANDOM_RANGE_RATIO": "0.8", "USE_CHAT_TEMPLATE": "true", "REQUIRE_POWER": "0",
        "SPEC_ALGORITHM": "mtp", "SPEC_DRAFT_MODEL": "native",
    }, "concurrencies": [4]}  # fmt: skip
    assert recipe["roles"]["agg"]["env"] == {"NCCL_SOCKET_IFNAME": "eth7"}


def test_a_label_on_several_clusters_keeps_fabric_references(project, tmp_path):
    runners = yaml.safe_load((project / "configs/runners.yaml").read_text())
    runners["labels"] = {
        "fixture": ["fixture_0", "spare_0"], "cluster:fixture": ["fixture_0"], "cluster:spare": ["spare_0"],
    }  # fmt: skip
    runners["clusters"]["spare"] = runners["clusters"]["fixture"]
    (project / "configs/runners.yaml").write_text(yaml.safe_dump(runners))
    output = tmp_path / "out"
    assert generate(project, output) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert [r["cluster"] for r in manifest["recipes"]] == [None, None]
    recipe = yaml.safe_load((output / manifest["recipes"][0]["file"]).read_text())
    assert recipe["roles"]["agg"]["env"] == {"NCCL_SOCKET_IFNAME": "@fabric.socket-ifname"}


def test_a_single_node_agentx_point_gets_the_agentic_client_and_no_sequence_lengths(
    project, tmp_path
):
    output = tmp_path / "out"
    assert generate(project, output, "agentx-single") == 0

    [(record, recipe)] = written(output)
    assert (record["variant"], record["cluster"]) == (None, "fixture")
    assert recipe["model"] == {"path": "hf:org/model", "container": "org/image:1", "precision": "fp8"}
    assert recipe["benchmark"] == {
        "type": "custom", "command": "bash srt_agentic.sh", "env": {"MODEL": "org/model", "CONC": "2", "REQUIRE_POWER": "0", "ENABLE_AGENTX_POWER": "1"},
        "concurrencies": [2],
    }  # fmt: skip


def test_a_multinode_agentx_power_point_gets_its_clusters_exporter_port_and_client_paths(
    project, power_lane, tmp_path
):
    output = tmp_path / "out"
    assert generate(project, output, "agentx-multi") == 0

    [(record, recipe)] = written(output)
    assert record["cluster"] == "fixture"
    assert recipe["telemetry"] == {"enabled": True, "required": True, "dcgm_exporter": {
        "container_image": "dcgm-exporter", "command": "dcgm-exporter --address :{port}", "port": 9555,
    }}  # fmt: skip
    assert recipe["benchmark"]["env"] == {
        "RESULT_DIR": "/results", "AIPERF_DATASET_MMAP_CACHE_DIR": "/aiperf", "HF_HUB_CACHE": "/hf",
        "IS_MULTINODE": "true", "ENABLE_AGENTX_POWER": "1", "REQUIRE_POWER": "0",
    }  # fmt: skip
    assert recipe["benchmark"]["concurrencies"] == [8]


@pytest.mark.parametrize("declared", [True, False])
def test_a_cpu_power_point_binds_the_clusters_grace_exporter_or_fails(
    project, power_lane, tmp_path, capsys, declared
):
    master = project / "configs/master.yaml"
    config = yaml.safe_load(master.read_text())
    config["agentx-multi"]["scenarios"]["agentic-coding"][0]["search-space"][0]["cpu-power"] = True
    master.write_text(yaml.safe_dump(config))
    if declared:
        runners = project / "configs/runners.yaml"
        inventory = yaml.safe_load(runners.read_text())
        srt = inventory["clusters"]["fixture"]["slurm"]["srt-slurm"]
        srt["cpu-power-exporter"] = {"source": "acpi"}
        runners.write_text(yaml.safe_dump(inventory))
    output = tmp_path / "out"

    if not declared:
        with pytest.raises(SystemExit):
            generate(project, output, "agentx-multi")
        assert "declares no srt-slurm.cpu-power-exporter" in capsys.readouterr().err
        assert not output.exists()
        return
    assert generate(project, output, "agentx-multi") == 0
    [(_, recipe)] = written(output)
    assert recipe["telemetry"]["cpu_power_exporter"] == {"source": "acpi"}
    assert recipe["telemetry"]["dcgm_exporter"]["port"] == 9555


@pytest.mark.parametrize("declared", [True, False])
def test_a_single_node_cpu_power_point_binds_the_clusters_rapl_exporter_or_fails(
    project, tmp_path, capsys, declared
):
    master = project / "configs/master.yaml"
    config = yaml.safe_load(master.read_text())
    config["fixture-sglang"]["scenarios"]["fixed-seq-len"][0]["search-space"][0]["cpu-power"] = True
    master.write_text(yaml.safe_dump(config))
    if declared:
        runners = project / "configs/runners.yaml"
        inventory = yaml.safe_load(runners.read_text())
        inventory["clusters"]["fixture"]["slurm"]["srt-slurm"]["cpu-power-exporter"] = {"source": "rapl"}
        runners.write_text(yaml.safe_dump(inventory))
    output = tmp_path / "out"

    if not declared:
        with pytest.raises(SystemExit):
            generate(project, output, "fixture-sglang")
        assert "declares no srt-slurm.cpu-power-exporter" in capsys.readouterr().err
        assert not output.exists()
        return
    assert generate(project, output, "fixture-sglang") == 0
    recipes = [recipe for _, recipe in written(output)]
    assert [recipe["telemetry"]["cpu_power_exporter"] for recipe in recipes] == [{"source": "rapl"}] * 2


@pytest.mark.parametrize("declared", [True, False])
def test_a_single_node_point_measures_cpu_power_by_default_where_its_cluster_declares_it(
    project, tmp_path, declared
):
    if declared:
        runners = project / "configs/runners.yaml"
        inventory = yaml.safe_load(runners.read_text())
        inventory["clusters"]["fixture"]["slurm"]["srt-slurm"]["cpu-power-exporter"] = {"source": "rapl"}
        runners.write_text(yaml.safe_dump(inventory))
    output = tmp_path / "out"

    assert generate(project, output, "fixture-sglang") == 0
    recipes = [recipe for _, recipe in written(output)]
    exporters = [recipe.get("telemetry", {}).get("cpu_power_exporter") for recipe in recipes]
    assert exporters == [{"source": "rapl"} if declared else None] * 2


def _drop_srt_slurm(project: Path, key: str) -> None:
    runners = project / "configs/runners.yaml"
    inventory = yaml.safe_load(runners.read_text())
    del inventory["clusters"]["fixture"]["slurm"]["srt-slurm"][key]
    runners.write_text(yaml.safe_dump(inventory))


@pytest.mark.parametrize("port", [True, False])
def test_a_multinode_agentx_point_without_power_measures_it_with_its_clusters_exporter(
    project, power_lane, tmp_path, port
):
    master = project / "configs/master.yaml"
    config = yaml.safe_load(master.read_text())
    del config["agentx-multi"]["scenarios"]["agentic-coding"][0]["search-space"][0]["power"]
    master.write_text(yaml.safe_dump(config))
    if not port:
        _drop_srt_slurm(project, "power-exporter-port")
    output = tmp_path / "out"
    assert generate(project, output, "agentx-multi") == 0

    [(_, recipe)] = written(output)
    env = recipe["benchmark"]["env"]
    assert (env["ENABLE_AGENTX_POWER"], env["REQUIRE_POWER"]) == ("1", "0")
    assert (recipe["telemetry"]["enabled"], recipe["telemetry"]["required"]) == (True, False)
    assert recipe["telemetry"].get("dcgm_exporter", {}).get("port") == (9555 if port else None)


def test_a_power_point_its_cluster_cannot_measure_fails_without_writing(
    project, power_lane, tmp_path, capsys
):
    _drop_srt_slurm(project, "extra")
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        generate(project, output, "agentx-multi")
    assert "measures no GPU power" in capsys.readouterr().err
    assert not output.exists()


def test_output_must_be_new_or_empty(project, tmp_path, capsys):
    output = tmp_path / "out"
    output.mkdir()
    (output / "stale.yaml").write_text("{}\n")
    with pytest.raises(SystemExit) as exit_info:
        generate(project, output)
    assert exit_info.value.code == 2
    assert "Output directory must be empty" in capsys.readouterr().err


def test_a_video_point_renders_its_bound_server_recipe(project, tmp_path):
    output = tmp_path / "out"
    assert generate(project, output, "video") == 0

    [(record, recipe)] = written(output)
    assert (record["variant"], record["cluster"]) == (None, "fixture")
    assert (record["matrix"]["scenario-type"], record["matrix"]["conc-list"]) == ("video-generation", [1])
    assert recipe["model"] == {"path": f"hf:{VIDEO_MODEL}", "container": "example/video:1", "precision": "bf16"}
    assert recipe["resources"]["gpus_per_node"] == 2
    [server] = recipe["services"]
    assert server["args"][server["args"].index("--num-gpus") + 1] == "2"
    assert server["env"]["VIDEO_SPEC_PATH"] == f"/video-package/specs/{VIDEO_SPEC}"
    assert recipe["benchmark"]["env"]["CONC_LIST"] == "1"


def test_a_recipe_srtctl_rejects_fails_without_writing(project, tmp_path, capsys):
    recipe = project / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
    recipe.write_text(yaml.safe_dump({**FRAGMENT, "base": {**FRAGMENT["base"], "no_such_field": 1}}))
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        generate(project, output)
    assert "srtctl rejects the bound recipe" in capsys.readouterr().err
    assert not output.exists()
