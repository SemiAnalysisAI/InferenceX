"""The video-generation binder: a services-only H3 recipe bound to one master-config point."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from infx.config import repository_root
from infx.srt_slurm import video

ROOT = repository_root()
H200 = ROOT / "benchmarks/single_node/srt-slurm-recipes/h3/sglang/h200-bf16/serving-client.yaml"
SPEC = json.loads((video.video_package(ROOT) / "specs/h3-serving-2gpu.json").read_text())
ENV = {
    "IMAGE": "hongccc/sglang-omni:h3-runtime-71de97b", "MODEL": "MiniMaxAI/MiniMax-H3",
    "PRECISION": "bf16", "GPU_COUNT": "2", "CONC_LIST": "1 2 4", "H3_SPEC": "h3-serving-2gpu.json",
    "HF_HUB_CACHE": "/mnt/hf_hub_cache/", "RESULT_FILENAME": "h3_video_h3-serving-2gpu_g2",
}  # fmt: skip


@pytest.mark.parametrize("sku", ["h200", "b200", "h100", "mi355x"])
def test_checked_in_fragments_bind_as_custom_benchmark_against_the_h3_service(sku):
    fragment = H200.with_name(H200.name).parent.parent / f"{sku}-bf16" / H200.name
    bound = video.prepare(fragment, ENV, root=ROOT)

    assert bound["model"] == {"path": "hf:MiniMaxAI/MiniMax-H3", "container": ENV["IMAGE"], "precision": "bf16"}
    assert bound["resources"]["gpus_per_node"] == 2 and "roles" not in bound
    assert bound["frontend"] == {"type": "none"}
    [service] = bound["services"]
    assert (service["name"], service["type"], service["nodes"], service["critical"]) == ("h3-server", "generic", 1, True)
    assert service["command"][-1].endswith("benchmarks/h3/run_server.sh")
    assert service["readiness"]["http"] == {"port": SPEC["port"], "path": "/health"}
    assert service["env"]["H3_SPEC_PATH"] == "/h3-package/specs/h3-serving-2gpu.json"
    benchmark = bound["benchmark"]
    assert benchmark["type"] == "custom" and benchmark["command"].endswith("benchmarks/h3/run_client.sh")
    assert benchmark["concurrencies"] == [1, 2, 4]
    assert benchmark["env"]["CONC_LIST"] == "1 2 4" and benchmark["env"]["H3_SERVER_SERVICE"] == "h3-server"
    assert benchmark["env"]["H3_MODEL_PATH"] == SPEC["model"]["path"] == (
        f"/mnt/hf_hub_cache/models--MiniMaxAI--MiniMax-H3/snapshots/{SPEC['model']['revision']}"
    )
    assert bound["telemetry"]["enabled"] is True and bound["telemetry"]["required"] is False
    assert benchmark["env"]["REQUIRE_POWER"] == "0"


def test_require_power_binds_required_telemetry():
    bound = video.prepare(H200, {**ENV, "REQUIRE_POWER": "1"}, root=ROOT)
    assert bound["telemetry"]["required"] is True
    assert bound["benchmark"]["env"]["REQUIRE_POWER"] == "1"


def test_runtime_arguments_bind_only_srun_values():
    options = json.dumps({"mem": "0", "cpus-per-task": "32"})
    assert video.runtime_arguments({**ENV, "SRT_SRUN_OPTIONS": options}) == [
        "--set", 'srun_options.gpus-per-node="2"',
        "--set", 'srun_options.mem="0"', "--set", 'srun_options.cpus-per-task="32"',
    ]  # fmt: skip
    with pytest.raises(ValueError, match="SRT_SRUN_OPTIONS"):
        video.runtime_arguments({**ENV, "SRT_SRUN_OPTIONS": json.dumps({"Mem": 1})})


def _fragment(**edits):
    raw = yaml.safe_load(H200.read_text())
    for key, value in edits.items():
        node = raw
        *parents, last = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[last] = value
    return raw


@pytest.mark.parametrize(("edits", "message"), [
    ({"roles": {"agg": {"nodes": 1}}}, "own no engine roles"),
    ({"model": {"path": "hf:x"}}, "launcher sets them"),
    ({"benchmark.concurrencies": [1]}, "launcher sets them"),
    ({"telemetry": {"enabled": False}}, "launcher sets them"),
    ({"benchmark.env.MODEL": "x"}, "launcher sets them"),
    ({"base": {}}, "variant bundles"),
])  # fmt: skip
def test_fragments_cannot_claim_launcher_owned_keys(tmp_path, edits, message):
    path = tmp_path / "serving-client.yaml"
    path.write_text(yaml.safe_dump(_fragment(**edits)))
    with pytest.raises(ValueError, match=message):
        video.prepare(path, ENV, root=ROOT)


@pytest.mark.parametrize(("edits", "message"), [
    ({"frontend": {"type": "disaggregated"}}, "services-only"),
    ({"benchmark.type": "aiperf"}, "benchmark.type custom"),
    ({"services": []}, "exactly one service"),
    ({"benchmark.env.H3_SERVER_PORT": "30000"}, "spec port"),
    ({"benchmark.env.H3_SERVER_SERVICE": "other"}, "must name the server service"),
])  # fmt: skip
def test_the_server_service_and_client_must_match_the_spec(tmp_path, edits, message):
    raw = _fragment(**edits)
    if edits.get("services") == []:
        raw["services"] = []
    path = tmp_path / "serving-client.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        video.prepare(path, ENV, root=ROOT)


def test_readiness_must_probe_health_on_the_spec_port(tmp_path):
    raw = _fragment()
    raw["services"][0]["readiness"]["http"]["path"] = "/ready"
    path = tmp_path / "serving-client.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="/health"):
        video.prepare(path, ENV, root=ROOT)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"MODEL": "other/model"}, "H3 spec serves"),
    ({"HF_HUB_CACHE": "/elsewhere/"}, "mounted hub cache"),
    ({"H3_SPEC": "../h3-serving-2gpu.json"}, "H3_SPEC must name"),
    ({"H3_SPEC": "missing.json"}, "H3 spec not found"),
    ({"GPU_COUNT": "0"}, "GPU_COUNT must be a positive integer"),
    ({"CONC_LIST": ""}, "Missing video input: CONC_LIST"),
])  # fmt: skip
def test_the_point_must_agree_with_the_frozen_spec(overrides, message):
    with pytest.raises(ValueError, match=message):
        video.prepare(H200, {**ENV, **overrides}, root=ROOT)


def test_spec_model_path_is_the_hub_cache_snapshot():
    assert video.spec_model_path("/mnt/hf_hub_cache/", "org/name", "a" * 40) == (
        "/mnt/hf_hub_cache/models--org--name/snapshots/" + "a" * 40
    )
    with pytest.raises(ValueError, match="40-hex"):
        video.spec_model_path("/mnt/hf_hub_cache/", "org/name", "main")
    with pytest.raises(ValueError, match="absolute"):
        video.spec_model_path("hf_hub_cache", "org/name", "a" * 40)


def test_prepare_cli_writes_the_bound_recipe_and_runtime_arguments(tmp_path, monkeypatch):
    import subprocess
    import sys

    output = tmp_path / "arguments"
    env = {**ENV, "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "infx.srt_slurm.video", "prepare", str(H200), str(output),
         "--fabric", "{}"],
        env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    selected, config, *arguments = output.read_bytes().decode().split("\0")[:-1]
    assert (Path(selected), Path(config)) == (H200, tmp_path / "recipe.yaml")
    assert arguments == ["--set", 'srun_options.gpus-per-node="2"']
    assert yaml.safe_load(Path(config).read_text())["benchmark"]["env"]["CONC_LIST"] == "1 2 4"
