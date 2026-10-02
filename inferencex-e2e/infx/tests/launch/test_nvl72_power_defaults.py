"""Cluster telemetry survives new models, recipe variants, and ordinary run policy."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

from infx.clusters.slurm import PowerTelemetrySettings
from infx.launch.drivers.srt.power import resolve_power, telemetry_arguments
from infx.launch.drivers.srt.recipe import recipe_mirror_path
from infx.launch.policy import LaunchPath
from infx.launch.request import LaunchRequest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
from srtctl.core.config import generate_override_configs, resolve_config_with_defaults
from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides
from srtctl.core.schema import SrtConfig


@pytest.fixture(params=["acpi", "dcgm"])
def settings(request):
    return PowerTelemetrySettings(dcgm_port=19317, cpu_port=19318, cpu_source=request.param)


@pytest.mark.parametrize("cluster", ["gb200-nv", "gb300-nv"])
@pytest.mark.parametrize("agentic,framework,path", [
    (False, "dynamo-trt", LaunchPath.SRT_MULTI),
    (True, "dynamo-trt", LaunchPath.SRT_MULTI),
    (True, "dynamo-vllm", LaunchPath.SRT_MULTI),
    (True, "vllm", LaunchPath.SRT_SINGLE),
])
def test_new_models_collect_without_recipe_or_model_allowlists(settings, cluster, agentic, framework, path):
    request = LaunchRequest.from_env({
        "RUNNER_NAME": "fixture_0", "IS_AGENTIC": str(int(agentic)),
        "MODEL_PREFIX": "future-model", "FRAMEWORK": framework,
    })
    decision = resolve_power(cluster, path, request, settings=settings)
    assert decision.dcgm and decision.agentx is agentic
    assert decision.expected_cpu_source == settings.cpu_source
    assert not decision.require_power
    evaluation = LaunchRequest.from_env({**request.env, "EVAL_ONLY": "true"})
    assert not resolve_power(cluster, path, evaluation, settings=settings).dcgm


@pytest.mark.parametrize("variant_file", [False, True])
@pytest.mark.parametrize("dedicated_frontend", [False, True])
def test_native_overrides_preserve_workload_and_worker_budget(settings, variant_file, dedicated_frontend, monkeypatch):
    from srtctl.cli.submit import planned_total_nodes
    from srtctl.core.runtime import Nodes

    recipe = {
        "schema": 2, "name": "fixture", "engine": "sglang",
        "model": {"path": "hf:test/model", "container": "test:tag", "precision": "fp8"},
        "resources": {"gpus_per_node": 4},
        "frontend": {"type": "sglang-router", "dedicated_node": dedicated_frontend},
        "roles": {"prefill": {"nodes": 2, "workers": 2}, "decode": {"nodes": 1, "workers": 1}},
        "benchmark": {"type": "custom", "command": "python /configs/client.py", "concurrencies": [1]},
    }
    original = copy.deepcopy(recipe)
    before = SrtConfig.Schema().load(resolve_config_with_defaults(original, None))
    document = {"base": recipe, "override_point": {"benchmark": {"concurrencies": [1]}}} if variant_file else recipe
    arguments = telemetry_arguments(settings, [2048, 4096])
    apply_overrides_to_recipe(document, parse_overrides(arguments[1::2], None))
    effective = generate_override_configs(document, selector="override_point")[0][1] if variant_file else document
    config = SrtConfig.Schema().load(resolve_config_with_defaults(effective, None))
    assert config.telemetry.enabled
    assert config.telemetry.cpu_power_exporter.source == settings.cpu_source
    assert config.telemetry.dcgm_exporter.port == settings.dcgm_port
    assert config.telemetry.cpu_power_exporter.port == settings.cpu_port
    assert "dcgm-counters-noprof.csv" in config.telemetry.dcgm_exporter.command
    assert config.benchmark.get_concurrency_list() == [2048, 4096]
    assert effective["roles"] == original["roles"]
    assert effective["benchmark"]["command"] == original["benchmark"]["command"]
    assert planned_total_nodes(config) == planned_total_nodes(before)
    assert config.engine_node_count == before.engine_node_count
    monkeypatch.setattr('srtctl.core.runtime.get_slurm_nodelist', lambda: [
        f'node-{i}' for i in range(planned_total_nodes(config))
    ])
    monkeypatch.setattr('srtctl.core.runtime.get_slurm_het_nodelists', lambda: None)
    nodes = Nodes.from_slurm(
        frontend_dedicated_node=config.frontend.dedicated_node,
        client_dedicated_node=config.benchmark.client_dedicated_node,
        etcd_nats_dedicated_node=config.infra.etcd_nats_dedicated_node,
        colocate_dedicated_nodes=config.benchmark.colocate_with_frontend,
    )
    assert nodes.head == nodes.bench
    assert len(nodes.compute) == len(nodes.worker) == config.engine_node_count


def test_existing_strict_agentx_recipe_remains_strict(settings, tmp_path):
    config_file = "recipes/kimik3/vllm/gb200-fp4/agentx/fixture.yaml"
    mirror = recipe_mirror_path(tmp_path, config_file)
    mirror.parent.mkdir(parents=True)
    mirror.write_text(yaml.safe_dump({"telemetry": {"enabled": True, "dcgm_exporter": {}}}))
    request = LaunchRequest.from_env({
        "RUNNER_NAME": "fixture_0", "GITHUB_WORKSPACE": str(tmp_path),
        "IS_AGENTIC": "1", "MODEL_PREFIX": "kimik3", "PRECISION": "fp4",
        "FRAMEWORK": "dynamo-vllm",
        "CONFIG_FILE": config_file,
    })
    assert resolve_power("gb200-nv", LaunchPath.SRT_MULTI, request, settings=settings).require_power


def test_multinode_power_bundle_retains_effective_cluster_overrides(tmp_path):
    from types import SimpleNamespace
    from infx.launch.drivers.srt.collect import _stage_logs
    from infx.launch.drivers.srt.power import PowerDecision
    logs, workspace = tmp_path / 'job/logs', tmp_path / 'workspace'
    logs.mkdir(parents=True)
    workspace.mkdir()
    for name in ('power-exporter-image.txt', 'power-producer-sha.txt'):
        (workspace / name).write_text('fixture\n')
    effective = 'telemetry:\n  enabled: true\n  cpu_power_exporter:\n    source: acpi\n'
    (logs.parent / 'config_resolved.yaml').write_text(effective)
    _stage_logs(SimpleNamespace(workspace=workspace), logs, PowerDecision(dcgm=True, agentx=True))
    assert (workspace / 'LOGS/power/config_resolved.yaml').read_text() == effective
