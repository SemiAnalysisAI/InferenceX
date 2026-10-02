"""Cluster telemetry survives new models, recipe variants, and ordinary run policy."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

from infx.clusters import load_inventory
from infx.launch.drivers.srt.power import resolve_power, telemetry_arguments
from infx.launch.policy import LaunchPath
from infx.launch.request import LaunchRequest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
from srtctl.core.config import generate_override_configs, resolve_config_with_defaults
from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides
from srtctl.core.schema import SrtConfig


@pytest.mark.parametrize("cluster", ["gb200-nv", "gb300-nv"])
@pytest.mark.parametrize("agentic,framework,path", [
    (False, "dynamo-trt", LaunchPath.SRT_MULTI),
    (True, "dynamo-trt", LaunchPath.SRT_MULTI),
    (True, "dynamo-vllm", LaunchPath.SRT_MULTI),
    (True, "vllm", LaunchPath.SRT_SINGLE),
])
def test_new_models_collect_without_recipe_or_model_allowlists(cluster, agentic, framework, path):
    settings = load_inventory(yaml.safe_load((ROOT / "configs/runners.yaml").read_text())).clusters[
        cluster
    ].scheduler_settings.srt_slurm
    request = LaunchRequest.from_env({
        "RUNNER_NAME": "fixture_0", "IS_AGENTIC": str(int(agentic)),
        "MODEL_PREFIX": "future-model", "FRAMEWORK": framework,
    })
    decision = resolve_power(cluster, path, request, settings=settings.power_telemetry)
    assert decision.dcgm and decision.agentx is agentic
    assert decision.expected_cpu_source == "acpi"
    assert not decision.require_power
    assert settings.env["CPU_POWER_EXPORTER_RELEASE"] == "v2.40.2"
    evaluation = LaunchRequest.from_env({**request.env, "EVAL_ONLY": "true"})
    assert not resolve_power(cluster, path, evaluation, settings=settings.power_telemetry).dcgm


@pytest.mark.parametrize("variant_file", [False, True])
def test_native_overrides_materialize_cpu_and_noprof_without_rewriting_workload(variant_file):
    settings = load_inventory(yaml.safe_load((ROOT / "configs/runners.yaml").read_text())).clusters[
        "gb300-nv"
    ].scheduler_settings.srt_slurm.power_telemetry
    recipe = yaml.safe_load((ROOT / (
        "benchmarks/multi_node/srt-slurm-recipes/qwen3.5/sglang/gb300-fp8/8k1k/"
        "disagg-8p1d-dep4-dep16-stp.yaml"
    )).read_text())
    recipe.pop("telemetry")
    original = copy.deepcopy(recipe)
    document = {"base": recipe, "override_point": {"benchmark": {"concurrencies": [1]}}} if variant_file else recipe
    arguments = telemetry_arguments(settings, [2048, 4096])
    apply_overrides_to_recipe(document, parse_overrides(arguments[1::2], None))
    effective = generate_override_configs(document, selector="override_point")[0][1] if variant_file else document
    config = SrtConfig.Schema().load(resolve_config_with_defaults(effective, None))
    assert config.telemetry.enabled
    assert config.telemetry.cpu_power_exporter.source == "acpi"
    assert config.telemetry.dcgm_exporter.port == 19401
    assert "dcgm-counters-noprof.csv" in config.telemetry.dcgm_exporter.command
    assert config.benchmark.get_concurrency_list() == [2048, 4096]
    assert effective["roles"] == original["roles"]
    assert effective["benchmark"]["command"] == original["benchmark"]["command"]


def test_existing_strict_agentx_recipe_remains_strict():
    settings = load_inventory(yaml.safe_load((ROOT / "configs/runners.yaml").read_text())).clusters[
        "gb200-nv"
    ].scheduler_settings.srt_slurm.power_telemetry
    request = LaunchRequest.from_env({
        "RUNNER_NAME": "fixture_0", "GITHUB_WORKSPACE": str(ROOT),
        "IS_AGENTIC": "1", "MODEL_PREFIX": "kimik3", "PRECISION": "fp4",
        "FRAMEWORK": "dynamo-vllm",
        "CONFIG_FILE": "recipes/kimik3/vllm/gb200-fp4/agentx/agg-dcp16-dspark4-maxseq2-mooncake.yaml",
    })
    assert resolve_power("gb200-nv", LaunchPath.SRT_MULTI, request, settings=settings).require_power


@pytest.mark.parametrize('relative,selector,total,workers', [
    ('dsv4/vllm/gb300-fp4/agentx/disagg-1p4d-dep4-tp8-c4-mtp.yaml', None, 9, 9),
    ('dsv4/trtllm/gb300-fp4/agentx/disagg-variants.yaml', 'override_1p4d_dep4_tep8_c4_b1', 9, 9),
    ('glm5.2/trtllm/gb300-fp4/agentx/disagg-variants.yaml', 'override_1p1d_tp8_c1_b1_mtp5', 4, 3),
])
def test_client_and_collector_share_head_without_changing_worker_budget(monkeypatch, relative, selector, total, workers):
    from srtctl.cli.submit import planned_total_nodes
    from srtctl.core.runtime import Nodes
    settings = load_inventory(yaml.safe_load((ROOT / 'configs/runners.yaml').read_text())).clusters[
        'gb300-nv'
    ].scheduler_settings.srt_slurm.power_telemetry
    raw = yaml.safe_load((ROOT / 'benchmarks/multi_node/srt-slurm-recipes' / relative).read_text())
    apply_overrides_to_recipe(raw, parse_overrides(telemetry_arguments(settings, [4])[1::2], None))
    effective = generate_override_configs(raw, selector)[0][1] if selector else raw
    config = SrtConfig.Schema().load(resolve_config_with_defaults(effective, None))
    assert planned_total_nodes(config) == total
    assert config.engine_node_count == workers
    assert not config.pool_services
    monkeypatch.setattr('srtctl.core.runtime.get_slurm_nodelist', lambda: [f'node-{i}' for i in range(total)])
    monkeypatch.setattr('srtctl.core.runtime.get_slurm_het_nodelists', lambda: None)
    nodes = Nodes.from_slurm(
        frontend_dedicated_node=config.frontend.dedicated_node,
        client_dedicated_node=config.benchmark.client_dedicated_node,
        etcd_nats_dedicated_node=config.infra.etcd_nats_dedicated_node,
        colocate_dedicated_nodes=config.benchmark.colocate_with_frontend,
    )
    assert nodes.head == nodes.bench == 'node-0'
    assert len(nodes.compute) == len(nodes.worker) == workers


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
