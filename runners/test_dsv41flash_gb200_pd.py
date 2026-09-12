import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / (
    "benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4.1-flash/"
    "agentic/disagg-gb200-1p1d-dep4-dep16-c64-c256-dspark5-agentic.yaml"
)


def test_dsv41flash_pd_uses_nixl_without_external_kv_store() -> None:
    recipe = yaml.safe_load(RECIPE.read_text())
    assert recipe["model"] == {
        "path": "deepseek-v4.1-flash",
        "container": "vllm/vllm-openai:deepseekv41-flash-0909",
        "precision": "fp4",
    }
    resources = recipe["resources"]
    assert resources["prefill_nodes"] == 1
    assert resources["decode_nodes"] == 4
    assert resources["gpus_per_prefill"] == 4
    assert resources["gpus_per_decode"] == 16

    backend = recipe["backend"]
    assert "mooncake_kv_store" not in backend
    for role, expected_role in (("prefill", "kv_both"), ("decode", "kv_consumer")):
        config = backend["vllm_config"][role]
        transfer = json.loads(config["kv-transfer-config"])
        assert transfer["kv_connector"] == "NixlConnector"
        assert transfer["kv_role"] == expected_role
        assert "connectors" not in transfer.get("kv_connector_extra_config", {})
        assert config["engram-config"] == '{"cpu_offload":true}'
        speculative = json.loads(config["speculative-config"])
        assert speculative["method"] == "dspark"
        assert speculative["num_speculative_tokens"] == 5
        assert speculative["rejection_sample_method"] == "block"

    raw = RECIPE.read_text().lower()
    assert "mooncakestoreconnector" not in raw
    assert "global_segment_size" not in raw
