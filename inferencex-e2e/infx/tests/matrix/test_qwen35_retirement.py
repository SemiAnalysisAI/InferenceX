"""Guard the workload boundary of Qwen3.5's single-turn retirement."""

from pathlib import Path

import yaml


def test_qwen35_keeps_agentic_coverage_without_fixed_sequence_sweeps():
    root = Path(__file__).resolve().parents[3]
    for vendor in ("amd", "nvidia"):
        configs = yaml.safe_load((root / "configs" / f"{vendor}-master.yaml").read_text())
        qwen_configs = [
            config for config in configs.values() if config.get("model-prefix") == "qwen3.5"
        ]
        assert qwen_configs, f"{vendor} must retain Qwen3.5 AgentX coverage"
        assert all(set(config["scenarios"]) == {"agentic-coding"} for config in qwen_configs)
