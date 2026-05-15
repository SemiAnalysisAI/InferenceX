from __future__ import annotations

import json
from pathlib import Path

from validate_reusable_sweep_artifacts import (
    expected_eval_artifact_prefixes,
    validate_eval_artifacts,
)


def write_eval_aggregate(root: Path) -> None:
    eval_dir = root / "eval_results_all"
    eval_dir.mkdir()
    (eval_dir / "agg_eval_all.json").write_text(json.dumps([{"task": "gsm8k"}]))


def single_eval_entry(conc: int) -> dict:
    return {
        "exp-name": "gptoss_8k1k",
        "precision": "fp4",
        "framework": "vllm",
        "tp": 2,
        "ep": 1,
        "dp-attn": False,
        "disagg": False,
        "spec-decoding": "none",
        "conc": conc,
    }


def test_eval_validation_requires_raw_result_dirs_not_eval_debug_dirs(
    tmp_path: Path,
) -> None:
    config = {
        "evals": [single_eval_entry(32), single_eval_entry(64)],
        "multinode_evals": [],
    }
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(tmp_path)

    (tmp_path / "eval_server_logs_gptoss_8k1k_runner").mkdir()
    (tmp_path / "eval_gpu_metrics_gptoss_8k1k_runner").mkdir()
    (tmp_path / f"{prefixes[0]}h100-dgxc-slurm_00").mkdir()

    errors = validate_eval_artifacts(tmp_path, prefixes)

    assert "missing 1 expected raw eval result artifact dir(s)" in errors
    assert f"  missing eval artifact prefix: {prefixes[1]}" in errors


def test_eval_validation_accepts_all_expected_raw_result_dirs(tmp_path: Path) -> None:
    config = {
        "evals": [single_eval_entry(32), single_eval_entry(64)],
        "multinode_evals": [],
    }
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(tmp_path)
    for index, prefix in enumerate(prefixes):
        (tmp_path / f"{prefix}h100-dgxc-slurm_{index:02d}").mkdir()

    assert validate_eval_artifacts(tmp_path, prefixes) == []
