"""Collector-compatible artifacts shared by the vendor verifier adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULT_FORMAT = "inferencex-eval-v1"


def error_dict(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def compatibility_result(
    *,
    adapter: str,
    task: str,
    model: str,
    score: float,
    original: int,
    effective: int,
    task_config: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    integration_error: BaseException | None = None,
) -> dict[str, Any]:
    """Project one verifier score into the collector's strict-match format."""
    config: dict[str, Any] = {
        "metric_list": [{"metric": "exact_match"}],
        "filter_list": [{"name": "strict-match"}],
    }
    if task_config is not None:
        config.update(task_config)
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": adapter,
        "model_name": model,
        "results": {
            task: {
                "exact_match,strict-match": score,
                "exact_match_stderr,strict-match": 0.0,
            }
        },
        "configs": {task: config},
        "n-samples": {task: {"original": original, "effective": effective}},
    }
    if source is not None:
        result["source"] = dict(source)
    if integration_error is not None:
        result["integration_error"] = error_dict(integration_error)
    return result


def prepare_compatibility_path(output_dir: Path, *, prefix: str, stale_glob: str) -> Path:
    """Remove earlier projections from this adapter before naming the next one."""
    for stale_path in output_dir.glob(stale_glob):
        stale_path.unlink()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S.%f")
    return output_dir / f"{prefix}{timestamp}.json"


def write_json(path: Path, value: Mapping[str, Any], *, ensure_ascii: bool) -> None:
    path.write_text(json.dumps(value, ensure_ascii=ensure_ascii, indent=2) + "\n", encoding="utf-8")
