#!/usr/bin/env python3
"""Generate dedicated Qwen 3.5 ISB1 export bundles for 8k/32k/64k lanes.

These files are derived from the committed generic export bundles by selecting only
GPT-OSS cells that are actually runnable (`supported` or `reviewed_preview`), then
rewriting model identity fields to the Qwen 3.5 replay identity while keeping trace
payloads unchanged.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPORT_ROOT = ROOT / "datasets" / "isb1" / "exports"

QWEN_MODEL_ID = "qwen3_5_397b_a17b"
GPTOSS_MODEL_ID = "gpt_oss_120b"
ALLOWED_SUPPORT_STATUSES = {"supported", "reviewed_preview"}

TARGETS = [
    ("core", "8k1k", "chat", "vllm"),
    ("core", "8k1k", "chat", "sglang"),
    ("core", "8k1k", "code", "vllm"),
    ("core", "8k1k", "code", "sglang"),
    ("extension_32k", "32k1k", "chat", "vllm"),
    ("extension_32k", "32k1k", "chat", "sglang"),
    ("extension_32k", "32k1k", "code", "vllm"),
    ("extension_32k", "32k1k", "code", "sglang"),
    ("extension_64k", "64k1k", "chat", "vllm"),
    ("extension_64k", "64k1k", "chat", "sglang"),
    ("extension_64k", "64k1k", "code", "vllm"),
    ("extension_64k", "64k1k", "code", "sglang"),
]


def _source_path(lane: str, shape: str, surface: str, engine: str) -> Path:
    return EXPORT_ROOT / lane / engine / f"{surface}_{shape}.json"


def _target_path(lane: str, shape: str, surface: str, engine: str) -> Path:
    return EXPORT_ROOT / lane / engine / f"{surface}_{shape}_qwen3.5.json"


def _rewrite_bundle_id(bundle_id: str, lane: str, engine: str, surface: str, shape: str) -> str:
    expected_prefix = f"isb1_{lane}_{engine}_{surface}_{shape}"
    if bundle_id != expected_prefix:
        raise ValueError(
            f"Unexpected bundle_id {bundle_id!r}; expected {expected_prefix!r} for {lane}/{engine}/{surface}_{shape}"
        )
    return f"{bundle_id}_qwen3_5"


def _rewrite_cell(cell: dict) -> dict:
    rewritten = deepcopy(cell)
    rewritten["canonical_model_id"] = QWEN_MODEL_ID
    rewritten["thinking_history_policy"] = "strip_reasoning"
    rewritten["history_projection_mode"] = "strip_reasoning_history"
    rewritten["support_status"] = "reviewed_preview"
    return rewritten


def build_export(lane: str, shape: str, surface: str, engine: str) -> tuple[Path, int]:
    source_path = _source_path(lane, shape, surface, engine)
    target_path = _target_path(lane, shape, surface, engine)

    payload = json.loads(source_path.read_text())
    exports = payload.get("exports")
    if not isinstance(exports, list):
        raise ValueError(f"Missing exports list in {source_path}")

    filtered = [
        _rewrite_cell(cell)
        for cell in exports
        if cell.get("canonical_model_id") == GPTOSS_MODEL_ID
        and cell.get("support_status") in ALLOWED_SUPPORT_STATUSES
    ]
    if not filtered:
        raise ValueError(f"No runnable GPT-OSS cells found in {source_path}")

    payload["bundle_id"] = _rewrite_bundle_id(payload.get("bundle_id"), lane, engine, surface, shape)
    payload["exports"] = filtered

    target_path.write_text(json.dumps(payload, indent=2) + "\n")
    return target_path, len(filtered)


def main() -> int:
    for lane, shape, surface, engine in TARGETS:
        target_path, count = build_export(lane, shape, surface, engine)
        print(f"wrote {target_path.relative_to(ROOT)} ({count} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
