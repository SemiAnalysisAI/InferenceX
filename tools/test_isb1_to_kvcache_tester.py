# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ``tools/isb1_to_kvcache_tester.py``.

These tests lock the bytes-level output schema so ISB1→kv-cache-tester
conversion can't silently drift from what
``callanjfox/kv-cache-tester``'s ``normalize_trace()`` expects.

We re-implement the minimal ``normalize_trace`` logic inline so tests do not
pull in ``transformers`` / ``numpy`` / ``openai`` just to exercise the
conversion shim. The actual upstream function is
`trace_replay_tester.py::normalize_trace` (see kv-cache-tester repo).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM = REPO_ROOT / "tools" / "isb1_to_kvcache_tester.py"


# ---------------------------------------------------------------------------
# Local mirror of kv-cache-tester::normalize_trace (sans tokenizer deps).
# Keep in sync with upstream manually — these tests flag drift.
# ---------------------------------------------------------------------------


def _normalize_request(req: dict, base_time: float = 0.0) -> dict:
    return {
        "timestamp": base_time + req.get("t", 0.0),
        "type": {"s": "streaming", "n": "non_streaming"}.get(
            req.get("type", ""), req.get("type", "streaming")
        ),
        "input_tokens": req.get("in", 0),
        "output_tokens": req.get("out", 0),
        "hash_ids": req.get("hash_ids", []),
        "stop_reason": req.get("stop", ""),
        "model": req.get("model", ""),
    }


def _normalize_trace(trace: dict) -> dict:
    raw = trace.get("requests", [])
    parent = [r for r in raw if r.get("type") != "subagent"]
    requests = [_normalize_request(r) for r in parent]

    total_input = sum(r["input_tokens"] for r in requests)

    cache_hits = 0
    total_blocks = 0
    for i, req in enumerate(requests):
        hash_ids = req["hash_ids"]
        if i > 0 and hash_ids:
            prev = set(requests[i - 1]["hash_ids"])
            for h in hash_ids:
                total_blocks += 1
                if h in prev:
                    cache_hits += 1
                else:
                    break
        elif hash_ids:
            total_blocks += len(hash_ids)

    return {
        "metadata": {
            "conversation_id": trace.get("id", "unknown"),
            "models": trace.get("models", []),
            "block_size": trace.get("block_size", 64),
            "hash_id_scope": trace.get("hash_id_scope", "per_context"),
            "tool_tokens": trace.get("tool_tokens", 0),
            "system_tokens": trace.get("system_tokens", 0),
            "request_count": len(requests),
            "total_input_tokens": total_input,
            "cache_hit_rate": cache_hits / total_blocks if total_blocks else 0.0,
        },
        "requests": requests,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _multiturn_bundle() -> dict:
    """Minimal ``inferencex_multiturn`` fixture — 1 cell, 2 turns."""
    return {
        "adapter_id": "inferencex_multiturn",
        "schema_version": "0.1.0",
        "exports": [
            {
                "trace_id": "core_chat_8k1k_001",
                "runtime_stack_id": "standalone:vllm",
                "hardware_profile_id": "nvidia:h200_sxm_141gb",
                "canonical_model_id": "qwen3_5_397b_a17b",
                "support_status": "supported",
                "benchmark_certification_status": "dataset_replay_verified",
                "context_band": "core_8k",
                "session": {
                    "session_id": "sess_001",
                    "turns": [
                        {
                            "turn_idx": 0,
                            "messages": [
                                {"role": "system", "content": "sys" * 16},
                                {"role": "user", "content": "u" * 3200},
                            ],
                            "expected_output_tokens": 128,
                            "wait_before_ms": 0,
                        },
                        {
                            "turn_idx": 1,
                            "messages": [
                                {"role": "system", "content": "sys" * 16},
                                {"role": "user", "content": "u" * 3200},
                                {"role": "assistant", "content": "a" * 400},
                                {"role": "user", "content": "f" * 400},
                            ],
                            "expected_output_tokens": 96,
                            "wait_before_ms": 1500,
                        },
                    ],
                },
            }
        ],
    }


def _trace_replay_bundle() -> dict:
    """Minimal ``inferencex_trace_replay`` fixture (schema 0.1, no prefix_ref)."""
    return {
        "adapter_id": "inferencex_trace_replay",
        "schema_version": "0.1.0",
        "exports": [
            {
                "trace_id": "ext_64k_001",
                "runtime_stack_id": "standalone:sglang",
                "hardware_profile_id": "nvidia:h200_sxm_141gb",
                "canonical_model_id": "qwen3_5_397b_a17b",
                "support_status": "supported",
                "benchmark_certification_status": "dataset_replay_verified",
                "context_band": "extension_64k",
                "trace_metadata": {"session_id": "sess_tr_001"},
                "events": [
                    {
                        "arrival_time_offset_ms": 0,
                        "input_messages": [
                            {"role": "system", "content": "s" * 64},
                            {"role": "user", "content": "x" * 12000},
                        ],
                        "target_output_tokens": 100,
                    },
                    {
                        "arrival_time_offset_ms": 4000,
                        "input_messages": [
                            {"role": "system", "content": "s" * 64},
                            {"role": "user", "content": "x" * 12000},
                            {"role": "assistant", "content": "r" * 400},
                            {"role": "user", "content": "follow" * 10},
                        ],
                        "target_output_tokens": 150,
                    },
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_shim(
    tmp_path: Path,
    bundle: dict,
    *,
    extra_args: list[str] | None = None,
) -> Path:
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(SHIM),
        "--export-file",
        str(bundle_path),
        "--output-dir",
        str(out_dir),
        "--quiet",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"shim failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    return out_dir


def _load_single_trace(out_dir: Path) -> dict:
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == 1, f"expected 1 trace file, got {len(files)}: {files}"
    return json.loads(files[0].read_text())


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestMultiturnBundle:
    def test_top_level_schema(self, tmp_path: Path) -> None:
        out = _run_shim(tmp_path, _multiturn_bundle())
        trace = _load_single_trace(out)

        # Cam's tester reads these exact fields (see trace_replay_tester.py::normalize_trace).
        assert trace["id"] == "sess_001"
        assert trace["models"] == ["qwen3_5_397b_a17b"]
        assert trace["block_size"] == 64
        assert trace["hash_id_scope"] == "local"
        assert trace["tool_tokens"] == 0
        assert trace["system_tokens"] == 0
        assert isinstance(trace["requests"], list)
        assert len(trace["requests"]) == 2

    def test_per_request_schema(self, tmp_path: Path) -> None:
        out = _run_shim(tmp_path, _multiturn_bundle())
        trace = _load_single_trace(out)

        required_keys = {"t", "type", "model", "in", "out", "hash_ids"}
        for req in trace["requests"]:
            missing = required_keys - req.keys()
            assert not missing, f"missing required request keys: {missing}"
            assert req["type"] in ("n", "s")
            assert isinstance(req["in"], int) and req["in"] > 0
            assert isinstance(req["out"], int) and req["out"] > 0
            assert isinstance(req["hash_ids"], list) and req["hash_ids"]

    def test_hash_ids_are_prefix_extending(self, tmp_path: Path) -> None:
        """hash_ids[i+1] must start with hash_ids[i] for cache hit rate > 0."""
        out = _run_shim(tmp_path, _multiturn_bundle())
        trace = _load_single_trace(out)
        reqs = trace["requests"]

        for i in range(1, len(reqs)):
            prev_hash_ids = reqs[i - 1]["hash_ids"]
            curr_hash_ids = reqs[i]["hash_ids"]
            assert len(curr_hash_ids) >= len(prev_hash_ids), (
                f"turn {i} has fewer hash_ids than turn {i-1}"
            )
            assert curr_hash_ids[: len(prev_hash_ids)] == prev_hash_ids, (
                f"turn {i} hash_ids do not start with turn {i-1} prefix — "
                "this breaks kv-cache-tester's cache-hit walker"
            )

    def test_block_size_mapping(self, tmp_path: Path) -> None:
        out = _run_shim(tmp_path, _multiturn_bundle(), extra_args=["--block-size", "32"])
        trace = _load_single_trace(out)
        assert trace["block_size"] == 32
        for req in trace["requests"]:
            expected_blocks = max(1, math.ceil(req["in"] / 32))
            assert req["hash_ids"] == list(range(1, expected_blocks + 1))

    def test_arrival_timing_monotonic(self, tmp_path: Path) -> None:
        out = _run_shim(tmp_path, _multiturn_bundle())
        trace = _load_single_trace(out)
        times = [r["t"] for r in trace["requests"]]
        assert times == sorted(times), "request arrival times must be monotonic"
        # second turn wait_before_ms=1500 → t>=1.5
        assert trace["requests"][1]["t"] >= 1.5

    def test_isb1_passthrough_tags(self, tmp_path: Path) -> None:
        out = _run_shim(tmp_path, _multiturn_bundle())
        trace = _load_single_trace(out)
        assert "isb1" in trace
        tags = trace["isb1"]
        assert tags["adapter_id"] == "inferencex_multiturn"
        assert tags["runtime_stack_id"] == "standalone:vllm"
        assert tags["hardware_profile_id"] == "nvidia:h200_sxm_141gb"
        assert tags["canonical_model_id"] == "qwen3_5_397b_a17b"
        assert tags["support_status"] == "supported"
        assert tags["benchmark_certification_status"] == "dataset_replay_verified"
        assert tags["context_band"] == "core_8k"

    def test_normalize_trace_compatibility(self, tmp_path: Path) -> None:
        """The emitted trace must round-trip through normalize_trace cleanly."""
        out = _run_shim(tmp_path, _multiturn_bundle())
        trace = _load_single_trace(out)
        normalized = _normalize_trace(trace)

        md = normalized["metadata"]
        assert md["conversation_id"] == "sess_001"
        assert md["request_count"] == 2
        assert md["total_input_tokens"] > 0
        # Two turns, second extends first's prefix → cache hit rate > 0.
        assert md["cache_hit_rate"] > 0.0


class TestTraceReplayBundle:
    def test_reads_events(self, tmp_path: Path) -> None:
        out = _run_shim(tmp_path, _trace_replay_bundle())
        trace = _load_single_trace(out)
        assert trace["id"] == "sess_tr_001"
        assert len(trace["requests"]) == 2

    def test_arrival_offsets(self, tmp_path: Path) -> None:
        out = _run_shim(tmp_path, _trace_replay_bundle())
        trace = _load_single_trace(out)
        # second event arrival_time_offset_ms=4000 → t=4.0
        assert trace["requests"][0]["t"] == 0.0
        assert trace["requests"][1]["t"] == 4.0

    def test_prefix_reuse_dominates(self, tmp_path: Path) -> None:
        out = _run_shim(tmp_path, _trace_replay_bundle())
        trace = _load_single_trace(out)
        normalized = _normalize_trace(trace)
        # 12k-token prefix shared across both events → cache hit rate ≈ 49-50%.
        assert 0.4 < normalized["metadata"]["cache_hit_rate"] < 0.6


class TestFilters:
    def test_runtime_stack_filter_excludes_non_match(self, tmp_path: Path) -> None:
        bundle_path = tmp_path / "bundle.json"
        bundle_path.write_text(json.dumps(_multiturn_bundle()))
        out_dir = tmp_path / "out"

        # Non-matching runtime filter should exit non-zero with "no traces".
        result = subprocess.run(
            [
                sys.executable,
                str(SHIM),
                "--export-file",
                str(bundle_path),
                "--output-dir",
                str(out_dir),
                "--runtime-stack-id",
                "standalone:trtllm",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "no traces written" in result.stderr.lower()

    def test_support_status_filter_allows_match(self, tmp_path: Path) -> None:
        out = _run_shim(
            tmp_path,
            _multiturn_bundle(),
            extra_args=["--support-status", "supported,reviewed_preview"],
        )
        assert sorted(out.glob("*.json"))

    def test_max_turns_truncates(self, tmp_path: Path) -> None:
        out = _run_shim(
            tmp_path,
            _multiturn_bundle(),
            extra_args=["--max-turns-per-conversation", "1"],
        )
        trace = _load_single_trace(out)
        assert len(trace["requests"]) == 1


class TestErrorHandling:
    def test_rejects_unknown_adapter(self, tmp_path: Path) -> None:
        bundle = {"adapter_id": "something_else", "exports": []}
        bundle_path = tmp_path / "bundle.json"
        bundle_path.write_text(json.dumps(bundle))
        out_dir = tmp_path / "out"
        result = subprocess.run(
            [
                sys.executable,
                str(SHIM),
                "--export-file",
                str(bundle_path),
                "--output-dir",
                str(out_dir),
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "unsupported isb1 adapter" in result.stderr.lower()

    def test_rejects_nonpositive_block_size(self, tmp_path: Path) -> None:
        bundle_path = tmp_path / "bundle.json"
        bundle_path.write_text(json.dumps(_multiturn_bundle()))
        result = subprocess.run(
            [
                sys.executable,
                str(SHIM),
                "--export-file",
                str(bundle_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--block-size",
                "0",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "--block-size must be positive" in result.stderr
