from __future__ import annotations

from pathlib import Path

import pytest

from infx.results.agentic.backends.dynamo_vllm import DynamoVllmBackend
from infx.results.agentic.backends.llamacpp import LlamaCppBackend
from infx.results.agentic.backends.sglang import SglangBackend
from infx.results.agentic.backends.vllm import VllmBackend
from infx.results.agentic.artifacts import (
    find_server_log_paths,
    load_server_log_head,
)


def test_kv_cache_pool_tokens_from_server_log_missing() -> None:
    assert VllmBackend.kv_cache_pool_tokens_from_server_log(None) is None
    assert VllmBackend.kv_cache_pool_tokens_from_server_log("") is None
    assert VllmBackend.kv_cache_pool_tokens_from_server_log("INFO no kv cache line") is None
    assert SglangBackend.kv_cache_pool_tokens_from_server_log(None) is None
    assert SglangBackend.kv_cache_pool_tokens_from_server_log("") is None
    assert SglangBackend.kv_cache_pool_tokens_from_server_log("INFO no kv cache line") is None


def test_kv_cache_pool_tokens_sums_bare_lines() -> None:
    log = "\n".join(
        [
            "INFO GPU KV cache size: 1,234,567 tokens",
            "INFO GPU KV cache size: 2,000,000 tokens",
        ]
    )

    assert VllmBackend.kv_cache_pool_tokens_from_server_log(log) == 3_234_567


def test_kv_cache_pool_tokens_from_sglang_per_rank_lines() -> None:
    log = "\n".join(
        [
            "[2026-06-23 01:10:14 DP0 TP0 EP0] max_total_num_tokens=1000",
            "[2026-06-23 01:10:14 DP1  TP1\tEP1] max_total_num_tokens=1200",
            "[2026-06-23 01:10:14 DP1 TP1 EP1] max_total_num_tokens=1200",
        ]
    )

    assert SglangBackend.kv_cache_pool_tokens_from_server_log(log) == 2200


def llamacpp_slots(slots: int, context: int, unified: bool) -> str:
    """b11910's slot line: ``n_ctx_slot`` is a slot's context after every cap."""
    shared = "true" if unified else "false"
    return f"0.01.029.530 I srv    load_model: initializing, n_slots = {slots}, n_ctx_slot = {context}, kv_unified = '{shared}'"


TRAINING_CAP = "0.01.029.529 W srv    load_model: the slot context ({}) exceeds the training context of the model (262144) - capping"
SEQ_OVER_TRAINING = "0.00.951.004 W llama_context: n_ctx_seq ({}) > n_ctx_train (262144) -- possible training context overflow"


@pytest.mark.parametrize(("lines", "tokens"), [
    ([llamacpp_slots(4, 9472, False)], 4 * 9472),
    ([llamacpp_slots(4, 32768, True)], 32768),
    ([SEQ_OVER_TRAINING.format(524288), TRAINING_CAP.format(524288), llamacpp_slots(2, 262144, False)], 2 * 524288),
    ([TRAINING_CAP.format(524288), llamacpp_slots(4, 262144, True)], 524288),
    (["0.00.199.760 I srv  llama_server: initializing ..."], None),
], ids=["a-cache-per-slot", "one-shared-cache", "per-slot-caches-capped-at-training", "shared-cache-capped-at-training-in-the-server-log", "no-slots-logged"])  # fmt: skip
def test_llamacpp_kv_pool_is_the_cache_its_slots_were_allocated(lines, tokens) -> None:
    assert LlamaCppBackend.kv_cache_pool_tokens_from_server_log("\n".join(lines)) == tokens


def test_dynamo_vllm_uses_vllm_server_log_capacity_parser(tmp_path: Path) -> None:
    worker_log = tmp_path / "watchtower-worker.out"
    worker_log.write_text(
        "\n".join(
            [
                "INFO (EngineCore_DP0 pid=100) GPU KV cache size: 5,000,000 tokens",
                "INFO (EngineCore_DP1 pid=101) GPU KV cache size: 6,500,000 tokens",
            ]
        )
    )

    assert DynamoVllmBackend().gpu_kv_capacity_tokens({}, [load_server_log_head(worker_log)]) == 11_500_000


def test_find_server_log_paths_includes_multinode_watchtower_logs(tmp_path: Path) -> None:
    result_dir = tmp_path / "logs" / "agentic" / "conc_128"
    result_dir.mkdir(parents=True)
    server_log = result_dir / "server.log"
    worker_log = tmp_path / "logs" / "watchtower-node_prefill_w0.out"
    ignored = tmp_path / "logs" / "benchmark.out"
    server_log.write_text("server")
    worker_log.write_text("worker")
    ignored.write_text("ignored")

    paths = find_server_log_paths(result_dir)

    assert server_log in paths
    assert worker_log in paths
    assert ignored not in paths


def test_load_server_log_head_missing_and_sanitized(tmp_path: Path) -> None:
    missing = tmp_path / "missing.log"
    assert load_server_log_head(missing) is None

    server_log = tmp_path / "server.log"
    server_log.write_bytes(b"abc\x00def")

    assert load_server_log_head(server_log) == "abcdef"
