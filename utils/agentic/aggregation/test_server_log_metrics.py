from __future__ import annotations

from pathlib import Path

from utils.agentic.aggregation.backends.dynamo_vllm import DynamoVllmBackend
from utils.agentic.aggregation.backends.sglang import SglangBackend
from utils.agentic.aggregation.backends.vllm import VllmBackend
from utils.agentic.aggregation.server_log_metrics import (
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


def test_kv_cache_pool_tokens_from_single_engine_server_log() -> None:
    log = "INFO (EngineCore pid=123) GPU KV cache size: 11,294,463 tokens"

    assert VllmBackend.kv_cache_pool_tokens_from_server_log(log) == 11_294_463


def test_kv_cache_pool_tokens_from_data_parallel_server_log() -> None:
    log = "\n".join(
        [
            "INFO (EngineCore_DP0 pid=123) GPU KV cache size: 11,577,333 tokens",
            "INFO (EngineCore_DP1 pid=124) GPU KV cache size: 11,577,333 tokens",
            "INFO (EngineCore_DP2 pid=125) GPU KV cache size: 11,577,333 tokens",
        ]
    )

    assert VllmBackend.kv_cache_pool_tokens_from_server_log(log) == 34_731_999


def test_kv_cache_pool_tokens_dedupes_engine_tags() -> None:
    log = "\n".join(
        [
            "INFO (EngineCore_DP0 pid=123) GPU KV cache size: 11,577,333 tokens",
            "INFO (EngineCore_DP0 pid=123) GPU KV cache size: 11,577,333 tokens",
            "INFO (EngineCore_DP1 pid=124) GPU KV cache size: 5,000,000 tokens",
        ]
    )

    assert VllmBackend.kv_cache_pool_tokens_from_server_log(log) == 16_577_333


def test_kv_cache_pool_tokens_sums_bare_lines() -> None:
    log = "\n".join(
        [
            "INFO GPU KV cache size: 1,234,567 tokens",
            "INFO GPU KV cache size: 2,000,000 tokens",
        ]
    )

    assert VllmBackend.kv_cache_pool_tokens_from_server_log(log) == 3_234_567


def test_kv_cache_pool_tokens_from_sglang_server_log() -> None:
    log = "\n".join(
        [
            "[2026-06-23 01:05:00] server_args=ServerArgs(dp_size=8, tp_size=8)",
            "[2026-06-23 01:10:14 DP0 TP0 EP0] max_total_num_tokens=1172224, "
            "chunked_prefill_size=4096",
        ]
    )

    assert SglangBackend.kv_cache_pool_tokens_from_server_log(log) == 9_377_792


def test_kv_cache_pool_tokens_from_sglang_per_rank_lines() -> None:
    log = "\n".join(
        [
            "[2026-06-23 01:10:14 DP0 TP0 EP0] max_total_num_tokens=1000",
            "[2026-06-23 01:10:14 DP1 TP1 EP1] max_total_num_tokens=1200",
            "[2026-06-23 01:10:14 DP1 TP1 EP1] max_total_num_tokens=1200",
        ]
    )

    assert SglangBackend.kv_cache_pool_tokens_from_server_log(log) == 2200


def test_kv_cache_pool_tokens_sums_multiple_log_files(tmp_path: Path) -> None:
    first = tmp_path / "watchtower-a.out"
    second = tmp_path / "watchtower-b.out"
    first.write_text(
        "\n".join(
            [
                "INFO (EngineCore_DP0 pid=100) GPU KV cache size: 5,000,000 tokens",
                "INFO (EngineCore_DP1 pid=101) GPU KV cache size: 6,500,000 tokens",
            ]
        )
    )
    second.write_text(
        "INFO (EngineCore_DP0 pid=200) GPU KV cache size: 7,000,000 tokens"
    )

    assert VllmBackend().gpu_kv_capacity_tokens({}, [first, second]) == 18_500_000


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

    assert DynamoVllmBackend().gpu_kv_capacity_tokens({}, [worker_log]) == 11_500_000


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
