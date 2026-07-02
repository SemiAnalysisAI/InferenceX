"""Server-log parsing helpers for agentic aggregate generation."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path


_ENGINE_TAG_RE = re.compile(r"\((?P<tag>EngineCore(?:_DP\d+)?)\s+pid=\d+\)")
_GPU_KV_SIZE_RE = re.compile(r"GPU KV cache size:\s*(?P<tokens>[\d,]+)\s*tokens")
_SGLANG_RANK_RE = re.compile(r"\b(?P<tag>DP\d+\s+TP\d+\s+EP\d+)\b")
_SGLANG_MAX_TOKENS_RE = re.compile(r"\bmax_total_num_tokens=(?P<tokens>\d+)\b")
_SGLANG_DP_SIZE_RE = re.compile(r"\bdp_size=(?P<dp_size>\d+)\b")


def load_server_log_head(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def kv_cache_pool_tokens_from_server_log(server_log: str | None) -> int | None:
    if not server_log:
        return None

    vllm_total = vllm_kv_cache_pool_tokens_from_server_log(server_log)
    if vllm_total is not None:
        return vllm_total

    return sglang_kv_cache_pool_tokens_from_server_log(server_log)


def vllm_kv_cache_pool_tokens_from_server_log(server_log: str | None) -> int | None:
    if not server_log:
        return None
    return _vllm_kv_cache_pool_tokens(server_log)


def sglang_kv_cache_pool_tokens_from_server_log(server_log: str | None) -> int | None:
    if not server_log:
        return None
    return _sglang_kv_cache_pool_tokens(server_log)


def kv_cache_pool_tokens_from_server_logs(paths: list[Path]) -> int | None:
    return _sum_server_log_capacities(paths, kv_cache_pool_tokens_from_server_log)


def vllm_kv_cache_pool_tokens_from_server_logs(paths: list[Path]) -> int | None:
    return _sum_server_log_capacities(paths, vllm_kv_cache_pool_tokens_from_server_log)


def sglang_kv_cache_pool_tokens_from_server_logs(paths: list[Path]) -> int | None:
    return _sum_server_log_capacities(paths, sglang_kv_cache_pool_tokens_from_server_log)


def _sum_server_log_capacities(
    paths: list[Path],
    parser: Callable[[str | None], int | None],
) -> int | None:
    total = 0
    found = False
    for path in paths:
        value = parser(load_server_log_head(path))
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def find_server_log_paths(result_dir: Path) -> list[Path]:
    paths: list[Path] = []
    direct = result_dir / "server.log"
    if direct.is_file():
        paths.append(direct)

    for root in (result_dir, *result_dir.parents[:3]):
        if not root.is_dir():
            continue
        paths.extend(sorted(root.glob("watchtower-*.out")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _vllm_kv_cache_pool_tokens(server_log: str) -> int | None:
    per_engine: dict[str, int] = {}
    bare_total = 0
    bare_found = False

    for line in server_log.splitlines():
        if "GPU KV cache size" not in line:
            continue

        size_match = _GPU_KV_SIZE_RE.search(line)
        if not size_match:
            continue

        try:
            tokens = int(size_match.group("tokens").replace(",", ""))
        except ValueError:
            continue
        if tokens <= 0:
            continue

        tag_match = _ENGINE_TAG_RE.search(line)
        if tag_match:
            per_engine[tag_match.group("tag")] = tokens
        else:
            bare_total += tokens
            bare_found = True

    if per_engine:
        return sum(per_engine.values())
    return bare_total if bare_found else None


def _sglang_kv_cache_pool_tokens(server_log: str) -> int | None:
    per_rank: dict[str, int] = {}
    bare_total = 0
    bare_count = 0
    dp_size = _sglang_dp_size(server_log)

    for line in server_log.splitlines():
        if "max_total_num_tokens" not in line:
            continue
        size_match = _SGLANG_MAX_TOKENS_RE.search(line)
        if not size_match:
            continue
        tokens = int(size_match.group("tokens"))
        if tokens <= 0:
            continue
        tag_match = _SGLANG_RANK_RE.search(line)
        if tag_match:
            per_rank[tag_match.group("tag")] = tokens
        else:
            bare_total += tokens
            bare_count += 1

    if per_rank:
        if dp_size is not None and len(per_rank) == 1 and dp_size > 1:
            return next(iter(per_rank.values())) * dp_size
        return sum(per_rank.values())
    if bare_count == 1 and dp_size is not None and dp_size > 1:
        return bare_total * dp_size
    return bare_total if bare_count else None


def _sglang_dp_size(server_log: str) -> int | None:
    match = _SGLANG_DP_SIZE_RE.search(server_log)
    if not match:
        return None
    dp_size = int(match.group("dp_size"))
    return dp_size if dp_size > 0 else None
