#!/usr/bin/env python3
"""KV-transfer workload: vLLM 0.32.0's DeepSeek-V4-Pro block layout, tables and verification."""

from __future__ import annotations

import functools
import math

import numpy as np

ALIGNMENT = 576  # vLLM pads every fp8_ds_mla-family page to this

# name: (block_tokens, tokens_per_state, bytes_per_state). page bytes =
# round_up(block_tokens / tokens_per_state * bytes_per_state, ALIGNMENT).
DSV4_CACHES = {
    "c4a": (256, 4, 584),            # CSA KV: 448 NoPE + 128 RoPE + 8 scale
    "c4a-indexer": (256, 4, 132),    # lightning indexer: 128 fp8 + 4 scale
    "c128a": (256, 128, 584),        # HCA KV
    "swa": (64, 1, 584),             # 128-token sliding window, block 64
    "c4a-state": (4, 1, 8192),       # C4 attention compressor, fp32 2x2x512
    "c4a-indexer-state": (4, 1, 2048),
    "c128a-state": (8, 1, 4096),     # C128 compressor, fp32 2x512
}

PRESETS = {
    "dsv4": dict(
        model_class="deepseek-v4-pro",
        precisions=("fp8",),
        model_layers=61,
        block_tokens=256,
        vllm_version="0.32.0",
        # vLLM's transfer groups, in its order (the descriptor order of a request):
        # the 61 sliding-window layers split 31/30, the three compressed-attention
        # MLA caches share one group, then the C128 and C4 compressor states.
        # (cache, layers) each, in the order vLLM lays each group's pages out from
        # row byte 0. Dumped from vLLM 0.32.0 itself (KV cache config, NIXL
        # registration and the per-request block ids it hands the connector).
        groups=(
            dict(name="swa-a", block_tokens=64, window=128, caches=(("swa", 31),)),
            dict(name="swa-b", block_tokens=64, window=128, caches=(("swa", 30),)),
            dict(name="mla", block_tokens=256, window=None,
                 caches=(("c128a", 31), ("c4a-indexer", 30), ("c4a", 30))),
            dict(name="c128a-state", block_tokens=8, window=128,
                 caches=(("c128a-state", 31),)),
            dict(name="c4a-state", block_tokens=4, window=8,
                 caches=(("c4a-indexer-state", 30), ("c4a-state", 30))),
        ),
    ),
}
DEFAULT_TABLE_SETS = 4


def view_bytes(cache: str) -> int:
    """Bytes one layer's view spans in each block: its states without the page's padding."""
    block, per_state, state_bytes = DSV4_CACHES[cache]
    return block // per_state * state_bytes


def page_bytes(cache: str) -> int:
    return -(-view_bytes(cache) // ALIGNMENT) * ALIGNMENT


def group_block_bytes(group: dict) -> int:
    return sum(layers * page_bytes(cache) for cache, layers in group["caches"])


def group_blocks(group: dict, isl: int) -> int:
    """Block ids vLLM moves for this group: every block, or for a windowed group the
    decode side's live blocks (isl - window + 1 tokens are skipped as null blocks)
    clipped to the window's tail, cdiv(window, block) + 1."""
    block = group["block_tokens"]
    blocks = math.ceil(isl / block)
    if group["window"] is None:
        return blocks
    live = blocks - max(isl - group["window"] + 1, 0) // block
    return min(math.ceil(group["window"] / block) + 1, live)


@functools.lru_cache(maxsize=None)
def mooncake_regions(preset: str) -> tuple[tuple[tuple[int, ...], int, int | None], ...]:
    """vLLM 0.32.0 MooncakeConnector's transfer regions as (group indices whose ids
    it moves, row offset, bytes per id; None = the whole row): layer views at an
    already-registered (offset, length) fold into that region, consecutive views of
    one group set merge, and a run from row byte 0 becomes the whole row."""
    groups = PRESETS[preset]["groups"]
    row = max(group_block_bytes(g) for g in groups)
    regions, seen = [], {}
    for index, group in enumerate(groups):
        offset = 0
        for cache, layers in group["caches"]:
            for _ in range(layers):
                key = (offset, view_bytes(cache))
                if key not in seen:
                    seen[key] = len(regions)
                    regions.append(([index], offset, view_bytes(cache)))
                elif index not in regions[seen[key]][0]:
                    regions[seen[key]][0].append(index)
                offset += page_bytes(cache)
    runs = []
    for members, offset, size in regions:
        if runs and runs[-1][0] == members and runs[-1][2] <= offset \
                and offset + size <= runs[-1][1] + row:
            runs[-1][2] = offset + size
        else:
            runs.append([members, offset, offset + size])
    return tuple((tuple(members), start, None if start == 0 else end - start)
                 for members, start, end in runs)


def mooncake_entries(cfg: dict, local_rows, remote_rows) -> tuple[np.ndarray, ...]:
    """(local offsets, remote offsets, sizes) as vLLM 0.32.0's MooncakeConnector
    posts one request: per region, whole-row ids whose local and remote rows both
    advance by one coalesce into a run (group_concurrent_contiguous); a page
    region posts one page per id."""
    local_rows = np.asarray(local_rows, dtype=np.int64)
    remote_rows = np.asarray(remote_rows, dtype=np.int64)
    ends = np.cumsum([g["blocks"] for g in cfg["groups"]])
    spans = list(zip(ends - [g["blocks"] for g in cfg["groups"]], ends))
    row = cfg["row_bytes"]
    local, remote, sizes = [], [], []
    for members, offset, size in mooncake_regions(cfg["preset"]):
        ids = np.concatenate([np.arange(*spans[index]) for index in members])
        lo, re = local_rows[ids], remote_rows[ids]
        if size is None:
            cuts = np.flatnonzero((np.diff(lo) != 1) | (np.diff(re) != 1)) + 1
            bounds = np.concatenate([[0], cuts, [lo.size]])
            lo, re, nbytes = lo[bounds[:-1]], re[bounds[:-1]], np.diff(bounds) * row
        else:
            nbytes = np.full(lo.size, size)
        local.append(lo * row + offset)
        remote.append(re * row + offset)
        sizes.append(nbytes)
    return tuple(np.concatenate(part).astype(np.uint64) for part in (local, remote, sizes))


def nixl_entries(cfg: dict, local_rows, remote_rows) -> tuple[np.ndarray, ...]:
    """(local offsets, remote offsets, sizes) as vLLM 0.32.0's NixlConnector posts
    one request: one whole-row descriptor per block id."""
    row = np.uint64(cfg["row_bytes"])
    return (np.asarray(local_rows, dtype=np.uint64) * row,
            np.asarray(remote_rows, dtype=np.uint64) * row, np.full(cfg["descs"], row))


def plan_config(preset: str, precision: str, isl: int, pool_slack: float = 2.0,
                batch_max: int = 1, table_sets: int = DEFAULT_TABLE_SETS) -> dict:
    """One (preset, isl) point at the preset's served block size: per-group block
    counts, the row size (the widest group's bytes per block), descriptors and
    bytes per request, and a pool of rows for ``table_sets`` disjoint sets of
    ``batch_max`` requests plus head-room."""
    shape = PRESETS[preset]
    if precision not in shape["precisions"]:
        raise ValueError(f"{preset} runs {shape['precisions']}, not {precision}")
    row_bytes = max(group_block_bytes(g) for g in shape["groups"])
    groups = [dict(name=g["name"], blocks=group_blocks(g, isl)) for g in shape["groups"]]
    descs = sum(g["blocks"] for g in groups)
    per_request = max(pool_slack, batch_max * 1.25)
    pool_rows = int(descs * table_sets * per_request) + 8
    return dict(
        preset=preset,
        isl=isl,
        page_tokens=shape["block_tokens"],
        layers=shape["model_layers"],
        row_bytes=row_bytes,
        groups=groups,
        descs=descs,
        req_bytes=descs * row_bytes,
        batch_max=batch_max,
        table_sets=table_sets,
        pool_rows=pool_rows,
        pool_bytes=pool_rows * row_bytes,
    )


def block_table(cfg: dict, seed: int, request: int = 0, table_set: int = 0) -> np.ndarray:
    """One request's row ids in group order (deterministic, seed-keyed): a slice
    of one random permutation of the pool's rows. Every (table set, request)
    slices a disjoint range, as a real allocator's live requests never alias
    blocks."""
    rng = np.random.default_rng(seed)
    low = (table_set * cfg["batch_max"] + request) * cfg["descs"]
    rows = rng.permutation(cfg["pool_rows"])[low : low + cfg["descs"]]
    if len(rows) < cfg["descs"]:
        raise ValueError(f"pool too small for table set {table_set} request {request}")
    return rows.astype(np.int64)


def table_seed(cfg: dict, side: str, seed: int = 0) -> int:
    """Both ranks derive both sides' tables from the config and the sweep seed — no exchange."""
    base = cfg["isl"] * 31 + cfg["page_tokens"] + len(cfg["preset"]) * 7 + seed * 7919
    return base + (1000 if side == "local" else 0)


_SALT_MIX = 0x5851F42D4C957F2D
FILL_CHUNK_WORDS = 1 << 27  # bounds the fill's arange temp at 1 GiB


def salt_mix(salt: int) -> int:
    return ((salt + 1) * _SALT_MIX) & 0x7FFFFFFFFFFFFFFF


def fill_pattern(words, salt: int) -> None:
    """Paint word i of an int64 torch view with ``i ^ salt_mix(salt)`` on-device."""
    import torch

    mix = salt_mix(salt)
    for start in range(0, words.numel(), FILL_CHUNK_WORDS):
        end = min(start + FILL_CHUNK_WORDS, words.numel())
        words[start:end] = torch.arange(start, end, dtype=torch.int64,
                                        device=words.device) ^ mix


def pattern_words(start: int, end: int, salt: int) -> np.ndarray:
    """Host reference of the pattern: words [start, end) of a pool painted
    with ``salt`` (what fill_pattern writes on-device)."""
    return np.arange(start, end, dtype=np.int64) ^ np.int64(salt_mix(salt))


def _gather(words, idx: np.ndarray) -> np.ndarray:
    """Probe words to host: only 3 words per entry cross PCIe, so the
    comparison runs in numpy (and a numpy pool stands in for tests)."""
    if isinstance(words, np.ndarray):
        return words[idx]
    import torch

    return words[torch.from_numpy(idx).to(words.device)].cpu().numpy()


def _check(dst_words, dst_idx: np.ndarray, src_idx: np.ndarray, src_salt: int):
    got = _gather(dst_words, dst_idx)
    expected = src_idx ^ np.int64(salt_mix(src_salt))
    bad = np.flatnonzero(got != expected)
    if bad.size == 0:
        return None
    k = int(bad[0])
    return k, int(expected[k]), int(got[k]), int(bad.size)


def verify_entries(dst_words, dst_offsets, src_offsets, sizes,
                   src_salt: int, seed: int = 7) -> tuple[bool, str]:
    """On-device: each destination entry must hold its source entry. Every
    entry (a row descriptor or a per-layer page) is probed at its first, last
    and one random interior word, each expected to equal the source word
    index XOR the source salt mix. Offsets and sizes are bytes, 8-aligned."""
    dst = np.asarray(dst_offsets, dtype=np.int64) // 8
    src = np.asarray(src_offsets, dtype=np.int64) // 8
    words = np.asarray(sizes, dtype=np.int64) // 8
    rng = np.random.default_rng(seed)
    interior = (rng.random(len(words)) * np.maximum(words - 2, 1)).astype(np.int64) + 1
    within = np.stack([np.zeros_like(words), np.minimum(interior, words - 1), words - 1],
                      axis=1)
    miss = _check(dst_words, (dst[:, None] + within).reshape(-1),
                  (src[:, None] + within).reshape(-1), src_salt)
    if miss is None:
        return True, ""
    k, expected, got, count = miss
    return False, (f"entry={k // 3} probe={('first', 'interior', 'last')[k % 3]} "
                   f"dst_offset={int(dst[k // 3]) * 8} src_offset={int(src[k // 3]) * 8} "
                   f"expected={expected} got={got} mismatched={count}/{3 * len(words)}")


def verify_bulk(dst_words, nbytes: int, src_salt: int, seed: int = 7,
                samples: int = 4096) -> tuple[bool, str]:
    """On-device: the contiguous range's first, last and ``samples`` random
    words must hold the source's pattern (same offsets on both sides)."""
    words = nbytes // 8
    rng = np.random.default_rng(seed)
    idx = np.unique(np.concatenate([[0, words - 1],
                                    rng.integers(0, words, size=samples)])).astype(np.int64)
    miss = _check(dst_words, idx, idx, src_salt)
    if miss is None:
        return True, ""
    k, expected, got, count = miss
    return False, (f"word={int(idx[k])} expected={expected} got={got} "
                   f"mismatched={count}/{len(idx)}")


def pcts(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    n = len(ordered)
    return {
        "p50": ordered[n // 2],
        "p95": ordered[min(n - 1, int(n * 0.95))],
        "min": ordered[0],
        "max": ordered[-1],
        "n": n,
    }


def library_version(dists, module=None) -> str | None:
    """The first installed distribution's version, else ``module.__version__``."""
    import importlib.metadata as md

    for name in dists:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            pass
    return getattr(module, "__version__", None)
