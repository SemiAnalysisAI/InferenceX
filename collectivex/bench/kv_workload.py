#!/usr/bin/env python3
"""KV-transfer workloads: the block layouts vLLM registers for Kimi-K3.1-Pro (the served preset)
and DeepSeek-V4-Pro, the descriptors its NixlConnector posts for one request, tables and
verification."""

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

# Kimi-K3.1-Pro on one rank of a TP16 engine, as the vllm/vllm-openai:kimi-k31 image (vLLM
# 0.1.dev24216) lays it out with kv-cache-dtype fp8, an mxfp4 indexer, float32 KDA state and the
# DSpark draft: 1024-token blocks, one allocation of 18,800,640-byte rows with every cache group
# overlaid from row byte 0. A GB200 production run logged the same shape (88 regions,
# block lengths {425472, 589824, 69632}, KDA state (32256, 393216)); docs/kv-transfer.md.
K31_INDEXER = 69_632          # 24 DSA indexers: (128/2 + 128/32) B x 1024 tokens
K31_MLA = 589_824             # 24 MLA layers and 5 draft layers: (512 + 64) B x 1024 tokens
K31_KDA = 425_472             # one KDA layer's state on one of 16 ranks
# NIXL reads a KDA state as its three conv projections and the recurrent state.
K31_KDA_PARTS = ((0, 10_752), (10_752, 10_752), (21_504, 10_752), (32_256, 393_216))
K31_ATTENTION = tuple(
    [(k * K31_INDEXER, K31_INDEXER) for k in range(24)]
    + [(24 * K31_INDEXER + m * K31_MLA, K31_MLA) for m in range(29)])
K31_KDA_SLOTS = 35            # 69 KDA layers in two overlaid groups of 35 and 34
PRESETS["k31"] = dict(
    model_class="kimi-k3.1-pro",
    precisions=("fp8",),
    model_layers=93,
    block_tokens=1024,
    vllm_version="0.1.dev24216 (vllm/vllm-openai:kimi-k31)",
    tp_size=16,
    row_bytes=18_800_640,     # the attention group's 18,776,064 B, padded to its stride alignment
    # What vLLM registers per block: its 88 regions, then each KDA region's four parts.
    kinds=(tuple((j * K31_KDA, K31_KDA) for j in range(K31_KDA_SLOTS)) + K31_ATTENTION
           + tuple((j * K31_KDA + off, size) for j in range(K31_KDA_SLOTS)
                   for off, size in K31_KDA_PARTS)),
    # A request moves every prompt block's attention pages and one running-state block per KDA
    # group. kinds index into the tuple above.
    groups=(
        dict(name="attention", state=False,
             kinds=tuple(range(K31_KDA_SLOTS, K31_KDA_SLOTS + len(K31_ATTENTION)))),
        dict(name="kda-a", state=True, kinds=tuple(range(88, 88 + 4 * 35))),
        dict(name="kda-b", state=True, kinds=tuple(range(88, 88 + 4 * 34))),
    ),
)


def _is_regions(preset: str) -> bool:
    """A preset whose connector posts per-layer regions (K3.1), not whole rows (DSV4)."""
    return "kinds" in PRESETS[preset]


def desc_kinds(cfg: dict) -> tuple[tuple[int, int], ...]:
    """(row offset, bytes) of every descriptor kind the connector preps for each pool row."""
    shape = PRESETS[cfg["preset"]]
    return shape["kinds"] if _is_regions(cfg["preset"]) else ((0, cfg["row_bytes"]),)


def request_descs(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(kind index, position in the request's row list) of each descriptor one request posts,
    in posting order."""
    if not _is_regions(cfg["preset"]):
        return np.zeros(cfg["descs"], dtype=np.int64), np.arange(cfg["descs"], dtype=np.int64)
    kinds, rows, first = [], [], 0
    for group, plan in zip(PRESETS[cfg["preset"]]["groups"], cfg["groups"]):
        members = np.asarray(group["kinds"], dtype=np.int64)
        for row in range(first, first + plan["blocks"]):
            kinds.append(members)
            rows.append(np.full(len(members), row, dtype=np.int64))
        first += plan["blocks"]
    return np.concatenate(kinds), np.concatenate(rows)


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
    if _is_regions(preset):
        raise ValueError(f"no vLLM MooncakeConnector plan is modelled for {preset}")
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
    """(local offsets, remote offsets, sizes) as vLLM's NixlConnector posts one request:
    for DSV4 one whole-row descriptor per block id, for K3.1 each block's regions."""
    table = np.asarray(desc_kinds(cfg), dtype=np.int64)
    kinds, rows = request_descs(cfg)
    row = cfg["row_bytes"]
    offsets, sizes = table[kinds, 0], table[kinds, 1]
    local = np.asarray(local_rows, dtype=np.int64)[rows] * row + offsets
    remote = np.asarray(remote_rows, dtype=np.int64)[rows] * row + offsets
    return local.astype(np.uint64), remote.astype(np.uint64), sizes.astype(np.uint64)


def plan_config(preset: str, precision: str, isl: int, pool_slack: float = 2.0,
                batch_max: int = 1, table_sets: int = DEFAULT_TABLE_SETS) -> dict:
    """One (preset, isl) point at the preset's served block size: per-group block
    counts, the row size (the widest group's bytes per block), descriptors and
    bytes per request, and a pool of rows for ``table_sets`` disjoint sets of
    ``batch_max`` requests plus head-room."""
    shape = PRESETS[preset]
    if precision not in shape["precisions"]:
        raise ValueError(f"{preset} runs {shape['precisions']}, not {precision}")
    if _is_regions(preset):
        # The decode side loads isl - 1 tokens (vLLM recomputes the last one); a state group
        # moves its one running-state block.
        row_bytes = shape["row_bytes"]
        groups = [dict(name=g["name"],
                       blocks=1 if g["state"] else max(1, math.ceil((isl - 1) / shape["block_tokens"])))
                  for g in shape["groups"]]
    else:
        row_bytes = max(group_block_bytes(g) for g in shape["groups"])
        groups = [dict(name=g["name"], blocks=group_blocks(g, isl)) for g in shape["groups"]]
    descs = sum(g["blocks"] for g in groups)
    per_request = max(pool_slack, batch_max * 1.25)
    pool_rows = int(descs * table_sets * per_request) + 8
    cfg = dict(
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
    if _is_regions(preset):  # the request's KV is the regions it moves, not whole rows
        sizes = np.asarray(desc_kinds(cfg), dtype=np.int64)[request_descs(cfg)[0], 1]
        cfg["req_bytes"] = int(sizes.sum())
    return cfg


def block_table(cfg: dict, seed: int, request: int = 0, table_set: int = 0) -> np.ndarray:
    """One request's row ids (``descs`` of them) in group order (deterministic, seed-keyed): a slice
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
    """Nearest-rank percentiles, the EP suite's rule (ep_harness.percentile)."""
    ordered = sorted(samples_ms)
    n = len(ordered)
    rank = lambda q: ordered[max(0, math.ceil(q * n) - 1)]  # noqa: E731
    return {
        "p50": rank(0.50),
        "p95": rank(0.95),
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
