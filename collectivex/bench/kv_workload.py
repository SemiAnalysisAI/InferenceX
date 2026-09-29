#!/usr/bin/env python3
"""Workload model for the KV-cache transfer suite, following vLLM.

A transfer is one request's KV in the shape vLLM's NixlConnector posts for
DeepSeek-V4-Pro (validated against vLLM 32ad1400d7). vLLM builds every
DSV4 cache (the three compressed-attention MLA caches, the sliding-window
caches, and the fp32 compressor states) into ONE backing allocation whose
block rows are as wide as the widest cache group; the groups overlay each
other from byte 0 because a block id is owned by one group at a time. Every
page is padded to the 576 B FlashMLA alignment, so no per-layer view is
contiguous and the connector registers one region whose descriptor is a
whole row (`(storage_addr, storage.nbytes() // num_blocks)` in
nixl/base_worker.py). A request therefore moves one full row per block id:
``ceil(L/256)`` rows for the MLA group plus each sliding-window group's tail
(``cdiv(window, block) + 1`` blocks, clipped to the prompt), all drawn from
one shared block pool so the rows are distinct. Seed-keyed random tables
scatter those rows over the pool, the block-granular fragmentation a real
allocator produces; each rep takes a fresh table set, as each request a
decode worker admits carries fresh block ids.

The dtype mix is architectural (fp8_ds_mla states, fp8 indexer, fp32
compressor states), so the preset pins precision to "fp8", and sparse MLA
serves only block size 256.

Pattern: every 8-byte word holds its own word index XORed with the owning
rank's salt mix, so a word's expected value follows from its SOURCE offset and
source rank alone. No two words in a pool match: a wrong block, a shifted copy,
a partial descriptor, or a loopback (own salt) all fail verification.
"""

from __future__ import annotations

import fcntl
import math
import socket
import struct

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
        vllm_commit="32ad1400d7",
        # vLLM's kv_cache_groups: the MLA specs unify into one group; the
        # sliding-window specs group by (block_size, window) and the 61 SWA
        # layers split 31/30 on the uniform-group gcd. (cache, layers) each,
        # in the order vLLM lays each group's pages out from row byte 0.
        groups=(
            dict(name="mla", block_tokens=256, window=None,
                 caches=(("c128a", 31), ("c4a-indexer", 30), ("c4a", 30))),
            dict(name="swa-a", block_tokens=64, window=128, caches=(("swa", 31),)),
            dict(name="swa-b", block_tokens=64, window=128, caches=(("swa", 30),)),
            dict(name="c4a-state", block_tokens=4, window=8,
                 caches=(("c4a-indexer-state", 30), ("c4a-state", 30))),
            dict(name="c128a-state", block_tokens=8, window=128,
                 caches=(("c128a-state", 31),)),
        ),
    ),
}
DEFAULT_TABLE_SETS = 4


def _round_up(value: int, align: int) -> int:
    return -(-value // align) * align


def page_bytes(cache: str) -> int:
    block, per_state, state_bytes = DSV4_CACHES[cache]
    return _round_up(block // per_state * state_bytes, ALIGNMENT)


def group_block_bytes(group: dict) -> int:
    return sum(layers * page_bytes(cache) for cache, layers in group["caches"])


def group_blocks(group: dict, isl: int) -> int:
    """Block ids vLLM transfers for this group: every block of a full-attention
    group, the window's tail (cdiv(window, block) + 1, clipped) otherwise."""
    blocks = math.ceil(isl / group["block_tokens"])
    if group["window"] is None:
        return blocks
    return min(math.ceil(group["window"] / group["block_tokens"]) + 1, blocks)


def layer_layout(preset: str) -> list[dict]:
    """Per group, each layer's page (offset inside the row, bytes) in the
    order vLLM packs them; the per-layer entries a layer-registering
    connector posts."""
    out = []
    for group in PRESETS[preset]["groups"]:
        offset, layers = 0, []
        for cache, count in group["caches"]:
            size = page_bytes(cache)
            for _ in range(count):
                layers.append((offset, size))
                offset += size
        out.append(dict(name=group["name"], layers=layers))
    return out


def layer_entries(cfg: dict, rows) -> tuple[np.ndarray, np.ndarray]:
    """(offsets, sizes) of one request's per-layer pages: for every layer of
    every group, one entry per block of that group (layer-major, the order
    vLLM's MooncakeConnector walks its per-layer regions). Pages carry no row
    padding, so a group narrower than the row leaves the row's tail unmoved."""
    rows = np.asarray(rows, dtype=np.uint64)
    offsets, sizes = [], []
    for (_, start, end), group in zip(group_slices(cfg), layer_layout(cfg["preset"])):
        block_base = rows[start:end] * np.uint64(cfg["row_bytes"])
        for offset, size in group["layers"]:
            offsets.append(block_base + np.uint64(offset))
            sizes.append(np.full(end - start, size, dtype=np.uint64))
    return np.concatenate(offsets), np.concatenate(sizes)


def plan_config(preset: str, precision: str, isl: int, block_tokens: int,
                pool_slack: float = 2.0, batch_max: int = 1,
                table_sets: int = DEFAULT_TABLE_SETS) -> dict:
    """Resolve one (preset, precision, isl, block size) point: per-group block
    counts, the row size (the widest group's bytes per block), descriptors and
    bytes per request, and a pool of rows large enough for ``table_sets``
    disjoint sets of ``batch_max`` requests plus fragmentation head-room."""
    shape = PRESETS[preset]
    if precision not in shape["precisions"]:
        raise ValueError(f"{preset} runs {shape['precisions']}, not {precision}")
    if block_tokens != shape["block_tokens"]:
        raise ValueError(f"{preset} is served at block size {shape['block_tokens']} "
                         f"(sparse MLA supports no other), not {block_tokens}")
    row_bytes = max(group_block_bytes(g) for g in shape["groups"])
    groups = [dict(name=g["name"], blocks=group_blocks(g, isl)) for g in shape["groups"]]
    descs = sum(g["blocks"] for g in groups)
    per_request = max(pool_slack, batch_max * 1.25)
    pool_rows = int(descs * table_sets * per_request) + 8
    return dict(
        preset=preset,
        precision=precision,
        isl=isl,
        page_tokens=block_tokens,
        layers=shape["model_layers"],
        row_bytes=row_bytes,
        page_bytes=row_bytes,  # one descriptor: a full block row
        groups=groups,
        descs=descs,
        req_bytes=descs * row_bytes,
        batch_max=batch_max,
        table_sets=table_sets,
        pool_rows=pool_rows,
        pool_bytes=pool_rows * row_bytes,
    )


def block_table(cfg: dict, seed: int, request: int = 0, table_set: int = 0) -> np.ndarray:
    """One request's row ids (group order, deterministic, seed-keyed): a slice
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


def page_offsets(cfg: dict, rows) -> np.ndarray:
    """Byte offsets (relative to the pool base) of a request's row descriptors."""
    return np.asarray(rows, dtype=np.uint64) * np.uint64(cfg["row_bytes"])


def desc_sizes(cfg: dict) -> np.ndarray:
    return np.full(cfg["descs"], cfg["row_bytes"], dtype=np.uint64)


def group_slices(cfg: dict) -> list[tuple[str, int, int]]:
    """(group name, start, end) of each group's rows inside a request's table."""
    out, start = [], 0
    for group in cfg["groups"]:
        out.append((group["name"], start, start + group["blocks"]))
        start += group["blocks"]
    return out


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


def iface_ipv4(iface: str) -> str:
    """IPv4 of a named interface (SIOCGIFADDR); the TCP bootstrap address."""
    packed = struct.pack("256s", iface.encode()[:15])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])
