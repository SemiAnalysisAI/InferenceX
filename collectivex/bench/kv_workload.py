#!/usr/bin/env python3
"""KV-transfer workload: the block layout vLLM registers for Kimi-K3.1-Pro, the descriptors its
NixlConnector posts for one request, tables and verification."""

from __future__ import annotations

import math

import numpy as np

DEFAULT_TABLE_SETS = 4

# Kimi-K3.1-Pro on one rank of a TP16 engine, as the vllm/vllm-openai:kimi-k31 image (vLLM
# 0.1.dev24216) lays it out with kv-cache-dtype fp8, an mxfp4 indexer, float32 KDA state and the
# DSpark draft: 1024-token blocks, one allocation of 18,800,640-byte rows with every cache group
# overlaid from row byte 0. A GB200 production run logged the same shape (88 regions,
# block lengths {425472, 589824, 69632}, KDA state (32256, 393216)); docs/kv-transfer.md.
INDEXER = 69_632          # 24 DSA indexers: (128/2 + 128/32) B x 1024 tokens
MLA = 589_824             # 24 MLA layers and 5 draft layers: (512 + 64) B x 1024 tokens
KDA = 425_472             # one KDA layer's state on one of 16 ranks
# NIXL reads a KDA state as its three conv projections and the recurrent state.
KDA_PARTS = ((0, 10_752), (10_752, 10_752), (21_504, 10_752), (32_256, 393_216))
ATTENTION = tuple([(k * INDEXER, INDEXER) for k in range(24)]
                  + [(24 * INDEXER + m * MLA, MLA) for m in range(29)])
KDA_SLOTS = 35            # 69 KDA layers in two overlaid groups of 35 and 34

PRESETS = {
    "k31": dict(
        model_class="kimi-k3.1-pro",
        precisions=("fp8",),
        model_layers=93,
        block_tokens=1024,
        vllm_version="0.1.dev24216 (vllm/vllm-openai:kimi-k31)",
        tp_size=16,
        row_bytes=18_800_640,  # the attention group's 18,776,064 B, padded to its stride alignment
        # (row offset, bytes) of what vLLM registers per block: its 88 regions, then each KDA
        # region's four parts.
        kinds=(tuple((j * KDA, KDA) for j in range(KDA_SLOTS)) + ATTENTION
               + tuple((j * KDA + off, size) for j in range(KDA_SLOTS) for off, size in KDA_PARTS)),
        # A request moves every prompt block's attention pages and one running-state block per
        # KDA group. kinds index into the tuple above.
        groups=(
            dict(name="attention", state=False,
                 kinds=tuple(range(KDA_SLOTS, KDA_SLOTS + len(ATTENTION)))),
            dict(name="kda-a", state=True, kinds=tuple(range(88, 88 + 4 * 35))),
            dict(name="kda-b", state=True, kinds=tuple(range(88, 88 + 4 * 34))),
        ),
    ),
}


def request_descs(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(kind index, position in the request's row list) of each descriptor one request posts,
    in posting order."""
    kinds, rows, first = [], [], 0
    for group, plan in zip(PRESETS[cfg["preset"]]["groups"], cfg["groups"]):
        members = np.asarray(group["kinds"], dtype=np.int64)
        for row in range(first, first + plan["blocks"]):
            kinds.append(members)
            rows.append(np.full(len(members), row, dtype=np.int64))
        first += plan["blocks"]
    return np.concatenate(kinds), np.concatenate(rows)


def nixl_entries(cfg: dict, local_rows, remote_rows) -> tuple[np.ndarray, ...]:
    """(local offsets, remote offsets, sizes) as vLLM's NixlConnector posts one request: each
    block's regions, in its own row on each side."""
    table = np.asarray(PRESETS[cfg["preset"]]["kinds"], dtype=np.int64)
    kinds, rows = request_descs(cfg)
    row = cfg["row_bytes"]
    offsets, sizes = table[kinds, 0], table[kinds, 1]
    local = np.asarray(local_rows, dtype=np.int64)[rows] * row + offsets
    remote = np.asarray(remote_rows, dtype=np.int64)[rows] * row + offsets
    return local.astype(np.uint64), remote.astype(np.uint64), sizes.astype(np.uint64)


def plan_config(preset: str, precision: str, isl: int, pool_slack: float = 2.0,
                batch_max: int = 1, table_sets: int = DEFAULT_TABLE_SETS) -> dict:
    """One (preset, isl) point: per-group block counts, the rows and bytes one request moves,
    and a pool of rows for ``table_sets`` disjoint sets of ``batch_max`` requests plus
    head-room. The decode side loads isl - 1 tokens (vLLM recomputes the last one); a state
    group moves its one running-state block."""
    shape = PRESETS[preset]
    if precision not in shape["precisions"]:
        raise ValueError(f"{preset} runs {shape['precisions']}, not {precision}")
    blocks = max(1, math.ceil((isl - 1) / shape["block_tokens"]))
    groups = [dict(name=g["name"], blocks=1 if g["state"] else blocks) for g in shape["groups"]]
    rows = sum(g["blocks"] for g in groups)
    pool_rows = int(rows * table_sets * max(pool_slack, batch_max * 1.25)) + 8
    cfg = dict(
        preset=preset,
        isl=isl,
        page_tokens=shape["block_tokens"],
        layers=shape["model_layers"],
        row_bytes=shape["row_bytes"],
        groups=groups,
        rows=rows,
        batch_max=batch_max,
        table_sets=table_sets,
        pool_rows=pool_rows,
        pool_bytes=pool_rows * shape["row_bytes"],
    )
    kinds = np.asarray(shape["kinds"], dtype=np.int64)[request_descs(cfg)[0]]
    cfg["descs"], cfg["req_bytes"] = len(kinds), int(kinds[:, 1].sum())
    return cfg


def block_table(cfg: dict, seed: int, request: int = 0, table_set: int = 0) -> np.ndarray:
    """One request's row ids in group order (deterministic, seed-keyed): a slice of one
    random permutation of the pool's rows. Every (table set, request) slices a disjoint
    range, as a real allocator's live requests never alias blocks."""
    rng = np.random.default_rng(seed)
    low = (table_set * cfg["batch_max"] + request) * cfg["rows"]
    rows = rng.permutation(cfg["pool_rows"])[low : low + cfg["rows"]]
    if len(rows) < cfg["rows"]:
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
