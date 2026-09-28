#!/usr/bin/env python3
"""Workload model for the KV-cache transfer suite.

A transfer is one request's paged KV in the shape vLLM's packed DSV4 NIXL path
actually registers and posts: per cache group, the physical block is the
transfer unit, and one contiguous descriptor covers ALL of that group's layers
for the block (block-major `[block][layer]` layout, `packed_bytes = layers x
page_bytes` per descriptor). Fragmentation is real but block-granular:
seed-keyed random block tables per side scatter each request's blocks over the
pool, exactly what a fragmented allocator hands a connector. What this model
deliberately does NOT do is explode each (layer, page) into its own descriptor
— vLLM's connector asserts one descriptor per packed physical block, and the
per-(layer, page) shape inflates descriptor counts by ~2 orders of magnitude,
which inverts backend and fabric conclusions on descriptor-bound lanes.

Geometry for ``dsv4`` is transcribed from vLLM (validated against commit
32ad1400d7): every token-state is 584 B of content (448 B NoPE + 128 B RoPE +
8 B fp8 scale, the ``fp8_ds_mla`` layout), and each block's page is padded to
a 576 B multiple (FlashMLA packing — alignment applies at PAGE granularity,
not per state). The config's ``compress_ratios`` interleave 30 Compressed
Sparse Attention layers (4 tokens per state) with 31 Heavily Compressed
Attention layers (128 tokens per state); CSA layers add a lightning-indexer
cache (132 B per state: 128 fp8 + 4 scale bytes); and all 61 layers keep a
128-token sliding window whose block size is FIXED at 64 tokens because the
window shares its physical tensor with the CSA cache (a 256-token CSA block is
64 states, so the shared tensor's block covers 64 window tokens — the window
page equals the CSA page byte for byte). HCA's 128-token states force the
model block size to a multiple of 128; vLLM serves DSV4 at 256. The dtype mix
is architectural, so the preset pins precision to "fp8".

Pattern correctness: byte at offset o of a pool is derived from o (constant per
256-byte chunk), so any block's expected contents follow from its offset alone,
at any alignment.
"""

from __future__ import annotations

import fcntl
import math
import socket
import struct

import numpy as np

PRESETS = {
    "dsv4": dict(
        model_class="deepseek-v4-pro",
        precisions=("fp8",),  # vLLM's fp8_ds_mla states + fp8 indexer, baked in
        model_layers=61,
        alignment=576,  # vLLM pads each block's page to this (FlashMLA packing)
        groups=(
            dict(name="c4a", layers=30, tokens_per_state=4, state_bytes=584),
            dict(name="c4a-idx", layers=30, tokens_per_state=4, state_bytes=132),
            dict(name="c128a", layers=31, tokens_per_state=128, state_bytes=584),
            dict(name="swa", layers=61, tokens_per_state=1, state_bytes=584,
                 block_tokens=64, window_tokens=128),
        ),
    ),
}


def _round_up(value: int, align: int) -> int:
    return -(-value // align) * align


def plan_config(preset: str, precision: str, isl: int, block_tokens: int,
                pool_slack: float = 2.0, batch_max: int = 1) -> dict:
    """Resolve one (preset, precision, isl, block size) point into regions.

    A region is one vLLM cache group. Every region gets: layers, page_bytes
    (one layer's padded page for one block), packed_bytes (the transfer unit —
    one descriptor covering all the group's layers for one physical block),
    blocks_req (descriptors for one request), pool_blocks (sized so
    ``batch_max`` concurrent requests hold disjoint blocks, plus fragmentation
    head-room), and its base offset in the one contiguous pool allocation.
    """
    shape = PRESETS[preset]
    if precision not in shape["precisions"]:
        raise ValueError(f"{preset} runs {shape['precisions']}, not {precision}")
    pool_slack = max(pool_slack, batch_max * 1.25)
    regions = []
    offset = 0
    for group in shape["groups"]:
        group_block = group.get("block_tokens", block_tokens)
        if group_block < group["tokens_per_state"] \
                or group_block % group["tokens_per_state"]:
            raise ValueError(
                f"{preset} block size {block_tokens} does not hold whole "
                f"{group['name']} states ({group['tokens_per_state']} tokens each)")
        states = group_block // group["tokens_per_state"]
        page_bytes = _round_up(states * group["state_bytes"], shape["alignment"])
        packed_bytes = group["layers"] * page_bytes
        tokens = min(isl, group["window_tokens"]) if "window_tokens" in group else isl
        blocks_req = math.ceil(tokens / group_block)
        pool_blocks = int(blocks_req * pool_slack) + 8
        regions.append(dict(name=group["name"], layers=group["layers"],
                            block_tokens=group_block, page_bytes=page_bytes,
                            packed_bytes=packed_bytes, blocks_req=blocks_req,
                            pool_blocks=pool_blocks, base=offset))
        offset += pool_blocks * packed_bytes

    return dict(
        preset=preset,
        precision=precision,
        isl=isl,
        page_tokens=block_tokens,  # row label: the model block size in tokens
        layers=shape["model_layers"],
        page_bytes=regions[0]["packed_bytes"],  # one primary-region descriptor
        regions=regions,
        pool_bytes=offset,
        req_bytes=sum(r["blocks_req"] * r["packed_bytes"] for r in regions),
        descs=sum(r["blocks_req"] for r in regions),
    )


def block_table(cfg: dict, seed: int, request: int = 0) -> dict:
    """Per-region block tables (deterministic, seed-keyed): region name -> the
    random block permutation a fragmented allocator would hand the request.
    Requests in one batch slice disjoint ranges of a single permutation, as a
    real allocator's live requests never alias blocks."""
    rng = np.random.default_rng(seed)
    tables = {}
    for region in cfg["regions"]:
        low = request * region["blocks_req"]
        tables[region["name"]] = (
            rng.permutation(region["pool_blocks"])[low : low + region["blocks_req"]]
        )
        if len(tables[region["name"]]) < region["blocks_req"]:
            raise ValueError(f"pool too small for batch request {request} "
                             f"in region {region['name']}")
    return tables


def table_seed(cfg: dict, side: str) -> int:
    """Both ranks derive both sides' tables from the config alone — no exchange."""
    base = cfg["isl"] * 31 + cfg["page_tokens"] + len(cfg["preset"]) * 7
    return base + (1000 if side == "local" else 0)


def page_offsets(cfg: dict, tables: dict) -> np.ndarray:
    """Block-major byte offsets (relative to the pool base) across all regions:
    one offset per packed physical block, the descriptor vLLM posts."""
    parts = []
    for region in cfg["regions"]:
        offsets = (tables[region["name"]].astype(np.uint64)
                   * np.uint64(region["packed_bytes"]) + np.uint64(region["base"]))
        parts.append(offsets)
    return np.concatenate(parts)


def desc_sizes(cfg: dict) -> np.ndarray:
    """Per-descriptor byte sizes aligned with page_offsets' ordering."""
    return np.concatenate([
        np.full(region["blocks_req"], region["packed_bytes"], dtype=np.uint64)
        for region in cfg["regions"]
    ])


def desc_array(base: int, cfg: dict, tables: dict, dev: int) -> np.ndarray:
    """(addr, len, devId) uint64 rows for descriptor-list APIs (NIXL's numpy form)."""
    out = np.empty((cfg["descs"], 3), dtype=np.uint64)
    out[:, 0] = np.uint64(base) + page_offsets(cfg, tables)
    out[:, 1] = desc_sizes(cfg)
    out[:, 2] = dev
    return out


def _chunk_byte(offset: int) -> int:
    return ((offset >> 8) * 131 + 7) & 0xFF


def fill_pattern(pool_u8) -> None:
    """Paint the offset-derived pattern over the whole pool (torch uint8 tensor)."""
    import torch

    chunks = pool_u8.numel() // 256
    view = pool_u8[: chunks * 256].view(chunks, 256)
    vals = ((torch.arange(chunks, device=pool_u8.device, dtype=torch.int64) * 131 + 7) & 0xFF)
    view.copy_(vals.to(torch.uint8)[:, None].expand(chunks, 256))


def verify_transfer(read8, cfg: dict, dst_tables: dict, src_tables: dict,
                    samples: int = 16, seed: int = 7) -> tuple[bool, str]:
    """On the destination pool: packed block (region, dst[i]) must hold the
    source pool's pattern at (region, src[i])'s offset. Each sample probes one
    layer's page inside the packed block, so the checks range over the whole
    descriptor. ``read8(offset)`` returns 8 destination-pool bytes (see
    kv_pool). Compared per byte, so any page alignment verifies exactly."""
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        region = cfg["regions"][int(rng.integers(len(cfg["regions"])))]
        dst, src = dst_tables[region["name"]], src_tables[region["name"]]
        layer = int(rng.integers(region["layers"]))
        i = int(rng.integers(len(dst)))
        delta = layer * region["page_bytes"]
        src_off = int(src[i]) * region["packed_bytes"] + region["base"] + delta
        dst_off = int(dst[i]) * region["packed_bytes"] + region["base"] + delta
        expected = bytes(_chunk_byte(src_off + j) for j in range(8))
        got = bytes(read8(dst_off))
        if got != expected:
            return False, (f"region={region['name']} layer={layer} i={i} "
                           f"expected={list(expected)} got={list(got)}")
    return True, ""


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
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packed = struct.pack("256s", iface.encode()[:15])
    return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])
