#!/usr/bin/env python3
"""Workload model for the KV-cache transfer suite.

A transfer is one request's paged KV, scattered over `[layer][page]` pools by
seed-keyed random block tables per side (the post-fragmentation layout
vLLM/SGLang address). A preset is a list of layer GROUPS because modern caches
are not uniform: DeepSeek-V4-Pro interleaves Compressed Sparse Attention
(30 layers, 4 tokens per compressed entry, plus an FP4 lightning-indexer key
cache) with Heavily Compressed Attention (31 layers, 128 tokens per entry),
and every layer keeps a 128-entry uncompressed sliding window. Each group is a
transfer region with its own entry granularity; the indexer and the window are
further regions.

Geometry for ``dsv4`` is what vLLM allocates when serving DeepSeek-V4-Pro: the
config's ``compress_ratios`` interleave 30 CSA layers (m=4) with 31 HCA layers
(m'=128); every compressed entry is vLLM's 576 B ``fp8_ds_mla`` slot (UE8M0
block-scaled fp8 packed with the bf16 RoPE dims — "page-size specs pick the
576B per-token slot"); the CSA lightning-indexer cache is 132 B per entry
(128 fp8 + 4 scale bytes, vLLM's default over the opt-in fp4 cache); and the
128-token sliding window is its own registered, paged cache spec on all 61
layers (DeepseekV4SWACache), so the connector transfers it too. The dtype mix
is architectural, so the preset pins precision to "fp8"; it computes to ~2% of
a GQA8-bf16 cache, matching the tech report.

Pattern correctness: byte at offset o of a pool is derived from o (constant per
256-byte chunk), so any page's expected contents follow from its offset alone,
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
        precisions=("fp8",),  # vLLM's fp8_ds_mla slots + fp8 indexer, baked in
        groups=(
            dict(name="csa", layers=30, tokens_per_entry=4, entry_bytes=576,
                 indexer_bytes=132),
            dict(name="hca", layers=31, tokens_per_entry=128, entry_bytes=576,
                 indexer_bytes=0),
        ),
        window=dict(tokens=128, entry_bytes=576),
    ),
}


def plan_config(preset: str, precision: str, isl: int, page_tokens: int,
                pool_slack: float = 2.0, batch_max: int = 1) -> dict:
    """Resolve one (preset, precision, isl, page) point into concrete regions.

    Every region gets: layers, page_bytes (one descriptor's size), pages_req
    (descriptors per layer for one request), pool_pages (sized so ``batch_max``
    concurrent requests hold disjoint pages, plus fragmentation head-room), and
    its base offset in the one contiguous pool allocation.
    """
    shape = PRESETS[preset]
    if precision not in shape["precisions"]:
        raise ValueError(f"{preset} runs {shape['precisions']}, not {precision}")
    pool_slack = max(pool_slack, batch_max * 1.25)
    regions = []
    offset = 0

    def add_region(name, layers, page_bytes, pages_req, pool_pages):
        nonlocal offset
        regions.append(dict(name=name, layers=layers, page_bytes=page_bytes,
                            pages_req=pages_req, pool_pages=pool_pages, base=offset))
        offset += layers * pool_pages * page_bytes

    for group in shape["groups"]:
        entry_bytes = group["entry_bytes"]
        entries_req = math.ceil(isl / group["tokens_per_entry"])
        entries_per_page = max(1, page_tokens // group["tokens_per_entry"])
        pages_req = math.ceil(entries_req / entries_per_page)
        pool_pages = int(pages_req * pool_slack) + 8
        add_region(group["name"], group["layers"],
                   entries_per_page * entry_bytes, pages_req, pool_pages)
        if group["indexer_bytes"]:
            add_region(group["name"] + "-idx", group["layers"],
                       entries_per_page * group["indexer_bytes"], pages_req, pool_pages)
    window = shape["window"]
    if window is not None:
        # The recent-token window cache: paged like everything else, capped at
        # its window size regardless of ISL.
        window_tokens = min(isl, window["tokens"])
        pages_req = math.ceil(window_tokens / page_tokens)
        add_region("window", sum(group["layers"] for group in shape["groups"]),
                   page_tokens * window["entry_bytes"], pages_req,
                   int(pages_req * pool_slack) + 8)

    return dict(
        preset=preset,
        precision=precision,
        isl=isl,
        page_tokens=page_tokens,
        layers=sum(group["layers"] for group in shape["groups"]),
        page_bytes=regions[0]["page_bytes"],  # primary kv region, for row labels
        regions=regions,
        pool_bytes=offset,
        req_bytes=sum(r["layers"] * r["pages_req"] * r["page_bytes"] for r in regions),
        descs=sum(r["layers"] * r["pages_req"] for r in regions),
    )


def block_table(cfg: dict, seed: int, request: int = 0) -> dict:
    """Per-region block tables (deterministic, seed-keyed): region name -> the
    random page permutation a fragmented allocator would hand the request.
    Requests in one batch slice disjoint ranges of a single permutation, as a
    real allocator's live requests never alias pages."""
    rng = np.random.default_rng(seed)
    tables = {}
    for region in cfg["regions"]:
        low = request * region["pages_req"]
        tables[region["name"]] = (
            rng.permutation(region["pool_pages"])[low : low + region["pages_req"]]
        )
        if len(tables[region["name"]]) < region["pages_req"]:
            raise ValueError(f"pool too small for batch request {request} "
                             f"in region {region['name']}")
    return tables


def table_seed(cfg: dict, side: str) -> int:
    """Both ranks derive both sides' tables from the config alone — no exchange."""
    base = cfg["isl"] * 31 + cfg["page_tokens"] + len(cfg["preset"]) * 7
    return base + (1000 if side == "local" else 0)


def page_offsets(cfg: dict, tables: dict) -> np.ndarray:
    """Layer-major byte offsets (relative to the pool base) across all regions."""
    parts = []
    for region in cfg["regions"]:
        offsets = (
            np.arange(region["layers"], dtype=np.uint64)[:, None]
            * np.uint64(region["pool_pages"])
            + tables[region["name"]][None, :].astype(np.uint64)
        ) * np.uint64(region["page_bytes"]) + np.uint64(region["base"])
        parts.append(offsets.reshape(-1))
    return np.concatenate(parts)


def desc_sizes(cfg: dict) -> np.ndarray:
    """Per-descriptor byte sizes aligned with page_offsets' ordering."""
    return np.concatenate([
        np.full(region["layers"] * region["pages_req"], region["page_bytes"],
                dtype=np.uint64)
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
    """On the destination pool: page (region, L, dst[i]) must hold the source
    pool's pattern at (region, L, src[i])'s offset. ``read8(offset)`` returns 8
    destination-pool bytes (see kv_pool). Compared per byte, so any region
    alignment (e.g. the 132 B indexer entries) verifies exactly."""
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        region = cfg["regions"][int(rng.integers(len(cfg["regions"])))]
        dst, src = dst_tables[region["name"]], src_tables[region["name"]]
        layer = int(rng.integers(region["layers"]))
        i = int(rng.integers(len(dst)))
        src_off = (layer * region["pool_pages"] + int(src[i])) * region["page_bytes"] \
            + region["base"]
        dst_off = (layer * region["pool_pages"] + int(dst[i])) * region["page_bytes"] \
            + region["base"]
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
