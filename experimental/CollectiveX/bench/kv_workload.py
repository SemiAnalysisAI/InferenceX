#!/usr/bin/env python3
"""Workload model for the KV-cache transfer suite.

A transfer is one request's paged KV: `isl` tokens in `page_tokens`-sized
blocks, per layer, scattered over a `[layer][page]` pool by a seed-keyed random
block table per side (the post-fragmentation layout vLLM/SGLang address).
Presets name production shapes: ``mla`` = DeepSeek-V4-Pro (61 layers; per token
per layer the MQA latent 1x512 + rope 64 + the DSA indexer k cache 128 that
vLLM's connector merges into the transfer regions = 704 elements); ``gqa`` =
Qwen3-235B class (1024, 94). fp8 halves page bytes, which deepens the
per-descriptor regime, so it is a measured dimension, never derived from bf16.

Pattern correctness: every 256-byte chunk holds a byte derived from its own
offset, so any page's expected contents follow from the offset alone and one
pool serves every config geometry. Page offsets are always 256-multiples.
"""

from __future__ import annotations

import fcntl
import socket
import struct

import numpy as np

PRESETS = {
    "mla": dict(layers=61, elems_per_token_layer=704, model_class="deepseek-v4-pro"),
    "gqa": dict(layers=94, elems_per_token_layer=1024, model_class="qwen3-235b"),
}
PRECISION_BYTES = {"bf16": 2, "fp8": 1}


def plan_config(preset: str, precision: str, isl: int, page_tokens: int,
                pool_slack: float = 2.0) -> dict:
    """Resolve one (preset, precision, isl, page) point into concrete geometry."""
    shape = PRESETS[preset]
    bytes_per_token_layer = shape["elems_per_token_layer"] * PRECISION_BYTES[precision]
    page_bytes = page_tokens * bytes_per_token_layer
    if page_bytes % 256:
        raise ValueError(f"page_bytes {page_bytes} must be a multiple of 256 for the pattern")
    pages_req = (isl + page_tokens - 1) // page_tokens
    pool_pages = int(pages_req * pool_slack) + 8  # per layer; slack models fragmentation head-room
    return dict(
        preset=preset,
        precision=precision,
        isl=isl,
        page_tokens=page_tokens,
        layers=shape["layers"],
        page_bytes=page_bytes,
        pages_req=pages_req,
        pool_pages=pool_pages,
        pool_bytes=shape["layers"] * pool_pages * page_bytes,
        req_bytes=shape["layers"] * pages_req * page_bytes,
        descs=shape["layers"] * pages_req,
    )


def block_table(cfg: dict, seed: int) -> np.ndarray:
    """A request's (deterministic, seed-keyed) random block table."""
    rng = np.random.default_rng(seed)
    return rng.permutation(cfg["pool_pages"])[: cfg["pages_req"]]


def table_seed(cfg: dict, side: str) -> int:
    """Both ranks derive both sides' tables from the config alone — no exchange."""
    base = cfg["isl"] * 31 + cfg["page_tokens"] + len(cfg["preset"]) * 7
    return base + (1000 if side == "local" else 0)


def page_offsets(cfg: dict, table: np.ndarray) -> np.ndarray:
    """Layer-major byte offsets (relative to the pool base) for a block table."""
    offsets = (
        np.arange(cfg["layers"], dtype=np.uint64)[:, None] * np.uint64(cfg["pool_pages"])
        + table[None, :].astype(np.uint64)
    ) * np.uint64(cfg["page_bytes"])
    return offsets.reshape(-1)


def desc_array(base: int, cfg: dict, table: np.ndarray, dev: int) -> np.ndarray:
    """(addr, len, devId) uint64 rows for descriptor-list APIs (NIXL's numpy form)."""
    out = np.empty((cfg["descs"], 3), dtype=np.uint64)
    out[:, 0] = np.uint64(base) + page_offsets(cfg, table)
    out[:, 1] = cfg["page_bytes"]
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


def verify_transfer(read8, cfg: dict, dst_table: np.ndarray, src_table: np.ndarray,
                    samples: int = 16, seed: int = 7) -> tuple[bool, str]:
    """On the destination pool: page (L, dst[i]) must hold the source pool's
    pattern at (L, src[i])'s offset. Covers both directions; ``read8(offset)``
    returns 8 destination-pool bytes (see kv_pool)."""
    rng = np.random.default_rng(seed)
    pool_pages, page_bytes = cfg["pool_pages"], cfg["page_bytes"]
    for _ in range(samples):
        layer = int(rng.integers(cfg["layers"]))
        i = int(rng.integers(len(dst_table)))
        expected = _chunk_byte((layer * pool_pages + int(src_table[i])) * page_bytes)
        got = bytes(read8((layer * pool_pages + int(dst_table[i])) * page_bytes))
        if any(value != expected for value in got):
            return False, f"layer={layer} i={i} expected={expected} got={list(got)}"
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
