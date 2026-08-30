#!/usr/bin/env python3
"""Log how vLLM buckets KV-cache layers, so padding blow-ups can be attributed.

Kimi-K3 on ROCM_AITER_MLA pays ~1.65x the KV bytes/token that TRITON_MLA pays
(55.1 vs 33.3 KB/token measured at conc 8, dummy weights, bf16 KV). Solving the
"Add N padding layers, may waste at most X%" warnings backwards pins it on the
layer bucketing in ``_get_kv_cache_groups_uniform_page_size``:

    TRITON  buckets {69, 29}      group_size 29 -> 29 MLA layer-slots
    AITER   buckets {69, 24, 5}   group_size 24 -> 48 MLA layer-slots

48/29 = 1.655, which is the whole difference. The 5-layer bucket is padded to 24
(19 wasted layer-slots, the "380%" warning).

What is NOT yet known is *why* the 24 and 5 stop sharing a bucket, because
MultiHeadLatentAttention.get_kv_cache_spec() reads nothing backend-specific and
MLAAttentionSpec.merge() already tolerates the draft's
``non_causal_multi_token_decode=True`` via ``any(...)``. So do not "fix" the
merge before seeing the specs: forcing layers with genuinely different page
layouts into one group would silently corrupt KV rather than save memory.

This patch is log-only. It prints every bucket's size, representative layer
names, and full spec repr just before the group size is chosen, which names both
the bucket that is the draft and the field that differs.
"""
from __future__ import annotations

from pathlib import Path

PATCH_MARKER = "INFERENCEX_PATCH_KV_GROUP_DEBUG"

OLD = """    bucket_sizes = [len(layers) for layers in layer_buckets]
    max_num_layers = max(bucket_sizes)"""

NEW = f"""    # {PATCH_MARKER}: attribute KV padding waste to a specific bucket/field.
    for _ix_i, (_ix_names, _ix_specs) in enumerate(zip(layer_buckets, spec_buckets)):
        logger.info(
            "[kv-group-debug] bucket %d: %d layers | first=%s last=%s | spec=%r",
            _ix_i,
            len(_ix_names),
            _ix_names[0],
            _ix_names[-1],
            _ix_specs[0],
        )

    bucket_sizes = [len(layers) for layers in layer_buckets]
    max_num_layers = max(bucket_sizes)"""

TAIL_OLD = """    grouped_layers = []
    for layers in layer_buckets:"""

TAIL_NEW = f"""    logger.info(
        "[kv-group-debug] bucket_sizes=%s group_size=%s",
        [len(layers) for layers in layer_buckets],
        group_size,
    )  # {PATCH_MARKER}
    grouped_layers = []
    for layers in layer_buckets:"""


def find_target() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent / "v1" / "core" / "kv_cache_utils.py"


def main() -> None:
    path = find_target()
    if not path.is_file():
        raise SystemExit(f"[kv-group-debug] kv_cache_utils.py not found: {path}")

    text = path.read_text()
    if PATCH_MARKER in text:
        print(f"[kv-group-debug] Already applied: {path}")
        return
    for anchor in (OLD, TAIL_OLD):
        if anchor not in text:
            raise SystemExit(f"[kv-group-debug] anchor not found in {path}")

    text = text.replace(OLD, NEW, 1).replace(TAIL_OLD, TAIL_NEW, 1)
    path.write_text(text)
    print(f"[kv-group-debug] Patched {path}")


if __name__ == "__main__":
    main()
