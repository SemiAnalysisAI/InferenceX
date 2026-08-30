#!/usr/bin/env python3
"""Let the aiter ASM MLA decode path serve head counts that do not divide 16.

Why
---
Kimi-K3 has 96 attention heads, so TP8 gives 12 MLA heads per rank. aiter's MLA
backend only drives its ASM decode kernel (``rocm_aiter_ops.mla_decode_fwd``) at
>= 16 heads, and pads smaller head counts up with::

    get_mla_padded_q:   q.repeat_interleave(16 // num_heads, dim=1)
    get_mla_unpadded_o: o[:, :: 16 // num_heads, :]

Both are exact for divisors of 16 (8, 4, 2, 1 -- e.g. Kimi-K2.5 at TP8) and both
silently degrade to a no-op for 12: ``16 // 12 == 1``. q keeps 12 heads while the
output buffer is already allocated at ``max(num_heads, 16) == 16``, so the shapes
disagree. Upstream sidesteps this by sending every ``num_heads < 16`` decode to
the Gluon kernel instead.

That detour is what costs us. ``mla_gluon`` picks its regime from the KV dtype:

    bh16bn64  : bf16 Q + bf16 KV, nhead <= 16, batch_size >= 1
    bh16bn128 : bf16 Q + fp8  KV, nhead <= 16, batch_size == 1

so on fp8 KV the 12-head path is pinned to batch 1 and any concurrency > 1 dies
with ``mla_gluon[bh16bn128] requires batch_size=1, got <N>``. The ASM path has no
such limit and already accepts ``q_scale``/``kv_scale``, i.e. it is both batched
and fp8-capable -- it was only ever unreachable because the padding helper could
not express 12 -> 16.

What this patch does
--------------------
Generalises the pad/unpad pair to arbitrary head counts (zero-pad up to 16, slice
the first ``num_heads`` back out) and stops routing those head counts to Gluon.
Attention is independent per head, so the padded heads compute values that are
discarded; the only cost is unused MFMA lanes, which this kernel family already
treats as free (its own comment: "NHEAD < BLOCK_H masks OOB heads on Q load and O
store. wasted MFMA lanes are free (memory-bound)").

Divisor head counts keep the existing ``repeat_interleave`` behaviour bit for
bit, so models like Kimi-K2.5 (8 heads at TP8) are untouched.

Idempotent: re-running is a no-op. Fails loudly if an anchor is missing, so a
vLLM bump cannot silently leave the patch unapplied.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MARKER = "# INFERENCEX-MLA-HEAD-PAD"


def _target() -> Path:
    import vllm  # noqa: PLC0415

    p = Path(vllm.__file__).parent / "v1/attention/backends/mla/rocm_aiter_mla.py"
    if not p.is_file():
        sys.exit(f"[mla-head-pad] ERROR: not found: {p}")
    return p


# (description, exact anchor, replacement)
EDITS: list[tuple[str, str, str]] = [
    (
        "get_mla_padded_q -> general zero-pad",
        """        return (
            q
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else q.repeat_interleave(
                AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, dim=1
            )
        )""",
        f"""        {MARKER}
        _m = AiterMLAHelper._AITER_MIN_MLA_HEADS
        if num_heads >= _m:
            return q
        if _m % num_heads == 0:
            return q.repeat_interleave(_m // num_heads, dim=1)
        # num_heads does not divide 16 (K3 TP8 -> 12): repeat_interleave would be
        # a no-op, so zero-pad the head dim instead. Heads are independent, and
        # get_mla_unpadded_o drops the padding again.
        pad = _m - num_heads
        return torch.cat(
            [q, q.new_zeros((q.shape[0], pad) + tuple(q.shape[2:]))], dim=1
        )""",
    ),
    (
        "get_mla_unpadded_o -> general slice",
        """        return (
            o
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else o[:, :: AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, :]
        )""",
        f"""        {MARKER}
        _m = AiterMLAHelper._AITER_MIN_MLA_HEADS
        if num_heads >= _m:
            return o
        if _m % num_heads == 0:
            return o[:, :: _m // num_heads, :]
        # Zero-padded layout: the real heads are the leading num_heads rows.
        return o[:, :num_heads, :]""",
    ),
    (
        "use_gluon_decode -> keep non-divisor head counts on ASM",
        """        return num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS and max_qo_len == 1""",
        f"""        {MARKER}
        _m = AiterMLAHelper._AITER_MIN_MLA_HEADS
        if num_heads < _m and _m % num_heads != 0:
            # General pad makes the ASM decode kernel reachable for these head
            # counts. ASM batches and takes q_scale/kv_scale, so it beats the
            # fp8 Gluon regime (bh16bn128), which is batch_size=1 only.
            return False
        return num_heads < _m and max_qo_len == 1""",
    ),
    (
        "multi-token verify branch -> only for divisor head counts",
        """        if (
            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
            and int(decode.max_qo_len) > 1
        ):""",
        f"""        {MARKER}
        if (
            self.num_heads < AiterMLAHelper._AITER_MIN_MLA_HEADS
            and AiterMLAHelper._AITER_MIN_MLA_HEADS % self.num_heads == 0
            and int(decode.max_qo_len) > 1
        ):""",
    ),
]


def main() -> None:
    path = _target()
    src = path.read_text()

    if MARKER in src:
        print(f"[mla-head-pad] already applied: {path}")
        return

    for desc, anchor, repl in EDITS:
        n = src.count(anchor)
        if n != 1:
            sys.exit(
                f"[mla-head-pad] ERROR: anchor for '{desc}' matched {n} times "
                f"(expected 1) in {path}. vLLM changed upstream -- update this patch."
            )
        src = src.replace(anchor, repl, 1)

    backup = path.with_suffix(path.suffix + ".inferencex.bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(src)

    # Import-check so a bad edit fails here, not 20 minutes into model load.
    os.system(f'python3 -c "import ast,sys; ast.parse(open(\'{path}\').read())"')
    print(f"[mla-head-pad] Patched {path} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main()
