#!/usr/bin/env python3
"""Let DSpark + padded-head decode use aiter's persistent MLA metadata.

The gfx950 fp8 ASM decode kernel reports::

    mla_decode_stage1_asm_fwd: only support gqa_ratio=16 fp8 mla decoding with
    qo_len <= 4 and qo_len>4 in persistent mode on gfx950

DSpark n=7 verifies 8 tokens per step, so persistent mode is mandatory. The K3
fork already builds persistent metadata for uniform multi-token decode (it added
``pad_uniform_mtp`` and the ``uni_seqlen_qo`` plumbing), but gates it on::

    self._mtp_decode_qlen = num_speculative_tokens + 1
        if speculative_config.method in ("mtp", "deepseek_mtp") else 1

    use_persistent_metadata = (
        self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
        and max_qo_len >= 1
        and max_qo_len <= self._mtp_decode_qlen
    )

Two assumptions in there exclude us:

1. ``method in ("mtp", "deepseek_mtp")`` -- K3 uses ``dspark``, whose
   verification step has exactly the same uniform ``num_spec + 1`` shape, so it
   falls to ``_mtp_decode_qlen = 1`` and every qlen > 1 is rejected.
2. ``num_heads >= 16`` -- written when every sub-16-head decode went to Gluon
   ("Small-head (<16) decode takes the Gluon paths and never consumes it").
   apply_vllm_aiter_mla_head_pad.py changes that: head counts that do not divide
   16 (K3 TP8 -> 12) are now padded and routed to the ASM kernel, which does
   consume the persistent metadata.

Apply AFTER apply_vllm_aiter_mla_head_pad.py. Divisor small-head models (e.g.
Kimi-K2.5 at 8 heads/rank) still go to Gluon and keep the original gate.

Idempotent. Fails loudly if an anchor moves.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MARKER = "# INFERENCEX-PERSISTENT-MTP"

EDITS: list[tuple[str, str, str]] = [
    (
        "_mtp_decode_qlen: treat dspark like mtp",
        """            and speculative_config.method in ("mtp", "deepseek_mtp")""",
        """            # INFERENCEX-PERSISTENT-MTP: dspark verification has the same
            # uniform num_spec+1 decode shape as mtp/deepseek_mtp.
            and speculative_config.method in ("mtp", "deepseek_mtp", "dspark")""",
    ),
    (
        "use_persistent_metadata: allow padded (non-divisor) head counts",
        """        use_persistent_metadata = (
            self.num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            and max_qo_len >= 1
            and max_qo_len <= self._mtp_decode_qlen
        )""",
        f"""        {MARKER}
        # Head counts that do not divide 16 are zero-padded to 16 and take the
        # ASM kernel (see apply_vllm_aiter_mla_head_pad.py), so they consume the
        # persistent metadata too. Divisor counts still go to Gluon.
        _m = AiterMLAHelper._AITER_MIN_MLA_HEADS
        use_persistent_metadata = (
            (self.num_heads >= _m or _m % self.num_heads != 0)
            and max_qo_len >= 1
            and max_qo_len <= self._mtp_decode_qlen
        )""",
    ),
]


def main() -> None:
    import vllm  # noqa: PLC0415

    path = Path(vllm.__file__).parent / "v1/attention/backends/mla/rocm_aiter_mla.py"
    if not path.is_file():
        sys.exit(f"[persistent-mtp] ERROR: not found: {path}")

    src = path.read_text()
    if MARKER in src:
        print(f"[persistent-mtp] already applied: {path}")
        return

    for desc, anchor, repl in EDITS:
        n = src.count(anchor)
        if n != 1:
            sys.exit(
                f"[persistent-mtp] ERROR: anchor for '{desc}' matched {n} times "
                f"(expected 1) in {path}. This patch targets the Kimi-K3 fork "
                f"(VLLM_K3_FORK_REF); update it if the fork moved."
            )
        src = src.replace(anchor, repl, 1)

    backup = path.with_suffix(path.suffix + ".persistmtp.bak")
    if not backup.exists():
        backup.write_text(path.read_text())
    path.write_text(src)
    ast.parse(src)
    print(f"[persistent-mtp] Patched {path} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main()
