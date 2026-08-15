#!/usr/bin/env python3
"""Replace PR #2585's fallback MLA mode stub with unified-v2's full helper."""

import os
import py_compile
from pathlib import Path


DIST = Path(os.environ.get("DIST", "/usr/local/lib/python3.12/dist-packages"))
path = DIST / "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
text = path.read_text()
old = '''def _aiter_mla_small_head_mode() -> str:
    """Return VLLM_ROCM_AITER_MLA_ASM_PADDING mode (auto/gluon/asm)."""
    import os

    return (os.environ.get("VLLM_ROCM_AITER_MLA_ASM_PADDING") or "auto").lower()
'''
new = '''@functools.lru_cache(maxsize=1)
def _gluon_mla_decode_supported() -> bool:
    """The small-head Gluon MLA decode kernel only has a gfx950 (CDNA4) build.

    Its tiling needs ~160 KiB of LDS, which exceeds CDNA3's 64 KiB, so on
    gfx942 there is no kernel to fall through to and selecting it asserts
    (``mla_gluon requires gfx950``). Restrict Gluon decode to gfx950; other
    archs use the asm persistent decode, which ``get_mla_padded_q`` makes
    correct for any 1..15 heads.
    """
    try:
        from vllm.platforms.rocm import on_gfx950
    except Exception:  # noqa: BLE001
        return False
    return on_gfx950()


def _aiter_mla_small_head_mode() -> str:
    """Small-head (<16) MLA decode kernel selection.

    Controlled by ``VLLM_ROCM_AITER_MLA_ASM_PADDING``:

    - ``"auto"`` (default): let the arch decide -- divisor head counts keep the
      Gluon decode where a build exists (gfx950), everything else (non-divisor
      counts and all counts on gfx942) uses the padded persistent-scheduling
      ASM decode.
    - ``"gluon"``: prefer the Gluon path wherever a build exists.
    - ``"asm"``: force the padded persistent-scheduling ASM decode.

    On gfx942 (no Gluon build) the ASM path is always used regardless of this
    setting; ``"gluon"`` there falls back to ASM with a one-time warning.
    """
    import os

    mode = (os.environ.get("VLLM_ROCM_AITER_MLA_ASM_PADDING") or "auto").lower()
    if mode == "gluon" and not _gluon_mla_decode_supported():
        logger.warning_once(
            "VLLM_ROCM_AITER_MLA_ASM_PADDING=gluon requested, but this device "
            "has no Gluon MLA decode build (Gluon requires gfx950); using the "
            "padded persistent-scheduling ASM decode instead."
        )
    return mode
'''

if "def _gluon_mla_decode_supported" not in text:
    if text.count(old) != 1:
        raise SystemExit("ERROR: MLA small-head helper stub missing or duplicated")
    text = text.replace(old, new, 1)
    # Match the measured unified-v2 source layout exactly: two blank lines
    # between top-level definitions.
    text = text.replace(
        "    return True\n\n\n\n\n@functools.lru_cache",
        "    return True\n\n\n@functools.lru_cache",
        1,
    )
    text = text.replace(
        "    return mode\n\nclass AiterMLABackend",
        "    return mode\n\n\nclass AiterMLABackend",
        1,
    )
    path.write_text(text)

if text.count("def _gluon_mla_decode_supported") != 1:
    raise SystemExit("ERROR: full MLA small-head helper missing or duplicated")
py_compile.compile(str(path), doraise=True)
print(f"full MLA small-head helper OK: {path}")
