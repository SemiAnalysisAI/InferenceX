#!/usr/bin/env python3
"""Force the in-repo Triton sparse paged-prefill (extend) kernel by disabling the
compiled AITER ``pa_sparse_prefill_opus`` path — DIAGNOSTIC ISOLATION TEST.

Why: dsv4 FP4 MTP (EAGLE) greedy accuracy regresses on the pure-TP8/EP1 mi355x
disagg *decode* topology (gsm8k 0.9613 base -> 0.8855@d1 / 0.8529@d2), while the
DEP8 MTP arms pass at ~0.96 and base spec-none passes on the *identical*
pure-TP8/EP1 topology. Elimination:

  * base spec-none PASSES on the same pure-TP8/EP1 topology -> everything SHARED
    by base and verify is exonerated: the o_proj all-reduce, EP1 MoE, LM-head
    vocab gather, attn_sink, hidden-state gather, AND the single-token *decode*
    attention kernel (base decode uses ``sparse_attn_v4_paged_decode`` only).
  * a disagg DECODE node never runs the extend/prefill attention kernel for base
    decode — only MTP's TARGET_VERIFY reaches ``sparse_attn_v4_paged_prefill``
    (runtime.prefill). So "base is lossless" does NOT exonerate the extend kernel.
  * every Python-level verify input (metadata, gather, sampling) is head-count
    agnostic. The ONLY thing differing between the PASSING DEP8 verify
    (H = n_local_heads = 128) and the FAILING pure-TP8 verify (H = 16) is the
    head count fed to the compiled ``pa_sparse_prefill_opus`` extend kernel.
  * base *prefill* also runs at H=16 on the prefill node and PASSES — but with an
    EMPTY prefix region (whole prompt is the extend). MTP verify is the only path
    that hits a large prefix + small extend *simultaneously* at H=16. block_h=16
    is the AMD MFMA minimal head tile, so H=16 is exactly one minimal head-block —
    the corner the AITER OPUS kernel evidently mishandles in the combined path.

This patch flips ``_HAS_OPUS = is_gfx95_supported()`` (paged_prefill.py) to
``_HAS_OPUS = False`` so the dispatch (``if _HAS_OPUS: ... else: return
_sparse_attn_v4_paged_prefill_triton(...)``) routes the extend through the
in-repo Triton kernel, which is correct for arbitrary H. This changes ONLY the
extend-kernel implementation and nothing else — an unambiguous one-run test:
  * recovers to ~0.96  -> the AITER OPUS combined prefix+extend kernel at H=16 is
    the root cause, and forcing Triton is a deployable (perf-costly) fix;
  * stays ~0.85        -> the extend kernel is exonerated; drop the arm.

Located WITHOUT importing sglang (importing the srt attention stack drags in the
distributed / dp_attention modules and hung the prefill node on the startup
critical path -- see patch_eagle_verify.py). Filesystem lookup only.

Exit codes: 0 = applied or already-present; 2 = source not located / shape
unrecognized (nothing changed) -- caller decides whether that is fatal.
"""
import glob
import os
import site
import sys

MARKER = "_HAS_OPUS = False  # forced: diagnostic (see patch_force_triton_extend.py)"

# Relative path of the target within any sglang install layout.
_REL = os.path.join(
    "sglang", "srt", "layers", "attention", "dsv4", "unified_kv_kernels",
    "paged_prefill.py",
)

# The gate assignment inside the `try:` that imports the AITER OPUS kernel.
# Only this occurrence sets _HAS_OPUS from is_gfx95_supported(); the except-branch
# already sets it False, so a single, unique anchor.
ANCHOR = "    _HAS_OPUS = is_gfx95_supported()\n"
REPLACEMENT = "    " + MARKER + "\n"


def _candidate_paths():
    """Ordered, de-duplicated candidate locations for paged_prefill.py.

    Never imports sglang -- only inspects env, well-known image paths, sys.path,
    site-packages, and a bounded glob over the editable-install root.
    """
    seen = []

    def add(p):
        if p and p not in seen:
            seen.append(p)

    add(os.environ.get("SGLANG_PAGED_PREFILL"))
    add(os.path.join("/sgl-workspace/sglang/python", _REL))
    dirs = []
    for getter in (getattr(site, "getsitepackages", None),):
        if getter:
            try:
                dirs.extend(getter())
            except Exception:  # noqa: BLE001
                pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    dirs.extend(d for d in sys.path if d)
    for d in dirs:
        add(os.path.join(d, _REL))
    for pat in (
        os.path.join("/sgl-workspace", "*", _REL),
        os.path.join("/sgl-workspace/sglang", "*", _REL),
    ):
        for hit in glob.glob(pat):
            add(hit)
    return seen


def _find_target():
    for p in _candidate_paths():
        try:
            if os.path.isfile(p):
                return p
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> int:
    path = _find_target()
    if path is None:
        print(
            "[PATCH] paged_prefill: source file not found on any candidate path "
            "(set SGLANG_PAGED_PREFILL to override); NOT patched",
            flush=True,
        )
        return 2

    with open(path, encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print(
            f"[PATCH] paged_prefill: OPUS extend already forced off in {path}; "
            f"skipping",
            flush=True,
        )
        return 0

    n = src.count(ANCHOR)
    if n != 1:
        print(
            f"[PATCH] paged_prefill: `_HAS_OPUS = is_gfx95_supported()` anchor found "
            f"{n}x (expected 1) in {path}; source shape unrecognized -- NOT patched",
            flush=True,
        )
        return 2

    new = src.replace(ANCHOR, REPLACEMENT, 1)
    if MARKER not in new or new == src:
        print(
            f"[PATCH] paged_prefill: post-transform verification failed for {path}; "
            f"NOT patched",
            flush=True,
        )
        return 2

    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    print(
        f"[PATCH] paged_prefill: forced OPUS extend kernel OFF (Triton fallback) "
        f"in {path} -- diagnostic isolation of the AITER pa_sparse_prefill_opus "
        f"H=16 combined prefix+extend path",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
